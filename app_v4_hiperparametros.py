# ============================================================
# ANÁLISE DE INTELIGÊNCIA COMPUTACIONAL — REGRESSÃO E CLASSIFICAÇÃO
# Versão reestruturada: pipeline seguro, CV sem vazamento, classificação, agrupamento e otimização de hiperparâmetros
# ============================================================

from __future__ import annotations

import io
import json
import math
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
try:
    import plotly.express as px
except ImportError as exc:
    raise ImportError(
        "A dependência 'plotly' não está instalada. "
        "Instale as dependências com: pip install -r requirements.txt"
    ) from exc

import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    ParameterGrid,
    ParameterSampler,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.discovery import all_estimators

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None


APP_TITLE = "Análise de Inteligência Computacional — Regressão, Classificação e Agrupamento"
DEFAULT_RANDOM_STATE = 42
CURATED_CLASSIFIERS = [
    "LogisticRegression",
    "LinearDiscriminantAnalysis",
    "GaussianNB",
    "KNeighborsClassifier",
    "SVC",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "AdaBoostClassifier",
    "RidgeClassifier",
]


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise IC — Regressão, Classificação e Agrupamento",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            border-left: 4px solid #2a5298;
            padding: 15px;
            border-radius: 8px;
            margin: 5px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ESTADO
# ============================================================

def init_state() -> None:
    defaults = {
        "df_raw": None,
        "encoding_config": None,
        "encoding_preview": None,
        "results": None,
        "task_type": "regression",
        "encoding_ready": False,
        "prediction_result": None,
        "clustering_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# TRANSFORMERS CUSTOMIZADOS
# ============================================================

class IQRTransformer(BaseEstimator, TransformerMixin):
    """Winsoriza ou substitui outliers de colunas numéricas.

    O ajuste dos limites ocorre apenas no conjunto recebido pelo fit.
    Quando usado dentro de Pipeline + CV, os limites são recalculados
    em cada dobra, evitando vazamento.
    """

    def __init__(self, mode: str = "keep", factor: float = 1.5):
        self.mode = mode
        self.factor = factor

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        self.lower_ = np.empty(self.n_features_in_, dtype=float)
        self.upper_ = np.empty(self.n_features_in_, dtype=float)
        self.median_ = np.empty(self.n_features_in_, dtype=float)

        for j in range(self.n_features_in_):
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                self.lower_[j] = -np.inf
                self.upper_[j] = np.inf
                self.median_[j] = 0.0
                continue

            q1, q3 = np.nanpercentile(finite, [25, 75])
            iqr = q3 - q1
            self.lower_[j] = q1 - self.factor * iqr
            self.upper_[j] = q3 + self.factor * iqr
            self.median_[j] = float(np.nanmedian(finite))

        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()

        if self.mode == "keep":
            return X_arr

        low = self.lower_
        up = self.upper_
        mask_low = X_arr < low
        mask_up = X_arr > up

        if self.mode == "winsorize":
            X_arr = np.where(mask_low, low, X_arr)
            X_arr = np.where(mask_up, up, X_arr)
        elif self.mode == "median":
            mask = mask_low | mask_up
            X_arr[mask] = np.broadcast_to(self.median_, X_arr.shape)[mask]

        return X_arr


class ManualMapper(BaseEstimator, TransformerMixin):
    """Aplica mapeamento categórico definido pelo usuário."""

    def __init__(self, mapping: dict[str, float] | None = None, unknown_value: float = -1.0):
        self.mapping = mapping or {}
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        out = np.empty(arr.shape, dtype=float)
        for idx, value in np.ndenumerate(arr):
            key = "missing" if pd.isna(value) else str(value)
            out[idx] = float(self.mapping.get(key, self.unknown_value))
        return out


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Seleciona variáveis por |correlação de Pearson| com o alvo.

    A classe é colocada dentro do Pipeline, portanto fit() ocorre somente
    com os dados disponíveis em cada dobra do treinamento/CV.
    """

    def __init__(self, threshold: float = 0.10, max_features: int | None = None):
        self.threshold = threshold
        self.max_features = max_features

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_sr = pd.Series(y).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        # y do modelo deve ser numérico neste ponto.
        y_num = pd.to_numeric(y_sr, errors="coerce")
        if y_num.isna().all():
            # Para classificação, transforma os rótulos em códigos estáveis
            # apenas dentro da dobra atual.
            y_num = pd.Series(pd.factorize(y_sr.astype(str))[0], dtype=float)
        else:
            y_num = y_num.fillna(y_num.median())

        corr_values = {}
        for col in X_df.columns:
            x = pd.to_numeric(X_df[col], errors="coerce")
            corr_values[col] = float(x.corr(y_num)) if x.notna().sum() > 1 else 0.0

        corr = pd.Series(corr_values).abs().fillna(0).sort_values(ascending=False)

        if corr.empty:
            raise ValueError("Nenhuma variável disponível após o pré-processamento.")

        if self.max_features is not None and int(self.max_features) > 0:
            chosen = list(corr.head(int(self.max_features)).index)
        else:
            chosen = list(corr[corr >= float(self.threshold)].index)

        if not chosen:
            chosen = [corr.index[0]]

        self.selected_indices_ = [X_df.columns.get_loc(c) for c in chosen]
        self.selected_features_ = [str(c) for c in chosen]
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X):
        arr = np.asarray(X)
        return arr[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


# ============================================================
# UTILITÁRIOS DE DADOS
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=float,
        )
    except TypeError:
        # Compatibilidade com versões mais antigas do scikit-learn.
        return OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
            dtype=float,
        )


def detect_categorical_columns(df: pd.DataFrame, max_unique_int: int = 20) -> list[str]:
    cats: list[str] = []
    for col in df.columns:
        try:
            s = df[col]
            if (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)
            ):
                cats.append(col)
                continue

            if pd.api.types.is_integer_dtype(s):
                nun = s.nunique(dropna=True)
                if 1 < nun <= max_unique_int:
                    cats.append(col)
        except Exception:
            continue
    return cats


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or str(series.dtype) == "string"
        ):
            continue

        s = series.astype(str).str.strip()
        s_num = (
            s.str.replace(r"R\$\s*", "", regex=True)
            .str.replace(r"%\s*$", "", regex=True)
            .str.replace(" ", "", regex=False)
        )

        direct = pd.to_numeric(s_num, errors="coerce")
        brazilian = pd.to_numeric(
            s_num.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        non_null = series.notna().sum()
        if non_null == 0:
            continue

        if direct.notna().sum() / non_null >= 0.90:
            df[col] = direct
        elif brazilian.notna().sum() / non_null >= 0.90:
            df[col] = brazilian

    return df


def load_dataframe(uploaded_file, source_type: str, uci_id: int | None = None) -> pd.DataFrame:
    if source_type == "UCI":
        if fetch_ucirepo is None:
            raise ImportError("Pacote 'ucimlrepo' não está instalado.")

        ds = fetch_ucirepo(id=int(uci_id))
        if getattr(ds.data, "features", None) is not None and getattr(ds.data, "targets", None) is not None:
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        else:
            df = ds.data.original

        return coerce_numeric(df)

    if uploaded_file is None:
        raise ValueError("Nenhum arquivo foi enviado.")

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        raw = uploaded_file.read().decode("utf-8", errors="replace").lstrip("\ufeff")
        first = raw.splitlines()[0] if raw else ""
        counts = {
            ";": first.count(";"),
            ",": first.count(","),
            "\t": first.count("\t"),
            "|": first.count("|"),
        }
        sep = max(counts, key=counts.get) if max(counts.values()) else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Use CSV, XLS/XLSX ou JSON.")

    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    return coerce_numeric(df)


def parse_list(text: str) -> list[Any]:
    values: list[Any] = []

    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue

        low = token.lower()
        if low in {"scale", "auto"}:
            values.append(low)
            continue

        try:
            value = float(token)
            if value.is_integer() and "." not in token and "e" not in low:
                value = int(value)
            values.append(value)
        except ValueError as exc:
            raise ValueError(
                f"Valor inválido na lista: '{token}'. Use números ou 'scale'/'auto'."
            ) from exc

    if not values:
        raise ValueError("A lista de hiperparâmetros não pode ficar vazia.")

    return values


def validate_feature_selection(df: pd.DataFrame, features: list[str], target: str) -> None:
    if target not in df.columns:
        raise ValueError("A variável alvo selecionada não existe no dataset.")

    if not features:
        raise ValueError("Selecione pelo menos uma variável preditora.")

    if target in features:
        raise ValueError("A variável alvo não pode estar entre as preditoras.")

    if len(df[features].columns) != len(set(features)):
        raise ValueError("Existem variáveis preditoras duplicadas.")


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.select_dtypes(include=np.number).columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > up)).sum())

        rows.append(
            [
                col,
                q1,
                q3,
                iqr,
                low,
                up,
                n,
                100 * n / len(s),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Variável",
            "Q1",
            "Q3",
            "IQR",
            "Limite inf.",
            "Limite sup.",
            "Outliers",
            "%",
        ],
    )


# ============================================================
# CODIFICAÇÃO — CONFIGURAÇÃO E PRÉVIA
# ============================================================

def apply_categorical_encoding_preview(
    df: pd.DataFrame,
    encoding_map: dict[str, str],
    ordinal_orders: dict[str, list[str]],
    manual_maps: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Somente para prévia da aba de codificação.

    O dataset exibido aqui não é usado diretamente no treino.
    O treinamento usa Pipeline + ColumnTransformer e aprende transformações
    somente nos dados de treinamento.
    """

    out = df.copy()
    report: list[tuple[str, str, str]] = []

    for col, method in encoding_map.items():
        if col not in out.columns:
            continue

        if method == "drop":
            out = out.drop(columns=[col])
            report.append((col, "drop", "coluna removida"))
            continue

        s = out[col].astype("string").fillna("missing")

        if method == "onehot":
            enc = make_onehot_encoder()
            arr = enc.fit_transform(s.to_frame())
            names = [f"{col}__{cat}" for cat in enc.categories_[0]]
            out = pd.concat(
                [
                    out.drop(columns=[col]),
                    pd.DataFrame(arr, columns=names, index=out.index),
                ],
                axis=1,
            )
            report.append((col, "onehot", f"{len(names)} colunas"))

        elif method == "label":
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "label", "1 coluna"))

        elif method == "ordinal":
            order = ordinal_orders.get(col, sorted(s.unique().tolist()))
            enc = OrdinalEncoder(
                categories=[order],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "ordinal", "1 coluna"))

        elif method == "manual":
            mapping = manual_maps.get(col, {})
            out[col] = s.map(mapping).fillna(-1).astype(float)
            report.append((col, "manual", f"{len(mapping)} regras"))

    return out, report


def build_preprocessor(
    X: pd.DataFrame,
    encoding_map: dict[str, str],
    ordinal_orders: dict[str, list[str]],
    manual_maps: dict[str, dict[str, float]],
    outlier_mode: str,
) -> ColumnTransformer:
    categorical_columns = []
    onehot_columns = []
    label_columns = []
    ordinal_columns = []
    manual_columns = []
    dropped_columns = []

    for col, method in encoding_map.items():
        if col not in X.columns:
            continue
        if method == "drop":
            dropped_columns.append(col)
        elif method == "onehot":
            onehot_columns.append(col)
            categorical_columns.append(col)
        elif method == "label":
            label_columns.append(col)
            categorical_columns.append(col)
        elif method == "ordinal":
            ordinal_columns.append(col)
            categorical_columns.append(col)
        elif method == "manual":
            manual_columns.append(col)
            categorical_columns.append(col)

    numeric_columns = [
        c
        for c in X.columns
        if c not in categorical_columns and c not in dropped_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipe = Pipeline(
            [
                (
                    "outliers",
                    IQRTransformer(mode=outlier_mode),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_columns))

    if onehot_columns:
        onehot_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    make_onehot_encoder(),
                ),
            ]
        )
        transformers.append(("onehot", onehot_pipe, onehot_columns))

    if label_columns:
        label_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("label", label_pipe, label_columns))

    for col in ordinal_columns:
        order = ordinal_orders.get(col)
        if not order:
            raise ValueError(
                f"A ordem ordinal da coluna '{col}' não foi informada."
            )
        ordinal_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[order],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append((f"ordinal_{col}", ordinal_pipe, [col]))

    for col in manual_columns:
        mapping = manual_maps.get(col, {})
        manual_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "mapper",
                    ManualMapper(mapping=mapping, unknown_value=-1.0),
                ),
            ]
        )
        transformers.append((f"manual_{col}", manual_pipe, [col]))

    if not transformers:
        raise ValueError("Nenhuma variável ficou disponível para o modelo.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# REGRESSÃO
# ============================================================

def make_regression_pipeline(
    estimator,
    X_train: pd.DataFrame,
    encoding_map: dict[str, str],
    ordinal_orders: dict[str, list[str]],
    manual_maps: dict[str, dict[str, float]],
    outlier_mode: str,
    threshold: float,
    max_features: int | None,
) -> Pipeline:
    preprocessor = build_preprocessor(
        X_train,
        encoding_map,
        ordinal_orders,
        manual_maps,
        outlier_mode,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", CorrelationSelector(threshold, max_features)),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]
    )


def build_regression_models(grids, use_linear=True, use_svr_lin=True, use_svr_rbf=True):
    models = {}

    if use_linear:
        models["Regressão Linear"] = {
            "estimator": LinearRegression(),
            "params": {},
        }

    if use_svr_lin:
        models["SVR Linear"] = {
            "estimator": SVR(kernel="linear"),
            "params": {
                "regressor__C": grids["C_lin"],
                "regressor__epsilon": grids["eps_lin"],
            },
        }

    if use_svr_rbf:
        models["SVR RBF"] = {
            "estimator": SVR(kernel="rbf"),
            "params": {
                "regressor__C": grids["C_rbf"],
                "regressor__gamma": grids["gamma_rbf"],
                "regressor__epsilon": grids["eps_rbf"],
            },
        }

    return models


def randomized_iteration_limit(params: dict[str, list[Any]], requested: int) -> int:
    if not params:
        return 1

    combinations = 1
    for values in params.values():
        if isinstance(values, list):
            combinations *= max(len(values), 1)

    return max(1, min(int(requested), combinations))


def evaluate_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    encoding_map,
    ordinal_orders,
    manual_maps,
    outlier_mode,
    threshold=0.10,
    max_features=None,
    cv=5,
    search_method="grid",
    n_iter=20,
    random_state=42,
):
    results = {}
    cv_strategy = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    progress = st.progress(0)
    status = st.empty()

    items = list(models.items())
    for idx_model, (name, cfg) in enumerate(items, start=1):
        status.info(f"Treinando **{name}** ({idx_model}/{len(items)})...")

        try:
            pipe = make_regression_pipeline(
                clone(cfg["estimator"]),
                X_train,
                encoding_map,
                ordinal_orders,
                manual_maps,
                outlier_mode,
                threshold,
                max_features,
            )

            cv_rows = None

            if cfg["params"]:
                if search_method == "grid":
                    search = GridSearchCV(
                        pipe,
                        cfg["params"],
                        scoring="r2",
                        cv=cv_strategy,
                        n_jobs=-1,
                        return_train_score=True,
                        refit=True,
                        error_score="raise",
                    )
                elif search_method == "random":
                    n_random = randomized_iteration_limit(cfg["params"], n_iter)
                    search = RandomizedSearchCV(
                        pipe,
                        cfg["params"],
                        n_iter=n_random,
                        scoring="r2",
                        cv=cv_strategy,
                        n_jobs=-1,
                        return_train_score=True,
                        refit=True,
                        random_state=random_state,
                        error_score="raise",
                    )
                else:  # manual
                    manual_params = {
                        key: [values[0]]
                        for key, values in cfg["params"].items()
                    }
                    search = GridSearchCV(
                        pipe,
                        manual_params,
                        scoring="r2",
                        cv=cv_strategy,
                        n_jobs=-1,
                        return_train_score=True,
                        refit=True,
                        error_score="raise",
                    )

                search.fit(X_train, y_train)
                best = search.best_estimator_
                best_params = search.best_params_
                best_cv = float(search.best_score_)
                best_std = float(
                    search.cv_results_["std_test_score"][search.best_index_]
                )
                idx_best = search.best_index_
                cv_scores = np.asarray(
                    [
                        search.cv_results_[f"split{k}_test_score"][idx_best]
                        for k in range(cv)
                    ],
                    dtype=float,
                )
                cv_rows = (
                    pd.DataFrame(search.cv_results_)
                    .sort_values("rank_test_score")
                    .head(15)
                )
            else:
                cv_res = cross_validate(
                    pipe,
                    X_train,
                    y_train,
                    scoring={
                        "r2": "r2",
                        "rmse": "neg_root_mean_squared_error",
                        "mae": "neg_mean_absolute_error",
                    },
                    cv=cv_strategy,
                    n_jobs=-1,
                    error_score="raise",
                )
                best = pipe.fit(X_train, y_train)
                best_params = {}
                best_cv = float(cv_res["test_r2"].mean())
                best_std = float(cv_res["test_r2"].std())
                cv_scores = np.asarray(cv_res["test_r2"], dtype=float)
                cv_rows = None

            prediction = best.predict(X_test)
            mse = float(mean_squared_error(y_test, prediction))

            results[name] = {
                "best_model": best,
                "best_params": best_params,
                "cv_mean": best_cv,
                "cv_std": best_std,
                "cv_scores": cv_scores,
                "test_r2": float(r2_score(y_test, prediction)),
                "test_rmse": float(np.sqrt(mse)),
                "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_mse": mse,
                "y_pred": prediction,
                "search_table": cv_rows,
            }

        except Exception as exc:
            results[name] = {
                "error": str(exc),
            }

        progress.progress(idx_model / len(items))

    status.success("✅ Regressão concluída!")
    progress.empty()
    return results


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

@st.cache_data(show_spinner=False)
def get_all_classifiers() -> list[str]:
    valid = []
    for name, klass in all_estimators(type_filter="classifier"):
        try:
            klass()
            valid.append(name)
        except Exception:
            continue
    return valid


def get_classifier_registry() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "GaussianNB": GaussianNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "SVC": SVC,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "RidgeClassifier": RidgeClassifier,
    }


def get_classifier_param_grid(name: str) -> list[dict[str, list[Any]]] | dict[str, list[Any]] | None:
    """Espaços conservadores de hiperparâmetros para modelos comuns."""
    grids = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__max_iter": [200, 500],
        },
        "LinearDiscriminantAnalysis": [
            {"classifier__solver": ["svd"]},
            {
                "classifier__solver": ["lsqr"],
                "classifier__shrinkage": [None, "auto"],
            },
        ],
        "GaussianNB": {
            "classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        "KNeighborsClassifier": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "SVC": {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
        "DecisionTreeClassifier": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 2, 3, 4, 5, 7, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "RandomForestClassifier": {
            "classifier__n_estimators": [100, 200, 400],
            "classifier__max_depth": [None, 3, 5, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "ExtraTreesClassifier": {
            "classifier__n_estimators": [100, 200, 400],
            "classifier__max_depth": [None, 3, 5, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "GradientBoostingClassifier": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_depth": [1, 2, 3],
            "classifier__subsample": [0.8, 1.0],
        },
        "HistGradientBoostingClassifier": {
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_iter": [100, 200, 300],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
        },
        "AdaBoostClassifier": {
            "classifier__n_estimators": [50, 100, 200, 400],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        "RidgeClassifier": {
            "classifier__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        },
    }
    return grids.get(name)


def make_classification_pipeline(
    estimator,
    X_train: pd.DataFrame,
    encoding_map,
    ordinal_orders,
    manual_maps,
) -> Pipeline:
    preprocessor = build_preprocessor(
        X_train,
        encoding_map,
        ordinal_orders,
        manual_maps,
        outlier_mode="keep",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def _set_classifier_seed(estimator, random_state):
    if random_state is not None and hasattr(estimator, "random_state"):
        try:
            estimator.set_params(random_state=random_state)
        except Exception:
            pass
    return estimator


def _format_best_params(params: dict[str, Any]) -> str:
    if not params:
        return "—"
    return ", ".join(
        f"{k.replace('classifier__', '')}={v}" for k, v in params.items()
    )


def evaluate_classifiers(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier_names,
    encoding_map,
    ordinal_orders,
    manual_maps,
    cv=5,
    random_state=42,
    tune=False,
    search_method="grid",
    n_iter=20,
    tuning_metric="f1_weighted",
):
    registry = dict(all_estimators(type_filter="classifier"))
    registry.update(get_classifier_registry())

    results = {}
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    tuning_scorer = scoring[tuning_metric]

    progress = st.progress(0)
    status = st.empty()
    names = list(dict.fromkeys(classifier_names))

    for idx_model, name in enumerate(names, start=1):
        status.info(f"Avaliando **{name}** ({idx_model}/{len(names)})...")
        try:
            if name not in registry:
                raise ValueError(f"Classificador '{name}' não encontrado.")

            estimator = _set_classifier_seed(registry[name](), random_state)
            pipe = make_classification_pipeline(
                estimator,
                X_train,
                encoding_map,
                ordinal_orders,
                manual_maps,
            )

            param_grid = get_classifier_param_grid(name) if tune else None
            best_params = {}
            tuning_table = None
            tuned = False

            if tune and param_grid:
                if search_method == "grid":
                    search = GridSearchCV(
                        pipe,
                        param_grid,
                        scoring=tuning_scorer,
                        cv=cv_strategy,
                        n_jobs=-1,
                        refit=True,
                        return_train_score=False,
                        error_score="raise",
                    )
                else:
                    combinations = list(ParameterGrid(param_grid))
                    n_to_draw = min(int(n_iter), len(combinations))
                    search = RandomizedSearchCV(
                        pipe,
                        param_distributions=param_grid,
                        n_iter=n_to_draw,
                        scoring=tuning_scorer,
                        cv=cv_strategy,
                        n_jobs=-1,
                        refit=True,
                        random_state=random_state,
                        return_train_score=False,
                        error_score="raise",
                    )

                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                tuned = True
                tuning_table = (
                    pd.DataFrame(search.cv_results_)
                    .sort_values("rank_test_score")
                    .head(15)
                )
                cv_best = float(search.best_score_)
                cv_best_std = float(
                    search.cv_results_["std_test_score"][search.best_index_]
                )
            else:
                model = pipe.fit(X_train, y_train)

                cv_res = cross_validate(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )
                cv_best = float(cv_res[f"test_{tuning_metric}"].mean())
                cv_best_std = float(cv_res[f"test_{tuning_metric}"].std())

            if tuned:
                # Avaliação descritiva dos demais indicadores da melhor configuração.
                cv_res = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )

            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            test_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            test_balanced = balanced_accuracy_score(y_test, y_pred)

            results[name] = {
                "best_model": model,
                "tuned": tuned,
                "tuning_method": search_method if tuned else "none",
                "tuning_metric": tuning_metric,
                "best_params": best_params,
                "best_params_text": _format_best_params(best_params),
                "tuning_table": tuning_table,
                "cv_tuning_mean": cv_best,
                "cv_tuning_std": cv_best_std,
                "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
                "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
                "cv_f1_weighted_mean": float(cv_res["test_f1_weighted"].mean()),
                "cv_f1_weighted_std": float(cv_res["test_f1_weighted"].std()),
                "cv_f1_macro_mean": float(cv_res["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(cv_res["test_f1_macro"].std()),
                "cv_balanced_accuracy_mean": float(cv_res["test_balanced_accuracy"].mean()),
                "cv_acc_scores": cv_res["test_accuracy"].tolist(),
                "cv_f1_scores": cv_res["test_f1_weighted"].tolist(),
                "test_accuracy": float(test_accuracy),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1": float(test_f1),
                "test_f1_macro": float(test_f1_macro),
                "test_balanced_accuracy": float(test_balanced),
                "y_pred": y_pred,
            }

            try:
                if hasattr(model, "predict_proba"):
                    score_values = model.predict_proba(X_test)
                    if score_values.shape[1] == 2:
                        results[name]["test_roc_auc"] = float(
                            roc_auc_score(y_test, score_values[:, 1])
                        )
                    else:
                        results[name]["test_roc_auc"] = float(
                            roc_auc_score(
                                y_test, score_values,
                                multi_class="ovr", average="weighted",
                            )
                        )
                elif hasattr(model, "decision_function"):
                    score_values = model.decision_function(X_test)
                    if np.ndim(score_values) == 1:
                        results[name]["test_roc_auc"] = float(
                            roc_auc_score(y_test, score_values)
                        )
                    else:
                        results[name]["test_roc_auc"] = float(
                            roc_auc_score(
                                y_test, score_values,
                                multi_class="ovr", average="weighted",
                            )
                        )
            except Exception:
                results[name]["test_roc_auc"] = None

        except Exception as exc:
            results[name] = {"error": str(exc)}

        progress.progress(idx_model / len(names))

    status.success("✅ Classificação concluída!")
    progress.empty()
    return results

# ============================================================
# GRÁFICOS
# ============================================================

def regression_boxplot(valid: dict[str, dict]):
    rows = []

    for name, result in valid.items():
        for fold, score in enumerate(result["cv_scores"], start=1):
            rows.append(
                {
                    "Modelo": name,
                    "Fold": fold,
                    "R²": float(score),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return None, df

    fig = px.box(
        df,
        x="Modelo",
        y="R²",
        color="Modelo",
        points="all",
        hover_data=["Fold"],
        title="📦 R² por dobra da validação cruzada",
    )
    fig.update_layout(height=500, showlegend=False)
    return fig, df


def regression_scatter_plot(y_test, valid):
    rows = []

    for name, result in valid.items():
        pred = np.asarray(result["y_pred"])
        real = np.asarray(y_test)
        resid = real - pred

        for y_real, y_pred, residual in zip(real, pred, resid):
            rows.append(
                {
                    "Modelo": name,
                    "Real": y_real,
                    "Predito": y_pred,
                    "Resíduo": residual,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None

    fig_real = px.scatter(
        df,
        x="Real",
        y="Predito",
        color="Modelo",
        facet_col="Modelo",
        facet_col_wrap=2,
        title="Real × Predito",
        hover_data=["Resíduo"],
    )

    fig_resid = px.scatter(
        df,
        x="Predito",
        y="Resíduo",
        color="Modelo",
        facet_col="Modelo",
        facet_col_wrap=2,
        title="Resíduos",
    )
    fig_resid.add_hline(y=0, line_dash="dash")
    return fig_real, fig_resid


def classification_cv_plot(valid):
    rows = []
    for name, result in valid.items():
        for fold, score in enumerate(result["cv_f1_scores"], start=1):
            rows.append(
                {
                    "Classificador": name,
                    "Fold": fold,
                    "F1 ponderado": float(score),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return None, df

    fig = px.box(
        df,
        x="Classificador",
        y="F1 ponderado",
        color="Classificador",
        points="all",
        hover_data=["Fold"],
        title="📦 F1 ponderado por dobra da CV",
    )
    fig.update_layout(
        height=550,
        showlegend=False,
        xaxis_tickangle=-40,
    )
    return fig, df


def confusion_matrix_plot(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        labels={
            "x": "Predito",
            "y": "Real",
            "color": "Quantidade",
        },
        title=title,
        aspect="auto",
    )
    return fig



# ============================================================
# CLASSIFICAÇÃO DE NOVOS DADOS
# ============================================================

def predict_new_classification(
    model: Pipeline,
    new_df: pd.DataFrame,
    features: list[str],
    classes: list[str],
) -> pd.DataFrame:
    """Aplica um Pipeline já treinado a um novo dataset."""
    missing = [col for col in features if col not in new_df.columns]
    if missing:
        raise ValueError(
            "O novo dataset não possui as variáveis necessárias: "
            + ", ".join(missing)
        )

    X_new = new_df[features].copy()
    prediction = model.predict(X_new)

    labels = []
    for value in prediction:
        try:
            idx = int(value)
            labels.append(classes[idx] if 0 <= idx < len(classes) else str(value))
        except (TypeError, ValueError):
            labels.append(str(value))

    result = new_df.copy()
    result["Classe_Prevista"] = labels

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)
            result["Confianca"] = proba.max(axis=1)
    except Exception:
        pass

    return result


# ============================================================
# AGRUPAMENTO
# ============================================================

def make_clustering_preprocessor(
    X: pd.DataFrame,
    encoding_map,
    ordinal_orders,
    manual_maps,
    outlier_mode="keep",
) -> Pipeline:
    preprocessor = build_preprocessor(
        X, encoding_map, ordinal_orders, manual_maps, outlier_mode=outlier_mode
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
    ])


def calculate_external_cluster_metrics(reference, labels):
    reference = pd.Series(reference).astype("string").fillna("missing").reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    if len(reference) != len(labels):
        return {"adjusted_rand": None, "normalized_mutual_info": None}
    try:
        ari = float(adjusted_rand_score(reference, labels))
    except Exception:
        ari = None
    try:
        nmi = float(normalized_mutual_info_score(reference, labels))
    except Exception:
        nmi = None
    return {"adjusted_rand": ari, "normalized_mutual_info": nmi}


def calculate_cluster_metrics(X_array, labels):
    X_array = np.asarray(X_array)
    labels = np.asarray(labels)
    noise = labels == -1
    valid = ~noise
    X_valid = X_array[valid]
    labels_valid = labels[valid]
    unique = np.unique(labels_valid)

    result = {
        "n_clusters": int(len(unique)),
        "n_noise": int(noise.sum()),
        "noise_pct": float(noise.mean() * 100) if len(labels) else 0.0,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    if len(unique) < 2 or len(labels_valid) <= len(unique):
        return result

    for key, fn in [
        ("silhouette", silhouette_score),
        ("calinski_harabasz", calinski_harabasz_score),
        ("davies_bouldin", davies_bouldin_score),
    ]:
        try:
            result[key] = float(fn(X_valid, labels_valid))
        except Exception:
            pass
    return result


def clustering_registry_params(random_state=None):
    """Espaços de busca separados por algoritmo."""
    return {
        "K-Means": {
            "n_clusters": list(range(2, 11)),
            "init": ["k-means++", "random"],
            "n_init": [10, 20],
        },
        "Hierárquico": {
            "n_clusters": list(range(2, 11)),
            "linkage": ["ward", "complete", "average"],
        },
        "DBSCAN": {
            "eps": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
            "min_samples": [3, 5, 7, 10],
        },
        "Gaussian Mixture": {
            "n_components": list(range(2, 11)),
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "reg_covar": [1e-6, 1e-4, 1e-2],
        },
        "Birch": {
            "n_clusters": list(range(2, 11)),
            "threshold": [0.3, 0.5, 0.7, 1.0],
            "branching_factor": [25, 50],
        },
    }


def make_cluster_estimator(name, params, random_state=None):
    p = dict(params)
    if name == "K-Means":
        return KMeans(
            n_clusters=int(p.get("n_clusters", 3)),
            init=p.get("init", "k-means++"),
            n_init=int(p.get("n_init", 10)),
            random_state=random_state,
        )
    if name == "Hierárquico":
        linkage = p.get("linkage", "ward")
        # Ward exige distância euclidiana; não incluímos outra métrica para manter
        # compatibilidade entre versões recentes do scikit-learn.
        return AgglomerativeClustering(
            n_clusters=int(p.get("n_clusters", 3)),
            linkage=linkage,
        )
    if name == "DBSCAN":
        return DBSCAN(
            eps=float(p.get("eps", 0.5)),
            min_samples=int(p.get("min_samples", 5)),
        )
    if name == "Gaussian Mixture":
        return GaussianMixture(
            n_components=int(p.get("n_components", 3)),
            covariance_type=p.get("covariance_type", "full"),
            reg_covar=float(p.get("reg_covar", 1e-6)),
            random_state=random_state,
        )
    if name == "Birch":
        return Birch(
            n_clusters=int(p.get("n_clusters", 3)),
            threshold=float(p.get("threshold", 0.5)),
            branching_factor=int(p.get("branching_factor", 50)),
        )
    raise ValueError(f"Algoritmo de agrupamento desconhecido: {name}")


def cluster_objective(metrics: dict, metric: str):
    value = metrics.get(metric)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def tune_single_clustering_algorithm(
    name,
    X_transformed,
    tune=True,
    search_method="grid",
    n_iter=20,
    objective="silhouette",
    random_state=42,
):
    search_spaces = clustering_registry_params(random_state)
    space = search_spaces[name]

    if not tune:
        default = {
            "K-Means": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
            "Hierárquico": {"n_clusters": 3, "linkage": "ward"},
            "DBSCAN": {"eps": 0.5, "min_samples": 5},
            "Gaussian Mixture": {"n_components": 3, "covariance_type": "full", "reg_covar": 1e-6},
            "Birch": {"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
        }[name]
        candidates = [default]
    elif search_method == "grid":
        candidates = list(ParameterGrid(space))
    else:
        candidates = list(
            ParameterSampler(
                space,
                n_iter=min(int(n_iter), max(1, len(list(ParameterGrid(space))))),
                random_state=random_state,
            )
        )

    rows = []
    best = None
    best_value = -np.inf if objective != "davies_bouldin" else np.inf

    for params in candidates:
        try:
            model = make_cluster_estimator(name, params, random_state)
            labels = model.fit_predict(X_transformed)
            metrics = calculate_cluster_metrics(X_transformed, labels)
            value = cluster_objective(metrics, objective)
            if value is None:
                continue

            row = dict(params)
            row.update({
                "n_clusters_real": metrics["n_clusters"],
                "noise_pct": metrics["noise_pct"],
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
                "objective": value,
            })
            rows.append(row)

            improved = (
                value < best_value if objective == "davies_bouldin" else value > best_value
            )
            if improved:
                best_value = value
                best = {
                    "estimator": model,
                    "labels": np.asarray(labels),
                    "metrics": metrics,
                    "best_params": params,
                }
        except Exception:
            continue

    if best is None:
        raise ValueError(
            f"Nenhuma configuração válida encontrada para {name} com a métrica {objective}."
        )

    tuning_table = pd.DataFrame(rows)
    if not tuning_table.empty:
        tuning_table = tuning_table.sort_values(
            "objective", ascending=(objective == "davies_bouldin")
        ).head(20)

    best["tuned"] = bool(tune)
    best["tuning_method"] = search_method if tune else "none"
    best["tuning_metric"] = objective
    best["tuning_table"] = tuning_table
    return best


def fit_clustering_models(
    X,
    encoding_map,
    ordinal_orders,
    manual_maps,
    selected_algorithms,
    n_clusters=3,
    eps=0.5,
    min_samples=5,
    outlier_mode="keep",
    random_state=42,
    reference=None,
    tune=False,
    search_method="grid",
    n_iter=20,
    objective="silhouette",
):
    pipeline = make_clustering_preprocessor(
        X, encoding_map, ordinal_orders, manual_maps, outlier_mode
    )
    X_transformed = pipeline.fit_transform(X)

    results = {}
    for name in selected_algorithms:
        try:
            if tune:
                result = tune_single_clustering_algorithm(
                    name,
                    X_transformed,
                    tune=True,
                    search_method=search_method,
                    n_iter=n_iter,
                    objective=objective,
                    random_state=random_state,
                )
            else:
                defaults = {
                    "K-Means": {"n_clusters": n_clusters, "init": "k-means++", "n_init": 10},
                    "Hierárquico": {"n_clusters": n_clusters, "linkage": "ward"},
                    "DBSCAN": {"eps": eps, "min_samples": min_samples},
                    "Gaussian Mixture": {"n_components": n_clusters, "covariance_type": "full", "reg_covar": 1e-6},
                    "Birch": {"n_clusters": n_clusters, "threshold": 0.5, "branching_factor": 50},
                }
                params = defaults[name]
                model = make_cluster_estimator(name, params, random_state)
                labels = model.fit_predict(X_transformed)
                result = {
                    "estimator": model,
                    "labels": np.asarray(labels),
                    "metrics": calculate_cluster_metrics(X_transformed, labels),
                    "best_params": params,
                    "tuned": False,
                    "tuning_method": "none",
                    "tuning_metric": objective,
                    "tuning_table": None,
                }

            if reference is not None:
                result["metrics"].update(
                    calculate_external_cluster_metrics(reference, result["labels"])
                )

            results[name] = result
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return X_transformed, results


def clustering_scatter_plot(X_transformed, labels, title):
    X_transformed = np.asarray(X_transformed)
    labels = np.asarray(labels)
    if X_transformed.shape[1] == 1:
        coords = np.column_stack([X_transformed[:, 0], np.zeros(len(X_transformed))])
    elif X_transformed.shape[1] == 2:
        coords = X_transformed[:, :2]
    else:
        coords = PCA(n_components=2, random_state=0).fit_transform(X_transformed)
    plot_df = pd.DataFrame({
        "Componente 1": coords[:, 0],
        "Componente 2": coords[:, 1],
        "Cluster": labels.astype(str),
    })
    return px.scatter(
        plot_df,
        x="Componente 1",
        y="Componente 2",
        color="Cluster",
        title=title,
    )

# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_regression_target(y: pd.Series) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().all():
        raise ValueError(
            "A variável alvo escolhida para regressão não é numérica."
        )

    if y_num.isna().any():
        st.warning(
            "Há valores inválidos na variável alvo. As linhas com alvo ausente "
            "ou não numérico serão removidas antes do treino."
        )

    y_num = y_num.dropna()
    if len(y_num) < 10:
        raise ValueError("A regressão requer pelo menos 10 observações válidas.")

    return y_num


def validate_classification_target(y: pd.Series, cv: int):
    y_clean = y.astype("string").fillna("missing")
    counts = y_clean.value_counts()

    if len(counts) < 2:
        raise ValueError("A classificação requer pelo menos 2 classes.")

    if counts.min() < cv:
        raise ValueError(
            f"A classe com menos observações possui {int(counts.min())} "
            f"amostra(s), mas a CV exige pelo menos {cv}."
        )

    if counts.min() < 2:
        raise ValueError(
            "Cada classe precisa aparecer pelo menos duas vezes para permitir "
            "uma divisão treino/teste estratificada."
        )

    return y_clean


def resolve_random_state(choice: str, custom_value: int | None = None):
    if choice == "fixa":
        return DEFAULT_RANDOM_STATE
    if choice == "aleatoria":
        return None
    return int(custom_value)


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0;">🧠 Análise de Inteligência Computacional</h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
            Regressão, classificação e agrupamento com pré-processamento seguro
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuração")

    task_options = {
        "Regressão": "regression",
        "Classificação": "classification",
        "Agrupamento": "clustering",
    }
    task_label = st.radio(
        "Tipo de tarefa",
        list(task_options.keys()),
        horizontal=False,
    )
    st.session_state["task_type"] = task_options[task_label]

    st.divider()
    st.subheader("📁 Fonte de dados")

    source = st.radio(
        "Origem",
        ["Arquivo Local", "UCI Repository"],
    )

    uploaded_file = None
    uci_id = None

    if source == "Arquivo Local":
        uploaded_file = st.file_uploader(
            "Envie o arquivo",
            type=["csv", "xlsx", "xls", "json"],
        )
    else:
        uci_id = st.number_input(
            "ID do Dataset (UCI)",
            min_value=1,
            value=186,
            step=1,
        )

    if st.button("📥 Carregar Dados", width="stretch"):
        with st.spinner("Carregando dados..."):
            try:
                df = (
                    load_dataframe(uploaded_file, "file")
                    if source == "Arquivo Local"
                    else load_dataframe(None, "UCI", int(uci_id))
                )

                if df.empty:
                    raise ValueError("O dataset carregado está vazio.")

                st.session_state["df_raw"] = df
                st.session_state["encoding_config"] = None
                st.session_state["encoding_preview"] = None
                st.session_state["encoding_ready"] = False
                st.session_state["results"] = None
                st.session_state["prediction_result"] = None
                st.session_state["clustering_result"] = None

                st.success(
                    f"✅ Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas."
                )
            except Exception as exc:
                st.error(f"Erro ao carregar dados: {exc}")


if st.session_state["df_raw"] is None:
    st.info("👈 Use a barra lateral para carregar um conjunto de dados.")
    st.stop()


df_raw: pd.DataFrame = st.session_state["df_raw"]

tabs = st.tabs(
    [
        "📊 Diagnóstico",
        "🏷️ Codificação",
        "🎯 Modelagem",
        "📈 Resultados",
    ]
)


# ============================================================
# TAB 1 — DIAGNÓSTICO
# ============================================================

with tabs[0]:
    st.header("📊 Descrição dos Dados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", df_raw.shape[0])
    c2.metric("Colunas", df_raw.shape[1])
    c3.metric("Numéricas", len(df_raw.select_dtypes(include=np.number).columns))
    c4.metric("Categóricas", len(detect_categorical_columns(df_raw)))

    info_rows = []
    for col in df_raw.columns:
        info_rows.append(
            {
                "Coluna": col,
                "dtype": str(df_raw[col].dtype),
                "Não-nulos": int(df_raw[col].notna().sum()),
                "Nulos": int(df_raw[col].isna().sum()),
                "Únicos": int(df_raw[col].nunique(dropna=True)),
                "Amostra": " | ".join(
                    df_raw[col].dropna().astype(str).head(3).tolist()
                )[:80],
            }
        )

    st.subheader("Tipos de dados por coluna")
    st.dataframe(
        pd.DataFrame(info_rows),
        width="stretch",
        height=320,
    )

    with st.expander("📋 Estatísticas descritivas", expanded=True):
        st.dataframe(
            df_raw.describe(include="all").T,
            width="stretch",
        )

    with st.expander("🔍 Outliers IQR — diagnóstico"):
        outlier_df = iqr_outlier_summary(df_raw)
        if outlier_df.empty:
            st.info("Nenhuma coluna numérica disponível para o diagnóstico de IQR.")
        else:
            st.dataframe(outlier_df.round(4), width="stretch")

    st.subheader("📈 Visualizações")

    numeric_cols = list(df_raw.select_dtypes(include=np.number).columns)

    if numeric_cols:
        selected_hist = st.multiselect(
            "Colunas para histogramas",
            numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))],
        )

        if selected_hist:
            hist_long = df_raw[selected_hist].melt(
                var_name="Variável",
                value_name="Valor",
            )
            fig = px.histogram(
                hist_long,
                x="Valor",
                facet_col="Variável",
                facet_col_wrap=2,
                marginal="box",
                title="Distribuições numéricas",
            )
            st.plotly_chart(fig, width="stretch")

    if len(numeric_cols) >= 2:
        corr = df_raw[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=len(numeric_cols) <= 12,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de correlação",
        )
        st.plotly_chart(fig_corr, width="stretch")


# ============================================================
# TAB 2 — CODIFICAÇÃO
# ============================================================

with tabs[1]:
    st.header("🏷️ Configuração de Variáveis Categóricas")
    st.caption(
        "A tabela abaixo é apenas uma prévia. O treinamento real usa "
        "ColumnTransformer/Pipeline para ajustar cada transformação somente no treino."
    )

    categorical_cols = detect_categorical_columns(df_raw)

    if not categorical_cols:
        st.success(
            "Nenhuma coluna categórica foi detectada automaticamente. "
            "Nesse caso, todas as colunas selecionadas como preditoras serão tratadas como numéricas."
        )
        st.session_state["encoding_config"] = {
            "encoding_map": {},
            "ordinal_orders": {},
            "manual_maps": {},
        }
        st.session_state["encoding_preview"] = df_raw.copy()
        st.session_state["encoding_ready"] = True
    else:
        encoding_map: dict[str, str] = {}
        ordinal_orders: dict[str, list[str]] = {}
        manual_maps: dict[str, dict[str, float]] = {}

        st.write(f"**{len(categorical_cols)} coluna(s) categórica(s) detectada(s).**")

        for col in categorical_cols:
            with st.expander(f"🔸 {col}", expanded=False):
                values = (
                    df_raw[col]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
                values_sorted = sorted(values)

                a, b, c = st.columns(3)
                a.metric("Únicos", len(values_sorted))
                b.metric("Ausentes", int(df_raw[col].isna().sum()))
                c.metric("dtype", str(df_raw[col].dtype))

                st.caption("Primeiros valores")
                st.code(" | ".join(values_sorted[:50]) or "(sem valores)", language=None)

                method = st.radio(
                    "Método",
                    ["onehot", "label", "ordinal", "manual", "drop"],
                    format_func=lambda x: {
                        "onehot": "One-Hot — recomendado para categorias nominais",
                        "label": "Label — compatibilidade; impõe ordem numérica",
                        "ordinal": "Ordinal — use quando existe ordem real",
                        "manual": "Manual — mapeamento próprio",
                        "drop": "Descartar coluna",
                    }[x],
                    key=f"encoding_{col}",
                )

                encoding_map[col] = method

                if method == "ordinal":
                    order_text = st.text_input(
                        "Ordem (separada por vírgula)",
                        value=", ".join(values_sorted),
                        key=f"ordinal_{col}",
                    )
                    ordinal_orders[col] = [
                        value.strip()
                        for value in order_text.split(",")
                        if value.strip()
                    ]

                elif method == "manual":
                    suggestion = ", ".join(
                        f"{value}={idx}" for idx, value in enumerate(values_sorted)
                    )

                    map_text = st.text_input(
                        "Mapeamento categoria=código",
                        value=suggestion,
                        key=f"manual_{col}",
                    )

                    mapping = {}
                    invalid_tokens = []

                    for token in map_text.split(","):
                        token = token.strip()
                        if not token:
                            continue

                        if "=" not in token:
                            invalid_tokens.append(token)
                            continue

                        key, value = token.split("=", 1)
                        try:
                            mapping[key.strip()] = float(value.strip())
                        except ValueError:
                            invalid_tokens.append(token)

                    if invalid_tokens:
                        st.warning(
                            "Entradas ignoradas no mapeamento: "
                            + ", ".join(invalid_tokens)
                        )

                    manual_maps[col] = mapping

        if st.button(
            "✅ Salvar configuração de codificação",
            width="stretch",
            type="primary",
        ):
            try:
                preview, report = apply_categorical_encoding_preview(
                    df_raw,
                    encoding_map,
                    ordinal_orders,
                    manual_maps,
                )

                st.session_state["encoding_config"] = {
                    "encoding_map": encoding_map,
                    "ordinal_orders": ordinal_orders,
                    "manual_maps": manual_maps,
                }
                st.session_state["encoding_preview"] = preview
                st.session_state["encoding_ready"] = True

                st.success(
                    "✅ Configuração salva. As transformações serão ajustadas "
                    "automaticamente em cada treino/CV."
                )
                st.dataframe(
                    pd.DataFrame(
                        report,
                        columns=["Coluna", "Método", "Resultado"],
                    ),
                    width="stretch",
                )

                with st.expander("🔎 Prévia codificada"):
                    st.dataframe(preview.head(100), width="stretch")

            except Exception as exc:
                st.error(f"Erro na configuração: {exc}")


# ============================================================
# TAB 3 — MODELAGEM
# ============================================================

with tabs[2]:
    st.header("🎯 Configuração e Execução")

    if not st.session_state["encoding_ready"]:
        st.warning(
            "Configure a codificação na aba anterior. Mesmo em datasets sem categóricos, "
            "a configuração é marcada automaticamente."
        )
    else:
        encoding_config = st.session_state["encoding_config"] or {}
        encoding_map = encoding_config.get("encoding_map", {})
        ordinal_orders = encoding_config.get("ordinal_orders", {})
        manual_maps = encoding_config.get("manual_maps", {})
        all_columns = list(df_raw.columns)

        # ========================================================
        # REGRESSÃO
        # ========================================================
        if st.session_state["task_type"] == "regression":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox(
                    "🎯 Variável alvo",
                    all_columns,
                    index=len(all_columns) - 1,
                    key="target_selection_reg",
                )
            with c2:
                features = st.multiselect(
                    "📊 Variáveis preditoras",
                    [c for c in all_columns if c != target],
                    default=[c for c in all_columns if c != target][:10],
                    key="feature_selection_reg",
                )

            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                st.subheader("⚙️ Parâmetros de Regressão")
                c1, c2, c3 = st.columns(3)
                with c1:
                    outlier_mode = st.selectbox(
                        "Tratamento de outliers numéricos",
                        ["keep", "winsorize", "median"],
                        format_func=lambda x: {
                            "keep": "Manter",
                            "winsorize": "Winsorizar",
                            "median": "Substituir por mediana",
                        }[x],
                    )
                with c2:
                    threshold = st.number_input(
                        "Threshold de correlação",
                        0.0, 1.0, 0.10, 0.05,
                    )
                with c3:
                    max_feat_input = st.number_input(
                        "Máx. features (0 = todas)",
                        0, 200, 0,
                    )

                st.markdown("**Modelos**")
                c1, c2, c3 = st.columns(3)
                use_linear = c1.checkbox("Regressão Linear", value=True)
                use_svr_lin = c2.checkbox("SVR Linear", value=True)
                use_svr_rbf = c3.checkbox("SVR RBF", value=True)

                st.markdown("**Busca de hiperparâmetros**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    C_lin = st.text_input("C (SVR linear)", "0.1, 1, 10", key="C_lin_v3")
                    eps_lin = st.text_input("epsilon (SVR linear)", "0.01, 0.1, 0.5", key="eps_lin_v3")
                with c2:
                    C_rbf = st.text_input("C (SVR RBF)", "0.1, 1, 10", key="C_rbf_v3")
                    gamma_rbf = st.text_input("gamma (SVR RBF)", "scale, auto, 0.1, 1", key="gamma_rbf_v3")
                    eps_rbf = st.text_input("epsilon (SVR RBF)", "0.01, 0.1, 0.5", key="eps_rbf_v3")
                with c3:
                    search_method = st.selectbox(
                        "Método",
                        ["grid", "random", "manual"],
                        format_func=lambda x: {
                            "grid": "Grid Search",
                            "random": "Random Search",
                            "manual": "Manual — primeiro valor",
                        }[x],
                    )
                    n_iter = st.number_input("n_iter (random)", 2, 500, 20)
                    cv_reg = st.number_input("Folds da CV", 3, 10, 5)

                seed_choice = st.radio(
                    "Semente",
                    ["fixa", "aleatoria", "custom"],
                    format_func=lambda x: {
                        "fixa": "Fixa (42)",
                        "aleatoria": "Aleatória",
                        "custom": "Customizada",
                    }[x],
                    horizontal=True,
                    key="seed_reg_v3",
                )
                custom_seed = st.number_input(
                    "Valor",
                    0, 999999, 42,
                    disabled=seed_choice != "custom",
                    key="custom_seed_reg_v3",
                )

                if st.button("▶️ Executar Análise de Regressão", width="stretch", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        data = df_raw[features + [target]].copy()
                        y = pd.to_numeric(data[target], errors="coerce")
                        valid_mask = y.notna()
                        X = data.loc[valid_mask, features].copy()
                        y = y.loc[valid_mask].copy()
                        if len(y) < 10:
                            raise ValueError("Poucas observações válidas para regressão.")

                        grids = {
                            "C_lin": parse_list(C_lin),
                            "eps_lin": parse_list(eps_lin),
                            "C_rbf": parse_list(C_rbf),
                            "gamma_rbf": parse_list(gamma_rbf),
                            "eps_rbf": parse_list(eps_rbf),
                        }
                        models = build_regression_models(
                            grids, use_linear, use_svr_lin, use_svr_rbf
                        )
                        if not models:
                            raise ValueError("Selecione pelo menos um modelo.")

                        seed = resolve_random_state(seed_choice, int(custom_seed))
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=seed
                        )

                        results = evaluate_regression(
                            X_train, y_train, X_test, y_test, models,
                            encoding_map, ordinal_orders, manual_maps,
                            outlier_mode,
                            threshold=threshold,
                            max_features=int(max_feat_input) if max_feat_input > 0 else None,
                            cv=int(cv_reg),
                            search_method=search_method,
                            n_iter=int(n_iter),
                            random_state=seed,
                        )
                        st.session_state["results"] = {
                            "task": "regression",
                            "results": results,
                            "y_test": y_test,
                            "features": features,
                            "target": target,
                            "train_n": len(X_train),
                            "test_n": len(X_test),
                            "seed": seed,
                            "cv": int(cv_reg),
                            "encoding_config": encoding_config,
                        }
                        st.success("🎉 Regressão concluída. Abra a aba **Resultados**.")
                    except Exception as exc:
                        st.error(f"Não foi possível executar a regressão: {exc}")

        # ========================================================
        # CLASSIFICAÇÃO
        # ========================================================
        elif st.session_state["task_type"] == "classification":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox(
                    "🎯 Variável alvo",
                    all_columns,
                    index=len(all_columns) - 1,
                    key="target_selection_clf",
                )
            with c2:
                features = st.multiselect(
                    "📊 Variáveis preditoras",
                    [c for c in all_columns if c != target],
                    default=[c for c in all_columns if c != target][:10],
                    key="feature_selection_clf",
                )

            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                st.subheader("⚙️ Parâmetros de Classificação")
                c1, c2, c3 = st.columns(3)
                with c1:
                    cv_clf = st.number_input("Folds da CV", 3, 10, 5)
                with c2:
                    test_size_clf = st.number_input("Fração de teste", 0.10, 0.40, 0.20, 0.05)
                with c3:
                    catalog = st.selectbox(
                        "Catálogo",
                        ["Lista recomendada", "Todos os classificadores"],
                    )

                h1, h2, h3 = st.columns(3)
                with h1:
                    tune_clf = st.checkbox("🔧 Otimizar hiperparâmetros", value=True, key="tune_clf_v4")
                with h2:
                    tuning_method_clf = st.selectbox(
                        "Busca", ["grid", "random"],
                        format_func=lambda x: "Grid Search" if x == "grid" else "Random Search",
                        disabled=not tune_clf,
                        key="tuning_method_clf_v4",
                    )
                with h3:
                    n_iter_clf = st.number_input(
                        "n_iter (Random)", 2, 100, 20, 1,
                        disabled=(not tune_clf or tuning_method_clf != "random"),
                        key="n_iter_clf_v4",
                    )
                tuning_metric_clf = st.selectbox(
                    "Métrica de seleção dos hiperparâmetros",
                    ["f1_weighted", "f1_macro", "accuracy", "balanced_accuracy"],
                    format_func=lambda x: {
                        "f1_weighted": "F1 ponderado",
                        "f1_macro": "F1 macro",
                        "accuracy": "Acurácia",
                        "balanced_accuracy": "Balanced accuracy",
                    }[x],
                    disabled=not tune_clf,
                    key="tuning_metric_clf_v4",
                )

                seed_choice = st.radio(
                    "Semente",
                    ["fixa", "aleatoria", "custom"],
                    format_func=lambda x: {
                        "fixa": "Fixa (42)",
                        "aleatoria": "Aleatória",
                        "custom": "Customizada",
                    }[x],
                    horizontal=True,
                    key="seed_clf_v3",
                )
                custom_seed_clf = st.number_input(
                    "Semente customizada", 0, 999999, 42,
                    disabled=seed_choice != "custom",
                    key="custom_seed_clf_v3",
                )

                classifiers = CURATED_CLASSIFIERS if catalog == "Lista recomendada" else get_all_classifiers()
                chosen = st.multiselect(
                    "Classificadores para avaliar",
                    classifiers,
                    default=classifiers[: min(12, len(classifiers))],
                    key="chosen_classifiers_v3",
                )

                if st.button("▶️ Executar Classificação", width="stretch", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        data = df_raw[features + [target]].copy()
                        y_raw = validate_classification_target(data[target], int(cv_clf))
                        data = data.loc[y_raw.index]
                        X, y = data[features], y_raw

                        if not chosen:
                            raise ValueError("Selecione pelo menos um classificador.")

                        class_counts = y.value_counts()
                        if class_counts.min() < 2:
                            raise ValueError("Cada classe precisa ter pelo menos 2 observações.")
                        seed = resolve_random_state(seed_choice, int(custom_seed_clf))

                        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
                            X, y,
                            test_size=float(test_size_clf),
                            random_state=seed,
                            stratify=y,
                        )
                        classes = sorted(y_train_raw.astype(str).unique().tolist())
                        if not set(y_test_raw.astype(str)).issubset(set(classes)):
                            raise ValueError("Há uma classe presente no teste que não aparece no treino.")

                        label_to_int = {label: i for i, label in enumerate(classes)}
                        y_train = y_train_raw.astype(str).map(label_to_int).to_numpy()
                        y_test = y_test_raw.astype(str).map(label_to_int).to_numpy()

                        results = evaluate_classifiers(
                            X_train, y_train, X_test, y_test, chosen,
                            encoding_map, ordinal_orders, manual_maps,
                            cv=int(cv_clf),
                            random_state=seed,
                            tune=bool(tune_clf),
                            search_method=tuning_method_clf,
                            n_iter=int(n_iter_clf),
                            tuning_metric=tuning_metric_clf,
                        )
                        st.session_state["results"] = {
                            "task": "classification",
                            "results": results,
                            "y_test": y_test,
                            "features": features,
                            "target": target,
                            "classes": classes,
                            "train_n": len(X_train),
                            "test_n": len(X_test),
                            "seed": seed,
                            "cv": int(cv_clf),
                            "encoding_config": encoding_config,
                        }
                        st.session_state["prediction_result"] = None
                        st.success("🎉 Classificação concluída. Abra a aba **Resultados**.")
                    except Exception as exc:
                        st.error(f"Não foi possível executar a classificação: {exc}")

                # ====================================================
                # INFERÊNCIA EM NOVOS DADOS
                # ====================================================
                trained = st.session_state.get("results")
                if trained and trained.get("task") == "classification":
                    st.divider()
                    st.subheader("🔮 Classificar novos dados")
                    st.caption(
                        "Envie outro dataset com as mesmas variáveis preditoras. "
                        "A coluna alvo é opcional e será ignorada na previsão."
                    )
                    valid_models = {
                        name: res for name, res in trained["results"].items()
                        if "error" not in res
                    }
                    if valid_models:
                        model_name = st.selectbox(
                            "Modelo treinado",
                            list(valid_models.keys()),
                            key="prediction_model_v3",
                        )
                        new_file = st.file_uploader(
                            "Novo dataset",
                            type=["csv", "xlsx", "xls", "json"],
                            key="new_classification_file_v3",
                        )
                        if new_file is not None:
                            try:
                                new_data = load_dataframe(new_file, "file")
                                missing = [
                                    col for col in trained["features"]
                                    if col not in new_data.columns
                                ]
                                if missing:
                                    st.error(
                                        "Variáveis obrigatórias ausentes: "
                                        + ", ".join(missing)
                                    )
                                else:
                                    st.dataframe(
                                        new_data[trained["features"]].head(20),
                                        width="stretch",
                                    )
                                    if st.button(
                                        "🔮 Classificar arquivo",
                                        type="primary",
                                        width="stretch",
                                        key="classify_new_file_v3",
                                    ):
                                        predicted = predict_new_classification(
                                            valid_models[model_name]["best_model"],
                                            new_data,
                                            trained["features"],
                                            trained["classes"],
                                        )
                                        st.session_state["prediction_result"] = predicted
                                        st.success(
                                            f"✅ {len(predicted)} registro(s) classificados usando {model_name}."
                                        )
                            except Exception as exc:
                                st.error(f"Não foi possível classificar o novo arquivo: {exc}")

                        if st.session_state.get("prediction_result") is not None:
                            st.markdown("### Resultado da previsão")
                            st.dataframe(
                                st.session_state["prediction_result"],
                                width="stretch",
                                height=350,
                            )
                            st.download_button(
                                "📥 Baixar previsões (.csv)",
                                st.session_state["prediction_result"].to_csv(index=False).encode("utf-8"),
                                file_name=f"classificacoes_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                mime="text/csv",
                                width="stretch",
                                key="download_new_classification_v3",
                            )
                    else:
                        st.warning("Nenhum classificador treinado com sucesso está disponível.")

        # ========================================================
        # AGRUPAMENTO
        # ========================================================
        else:
            st.subheader("🧩 Agrupamento não supervisionado")
            st.caption(
                "Nenhuma variável alvo é exigida para treinar os clusters. "
                "Caso o dataset possua rótulos conhecidos, você pode fornecê-los "
                "somente para avaliação externa (ARI/NMI)."
            )

            cluster_features = st.multiselect(
                "📊 Variáveis para agrupamento",
                all_columns,
                default=all_columns,
                key="cluster_features_v3",
            )
            reference_options = ["Nenhuma (sem rótulo)"] + all_columns
            reference_target = st.selectbox(
                "🏷️ Rótulo de referência (opcional)",
                reference_options,
                key="cluster_reference_v3",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                n_clusters = st.number_input("Número de clusters", 2, 15, 3, 1, key="cluster_n_v4")
                cluster_outlier = st.selectbox(
                    "Tratamento de outliers numéricos",
                    ["keep", "winsorize", "median"],
                    format_func=lambda x: {
                        "keep": "Manter",
                        "winsorize": "Winsorizar",
                        "median": "Substituir por mediana",
                    }[x],
                    key="cluster_outlier_v4",
                )
            with c2:
                eps = st.number_input("DBSCAN — eps", 0.01, 10.0, 0.50, 0.05, key="dbscan_eps_v4")
                min_samples = st.number_input("DBSCAN — min_samples", 2, 50, 5, 1, key="dbscan_min_samples_v4")
            with c3:
                algorithms = st.multiselect(
                    "Algoritmos",
                    ["K-Means", "Hierárquico", "DBSCAN", "Gaussian Mixture", "Birch"],
                    default=["K-Means", "Hierárquico", "DBSCAN", "Gaussian Mixture"],
                    key="cluster_algorithms_v4",
                )
                seed_choice = st.radio(
                    "Semente", ["fixa", "aleatoria", "custom"],
                    format_func=lambda x: {
                        "fixa": "Fixa (42)", "aleatoria": "Aleatória", "custom": "Customizada"
                    }[x],
                    horizontal=True, key="cluster_seed_v4",
                )
                custom_seed = st.number_input(
                    "Semente customizada", 0, 999999, 42,
                    disabled=seed_choice != "custom", key="cluster_custom_seed_v4",
                )

            t1, t2, t3 = st.columns(3)
            with t1:
                tune_cluster = st.checkbox("🔧 Otimizar hiperparâmetros", value=True, key="tune_cluster_v4")
            with t2:
                tuning_method_cluster = st.selectbox(
                    "Busca", ["grid", "random"],
                    format_func=lambda x: "Grid Search" if x == "grid" else "Random Search",
                    disabled=not tune_cluster, key="tuning_method_cluster_v4",
                )
            with t3:
                n_iter_cluster = st.number_input(
                    "n_iter (Random)", 2, 100, 20, 1,
                    disabled=(not tune_cluster or tuning_method_cluster != "random"),
                    key="n_iter_cluster_v4",
                )
            tuning_metric_cluster = st.selectbox(
                "Métrica para escolher os hiperparâmetros",
                ["silhouette", "calinski_harabasz", "davies_bouldin"],
                format_func=lambda x: {
                    "silhouette": "Silhouette (maior é melhor)",
                    "calinski_harabasz": "Calinski-Harabasz (maior é melhor)",
                    "davies_bouldin": "Davies-Bouldin (menor é melhor)",
                }[x],
                disabled=not tune_cluster, key="tuning_metric_cluster_v4",
            )

            if st.button("▶️ Executar Agrupamento", width="stretch", type="primary", key="run_cluster_v3"):
                try:
                    if len(cluster_features) < 2:
                        raise ValueError("Selecione pelo menos duas variáveis.")
                    if not algorithms:
                        raise ValueError("Selecione pelo menos um algoritmo.")

                    reference = None if reference_target == "Nenhuma (sem rótulo)" else df_raw[reference_target]
                    seed = resolve_random_state(seed_choice, int(custom_seed))
                    X_cluster = df_raw[cluster_features].copy()

                    X_transformed, cluster_results = fit_clustering_models(
                        X_cluster,
                        encoding_map,
                        ordinal_orders,
                        manual_maps,
                        algorithms,
                        n_clusters=int(n_clusters),
                        eps=float(eps),
                        min_samples=int(min_samples),
                        outlier_mode=cluster_outlier,
                        random_state=seed,
                        reference=reference,
                        tune=bool(tune_cluster),
                        search_method=tuning_method_cluster,
                        n_iter=int(n_iter_cluster),
                        objective=tuning_metric_cluster,
                    )
                    state = {
                        "task": "clustering",
                        "results": cluster_results,
                        "features": cluster_features,
                        "transformed": X_transformed,
                        "source_data": X_cluster,
                        "seed": seed,
                        "n_clusters": int(n_clusters),
                        "reference_target": None if reference_target == "Nenhuma (sem rótulo)" else reference_target,
                    }
                    st.session_state["clustering_result"] = state
                    st.session_state["results"] = state
                    st.success("🎉 Agrupamento concluído. Veja também a aba **Resultados**.")
                except Exception as exc:
                    st.error(f"Não foi possível executar o agrupamento: {exc}")

            cluster_state = st.session_state.get("clustering_result")
            if cluster_state:
                valid_cluster = {
                    name: res for name, res in cluster_state["results"].items()
                    if "error" not in res
                }
                if valid_cluster:
                    rows = []
                    for name, res in valid_cluster.items():
                        m = res["metrics"]
                        rows.append(
                            {
                                "Algoritmo": name,
                                "Clusters": m["n_clusters"],
                                "Silhouette": m["silhouette"],
                                "Calinski-Harabasz": m["calinski_harabasz"],
                                "Davies-Bouldin": m["davies_bouldin"],
                                "Ruído (%)": m["noise_pct"],
                                "ARI": m.get("adjusted_rand"),
                                "NMI": m.get("normalized_mutual_info"),
                            }
                        )
                    st.subheader("📋 Métricas")
                    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)


# ============================================================
# TAB 4 — RESULTADOS
# ============================================================

with tabs[3]:
    st.header("📈 Resultados e avaliação")
    result_state = st.session_state.get("results")

    if result_state is None:
        st.info("Execute uma análise na aba **Modelagem** para visualizar os resultados.")
    else:
        task = result_state["task"]
        results = result_state["results"]
        valid = {name: value for name, value in results.items() if "error" not in value}
        errors = {name: value["error"] for name, value in results.items() if "error" in value}

        if errors:
            with st.expander("⚠️ Modelos/algoritmos que falharam"):
                st.dataframe(
                    pd.DataFrame([{"Modelo": name, "Erro": error} for name, error in errors.items()]),
                    width="stretch",
                )

        if task == "regression":
            if not valid:
                st.error("Nenhum modelo de regressão foi executado com sucesso.")
            else:
                rows=[]
                for name,res in valid.items():
                    params = ", ".join(
                        f"{k.replace('regressor__','')}={v}" for k,v in res["best_params"].items()
                    ) or "—"
                    rows.append({
                        "Modelo": name,
                        "CV R² médio": round(res["cv_mean"],4),
                        "CV R² desvio": round(res["cv_std"],4),
                        "R² teste": round(res["test_r2"],4),
                        "RMSE teste": round(res["test_rmse"],4),
                        "MAE teste": round(res["test_mae"],4),
                        "Hiperparâmetros": params,
                    })
                result_table=pd.DataFrame(rows).sort_values("CV R² médio",ascending=False).reset_index(drop=True)
                st.dataframe(result_table,width="stretch",hide_index=True)
                selected=result_table.iloc[0]["Modelo"]
                st.info(f"Modelo selecionado pela média de R² na CV: **{selected}**. O teste permanece reservado para avaliação final.")

                fig_real,fig_resid=regression_scatter_plot(result_state["y_test"],valid)
                if fig_real is not None:
                    st.plotly_chart(fig_real,width="stretch")
                    st.plotly_chart(fig_resid,width="stretch")
                fig_box,box_df=regression_boxplot(valid)
                if fig_box is not None:
                    st.plotly_chart(fig_box,width="stretch")

        elif task == "classification":
            if not valid:
                st.error("Nenhum classificador foi executado com sucesso.")
            else:
                rows=[]
                for name,res in valid.items():
                    rows.append({
                        "Classificador": name,
                        "CV F1 ponderado": f"{res['cv_f1_weighted_mean']:.4f} ± {res['cv_f1_weighted_std']:.4f}",
                        "CV F1 macro": f"{res['cv_f1_macro_mean']:.4f} ± {res['cv_f1_macro_std']:.4f}",
                        "Média CV (métrica de tuning)": round(res.get("cv_tuning_mean", res["cv_f1_weighted_mean"]), 4),
                        "Desvio CV (tuning)": round(res.get("cv_tuning_std", res["cv_f1_weighted_std"]), 4),
                        "Hiperparâmetros": res.get("best_params_text", "—"),
                        "CV balanced accuracy": round(res["cv_balanced_accuracy_mean"],4),
                        "Acurácia teste": round(res["test_accuracy"],4),
                        "F1 ponderado teste": round(res["test_f1"],4),
                        "F1 macro teste": round(res["test_f1_macro"],4),
                        "Balanced accuracy teste": round(res["test_balanced_accuracy"],4),
                        "ROC-AUC teste": None if res.get("test_roc_auc") is None else round(res["test_roc_auc"],4),
                    })
                result_table=pd.DataFrame(rows).sort_values("_score_selection",ascending=False).reset_index(drop=True)
                display_table=result_table.drop(columns=["_score_selection"])
                st.dataframe(display_table,width="stretch",height=450,hide_index=True)
                selected=result_table.iloc[0]["Classificador"]
                selected_metric=valid[selected].get("tuning_metric","f1_weighted")
                st.info(f"Classificador selecionado pela métrica de tuning **{selected_metric}** na CV: **{selected}**. As métricas do teste são avaliação final.")

                selected_result=valid[selected]
                cm_fig=confusion_matrix_plot(
                    result_state["y_test"],
                    selected_result["y_pred"],
                    result_state["classes"],
                    f"Matriz de confusão — {selected}",
                )
                st.plotly_chart(cm_fig,width="stretch")
                st.write({
                    "Acurácia": round(selected_result["test_accuracy"],4),
                    "F1 ponderado": round(selected_result["test_f1"],4),
                    "F1 macro": round(selected_result["test_f1_macro"],4),
                    "Balanced accuracy": round(selected_result["test_balanced_accuracy"],4),
                    "ROC-AUC": selected_result.get("test_roc_auc"),
                })
                fig_cv,cv_df=classification_cv_plot(valid)
                if fig_cv is not None:
                    st.plotly_chart(fig_cv,width="stretch")

                st.subheader("🔧 Hiperparâmetros selecionados")
                tuning_rows = []
                for model_name, model_res in valid.items():
                    tuning_rows.append({
                        "Classificador": model_name,
                        "Otimizou": "Sim" if model_res.get("tuned") else "Não",
                        "Busca": model_res.get("tuning_method", "none"),
                        "Métrica": model_res.get("tuning_metric", "—"),
                        "Hiperparâmetros": model_res.get("best_params_text", "—"),
                    })
                st.dataframe(pd.DataFrame(tuning_rows), width="stretch", hide_index=True)
                for model_name, model_res in valid.items():
                    tuning_table = model_res.get("tuning_table")
                    if tuning_table is not None and not tuning_table.empty:
                        with st.expander(f"🔎 Configurações testadas — {model_name}"):
                            st.dataframe(tuning_table, width="stretch", hide_index=True)

                st.divider()
                st.subheader("🔮 Classificar novos dados")
                st.caption(
                    "Também é possível enviar um segundo dataset sem a coluna alvo. "
                    "O mesmo Pipeline treinado será reaplicado às novas linhas."
                )
                prediction_models=list(valid.keys())
                pred_model_name=st.selectbox(
                    "Modelo treinado",
                    prediction_models,
                    key="prediction_result_model_v3",
                )
                prediction_file=st.file_uploader(
                    "Novo arquivo para previsão",
                    type=["csv","xlsx","xls","json"],
                    key="prediction_result_file_v3",
                )
                if prediction_file is not None:
                    try:
                        new_data=load_dataframe(prediction_file,"file")
                        missing=[c for c in result_state["features"] if c not in new_data.columns]
                        if missing:
                            st.error("Variáveis ausentes: "+", ".join(missing))
                        elif st.button("🔮 Executar previsão",type="primary",width="stretch",key="predict_result_v3"):
                            predicted=predict_new_classification(
                                valid[pred_model_name]["best_model"],
                                new_data,
                                result_state["features"],
                                result_state["classes"],
                            )
                            st.session_state["prediction_result"]=predicted
                    except Exception as exc:
                        st.error(f"Erro na previsão: {exc}")

                if st.session_state.get("prediction_result") is not None:
                    st.dataframe(st.session_state["prediction_result"],width="stretch",height=350)
                    st.download_button(
                        "📥 Baixar previsões (.csv)",
                        st.session_state["prediction_result"].to_csv(index=False).encode("utf-8"),
                        file_name=f"classificacoes_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        mime="text/csv",
                        width="stretch",
                        key="download_prediction_result_v3",
                    )

        else:
            if not valid:
                st.error("Nenhum algoritmo de agrupamento foi executado com sucesso.")
            else:
                rows=[]
                for name,res in valid.items():
                    m=res["metrics"]
                    objective_key=res.get("tuning_metric","silhouette")
                    objective_value=m.get(objective_key)
                    rows.append({
                        "Algoritmo": name,
                        "_score_selection": objective_value,
                        "Clusters": m["n_clusters"],
                        "Silhouette": m["silhouette"],
                        "Calinski-Harabasz": m["calinski_harabasz"],
                        "Davies-Bouldin": m["davies_bouldin"],
                        "Ruído (%)": m["noise_pct"],
                        "ARI": m.get("adjusted_rand"),
                        "NMI": m.get("normalized_mutual_info"),
                        "Hiperparâmetros": str(res.get("best_params", {})),
                    })
                sort_ascending = True if valid[next(iter(valid))].get("tuning_metric") == "davies_bouldin" else False
                result_table=pd.DataFrame(rows).sort_values("_score_selection",ascending=sort_ascending,na_position="last")
                st.dataframe(result_table.drop(columns=["_score_selection"]).round(4),width="stretch",hide_index=True)
                st.info(
                    "Silhouette e Calinski-Harabasz: quanto maiores, melhor tende a ser a separação. "
                    "Davies-Bouldin: quanto menor, melhor. ARI/NMI só aparecem quando você forneceu um rótulo de referência; esse rótulo não é usado para formar os clusters. "
                    "A otimização de agrupamento usa a métrica interna selecionada e deve ser interpretada como busca heurística, não como validação supervisionada."
                )
                st.subheader("🔧 Hiperparâmetros selecionados")
                cluster_tuning_rows=[]
                for name,res in valid.items():
                    cluster_tuning_rows.append({
                        "Algoritmo": name,
                        "Otimizou": "Sim" if res.get("tuned") else "Não",
                        "Busca": res.get("tuning_method", "none"),
                        "Métrica": res.get("tuning_metric", "—"),
                        "Hiperparâmetros": str(res.get("best_params", {})),
                    })
                st.dataframe(pd.DataFrame(cluster_tuning_rows), width="stretch", hide_index=True)
                for name,res in valid.items():
                    tuning_table=res.get("tuning_table")
                    if tuning_table is not None and not tuning_table.empty:
                        with st.expander(f"🔎 Configurações testadas — {name}"):
                            st.dataframe(tuning_table, width="stretch", hide_index=True)

                for name,res in valid.items():
                    fig=clustering_scatter_plot(
                        result_state["transformed"],
                        res["labels"],
                        f"{name} — clusters",
                    )
                    st.plotly_chart(fig,width="stretch")
                    assignment=result_state["source_data"].copy()
                    assignment["Cluster"]=res["labels"]
                    with st.expander(f"🔎 Atribuições — {name}"):
                        st.dataframe(assignment,width="stretch")

        # ---------------- exportação ----------------
        st.divider()
        st.subheader("💾 Exportar resultados")

        if task == "clustering":
            export_rows=[]
            for name,res in valid.items():
                export_rows.append({"algoritmo":name,**res["metrics"]})
        else:
            export_rows=[]
            for name,res in valid.items():
                clean={}
                for key,item in res.items():
                    if key in {"best_model","y_pred","cv_scores","cv_acc_scores","cv_f1_scores","search_table","tuning_table"}:
                        continue
                    if isinstance(item,(np.floating,np.integer)):
                        clean[key]=item.item()
                    else:
                        clean[key]=item
                clean["modelo"]=name
                export_rows.append(clean)

        csv_bytes=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar resultados (.csv)",
            csv_bytes,
            file_name=f"resultados_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            width="stretch",
        )

        report={
            "tarefa": task,
            "features": result_state.get("features",[]),
            "seed": result_state.get("seed"),
            "modelos": export_rows,
        }
        if task == "clustering":
            report["n_clusters"]=result_state.get("n_clusters")
            report["reference_target"]=result_state.get("reference_target")
        else:
            report.update({
                "alvo":result_state.get("target"),
                "train_n":result_state.get("train_n"),
                "test_n":result_state.get("test_n"),
                "cv":result_state.get("cv"),
            })
        st.download_button(
            "📥 Baixar relatório (.json)",
            json.dumps(report,indent=2,ensure_ascii=False,default=str).encode("utf-8"),
            file_name=f"relatorio_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            width="stretch",
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()
st.caption(
    "🧠 Análise de Inteligência Computacional — Streamlit + Plotly | "
    "Pré-processamento dentro do pipeline e validação cruzada sem "
    "ajuste sobre o conjunto de teste."
)
