"""
classifiers.py — núcleo de la biblioteca de clasificación (solo NumPy)
=====================================================================

Contiene todo lo común y los clasificadores de referencia:

  1. Métricas de distancia con registro extensible (register_metric)
  2. Preprocesamiento: StandardScaler, PCA, train_test_split
  3. BaseClassifier + registro de clasificadores (register_classifier)
  4. Clasificadores base: KNN, CV-KNN, Naive Bayes gaussiano,
     Mínima Distancia y un clasificador trivial de clase mayoritaria
  5. Métricas de evaluación para clases desbalanceadas (con sample_weight):
     accuracy, balanced accuracy, sensibilidad, especificidad, precisión,
     F1 / F-beta, G-Mean, PR-AUC (average precision) y costo total (TMC)
  6. Validación cruzada estratificada y comparaciones

Los clasificadores avanzados viven en módulos aparte que se registran aquí
al importarse:
    cs_knn.py              -> DirectCSKNN, DistancesCSKNN
    reconstruction_knn.py  -> CMKNN, OneStepKNN

-------------------------------------------------------------------------------
USO RÁPIDO
-------------------------------------------------------------------------------
    from classifiers import *
    import cs_knn, reconstruction_knn        # registran los modelos avanzados

    clf = KNNClassifier(n_neighbors=5, metric="manhattan").fit(X_train, y_train)
    clf.predict(X_test); clf.predict_proba(X_test); clf.score(X_test, y_test)

    clf = make_classifier("cm_knn", rho1=1.0)          # por nombre
    list_classifiers(), list_metrics()

    evaluate_binary(y_test, clf.predict(X_test), clf.predict_proba(X_test)[:, 1],
                    pos_label=1, c_fn=10, c_fp=1)

-------------------------------------------------------------------------------
AGREGAR UNA MÉTRICA DE DISTANCIA
-------------------------------------------------------------------------------
    @register_metric("mi_dist")                  # vectorizada: (A, B) -> (n_A, n_B)
    def mi_dist(A, B, escala=1.0):
        return pairwise_apply(lambda a, b: np.abs(a - b).sum(-1) / escala, A, B)

    @register_metric("mi_dist2", pointwise=True) # punto a punto: (u, v) -> float
    def mi_dist2(u, v):
        return np.abs(u - v).max()

    KNNClassifier(metric="mi_dist", metric_params={"escala": 2.0})

-------------------------------------------------------------------------------
AGREGAR UN CLASIFICADOR
-------------------------------------------------------------------------------
Hereda de BaseClassifier e implementa:
    __init__(self, ...)       guarda cada parámetro como self.<mismo nombre>
    _fit(self, X, y)          X float 2D; y codificada 0..n_classes_-1
    _predict_proba(self, X)   (n, n_classes_), mayor = más probable
    _predict(self, X)         [opcional] índices de clase (p. ej. mínimo costo)

Si __init__ tiene `metric` y `metric_params`, usa self._dist(A, B).
Disponibles tras fit: classes_, n_classes_, n_features_in_, X_, y_.

    @register_classifier("nearest_median")
    class NearestMedian(BaseClassifier):
        def __init__(self, metric="euclidean", metric_params=None):
            self.metric = metric
            self.metric_params = metric_params
        def _fit(self, X, y):
            self.medians_ = np.array([np.median(X[y == c], 0) for c in range(self.n_classes_)])
        def _predict_proba(self, X):
            return distances_to_proba(self._dist(X, self.medians_))

Convención de costos en todo el proyecto:
    cost_matrix[i, j] = costo de PREDECIR la clase i cuando la REAL es j
    (orden de clf.classes_). Binario con ω1 positiva: C[0,1] = C01 = costo de
    un Falso Negativo, C[1,0] = C10 = costo de un Falso Positivo.
"""

import inspect
import numpy as np

__all__ = [
    # métricas de distancia
    "register_metric", "list_metrics", "pairwise_distances", "pairwise_apply",
    "resolve_metric_params",
    # preprocesamiento
    "StandardScaler", "PCA", "train_test_split", "distances_to_proba",
    # clasificadores
    "BaseClassifier", "register_classifier", "list_classifiers", "make_classifier",
    "KNNClassifier", "CVKNNClassifier", "GaussianNaiveBayes",
    "MinimumDistanceClassifier", "MajorityClassifier", "binary_cost_matrix",
    # métricas de evaluación
    "accuracy_score", "confusion_matrix", "binary_confusion", "recall_score",
    "specificity_score", "precision_score", "fbeta_score", "f1_score",
    "balanced_accuracy_score", "geometric_mean_score", "average_precision_score",
    "precision_recall_curve", "roc_curve", "roc_auc_score",
    "total_cost", "total_misclassification_cost", "classification_report",
    "evaluate_binary", "get_scorer", "SCORERS",
    # validación
    "clone", "stratified_kfold", "cross_val_score", "compare_classifiers",
    "compare_metrics", "print_results_table",
]

_EPS = 1e-12


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


# #############################################################################
# 1. MÉTRICAS DE DISTANCIA
# #############################################################################

_METRICS = {}          # nombre -> (función, pointwise)


def register_metric(name, func=None, pointwise=False, overwrite=False):
    """
    Registra una métrica de distancia.
        pointwise=False : func(A, B, **params) -> (n_A, n_B)   (vectorizada, rápida)
        pointwise=True  : func(u, v, **params) -> float        (simple, lenta)
    Uso como decorador: @register_metric("nombre").
    """
    def _add(f):
        if not callable(f):
            raise TypeError("La métrica debe ser una función")
        if name in _METRICS and not overwrite:
            raise ValueError(f"La métrica {name!r} ya existe (usa overwrite=True)")
        _METRICS[name] = (f, pointwise)
        return f
    return _add(func) if func is not None else _add


def list_metrics():
    """Nombres de las métricas registradas."""
    return sorted(_METRICS)


def pairwise_apply(fn, A, B, max_elems=2e7):
    """
    Matriz (n_A, n_B) aplicando fn(a, b) con broadcasting por bloques.
    a: (c, 1, d), b: (1, n_B, d); fn debe reducir el último eje.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    D = np.empty((len(A), len(B)))
    step = max(1, int(max_elems // max(1, len(B) * A.shape[1])))
    for i in range(0, len(A), step):
        D[i:i + step] = fn(A[i:i + step, None, :], B[None, :, :])
    return D


def _pointwise_matrix(f, A, B, **params):
    D = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            D[i, j] = f(a, b, **params)
    return D


def pairwise_distances(A, B, metric="euclidean", **params):
    """Distancias (n_A, n_B). metric: nombre registrado o función f(u, v)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if callable(metric):
        return _pointwise_matrix(metric, A, B, **params)
    if metric not in _METRICS:
        raise ValueError(f"Métrica no soportada: {metric!r}. Disponibles: {list_metrics()}")
    f, pointwise = _METRICS[metric]
    return _pointwise_matrix(f, A, B, **params) if pointwise else f(A, B, **params)


def resolve_metric_params(metric, metric_params, X):
    """Completa VI (mahalanobis) y V (seuclidean) a partir de X si no se dan."""
    params = dict(metric_params or {})
    if metric == "mahalanobis" and "VI" not in params:
        C = np.atleast_2d(np.cov(X, rowvar=False))
        params["VI"] = np.linalg.pinv(C + 1e-9 * np.eye(C.shape[0]))
    if metric == "seuclidean" and "V" not in params:
        V = X.var(0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
        params["V"] = np.where(V > 0, V, 1.0)
    return params


@register_metric("sqeuclidean")
def _sqeuclidean(A, B, w=None):
    if w is not None:
        s = np.sqrt(np.asarray(w, dtype=float))
        A, B = A * s, B * s
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(sq, 0.0)


@register_metric("euclidean")
def _euclidean(A, B, w=None):
    """w: pesos por característica (euclidiana ponderada)."""
    return np.sqrt(_sqeuclidean(A, B, w))


@register_metric("manhattan")
def _manhattan(A, B, w=None):
    w = 1.0 if w is None else np.asarray(w, dtype=float)
    return pairwise_apply(lambda a, b: (w * np.abs(a - b)).sum(-1), A, B)


register_metric("cityblock", _manhattan)


@register_metric("chebyshev")
def _chebyshev(A, B):
    return pairwise_apply(lambda a, b: np.abs(a - b).max(-1), A, B)


@register_metric("minkowski")
def _minkowski(A, B, p=2, w=None):
    if p == 1:
        return _manhattan(A, B, w)
    if p == 2:
        return _euclidean(A, B, w)
    if np.isinf(p):
        return _chebyshev(A, B)
    w = 1.0 if w is None else np.asarray(w, dtype=float)
    return pairwise_apply(lambda a, b: (w * np.abs(a - b) ** p).sum(-1) ** (1.0 / p), A, B)


@register_metric("seuclidean")
def _seuclidean(A, B, V):
    """Euclidiana estandarizada: V = varianza de cada característica."""
    return _euclidean(A, B, w=1.0 / np.asarray(V, dtype=float))


@register_metric("mahalanobis")
def _mahalanobis(A, B, VI):
    """VI = inversa de la covarianza (se calcula en fit si no se da)."""
    VI = np.asarray(VI, dtype=float)
    AV, BV = A @ VI, B @ VI
    sq = (AV * A).sum(1)[:, None] + (BV * B).sum(1)[None, :] - 2.0 * AV @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


@register_metric("cosine")
def _cosine(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + _EPS)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + _EPS)
    return np.maximum(1.0 - An @ Bn.T, 0.0)


@register_metric("correlation")
def _correlation(A, B):
    return _cosine(A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True))


@register_metric("canberra")
def _canberra(A, B):
    return pairwise_apply(lambda a, b: _safe_ratio(np.abs(a - b), np.abs(a) + np.abs(b)).sum(-1), A, B)


@register_metric("braycurtis")
def _braycurtis(A, B):
    return pairwise_apply(lambda a, b: np.abs(a - b).sum(-1) / (np.abs(a + b).sum(-1) + _EPS), A, B)


@register_metric("chi2")
def _chi2(A, B):
    return pairwise_apply(lambda a, b: _safe_ratio((a - b) ** 2, np.abs(a) + np.abs(b)).sum(-1), A, B)


@register_metric("lorentzian")
def _lorentzian(A, B):
    return pairwise_apply(lambda a, b: np.log1p(np.abs(a - b)).sum(-1), A, B)


@register_metric("hassanat")
def _hassanat(A, B):
    def f(a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        shift = np.where(lo < 0, -lo, 0.0)
        return (1.0 - (1.0 + lo + shift) / (1.0 + hi + shift)).sum(-1)
    return pairwise_apply(f, A, B)


@register_metric("hamming")
def _hamming(A, B):
    return pairwise_apply(lambda a, b: (a != b).mean(-1), A, B)


@register_metric("jaccard")
def _jaccard(A, B):
    A, B = (A != 0).astype(float), (B != 0).astype(float)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, 1.0 - _safe_ratio(inter, union), 0.0)


# #############################################################################
# 2. PREPROCESAMIENTO
# #############################################################################

class StandardScaler:
    """Estandarización (media 0, desviación 1)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class PCA:
    """
    Análisis de Componentes Principales vía SVD.

    n_components : int (número de componentes), float en (0, 1) (fracción
                   mínima de varianza explicada, p. ej. 0.85) o None (todas)
    Atributos: mean_, components_ (k, d), explained_variance_,
               explained_variance_ratio_, n_components_
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # signo determinista: la entrada de mayor |valor| de cada componente es positiva
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]
        var = S ** 2 / max(len(X) - 1, 1)
        ratio = var / var.sum()
        nc = self.n_components
        if nc is None:
            k = len(S)
        elif isinstance(nc, float) and 0 < nc < 1:
            k = int(np.searchsorted(np.cumsum(ratio), nc - 1e-12) + 1)
        else:
            k = int(nc)
        self.n_components_ = k
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = ratio[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_


def train_test_split(X, y, test_size=0.3, random_state=42, stratify=True):
    """Divide en entrenamiento y prueba (estratificado por defecto)."""
    X = np.asarray(X); y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    groups = [np.where(y == c)[0] for c in np.unique(y)] if stratify else [np.arange(len(y))]
    test_mask = np.zeros(len(y), dtype=bool)
    for g in groups:
        g = rng.permutation(g)
        test_mask[g[:int(round(len(g) * test_size))]] = True
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


def distances_to_proba(D):
    """Distancias (n, n_clases) -> puntajes que suman 1: softmax(-D)."""
    z = -(D - D.min(1, keepdims=True))
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _normalize_rows(P):
    P = np.asarray(P, dtype=float)
    s = P.sum(1, keepdims=True)
    return np.where(s > 0, P / np.where(s > 0, s, 1.0), 1.0 / P.shape[1])


# #############################################################################
# 3. CLASE BASE Y REGISTRO
# #############################################################################

_CLASSIFIERS = {}


def register_classifier(name, overwrite=False):
    """Decorador: registra una clase de clasificador con un nombre."""
    def _add(cls):
        if name in _CLASSIFIERS and not overwrite and _CLASSIFIERS[name] is not cls:
            raise ValueError(f"El clasificador {name!r} ya existe (usa overwrite=True)")
        _CLASSIFIERS[name] = cls
        cls._registry_name = name
        return cls
    return _add


def list_classifiers():
    return sorted(_CLASSIFIERS)


def make_classifier(name, **params):
    """make_classifier('knn', n_neighbors=3)."""
    if name not in _CLASSIFIERS:
        raise ValueError(f"Clasificador desconocido: {name!r}. Disponibles: {list_classifiers()} "
                         "(¿importaste cs_knn / reconstruction_knn?)")
    return _CLASSIFIERS[name](**params)


class BaseClassifier:
    """
    Base de todos los clasificadores. Implementa en la subclase:
        __init__, _fit(X, y), _predict_proba(X) y opcionalmente _predict(X).
    """

    _special_metrics = ()      # métricas que la subclase maneja por su cuenta

    @classmethod
    def _get_param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [n for n, p in sig.parameters.items()
                if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]

    def get_params(self, deep=True):
        return {n: getattr(self, n) for n in self._get_param_names()}

    def set_params(self, **params):
        valid = self._get_param_names()
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Parámetro inválido {k!r} para {type(self).__name__}. "
                                 f"Válidos: {valid}")
            setattr(self, k, v)
        return self

    def __repr__(self):
        def fmt(v):
            if isinstance(v, np.ndarray):
                return f"array(shape={v.shape})"
            if callable(v):
                return getattr(v, "__name__", "función")
            return repr(v)
        defaults = {n: p.default for n, p in inspect.signature(type(self).__init__).parameters.items()}
        args = []
        for k, v in self.get_params().items():
            d = defaults.get(k, inspect.Parameter.empty)
            try:
                same = (d is not inspect.Parameter.empty and not isinstance(v, np.ndarray)
                        and not isinstance(d, np.ndarray) and bool(v == d))
            except Exception:
                same = False
            if not same:
                args.append(f"{k}={fmt(v)}")
        return f"{type(self).__name__}({', '.join(args)})"

    # ------------------------------ API pública ------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (n_muestras, n_características)")
        if len(X) != len(y):
            raise ValueError("X e y deben tener el mismo número de muestras")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene NaN o infinitos")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.X_, self.y_ = X, y_enc
        if self._uses_metric():
            self._setup_metric(X, y_enc)
        self._fit(X, y_enc)
        return self

    def predict_proba(self, X):
        return self._predict_proba(self._check_X(X))

    def predict(self, X):
        X = self._check_X(X)
        return self.classes_[self._predict(X)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # ------------------------ a implementar en subclases ---------------------
    def _fit(self, X, y):
        """Por defecto no hace nada (aprendiz perezoso)."""

    def _predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__} debe implementar _predict_proba(X)")

    def _predict(self, X):
        return np.argmax(self._predict_proba(X), axis=1)

    # ------------------------------- auxiliares ------------------------------
    def _check_X(self, X):
        if not hasattr(self, "classes_"):
            raise RuntimeError(f"{type(self).__name__} no está entrenado; llama fit() primero.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} características; se esperaban {self.n_features_in_}")
        return X

    def _uses_metric(self):
        m = getattr(self, "metric", None)
        return (m is not None and "metric" in self._get_param_names()
                and not (isinstance(m, str) and m in self._special_metrics))

    def _metric_params_for_fit(self, X, y):
        params = dict(getattr(self, "metric_params", None) or {})
        if self.metric == "minkowski" and "p" not in params and getattr(self, "p", None) is not None:
            params["p"] = self.p
        return params

    def _setup_metric(self, X, y):
        m = self.metric
        if not callable(m) and m not in _METRICS:
            raise ValueError(f"Métrica no soportada: {m!r}. Disponibles: {list_metrics()}")
        self._metric_ = m
        self._metric_params_ = resolve_metric_params(m, self._metric_params_for_fit(X, y), X)

    def _dist(self, A, B):
        return pairwise_distances(A, B, self._metric_, **self._metric_params_)


def binary_cost_matrix(c_fn=10.0, c_fp=1.0):
    """
    Matriz de costos binaria con la convención C[pred, real] y ω1 = positiva:
        [[0,    C01],     C01 = costo de un Falso Negativo (predecir 0, real 1)
         [C10,  0  ]]     C10 = costo de un Falso Positivo (predecir 1, real 0)
    """
    return np.array([[0.0, float(c_fn)], [float(c_fp), 0.0]])


# #############################################################################
# 4. CLASIFICADORES BASE
# #############################################################################

@register_classifier("knn")
class KNNClassifier(BaseClassifier):
    """
    KNN convencional.

    n_neighbors   : int
    weights       : 'uniform' | 'distance' | función(dist) -> pesos
    metric        : nombre registrado o función f(u, v)
    p             : potencia para metric='minkowski'
    metric_params : parámetros extra de la métrica

    Punto de extensión: sobrescribe _distance_matrix(X) para deformar las
    distancias (lo usa DistancesCSKNN).
    """

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", p=2,
                 metric_params=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _fit(self, X, y):
        if not 1 <= self.n_neighbors <= len(X):
            raise ValueError(f"n_neighbors debe estar entre 1 y {len(X)}")

    def _distance_matrix(self, X):
        """Distancias (n_test, n_train) usadas para buscar vecinos."""
        return self._dist(X, self.X_)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Distancias e índices (en X_) de los k vecinos más cercanos, ordenados."""
        X = self._check_X(X)
        k = min(n_neighbors or self.n_neighbors, len(self.X_))
        D = self._distance_matrix(X)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(d, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)
        d = np.take_along_axis(d, order, axis=1)
        return (d, idx) if return_distance else idx

    def _neighbor_weights(self, dist):
        if self.weights == "uniform":
            return np.ones_like(dist)
        if self.weights == "distance":
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            zero = np.isinf(w)
            rows = zero.any(1)
            w[rows] = zero[rows].astype(float)
            return w
        if callable(self.weights):
            return self.weights(dist)
        raise ValueError("weights debe ser 'uniform', 'distance' o una función")

    def _class_votes(self, weights, idx, labels=None):
        labels = self.y_[idx] if labels is None else labels
        return np.stack([(weights * (labels == c)).sum(1) for c in range(self.n_classes_)], axis=1)

    def _predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return _normalize_rows(self._class_votes(self._neighbor_weights(dist), idx))


@register_classifier("cv_knn")
class CVKNNClassifier(BaseClassifier):
    """
    CV-KNN: elige k por búsqueda en cuadrícula con validación cruzada
    estratificada interna (solo con los datos de entrenamiento).

    k_grid       : valores de k a probar
    cv           : pliegues internos
    scoring      : 'accuracy' | 'balanced_accuracy' | 'g_mean' | función(y, y_pred)
    weights, metric, p, metric_params : como KNNClassifier
    random_state : semilla de los pliegues internos

    Atributos: best_k_, cv_scores_ (media por k), best_estimator_
    Eficiente: calcula distancias una vez por pliegue y evalúa todos los k.
    """

    def __init__(self, k_grid=(1, 3, 5, 7, 9, 11, 15), cv=5, scoring="balanced_accuracy",
                 weights="uniform", metric="euclidean", p=2, metric_params=None,
                 random_state=42):
        self.k_grid = k_grid
        self.cv = cv
        self.scoring = scoring
        self.weights = weights
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.random_state = random_state

    def _knn(self, k):
        return KNNClassifier(k, self.weights, self.metric, self.p, self.metric_params)

    def _fit(self, X, y):
        scorer = get_scorer(self.scoring)
        n_min = np.bincount(y).min()
        n_splits = int(min(self.cv, max(n_min, 2)))
        ks = sorted(set(int(k) for k in self.k_grid))
        scores = []
        for tr, te in stratified_kfold(y, n_splits, self.random_state):
            kmax = min(max(ks), len(tr))
            knn = self._knn(kmax).fit(X[tr], y[tr])
            d, idx = knn.kneighbors(X[te], n_neighbors=kmax)
            lab = y[tr][idx]                               # etiquetas globales 0..C-1
            row = []
            for k in ks:
                kk = min(k, kmax)
                w = knn._neighbor_weights(d[:, :kk])
                votes = np.stack([(w * (lab[:, :kk] == c)).sum(1) for c in range(self.n_classes_)], 1)
                row.append(scorer(y[te], votes.argmax(1)))
            scores.append(row)
        self.cv_scores_ = dict(zip(ks, np.mean(scores, axis=0)))
        self.best_k_ = max(ks, key=lambda k: (round(self.cv_scores_[k], 12), -k))
        self.best_estimator_ = self._knn(min(self.best_k_, len(X))).fit(X, y)

    def _predict_proba(self, X):
        return self.best_estimator_._predict_proba(X)


@register_classifier("gaussian_nb")
class GaussianNaiveBayes(BaseClassifier):
    """
    Naive Bayes gaussiano (equivalente a sklearn GaussianNB).
    priors: (n_clases,) o None; var_smoothing como en sklearn.
    Atributos: theta_, var_, class_prior_.
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def _fit(self, X, y):
        n_c = self.n_classes_
        self.theta_ = np.array([X[y == c].mean(0) for c in range(n_c)])
        eps = self.var_smoothing * X.var(0).max()
        self.var_ = np.array([X[y == c].var(0) for c in range(n_c)]) + eps
        if self.priors is None:
            self.class_prior_ = np.bincount(y, minlength=n_c) / len(y)
        else:
            self.class_prior_ = np.asarray(self.priors, dtype=float)
            if self.class_prior_.shape != (n_c,) or not np.isclose(self.class_prior_.sum(), 1):
                raise ValueError("priors debe tener n_clases elementos y sumar 1")

    def _joint_log_likelihood(self, X):
        jll = np.empty((len(X), self.n_classes_))
        for c in range(self.n_classes_):
            ll = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]))
            ll = ll - 0.5 * np.sum((X - self.theta_[c]) ** 2 / self.var_[c], axis=1)
            jll[:, c] = np.log(self.class_prior_[c]) + ll
        return jll

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(self._check_X(X))
        m = jll.max(1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(1, keepdims=True)))

    def _predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def _predict(self, X):
        return np.argmax(self._joint_log_likelihood(X), axis=1)


@register_classifier("min_distance")
class MinimumDistanceClassifier(BaseClassifier):
    """
    Mínima distancia al centroide de cada clase (sklearn NearestCentroid).

    metric : métrica registrada o función, y además
             'mahalanobis'       -> covarianza intra-clase común (pooled) por defecto
             'mahalanobis_class' -> una covarianza por clase
    reg    : regularización en la diagonal de las covarianzas
    predict_proba = softmax(-distancia) (puntaje, no probabilidad calibrada).
    """

    _special_metrics = ("mahalanobis_class",)

    def __init__(self, metric="euclidean", p=2, metric_params=None, reg=1e-6):
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.reg = reg

    def _metric_params_for_fit(self, X, y):
        params = super()._metric_params_for_fit(X, y)
        if self.metric == "mahalanobis" and "VI" not in params:
            cents = np.array([X[y == c].mean(0) for c in range(self.n_classes_)])
            Xc = X - cents[y]
            S = Xc.T @ Xc / max(len(X) - self.n_classes_, 1)
            params["VI"] = np.linalg.pinv(S + self.reg * np.eye(X.shape[1]))
        return params

    def _fit(self, X, y):
        d = X.shape[1]
        self.centroids_ = np.array([X[y == c].mean(0) for c in range(self.n_classes_)])
        if self.metric == "mahalanobis_class":
            self._inv_cov = []
            for c in range(self.n_classes_):
                Xi = X[y == c]
                S = np.cov(Xi, rowvar=False, bias=len(Xi) < 2).reshape(d, d)
                self._inv_cov.append(np.linalg.pinv(S + self.reg * np.eye(d)))

    def decision_distances(self, X):
        X = self._check_X(X)
        if self.metric == "mahalanobis_class":
            D = np.empty((len(X), self.n_classes_))
            for c, mu in enumerate(self.centroids_):
                diff = X - mu
                D[:, c] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, self._inv_cov[c], diff), 0))
            return D
        return self._dist(X, self.centroids_)

    def _predict_proba(self, X):
        return distances_to_proba(self.decision_distances(X))

    def _predict(self, X):
        return np.argmin(self.decision_distances(X), axis=1)


@register_classifier("majority")
class MajorityClassifier(BaseClassifier):
    """Línea base trivial: siempre predice la clase más frecuente del entrenamiento."""

    def __init__(self):
        pass

    def _fit(self, X, y):
        self.class_prior_ = np.bincount(y, minlength=self.n_classes_) / len(y)

    def _predict_proba(self, X):
        return np.tile(self.class_prior_, (len(X), 1))


# #############################################################################
# 5. MÉTRICAS DE EVALUACIÓN (aceptan sample_weight)
# #############################################################################

def _weights(y, sample_weight):
    return np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)


def accuracy_score(y_true, y_pred, sample_weight=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = _weights(y_true, sample_weight)
    return float(np.sum(w * (y_true == y_pred)) / np.sum(w))


def confusion_matrix(y_true, y_pred, classes=None, sample_weight=None):
    """Filas = clase real, columnas = clase predicha (ponderada si hay pesos)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=float if sample_weight is not None else int)
    w = _weights(y_true, sample_weight)
    for t, p, wi in zip(y_true, y_pred, w):
        M[idx[t], idx[p]] += wi if sample_weight is not None else 1
    return M


def binary_confusion(y_true, y_pred, pos_label=1, sample_weight=None):
    """TP, FN, FP, TN (ponderados si hay pesos)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = _weights(y_true, sample_weight)
    t, p = y_true == pos_label, y_pred == pos_label
    return {"tp": float(np.sum(w * (t & p))), "fn": float(np.sum(w * (t & ~p))),
            "fp": float(np.sum(w * (~t & p))), "tn": float(np.sum(w * (~t & ~p)))}


def recall_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Sensibilidad = TP / (TP + FN)."""
    c = binary_confusion(y_true, y_pred, pos_label, sample_weight)
    return float(_safe_ratio(c["tp"], c["tp"] + c["fn"]))


def specificity_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Especificidad = TN / (TN + FP)."""
    c = binary_confusion(y_true, y_pred, pos_label, sample_weight)
    return float(_safe_ratio(c["tn"], c["tn"] + c["fp"]))


def precision_score(y_true, y_pred, pos_label=1, sample_weight=None):
    c = binary_confusion(y_true, y_pred, pos_label, sample_weight)
    return float(_safe_ratio(c["tp"], c["tp"] + c["fp"]))


def fbeta_score(y_true, y_pred, beta=1.0, pos_label=1, sample_weight=None):
    """F_beta = (1+β²) P R / (β² P + R). β=2 prioriza el recall."""
    P = precision_score(y_true, y_pred, pos_label, sample_weight)
    R = recall_score(y_true, y_pred, pos_label, sample_weight)
    b2 = beta ** 2
    return float(_safe_ratio((1 + b2) * P * R, b2 * P + R))


def f1_score(y_true, y_pred, pos_label=1, sample_weight=None):
    return fbeta_score(y_true, y_pred, 1.0, pos_label, sample_weight)


def _per_class_recall(y_true, y_pred, sample_weight=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = _weights(y_true, sample_weight)
    rec = []
    for c in np.unique(y_true):
        m = y_true == c
        rec.append(np.sum(w[m] * (y_pred[m] == c)) / np.sum(w[m]))
    return np.array(rec)


def balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    """Promedio del recall por clase (binario: (sens + espec) / 2)."""
    return float(np.mean(_per_class_recall(y_true, y_pred, sample_weight)))


def geometric_mean_score(y_true, y_pred, sample_weight=None):
    """G-Mean: media geométrica del recall por clase (binario: sqrt(sens * espec))."""
    r = _per_class_recall(y_true, y_pred, sample_weight)
    return float(np.prod(r) ** (1.0 / len(r)))


def average_precision_score(y_true, y_score, pos_label=1, sample_weight=None):
    """
    PR-AUC como Average Precision:  AP = sum_n (R_n - R_{n-1}) P_n
    sobre los umbrales distintos del puntaje (igual que sklearn).
    """
    y_true = np.asarray(y_true) == pos_label
    s = np.asarray(y_score, dtype=float)
    w = _weights(y_true, sample_weight)
    order = np.argsort(-s, kind="mergesort")
    s, t, w = s[order], y_true[order], w[order]
    thr = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(w * t)[thr]
    fps = np.cumsum(w * ~t)[thr]
    if tps[-1] <= 0:
        return float("nan")
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _ranked_counts(y_true, y_score, pos_label, sample_weight):
    """TP y FP acumulados en cada umbral distinto del puntaje (orden descendente)."""
    t = np.asarray(y_true) == pos_label
    s = np.asarray(y_score, dtype=float)
    w = _weights(t, sample_weight)
    order = np.argsort(-s, kind="mergesort")
    s, t, w = s[order], t[order], w[order]
    thr = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    return np.cumsum(w * t)[thr], np.cumsum(w * ~t)[thr], s[thr]


def precision_recall_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """
    Curva Precision-Recall. Regresa (precision, recall, umbrales), empezando en
    recall = 0 (precisión 1) y con umbrales decrecientes.
    """
    tps, fps, thr = _ranked_counts(y_true, y_score, pos_label, sample_weight)
    precision = _safe_ratio(tps, tps + fps)
    recall = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    return np.r_[1.0, precision], np.r_[0.0, recall], thr


def roc_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """
    Curva ROC. Regresa (fpr, tpr, umbrales) con el punto (0, 0) al inicio
    (umbral = +inf) y umbrales decrecientes.
        TPR = TP / (TP + FN) = sensibilidad,   FPR = FP / (FP + TN) = 1 - especificidad
    """
    tps, fps, thr = _ranked_counts(y_true, y_score, pos_label, sample_weight)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.full_like(tps, np.nan)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.full_like(fps, np.nan)
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thr]


def roc_auc_score(y_true, y_score, pos_label=1, sample_weight=None):
    """
    Área bajo la curva ROC (regla del trapecio). Equivale a la probabilidad de
    que una mina elegida al azar reciba mayor puntaje que una roca al azar
    (los empates cuentan 1/2). No depende de la prevalencia.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label, sample_weight)
    if np.isnan(fpr).any() or np.isnan(tpr).any():
        return float("nan")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def total_cost(y_true, y_pred, cost_matrix, classes, sample_weight=None):
    """Costo total general: sum_n C[pred_n, real_n] (ponderado si hay pesos)."""
    idx = {c: i for i, c in enumerate(np.asarray(classes))}
    C = np.asarray(cost_matrix, dtype=float)
    w = _weights(y_true, sample_weight)
    return float(sum(wi * C[idx[p], idx[t]] for t, p, wi in zip(y_true, y_pred, w)))


def total_misclassification_cost(y_true, y_pred, c_fn=10.0, c_fp=1.0, pos_label=1,
                                 sample_weight=None):
    """TMC = FN · C01 + FP · C10."""
    c = binary_confusion(y_true, y_pred, pos_label, sample_weight)
    return float(c["fn"] * c_fn + c["fp"] * c_fp)


def evaluate_binary(y_true, y_pred, y_score=None, pos_label=1, c_fn=10.0, c_fp=1.0,
                    sample_weight=None):
    """Todas las métricas del ABP en un dict."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight),
        "sensitivity": recall_score(y_true, y_pred, pos_label, sample_weight),
        "specificity": specificity_score(y_true, y_pred, pos_label, sample_weight),
        "precision": precision_score(y_true, y_pred, pos_label, sample_weight),
        "g_mean": geometric_mean_score(y_true, y_pred, sample_weight),
        "f1": f1_score(y_true, y_pred, pos_label, sample_weight),
        "f2": fbeta_score(y_true, y_pred, 2.0, pos_label, sample_weight),
        "pr_auc": (average_precision_score(y_true, y_score, pos_label, sample_weight)
                   if y_score is not None else float("nan")),
        "roc_auc": (roc_auc_score(y_true, y_score, pos_label, sample_weight)
                    if y_score is not None else float("nan")),
        "tmc": total_misclassification_cost(y_true, y_pred, c_fn, c_fp, pos_label, sample_weight),
    }
    out.update(binary_confusion(y_true, y_pred, pos_label, sample_weight))
    return out


def classification_report(y_true, y_pred, classes=None, digits=3):
    """Precisión, recall, F1 y soporte por clase."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    M = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(M).astype(float)
    prec, rec = _safe_ratio(tp, M.sum(0)), _safe_ratio(tp, M.sum(1))
    f1 = _safe_ratio(2 * prec * rec, prec + rec)
    sup = M.sum(1)
    width = max(12, max(len(str(c)) for c in classes) + 2)
    lines = [f"{'':>{width}}{'precision':>11}{'recall':>9}{'f1':>9}{'soporte':>9}"]
    for i, c in enumerate(classes):
        lines.append(f"{str(c):>{width}}{prec[i]:>11.{digits}f}{rec[i]:>9.{digits}f}"
                     f"{f1[i]:>9.{digits}f}{sup[i]:>9}")
    lines.append("")
    lines.append(f"{'accuracy':>{width}}{'':>20}{accuracy_score(y_true, y_pred):>9.{digits}f}{sup.sum():>9}")
    lines.append(f"{'balanced acc':>{width}}{'':>20}{balanced_accuracy_score(y_true, y_pred):>9.{digits}f}")
    lines.append(f"{'G-mean':>{width}}{'':>20}{geometric_mean_score(y_true, y_pred):>9.{digits}f}")
    return "\n".join(lines)


SCORERS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "g_mean": geometric_mean_score,
}


def get_scorer(scoring):
    """Nombre en SCORERS o función scoring(y_true, y_pred)."""
    if scoring is None:
        return accuracy_score
    if callable(scoring):
        return scoring
    if scoring not in SCORERS:
        raise ValueError(f"scoring desconocido {scoring!r}. Opciones: {sorted(SCORERS)}")
    return SCORERS[scoring]


# #############################################################################
# 6. VALIDACIÓN CRUZADA Y COMPARACIONES
# #############################################################################

def clone(estimator, **params):
    """Copia sin entrenar, con cambios opcionales de parámetros."""
    new = type(estimator)(**estimator.get_params())
    return new.set_params(**params) if params else new


def _as_estimator(e):
    return make_classifier(e) if isinstance(e, str) else e


def _as_estimator_dict(estimators):
    if isinstance(estimators, dict):
        return {k: _as_estimator(v) for k, v in estimators.items()}
    if not isinstance(estimators, (list, tuple)):
        estimators = [estimators]
    out = {}
    for e in estimators:
        est = _as_estimator(e)
        name = e if isinstance(e, str) else type(est).__name__
        base, i = name, 2
        while name in out:
            name, i = f"{base}_{i}", i + 1
        out[name] = est
    return out


def stratified_kfold(y, n_splits=5, random_state=None):
    """Genera (idx_train, idx_test) estratificados."""
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for f, part in enumerate(np.array_split(idx, n_splits)):
            folds[f].extend(part)
    all_idx = np.arange(len(y))
    for f in range(n_splits):
        test = np.sort(np.array(folds[f], dtype=int))
        yield np.setdiff1d(all_idx, test), test


def cross_val_score(estimator, X, y, cv=5, scale=True, scoring=None, random_state=42):
    """Puntaje por pliegue con CV estratificada (estandariza por pliegue si scale=True)."""
    estimator = _as_estimator(estimator)
    scorer = get_scorer(scoring)
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    scores = []
    for tr, te in stratified_kfold(y, cv, random_state):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        est = clone(estimator).fit(Xtr, y[tr])
        scores.append(scorer(y[te], est.predict(Xte)))
    return np.array(scores)


def compare_classifiers(estimators, X, y, cv=5, scale=True, scoring=None,
                        random_state=42, verbose=True):
    """Compara clasificadores con CV; regresa {nombre: (media, desv)} ordenado."""
    ests = _as_estimator_dict(estimators)
    res = {}
    for name, est in ests.items():
        s = cross_val_score(est, X, y, cv, scale, scoring, random_state)
        res[name] = (s.mean(), s.std())
    res = dict(sorted(res.items(), key=lambda kv: -kv[1][0]))
    if verbose:
        w = max(len(n) for n in res) + 2
        print(f"{'Clasificador':<{w}}{'media':>8}{'desv':>8}")
        for n, (mu, sd) in res.items():
            print(f"{n:<{w}}{mu:>8.3f}{sd:>8.3f}")
    return res


def _metric_label(m):
    if isinstance(m, tuple):
        return m[0]
    return m if isinstance(m, str) else getattr(m, "__name__", repr(m))


def compare_metrics(estimators, metrics, X, y, cv=5, scale=True, scoring=None,
                    random_state=0, verbose=True):
    """
    Cada clasificador con cada métrica de distancia. metrics: nombres, funciones
    f(u, v) o tuplas (etiqueta, métrica, params).
    Regresa {(clasificador, etiqueta): (media, desv)}.
    """
    ests = _as_estimator_dict(estimators)
    labels = [_metric_label(m) for m in metrics]
    results = {}
    for name, est in ests.items():
        if "metric" not in est.get_params():
            s = cross_val_score(est, X, y, cv, scale, scoring, random_state)
            for lab in labels:
                results[(name, lab)] = (s.mean(), s.std())
            continue
        for m, lab in zip(metrics, labels):
            metric, params = (m[1], m[2]) if isinstance(m, tuple) else (m, None)
            try:
                e = clone(est, metric=metric, metric_params=params)
                s = cross_val_score(e, X, y, cv, scale, scoring, random_state)
                results[(name, lab)] = (s.mean(), s.std())
            except Exception as ex:
                results[(name, lab)] = (np.nan, np.nan)
                if verbose:
                    print(f"  [aviso] {name} + {lab}: {ex}")
    if verbose:
        print_results_table(results, list(ests), labels)
    return results


def print_results_table(results, est_names, metric_labels):
    best = {e: np.nanmax([results[(e, m)][0] for m in metric_labels]) for e in est_names}
    w = max(12, max(len(m) for m in metric_labels) + 2)
    cw = max(16, max(len(e) for e in est_names) + 2)
    print("".ljust(w) + "".join(e.rjust(cw) for e in est_names))
    for m in metric_labels:
        row = m.ljust(w)
        for e in est_names:
            mu, sd = results[(e, m)]
            txt = "—" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}" + ("*" if np.isclose(mu, best[e]) else " ")
            row += txt.rjust(cw)
        print(row)
