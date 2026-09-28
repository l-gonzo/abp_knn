"""
cs_knn.py — KNN sensibles al costo (Qin, Wang, Zhang & Zhang, 2013)
===================================================================

    DirectCSKNN     'direct_cs_knn'     riesgo Bayesiano mínimo dentro de la vecindad
    DistancesCSKNN  'distances_cs_knn'  deformación de la métrica por costo/frecuencia

Convención de costos (igual que el ABP):
    C[i, j] = costo de clasificar una muestra de la clase ω_j como ω_i,
    C[i, i] = 0. Binario (ω1 = mina, positiva):
        C[0, 1] = C01 -> Falso Negativo (mina clasificada como roca)
        C[1, 0] = C10 -> Falso Positivo (roca clasificada como mina)
    Usa classifiers.binary_cost_matrix(c_fn, c_fp) para construirla.

cost_matrix también acepta:
    None        -> costo 0-1 (el modelo se reduce a KNN estándar)
    'balanced'  -> costo dinámico por desbalance: C[i, j] = n_max / n_j (i != j)
"""

import numpy as np

from classifiers import (KNNClassifier, register_classifier, total_cost,
                         _normalize_rows)

__all__ = ["DirectCSKNN", "DistancesCSKNN", "DistanceCSKNN", "resolve_cost_matrix"]


def resolve_cost_matrix(cost_matrix, y, n_classes):
    """Convierte cost_matrix (None | 'balanced' | array) en un arreglo (C, C) validado."""
    if cost_matrix is None:
        return 1.0 - np.eye(n_classes)
    if isinstance(cost_matrix, str):
        if cost_matrix != "balanced":
            raise ValueError("cost_matrix como texto solo admite 'balanced'")
        counts = np.bincount(y, minlength=n_classes).astype(float)
        col = counts.max() / np.maximum(counts, 1.0)      # costo de equivocarse con la clase j
        C = np.tile(col, (n_classes, 1))
        np.fill_diagonal(C, 0.0)
        return C
    C = np.asarray(cost_matrix, dtype=float)
    if C.shape != (n_classes, n_classes):
        raise ValueError(f"cost_matrix debe ser {n_classes}x{n_classes}")
    if np.any(C < 0) or np.any(np.diag(C) != 0):
        raise ValueError("cost_matrix debe ser no negativa con diagonal cero")
    return C


class _CostSensitiveMixin:
    """cost_matrix_, expected_cost(), total_cost() y decisión por mínimo riesgo."""

    def _resolve_costs(self, y):
        self.cost_matrix_ = resolve_cost_matrix(self.cost_matrix, y, self.n_classes_)
        if self.n_classes_ == 2:
            C01, C10 = self.cost_matrix_[0, 1], self.cost_matrix_[1, 0]
            # Se predice ω1 si P(ω1|x) > C10 / (C01 + C10)  <=>  ω0 solo si P(ω0|x) > C01 / (C01 + C10)
            self.threshold_ = C10 / (C01 + C10)

    def expected_cost(self, X):
        """R(ω_i|x) = sum_j C_ij P(ω_j|x); matriz (n, n_clases)."""
        return self._expected_cost(self._check_X(X))

    def _expected_cost(self, X):
        return self._predict_proba(X) @ self.cost_matrix_.T

    def _predict(self, X):
        return np.argmin(self._expected_cost(X), axis=1)

    def total_cost(self, X, y):
        return total_cost(y, self.predict(X), self.cost_matrix_, self.classes_)


@register_classifier("direct_cs_knn")
class DirectCSKNN(_CostSensitiveMixin, KNNClassifier):
    """
    Direct-CS-KNN.

    1) Estima P(ω_j|x) con los k vecinos, suavizada:
         m-estimate : (k_j + m p_j) / (k + m)
         laplace    : (k_j + 1) / (k + C)            (= m-estimate con m=C y p uniforme)
         none       : k_j / k
    2) Elige la clase de mínimo riesgo condicional
         R(ω_i|x) = sum_j C_ij P(ω_j|x).
    En binario equivale a predecir ω1 cuando P(ω1|x) > τ1 = C10/(C01+C10)
    (atributo threshold_), en vez del umbral 0.5 del voto mayoritario.

    Parámetros
    ----------
    n_neighbors : int
    cost_matrix : None | 'balanced' | array (C, C)
    smoothing   : 'm-estimate' | 'laplace' | 'none'
    m           : peso del prior en el m-estimate
    prior       : 'uniform' (p_j = 1/C) | 'train' (frecuencias de entrenamiento)
    k_grid      : None (usa n_neighbors) o valores de k; si se da, se elige el k de
                  MÍNIMO COSTO total con validación cruzada interna (una de las
                  mejoras que proponen Qin et al., 2013). Atributo: best_k_.
    cv, random_state : pliegues y semilla de esa selección
    weights, metric, p, metric_params : como KNNClassifier
    """

    def __init__(self, n_neighbors=5, cost_matrix=None, smoothing="m-estimate", m=1.0,
                 prior="uniform", k_grid=None, cv=5, random_state=42, weights="uniform",
                 metric="euclidean", p=2, metric_params=None):
        super().__init__(n_neighbors, weights, metric, p, metric_params)
        self.cost_matrix = cost_matrix
        self.smoothing = smoothing
        self.m = m
        self.prior = prior
        self.k_grid = k_grid
        self.cv = cv
        self.random_state = random_state

    def _fit(self, X, y):
        super()._fit(X, y)
        self._resolve_costs(y)
        if self.prior == "train":
            self.class_prior_ = np.bincount(y, minlength=self.n_classes_) / len(y)
        elif self.prior == "uniform":
            self.class_prior_ = np.full(self.n_classes_, 1.0 / self.n_classes_)
        else:
            raise ValueError("prior debe ser 'uniform' o 'train'")
        self.best_k_ = self.n_neighbors
        if self.k_grid is not None:
            self.best_k_ = self._select_k_min_cost(X, y)

    def _smooth(self, counts):
        total = counts.sum(1, keepdims=True)
        if self.smoothing == "none":
            return _normalize_rows(counts)
        if self.smoothing == "laplace":
            return (counts + 1.0) / (total + self.n_classes_)
        if self.smoothing == "m-estimate":
            return (counts + self.m * self.class_prior_) / (total + self.m)
        raise ValueError("smoothing debe ser 'm-estimate', 'laplace' o 'none'")

    def _select_k_min_cost(self, X, y):
        """k con el menor costo total fuera de pliegue (desempate: k menor)."""
        from classifiers import stratified_kfold
        ks = sorted(set(int(k) for k in self.k_grid))
        n_splits = int(min(self.cv, max(np.bincount(y).min(), 2)))
        cost = np.zeros(len(ks))
        for tr, te in stratified_kfold(y, n_splits, self.random_state):
            kmax = min(max(ks), len(tr))
            knn = KNNClassifier(kmax, self.weights, self.metric, self.p, self.metric_params).fit(X[tr], y[tr])
            d, idx = knn.kneighbors(X[te], n_neighbors=kmax)
            lab = y[tr][idx]
            for a, k in enumerate(ks):
                kk = min(k, kmax)
                w = knn._neighbor_weights(d[:, :kk])
                counts = np.stack([(w * (lab[:, :kk] == c)).sum(1) for c in range(self.n_classes_)], 1)
                pred = np.argmin(self._smooth(counts) @ self.cost_matrix_.T, axis=1)
                cost[a] += self.cost_matrix_[pred, y[te]].sum()
        self.cv_costs_ = dict(zip(ks, cost))
        return ks[int(np.argmin(cost))]

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return super().kneighbors(X, n_neighbors or getattr(self, "best_k_", self.n_neighbors),
                                  return_distance)

    def _predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return self._smooth(self._class_votes(self._neighbor_weights(dist), idx))


@register_classifier("distances_cs_knn")
class DistancesCSKNN(_CostSensitiveMixin, KNNClassifier):
    """
    Distances-CS-KNN: deforma la métrica escalando la distancia a cada muestra de
    entrenamiento x_n ∈ ω_j por un factor de costo/desbalance de su clase:

        d_CS(x, x_n) = d(x, x_n) / [ω(c_j)]^γ

    y aplica KNN sobre d_CS. Las muestras de clases caras/minoritarias parecen
    más cercanas, lo que agranda sus celdas de Voronoi.

    Factor ω(c_j) (parámetro `factor`):
        'cost'           : C(pred, ω_j) = max_{i≠j} C[i, j]  (binario: ω(c1)=C01, ω(c0)=C10)
        'frequency'      : n_max / n_j
        'cost_frequency' : producto de ambos

    Formulación alternativa (formulation='weighted_cost'): no deforma la métrica;
    cada uno de los k vecinos aporta su costo ponderado por la inversa de su
    distancia y se elige la clase de menor costo acumulado,
        R(ω_i|x) = Σ_{n ∈ N_k(x)} f(d_n) C[i, y_n],   f(d) = 1/(d + eps)^power.
    Es como recordamos la versión de Qin et al. (2013); no pudimos consultar el
    texto completo, así que la formulación por defecto es la que define el ABP.

    Parámetros
    ----------
    n_neighbors : int
    cost_matrix : None | 'balanced' | array (C, C)
    formulation : 'metric' (d_CS del ABP) | 'weighted_cost' (ver arriba)
    gamma       : severidad γ > 0 (γ = 0 -> KNN estándar)       [formulation='metric']
    factor      : 'cost' | 'frequency' | 'cost_frequency'        [formulation='metric']
    decision    : 'vote' (mayoría sobre la vecindad deformada)
                  | 'risk' (además, mínimo riesgo con P suavizada por Laplace)
    power, eps  : ponderación por distancia                     [formulation='weighted_cost']
    weights, metric, p, metric_params : como KNNClassifier

    Atributo: class_factor_ (ω(c_j) por clase).

    Nota: en alta dimensión las distancias se concentran (su razón máx/mín es
    cercana a 1), así que γ debe ser pequeño: con C01 = 10, γ = 1 divide entre 10
    la distancia a toda mina y el modelo predice "mina" siempre; γ ≈ 0.1
    (factor 10^0.1 ≈ 1.26) desplaza la frontera sin colapsarla.
    """

    def __init__(self, n_neighbors=5, cost_matrix=None, formulation="metric", gamma=0.1,
                 factor="cost", decision="vote", power=1.0, eps=1e-8, weights="uniform",
                 metric="euclidean", p=2, metric_params=None):
        super().__init__(n_neighbors, weights, metric, p, metric_params)
        self.cost_matrix = cost_matrix
        self.formulation = formulation
        self.gamma = gamma
        self.factor = factor
        self.decision = decision
        self.power = power
        self.eps = eps

    def _fit(self, X, y):
        super()._fit(X, y)
        self._resolve_costs(y)
        C = self.cost_matrix_
        cost_f = np.array([np.max(np.delete(C[:, j], j)) for j in range(self.n_classes_)])
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        freq_f = counts.max() / np.maximum(counts, 1.0)
        if self.factor == "cost":
            f = cost_f
        elif self.factor == "frequency":
            f = freq_f
        elif self.factor == "cost_frequency":
            f = cost_f * freq_f
        else:
            raise ValueError("factor debe ser 'cost', 'frequency' o 'cost_frequency'")
        if np.any(f <= 0):
            raise ValueError("Los factores de clase deben ser > 0")
        self.class_factor_ = f

    def _distance_matrix(self, X):
        if self.formulation == "weighted_cost":
            return self._dist(X, self.X_)
        scale = self.class_factor_[self.y_] ** self.gamma      # (n_train,)
        return self._dist(X, self.X_) / scale[None, :]

    def _dist_weights(self, dist):
        return _normalize_rows(1.0 / (dist + self.eps) ** self.power)

    def _predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        if self.formulation == "weighted_cost":
            return _normalize_rows(self._class_votes(self._dist_weights(dist), idx))
        votes = self._class_votes(self._neighbor_weights(dist), idx)
        if self.decision == "risk":
            return (votes + 1.0) / (votes.sum(1, keepdims=True) + self.n_classes_)
        return _normalize_rows(votes)

    def _expected_cost(self, X):
        if self.formulation == "weighted_cost":
            dist, idx = self.kneighbors(X)
            w = self._dist_weights(dist)
            return np.einsum("mn,imn->mi", w, self.cost_matrix_[:, self.y_[idx]])
        return super()._expected_cost(X)

    def _predict(self, X):
        if self.formulation not in ("metric", "weighted_cost"):
            raise ValueError("formulation debe ser 'metric' o 'weighted_cost'")
        if self.formulation == "weighted_cost" or self.decision == "risk":
            return np.argmin(self._expected_cost(X), axis=1)
        if self.decision == "vote":
            return np.argmax(self._predict_proba(X), axis=1)
        raise ValueError("decision debe ser 'vote' o 'risk'")


# Alias con el nombre usado en versiones anteriores
DistanceCSKNN = DistancesCSKNN
