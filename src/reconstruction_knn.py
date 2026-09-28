"""
reconstruction_knn.py — KNN por reconstrucción matricial
========================================================

    CMKNN       'cm_knn'        Zhang, Li, Zong, Zhu & Cheng, "Learning k for kNN
                                Classification", ACM TIST 8(3), 2017.
    OneStepKNN  'one_step_knn'  Zhang & Li, "KNN Classification With One-Step
                                Computation", IEEE TKDE 35(3), 2023 (arXiv:2012.06047).

Notación (la de ambos artículos y del ABP):
    X ∈ R^{n×d}  entrenamiento (filas = muestras)
    Y ∈ R^{d×m}  prueba (columnas = muestras)      [en código: Y = X_test.T]
    W ∈ R^{n×m}  matriz de correlación / relación:  y_j ≈ X^T w_j

Ambos se resuelven con IRLS (Iteratively Reweighted Least Squares), tal como
en los artículos: cada norma no diferenciable se sustituye en cada iteración
por una cuadrática con pesos diagonales calculados en la iteración anterior.

    CM-KNN  (Alg. 1 del artículo; una columna a la vez)
        w_j ← (X X^T + ρ1 D_j + ρ2 D̃ + ρ3 X L X^T)^{-1} X y_j
        (D_j)_kk = 1 / (2|w_kj|),   D̃_kk = 1 / (2‖w^k‖_2)

    One-Step (Ec. 18 del artículo; todas las columnas a la vez)
        W̃ ← (X X^T + α Q + β F)^{-1} X Y^T
        q_jj = Σ_v ‖W̃^v‖_2 / ‖W̃^j‖_2
        f_ii = Σ_g I_Gg(i) ‖W̃_Gg‖_1 / ‖W̃^i‖_1

IRLS no produce ceros exactos: los coeficientes que tiende a anular se
consideran cero con un umbral (ver `zero_tol` / `sparsify`).
"""

import numpy as np

from classifiers import (BaseClassifier, KNNClassifier, register_classifier,
                         pairwise_distances, _normalize_rows, _EPS)

__all__ = ["CMKNN", "OneStepKNN", "feature_laplacian", "fuzzy_cmeans"]


# =============================================================================
# Utilidades comunes
# =============================================================================

def feature_laplacian(X, n_neighbors=5, sigma=None):
    """
    Laplaciano L = D - S (d×d) del grafo entre CARACTERÍSTICAS (columnas de X)
    para el término LPP tr(W^T X L X^T W). El artículo de CM-KNN no especifica
    la similitud; se usa el kernel de calor de LPP, S_ab = exp(-‖f_a - f_b‖²/2σ²),
    sobre un grafo kNN simétrico.
    """
    F = np.asarray(X, dtype=float).T
    D = pairwise_distances(F, F, "euclidean")
    sigma = sigma or (D[D > 0].mean() if np.any(D > 0) else 1.0)
    S = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(S, 0.0)
    if n_neighbors is not None and n_neighbors < F.shape[0] - 1:
        keep = np.zeros_like(S, dtype=bool)
        np.put_along_axis(keep, np.argsort(-S, axis=1)[:, :n_neighbors], True, axis=1)
        S = np.where(keep | keep.T, S, 0.0)
    return np.diag(S.sum(1)) - S


def fuzzy_cmeans(X, n_clusters=2, m=2.0, max_iter=100, tol=1e-5, random_state=42):
    """
    Fuzzy C-Means (Bezdek). Regresa U (n_clusters, n): grados de pertenencia.
    Equivale a `fcm` de MATLAB con exponente 2, que usa el código de los autores
    de One-Step KNN para formar los grupos.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    U = rng.random((n_clusters, n))
    U /= U.sum(0, keepdims=True)
    prev = np.inf
    for _ in range(max_iter):
        Um = U ** m
        C = Um @ X / Um.sum(1, keepdims=True)
        D = np.maximum(pairwise_distances(C, X), _EPS)          # (c, n)
        obj = float(np.sum(Um * D ** 2))
        U = 1.0 / np.sum((D[:, None, :] / D[None, :, :]) ** (2.0 / (m - 1)), axis=1)
        if np.isfinite(prev) and abs(prev - obj) < tol * max(prev, _EPS):
            break
        prev = obj
    return U


def _batched_solve(A0, diag, B, chunk=256):
    """Resuelve (A0 + diag(diag[:, j])) w_j = B[:, j] para cada columna j."""
    n, m = B.shape
    W = np.empty((n, m))
    idx = np.arange(n)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        A = np.broadcast_to(A0, (e - s, n, n)).copy()
        A[:, idx, idx] += diag[:, s:e].T
        W[:, s:e] = np.linalg.solve(A, B[:, s:e].T[..., None])[..., 0].T
    return W


def _fista(grad, prox, W0, lipschitz, max_iter=300, tol=1e-5):
    step = 1.0 / lipschitz
    W = W0.copy(); Z = W0.copy(); t = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        W_new = prox(Z - step * grad(Z), step)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = W_new + ((t - 1) / t_new) * (W_new - W)
        diff = np.linalg.norm(W_new - W) / (np.linalg.norm(W) + _EPS)
        W, t = W_new, t_new
        if diff < tol:
            break
    return W, it


class _ReconstructionBase(BaseClassifier):
    """Resuelve la prueba por lotes (batch_size) y guarda W_, n_iter_."""

    def _solve_all(self, X):
        m = len(X)
        bs = self.batch_size or m
        Ws, iters = [], []
        for s in range(0, m, bs):
            W, it = self._solve(X[s:s + bs])
            Ws.append(W); iters.append(it)
        self.n_iter_ = int(np.max(iters)) if iters else 0
        return np.hstack(Ws) if Ws else np.zeros((len(self.X_), 0))

    def learn_W(self, X):
        """W (n_train, n_test) para las muestras de prueba X (n_test, d)."""
        return self._solve_all(self._check_X(X))


# =============================================================================
# CM-KNN
# =============================================================================

@register_classifier("cm_knn")
class CMKNN(_ReconstructionBase):
    """
    CM-KNN (Zhang et al., 2017): aprende un k distinto para cada muestra de prueba.

        min_W ‖X^T W - Y‖_F² + ρ1 ‖W‖_1 + ρ2 ‖W‖_{2,1} + ρ3 tr(W^T X L X^T W)

    Tras optimizar, las muestras de entrenamiento con w_ij ≠ 0 son los vecinos
    de y_j, k_j = ‖w_j‖_0, y la clase se decide por mayoría entre ellos
    (Alg. 2 del artículo). Las filas que ℓ2,1 anula (muestras ruidosas) no
    participan en ninguna predicción.

    CON LOS VALORES POR DEFECTO ES LA FORMULACIÓN DEL ARTÍCULO (solver IRLS,
    ℓ1 sin ponderar, coeficientes de cualquier signo, voto mayoritario).
    Extensiones opcionales propias (no están en el artículo):
        locality=True  : ℓ1 ponderada por distancia, Σ s_ij |w_ij|,
                         s_ij = d(x_i, y_j) / media_i d(x_i, y_j)
        positive=True  : solo los coeficientes > 0 cuentan como vecinos
        vote='weighted': voto ponderado por |w_ij|
        neighbor_rule='k_nearest': usa k_j = ‖w_j‖_0 pero votan los k_j más cercanos
        solver='fista' : gradiente proximal en vez de IRLS

    Parámetros
    ----------
    rho1, rho2, rho3 : pesos de ℓ1, ℓ2,1 y LPP
    n_feature_neighbors, sigma : grafo entre características para L
    zero_tol   : |w_ij| ≤ zero_tol · max_i |w_ij| se considera cero (IRLS no da ceros exactos)
    fallback_k : k del KNN de respaldo si w_j queda vacío
    batch_size : muestras de prueba resueltas juntas (ℓ2,1 acopla las columnas del lote;
                 None = todas, como en el artículo)
    max_iter, tol : paro de IRLS (cambio relativo de W)

    Atributos tras predict: W_, k_, n_iter_.
    """

    def __init__(self, rho1=1.0, rho2=0.1, rho3=0.01, solver="irls", positive=False,
                 locality=False, neighbor_rule="coefficients", vote="majority",
                 metric="euclidean", metric_params=None, n_feature_neighbors=5, sigma=None,
                 fallback_k=5, zero_tol=0.05, batch_size=None, max_iter=50, tol=1e-4):
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.solver = solver
        self.positive = positive
        self.locality = locality
        self.neighbor_rule = neighbor_rule
        self.vote = vote
        self.metric = metric
        self.metric_params = metric_params
        self.n_feature_neighbors = n_feature_neighbors
        self.sigma = sigma
        self.fallback_k = fallback_k
        self.zero_tol = zero_tol
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, X, y):
        L = feature_laplacian(X, self.n_feature_neighbors, self.sigma)
        self._G = X @ X.T
        self._M = X @ L @ X.T
        self._A0 = self._G + self.rho3 * self._M
        self._fallback = None

    def _locality(self, Y):
        if not self.locality:
            return np.ones((len(self.X_), len(Y)))
        D = self._dist(self.X_, Y)
        return D / (D.mean(0, keepdims=True) + _EPS)

    def _solve(self, Y):
        B = self.X_ @ Y.T
        S = self._locality(Y)
        n = len(self.X_)
        if self.solver == "fista":
            lip = 2.0 * np.linalg.eigvalsh(self._A0)[-1] + _EPS

            def grad(W):
                return 2.0 * (self._A0 @ W - B)

            def prox(V, t):
                thr = t * self.rho1 * S
                V = (np.maximum(V - thr, 0.0) if self.positive
                     else np.sign(V) * np.maximum(np.abs(V) - thr, 0.0))
                nr = np.linalg.norm(V, axis=1, keepdims=True)
                return V * np.maximum(0.0, 1.0 - t * self.rho2 / (nr + _EPS))

            return _fista(grad, prox, np.zeros((n, len(Y))), lip, self.max_iter * 10, self.tol * 0.1)
        if self.solver != "irls":
            raise ValueError("solver debe ser 'irls' o 'fista'")
        lam = 1e-3 * np.trace(self._G) / n
        W = np.linalg.solve(self._A0 + lam * np.eye(n), B)          # inicio (ridge)
        it = 0
        for it in range(1, self.max_iter + 1):
            eps = 1e-6 * (np.abs(W).max() + _EPS)
            r = np.linalg.norm(W, axis=1)                              # ‖w^k‖_2
            diag = (self.rho2 / (2.0 * np.maximum(r, eps)))[:, None] \
                + self.rho1 * S / (2.0 * np.maximum(np.abs(W), eps))
            W_new = _batched_solve(self._A0, diag, B)
            diff = np.linalg.norm(W_new - W) / (np.linalg.norm(W) + _EPS)
            W = W_new
            if diff < self.tol:
                break
        return W, it

    def _support(self, W):
        A = np.abs(W)
        mx = A.max(0, keepdims=True)
        mask = A > self.zero_tol * np.where(mx > 0, mx, np.inf)
        if self.positive:
            mask &= W > 0
        return mask

    def _neighbor_sets(self, W, X):
        mask = self._support(W)
        D = None
        out = []
        for j in range(W.shape[1]):
            idx = np.where(mask[:, j])[0]
            if len(idx) == 0:
                if self._fallback is None:
                    self._fallback = KNNClassifier(
                        n_neighbors=min(self.fallback_k, len(self.X_)),
                        metric=self._metric_, metric_params=self._metric_params_,
                    ).fit(self.X_, self.y_)
                idx = self._fallback.kneighbors(X[j:j + 1], return_distance=False)[0]
                out.append((idx, np.ones(len(idx))))
                continue
            if self.neighbor_rule == "k_nearest":
                if D is None:
                    D = self._dist(X, self.X_)
                idx = np.argsort(D[j], kind="stable")[:len(idx)]
                out.append((idx, np.ones(len(idx))))
            elif self.neighbor_rule == "coefficients":
                out.append((idx, np.abs(W[idx, j])))
            else:
                raise ValueError("neighbor_rule debe ser 'coefficients' o 'k_nearest'")
        return out

    def _predict_proba(self, X):
        self.W_ = W = self._solve_all(X)
        neigh = self._neighbor_sets(W, X)
        self.k_ = np.array([len(i) for i, _ in neigh])
        P = np.zeros((len(X), self.n_classes_))
        for j, (idx, w) in enumerate(neigh):
            ww = w if self.vote == "weighted" else np.ones_like(w)
            P[j] = np.bincount(self.y_[idx], weights=ww, minlength=self.n_classes_)
        return _normalize_rows(P)


# =============================================================================
# One-Step KNN
# =============================================================================

@register_classifier("one_step_knn")
class OneStepKNN(_ReconstructionBase):
    """
    One-Step KNN (Zhang & Li, 2023). Formulación del artículo:

        min_{W,θ} ‖Y^T - X^T Θ W‖_F² + α ‖W‖²_F + β φ(W)    s.t. θ > 0, 1^T θ = 1   (Ec. 4)

    con Θ = diag(θ) y el Group Lasso "exclusivo" φ(W) = Σ_g ‖W_{Gg}‖²_1 (dispersión
    dentro de cada grupo, no dispersión entre grupos). Con W̃ = ΘW se resuelve

        min_W̃ ‖Y^T - X^T W̃‖_F² + α ‖W̃‖²_{2,1} + β Σ_g ‖W̃_{Gg}‖²_1          (Ec. 7)

    (el ABP escribe β‖W̃‖²_1, que es el caso de un solo grupo: groups='single').
    IRLS (Ec. 18):  W̃ ← (X X^T + α Q + β F)^{-1} X Y^T,   un solo sistema n×n.
    Pesos de muestra (Ec. 8):  θ_j = (‖W̃^j‖_2 + ‖W̃^j‖_1) / Σ_j' (‖W̃^j'‖_2 + ‖W̃^j'‖_1).
    Regla (Ec. 6):  v(y_i) = argmax_c (1/‖W_i‖_0) Σ_{j: W_ji ≠ 0} Θ_jj W_ji I(l_j = c),
    con W = Θ^{-1} W̃ (por lo que Θ_jj W_ji = W̃_ji) y K_i = ‖W_i‖_0.

    Parámetros
    ----------
    alpha, beta : pesos de ‖W̃‖²_{2,1} y del Group Lasso. Como las normas van al
                  cuadrado sobre toda la matriz, su efecto crece con el número de
                  muestras de prueba resueltas juntas (batch_size).
    groups      : 'fcm'    grupos de muestras de entrenamiento por Fuzzy C-Means
                           (código de los autores: 2 grupos, pertenencia > 0.01)
                  'single' un solo grupo (la ecuación tal como la escribe el ABP)
    n_groups, membership_threshold : parámetros de 'fcm'
    sparsify    : 'mean'     W̃_ji < media de la columna -> 0 (lo que hace el código de
                             los autores, pues IRLS no da ceros exactos)
                  'relative' |W̃_ji| ≤ zero_tol · max_j |W̃_ji| -> 0
    batch_size  : None = todas las muestras de prueba juntas (como en el artículo)
    max_iter, tol : paro por cambio relativo del objetivo (artículo: 1e-5)
    random_state: semilla de Fuzzy C-Means

    Atributos: groups_ (fit); W_ (=W̃), theta_, k_, class_scores_, n_iter_ (predict).
    predict_proba = softmax de los puntajes por clase (puntaje, no probabilidad).

    Nota sobre la complejidad O(1): W̃ depende de Y, así que las muestras de
    prueba deben conocerse cuando se "entrena"; el costo real por lote es
    O(n³ + n²(d + m)) por iteración y en la prueba solo queda la agregación de
    etiquetas O(K).
    """

    def __init__(self, alpha=1.0, beta=0.1, groups="fcm", n_groups=2,
                 membership_threshold=0.01, sparsify="mean", zero_tol=0.05,
                 batch_size=None, max_iter=100, tol=1e-5, random_state=42):
        self.alpha = alpha
        self.beta = beta
        self.groups = groups
        self.n_groups = n_groups
        self.membership_threshold = membership_threshold
        self.sparsify = sparsify
        self.zero_tol = zero_tol
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fit(self, X, y):
        self._G = X @ X.T
        n = len(X)
        if self.groups == "single":
            I = np.ones((1, n), dtype=bool)
        elif self.groups == "fcm":
            U = fuzzy_cmeans(X, self.n_groups, random_state=self.random_state)
            I = U > self.membership_threshold
        else:
            raise ValueError("groups debe ser 'fcm' o 'single'")
        self.groups_ = I                                     # (n_groups, n) indicadoras I_Gg

    def _objective(self, Wt, B_res):
        r2 = np.linalg.norm(Wt, axis=1)
        gl1 = np.array([np.abs(Wt[g]).sum() for g in self.groups_])
        return B_res + self.alpha * r2.sum() ** 2 + self.beta * np.sum(gl1 ** 2)

    def _solve(self, Y):
        X = self.X_
        B = X @ Y.T
        n = len(X)
        lam = 1e-3 * np.trace(self._G) / n
        Wt = np.linalg.solve(self._G + lam * np.eye(n), B)           # inicio (ridge)
        prev = None
        it = 0
        for it in range(1, self.max_iter + 1):
            eps = 1e-8 * (np.abs(Wt).max() + _EPS)
            r2 = np.maximum(np.linalg.norm(Wt, axis=1), eps)
            r1 = np.maximum(np.abs(Wt).sum(1), eps)
            q = r2.sum() / r2
            gl1 = np.array([np.abs(Wt[g]).sum() for g in self.groups_])
            f = (self.groups_ * gl1[:, None]).sum(0) / r1
            Wt = np.linalg.solve(self._G + np.diag(self.alpha * q + self.beta * f), B)
            res = np.linalg.norm(Y.T - X.T @ Wt) ** 2
            obj = self._objective(Wt, res)
            if prev is not None and abs(obj - prev) / max(prev, _EPS) <= self.tol:
                break
            prev = obj
        return Wt, it

    def decision_scores(self, X):
        """Puntajes (n_test, C) de la Ec. 6."""
        X = self._check_X(X)
        self.W_ = Wt = self._solve_all(X)
        r2, r1 = np.linalg.norm(Wt, axis=1), np.abs(Wt).sum(1)
        theta = (r2 + r1) / max((r2 + r1).sum(), _EPS)              # Ec. 8
        self.theta_ = theta
        if self.sparsify == "mean":
            mask = Wt >= Wt.mean(0, keepdims=True)
        elif self.sparsify == "relative":
            A = np.abs(Wt)
            mask = A > self.zero_tol * A.max(0, keepdims=True)
        else:
            raise ValueError("sparsify debe ser 'mean' o 'relative'")
        self.k_ = mask.sum(0)
        W = Wt / np.maximum(theta, _EPS)[:, None]                    # W = Θ^{-1} W̃
        contrib = np.where(mask, theta[:, None] * W, 0.0)             # Θ_jj W_ji
        S = np.stack([contrib[self.y_ == c].sum(0) for c in range(self.n_classes_)], axis=1)
        S = S / np.maximum(self.k_, 1)[:, None]
        self.class_scores_ = S
        return S

    def _predict_proba(self, X):
        S = self.decision_scores(X)
        Z = S - S.max(1, keepdims=True)
        T = np.median(np.abs(S - S.mean(1, keepdims=True))) + _EPS
        E = np.exp(Z / T)
        return E / E.sum(1, keepdims=True)

    def _predict(self, X):
        return np.argmax(self.decision_scores(X), axis=1)
