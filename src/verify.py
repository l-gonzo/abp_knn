"""
verify.py — Verificación de las implementaciones desde cero contra scikit-learn / scipy.

scikit-learn y scipy se usan AQUÍ solo como referencia de validación; ningún
clasificador ni métrica del proyecto depende de ellos.

    python verify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifiers import *                                    # noqa: F401,F403
from cs_knn import DirectCSKNN, DistancesCSKNN
from reconstruction_knn import CMKNN, OneStepKNN
from pipeline import load_sonar

from scipy.spatial.distance import cdist
from sklearn import metrics as M
from sklearn.decomposition import PCA as SKPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

FAILS = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALLA ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    rng = np.random.default_rng(42)
    X, y = load_sonar()
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xs, Xts = sc.transform(Xtr), sc.transform(Xte)

    print("Métricas de distancia vs scipy.cdist")
    A, B = rng.normal(size=(7, 5)), rng.normal(size=(9, 5))
    VI = np.linalg.inv(np.cov(np.vstack([A, B]).T))
    for m, pr in [("euclidean", {}), ("sqeuclidean", {}), ("cityblock", {}), ("chebyshev", {}),
                  ("minkowski", {"p": 3}), ("seuclidean", {"V": rng.random(5) + .1}),
                  ("mahalanobis", {"VI": VI}), ("cosine", {}), ("correlation", {}),
                  ("canberra", {}), ("braycurtis", {})]:
        check(np.allclose(pairwise_distances(A, B, m, **pr), cdist(A, B, m, **pr)), m)

    print("Clasificadores vs scikit-learn (Sonar)")
    for w in ("uniform", "distance"):
        a = KNNClassifier(5, weights=w).fit(Xs, ytr).predict(Xts)
        b = KNeighborsClassifier(5, weights=w).fit(Xs, ytr).predict(Xts)
        check((a == b).all(), f"KNN k=5 weights={w}")
    a, b = GaussianNaiveBayes().fit(Xtr, ytr), GaussianNB().fit(Xtr, ytr)
    check(np.allclose(a.predict_proba(Xte), b.predict_proba(Xte)), "Naive Bayes (probabilidades)")
    check((MinimumDistanceClassifier().fit(Xtr, ytr).predict(Xte)
           == NearestCentroid().fit(Xtr, ytr).predict(Xte)).all(), "Mínima distancia")
    p, s = PCA(0.85).fit(Xs), SKPCA().fit(Xs)
    k = p.n_components_
    check(k == np.searchsorted(np.cumsum(s.explained_variance_ratio_), 0.85) + 1, f"PCA k={k} (85 %)")
    check(np.allclose(np.abs(p.transform(Xts)), np.abs(s.transform(Xts)[:, :k])), "PCA transform")

    print("Métricas de evaluación vs scikit-learn (con y sin sample_weight)")
    yt, yp = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    sc_ = rng.random(300) + 0.3 * yt
    for sw in (None, rng.random(300) * 3):
        t = " (ponderada)" if sw is not None else ""
        check(np.isclose(balanced_accuracy_score(yt, yp, sw), M.balanced_accuracy_score(yt, yp, sample_weight=sw)), "balanced accuracy" + t)
        check(np.isclose(fbeta_score(yt, yp, 2, 1, sw), M.fbeta_score(yt, yp, beta=2, sample_weight=sw)), "F2" + t)
        check(np.isclose(f1_score(yt, yp, 1, sw), M.f1_score(yt, yp, sample_weight=sw)), "F1" + t)
        check(np.isclose(average_precision_score(yt, sc_, 1, sw), M.average_precision_score(yt, sc_, sample_weight=sw)), "PR-AUC (AP)" + t)
        check(np.isclose(roc_auc_score(yt, sc_, 1, sw), M.roc_auc_score(yt, sc_, sample_weight=sw)), "ROC-AUC" + t)
        f1_, t1_, _ = roc_curve(yt, sc_, 1, sw)
        f2_, t2_, _ = M.roc_curve(yt, sc_, sample_weight=sw, drop_intermediate=False)
        check(np.allclose(f1_, f2_) and np.allclose(t1_, t2_), "curva ROC" + t)
    tn, fp, fn, tp = M.confusion_matrix(yt, yp).ravel()
    c = binary_confusion(yt, yp)
    check((c["tp"], c["tn"], c["fp"], c["fn"]) == (tp, tn, fp, fn), "matriz de confusión TP/TN/FP/FN")
    check(np.isclose(geometric_mean_score(yt, yp), np.sqrt(tp / (tp + fn) * tn / (tn + fp))), "G-Mean")
    check(total_misclassification_cost(yt, yp, 10, 1) == 10 * fn + fp, "TMC = 10·FN + 1·FP")

    print("Propiedades teóricas")
    C = binary_cost_matrix(10, 1)
    d = DirectCSKNN(5, cost_matrix=C).fit(Xs, ytr)
    P = d.predict_proba(Xts)
    check((d.predict(Xts) == (P[:, 1] > d.threshold_).astype(int)).all(),
          f"Direct-CS-KNN = umbral τ1 = C10/(C01+C10) = {d.threshold_:.3f} sobre P(ω1|x)")
    check((DirectCSKNN(5, cost_matrix=None, smoothing="none").fit(Xs, ytr).predict(Xts)
           == KNNClassifier(5).fit(Xs, ytr).predict(Xts)).all(), "Direct-CS-KNN con costo 0-1 = KNN")
    check((DistancesCSKNN(5, cost_matrix=C, gamma=0).fit(Xs, ytr).predict(Xts)
           == KNNClassifier(5).fit(Xs, ytr).predict(Xts)).all(), "Distances-CS-KNN con γ=0 = KNN")
    cm = CMKNN().fit(Xs, ytr); cm.predict(Xts)
    check(cm.k_.min() >= 1 and cm.k_.max() < len(Xs), f"CM-KNN: k_j ∈ [{cm.k_.min()}, {cm.k_.max()}]")
    check(DirectCSKNN(5, cost_matrix=C, k_grid=(1, 3, 5, 7, 9)).fit(Xs, ytr).best_k_ in (1, 3, 5, 7, 9),
          "Direct-CS-KNN: selección de k por mínimo costo")
    dw = DistancesCSKNN(5, cost_matrix=C, formulation="weighted_cost").fit(Xs, ytr)
    check(dw.predict(Xts).shape == yte.shape, "Distances-CS-KNN formulación 'weighted_cost'")
    os_ = OneStepKNN().fit(Xs, ytr)
    S = os_.decision_scores(Xts)
    check((os_.predict(Xts) == os_.classes_[S.argmax(1)]).all(), "One-Step: regla Ec. 6 argmax (1/‖W_i‖0) Σ Θ_jj W_ji")
    check(np.isclose(os_.theta_.sum(), 1) and (os_.theta_ >= 0).all(), "One-Step: θ en el símplex (Ec. 8)")
    # IRLS de One-Step debe hacer decrecer el objetivo de la Ec. 7
    o = OneStepKNN(max_iter=1).fit(Xs, ytr)
    objs = []
    for it in (1, 3, 10, 30):
        o.set_params(max_iter=it, tol=0.0)
        Wt = o.learn_W(Xts)
        objs.append(o._objective(Wt, np.linalg.norm(Xts.T - Xs.T @ Wt) ** 2))
    check(all(a >= b - 1e-6 * a for a, b in zip(objs, objs[1:])),
          "One-Step: IRLS decrece el objetivo " + " > ".join(f"{v:.1f}" for v in objs))

    print(f"\n{'TODO CORRECTO' if not FAILS else f'{len(FAILS)} FALLAS'}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(main())
