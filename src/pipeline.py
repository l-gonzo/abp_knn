"""
pipeline.py — Pipeline experimental del ABP (Fases I, II, III y V)
==================================================================

Fase I   Carga de Sonar, estandarización, desbalance sintético (90:10, 95:5),
         inyección de 5 % de outliers y PCA (≥ 85 % de varianza).
Fase II  Evaluación con Stratified 10-Fold CV de:
         Mínima Distancia, Naive Bayes, KNN (k=5), CV-KNN, Direct-CS-KNN,
         Distances-CS-KNN, CM-KNN (IRLS) y One-Step KNN (+ línea base trivial).
Fase III Accuracy, Balanced Accuracy, G-Mean, F1, F2, PR-AUC, TMC y tiempos
         de entrenamiento / inferencia por muestra (time.perf_counter).
Fase V   "Vectores rebeldes": muestras que TODOS los modelos clasifican mal;
         distancia de Mahalanobis a los centroides real/predicho y densidad
         local k-NN en el espacio latente.

Protocolo (sin fuga de información), por cada pliegue:
    1. El pliegue de prueba NO se toca.
    2. En entrenamiento se submuestrea la clase Mina (ω1) hasta la razón pedida.
    3. StandardScaler se ajusta solo con entrenamiento.
    4. Se inyecta ruido atípico solo en entrenamiento (unidades estandarizadas).
    5. PCA (si aplica) se ajusta solo con entrenamiento.
    6. Métricas en prueba con pesos de "prevalencia operativa": se pondera la
       prueba para que las minas pesen lo mismo que en el escenario (p. ej. 5 %).
       Así la exactitud y el costo reflejan el patrullaje real (95 % rocas) sin
       desechar minas de prueba. Sensibilidad, especificidad, G-Mean y Balanced
       Accuracy no dependen de la prevalencia; accuracy, precisión, F-scores,
       PR-AUC y TMC sí.

Uso:
    python pipeline.py                 # todos los escenarios
    python pipeline.py --quick         # 3 pliegues, prueba rápida
    python pipeline.py --scenarios 90:10 95:5
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifiers import (StandardScaler, PCA, stratified_kfold, clone, evaluate_binary, binary_confusion,
                         binary_cost_matrix, pairwise_distances, KNNClassifier,
                         CVKNNClassifier, GaussianNaiveBayes, MinimumDistanceClassifier,
                         MajorityClassifier)
from cs_knn import DirectCSKNN, DistancesCSKNN
from reconstruction_knn import CMKNN, OneStepKNN

# ----------------------------------------------------------------------------
# Configuración global
# ----------------------------------------------------------------------------
SEED = 42
N_SPLITS = 10
# Costos (ω0 = roca, ω1 = mina):
#   Falso Negativo (mina -> roca, destrucción del AUV)        = 10
#   Falso Positivo (roca -> mina, verificación secundaria)   =  1
# Convención del Bloque B, C_ij = costo de clasificar ω_j como ω_i:  C01 = 10, C10 = 1,
# que hace coherentes TMC = FN·C01 + FP·C10 y τ = C01/(C01+C10). La Fase II escribe
# "C01 = 1, C10 = 10" con los índices (real, predicha): mismo significado.
C_FN = 10.0      # C01 en la convención del Bloque B
C_FP = 1.0       # C10 en la convención del Bloque B
C01, C10 = C_FN, C_FP
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sonar.csv")
RESULTS = os.path.join(ROOT, "results")
SONAR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
METRIC_COLS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision",
               "g_mean", "f1", "f2", "pr_auc", "roc_auc", "tmc"]


# ----------------------------------------------------------------------------
# Fase I: datos y escenarios
# ----------------------------------------------------------------------------

def load_sonar(path=DATA_PATH):
    """
    Sonar: Mines vs. Rocks (208 × 60). Regresa X (float) e y con
    1 = Mina (ω1, clase crítica/positiva) y 0 = Roca (ω0).
    """
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SONAR_URL, path)
    raw = np.genfromtxt(path, delimiter=",", dtype=str)
    X = raw[:, :-1].astype(float)
    y = (raw[:, -1] == "M").astype(int)
    return X, y


def induce_imbalance(idx, y, minority_ratio, rng, minority=1, min_count=2):
    """
    Submuestrea la clase minoritaria dentro de idx para que represente
    `minority_ratio` del total (p. ej. 0.10 -> 90:10). Conserva toda la mayoritaria.
    """
    if minority_ratio is None:
        return idx
    maj = idx[y[idx] != minority]
    mino = idx[y[idx] == minority]
    n_keep = int(round(len(maj) * minority_ratio / (1.0 - minority_ratio)))
    n_keep = int(np.clip(n_keep, min_count, len(mino)))
    return np.sort(np.r_[maj, rng.choice(mino, n_keep, replace=False)])


def inject_outliers(X, y, frac=0.05, mode="gaussian", scale=3.0, rng=None, k=5):
    """
    Inyecta ruido atípico en `frac` de las muestras (datos ya estandarizados).

    mode='gaussian'   : suma N(0, scale²) a todas las características de muestras
                        al azar (reflexiones múltiples / interferencia acústica).
    mode='label_flip' : invierte la etiqueta de las muestras más limítrofes
                        (mayor proporción de vecinos de la otra clase entre sus k-NN).
    Regresa X', y', máscara booleana de las muestras alteradas.
    """
    rng = np.random.default_rng(rng)
    X, y = X.copy(), y.copy()
    n = len(X)
    n_out = int(round(frac * n))
    mask = np.zeros(n, dtype=bool)
    if mode is None or n_out == 0:
        return X, y, mask
    if mode == "gaussian":
        idx = rng.choice(n, n_out, replace=False)
        X[idx] += rng.normal(0.0, scale, size=(n_out, X.shape[1]))
    elif mode == "label_flip":
        D = pairwise_distances(X, X)
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :k]
        border = (y[nn] != y[:, None]).mean(1) + 1e-6 * rng.random(n)   # desempate aleatorio
        idx = np.argsort(-border)[:n_out]
        y[idx] = 1 - y[idx]
    else:
        raise ValueError("mode debe ser 'gaussian', 'label_flip' o None")
    mask[idx] = True
    return X, y, mask


def deployment_weights(y_test, minority_ratio, minority=1):
    """
    Pesos de prueba para emular la prevalencia operativa: las minas suman
    `minority_ratio` del peso total. Se normalizan para sumar len(y_test).
    """
    n = len(y_test)
    if minority_ratio is None:
        return np.ones(n)
    pos = y_test == minority
    w = np.where(pos, minority_ratio / max(pos.sum(), 1),
                 (1 - minority_ratio) / max((~pos).sum(), 1))
    return w * n / w.sum()


@dataclass
class Scenario:
    name: str
    minority_ratio: float = None       # None = distribución original
    noise: str = "gaussian"            # 'gaussian' | 'label_flip' | None
    noise_frac: float = 0.05
    noise_scale: float = 3.0
    pca: float = None                  # None o fracción de varianza (0.85)
    weight_test: bool = True           # prevalencia operativa en prueba


SCENARIOS = {
    "original":  Scenario("Original", None, None, pca=None),
    "original+pca": Scenario("Original + PCA", None, None, pca=0.85),
    "90:10":     Scenario("90:10 + ruido", 0.10, "gaussian"),
    "90:10+pca": Scenario("90:10 + ruido + PCA", 0.10, "gaussian", pca=0.85),
    "95:5":      Scenario("95:5 + ruido", 0.05, "gaussian"),
    "95:5+pca":  Scenario("95:5 + ruido + PCA", 0.05, "gaussian", pca=0.85),
    "90:10+flip": Scenario("90:10 + etiquetas invertidas", 0.10, "label_flip"),
}


# ----------------------------------------------------------------------------
# Fase II: modelos
# ----------------------------------------------------------------------------

def build_models(c_fn=C_FN, c_fp=C_FP, include_baseline=True):
    """Modelos del ABP con los hiperparámetros del reporte."""
    C = binary_cost_matrix(c_fn, c_fp)
    models = {}
    if include_baseline:
        models["Trivial (mayoritaria)"] = MajorityClassifier()
    models.update({
        "Mín. Distancia": MinimumDistanceClassifier(metric="euclidean"),
        "Naive Bayes": GaussianNaiveBayes(),
        "KNN (k=5)": KNNClassifier(n_neighbors=5),
        "CV-KNN": CVKNNClassifier(k_grid=(1, 3, 5, 7, 9, 11, 15), cv=5,
                                  scoring="balanced_accuracy", random_state=SEED),
        "Direct-CS-KNN": DirectCSKNN(n_neighbors=5, cost_matrix=C, smoothing="m-estimate",
                                     m=1.0, prior="uniform"),
        # Variante de la Fase II-4: costo = "ratio dinámico de desbalance" (n_max / n_j).
        # Con 95:5 el umbral es τ1 = 1/20 = 0.05 < piso del m-estimate uniforme
        # (m·p/(k+m) = 0.5/6 ≈ 0.083): se usa el prior de entrenamiento (piso ≈ 0.017).
        "Direct-CS-KNN (ratio)": DirectCSKNN(n_neighbors=5, cost_matrix="balanced",
                                             smoothing="m-estimate", m=1.0, prior="train"),
        "Distances-CS-KNN": DistancesCSKNN(n_neighbors=5, cost_matrix=C, gamma=0.1,
                                           factor="cost"),
        # Formulación de Zhang et al. (2017): IRLS, ℓ1 simple, voto mayoritario
        "CM-KNN": CMKNN(rho1=30.0, rho2=0.1, rho3=0.01, solver="irls"),
        # Extensión propia: ℓ1 ponderada por distancia, solo w_ij > 0, voto por |w_ij|
        "CM-KNN (ext.)": CMKNN(rho1=30.0, rho2=0.1, rho3=0.01, solver="irls", locality=True,
                               positive=True, vote="weighted"),
        # Formulación de Zhang & Li (2023): Ec. 7 (IRLS, Ec. 18), Θ por Ec. 8, regla Ec. 6
        "One-Step KNN": OneStepKNN(alpha=0.1, beta=0.1, groups="fcm", sparsify="mean",
                                   random_state=SEED),
    })
    return models


# ----------------------------------------------------------------------------
# Fase II–III: validación cruzada
# ----------------------------------------------------------------------------

def run_scenario(X, y, scenario, models, n_splits=N_SPLITS, seed=SEED, verbose=True):
    """
    Ejecuta la CV de un escenario.
    Regresa (registros por pliegue, predicciones fuera de pliegue, info extra).
    """
    rng = np.random.default_rng(seed)
    records = []
    oof = {m: {"pred": np.full(len(y), -1), "score": np.full(len(y), np.nan)} for m in models}
    pca_dims = []
    for fold, (tr, te) in enumerate(stratified_kfold(y, n_splits, seed)):
        tr = induce_imbalance(tr, y, scenario.minority_ratio, rng)
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        Xtr, ytr, _ = inject_outliers(Xtr, ytr, scenario.noise_frac, scenario.noise,
                                      scenario.noise_scale, rng)
        if scenario.pca:
            pca = PCA(scenario.pca).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
            pca_dims.append(pca.n_components_)
        w = deployment_weights(yte, scenario.minority_ratio) if scenario.weight_test else None

        for name, model in models.items():
            est = clone(model)
            t0 = time.perf_counter()
            est.fit(Xtr, ytr)
            t1 = time.perf_counter()
            yp = est.predict(Xte)
            t2 = time.perf_counter()
            proba = est.predict_proba(Xte)
            pos_col = int(np.where(est.classes_ == 1)[0][0]) if 1 in est.classes_ else None
            score = proba[:, pos_col] if pos_col is not None else np.zeros(len(yte))
            met = evaluate_binary(yte, yp, score, pos_label=1, c_fn=C_FN, c_fp=C_FP,
                                  sample_weight=w)
            rec = {"scenario": scenario.name, "model": name, "fold": fold,
                   **met, "fit_time_s": t1 - t0,
                   "infer_ms_per_sample": 1000.0 * (t2 - t1) / len(yte),
                   "n_train": len(ytr), "n_train_minority": int((ytr == 1).sum()),
                   "n_features": Xtr.shape[1]}
            if hasattr(est, "k_"):
                rec["k_mean"] = float(np.mean(est.k_))
            if hasattr(est, "best_k_"):
                rec["k_mean"] = float(est.best_k_)
            records.append(rec)
            oof[name]["pred"][te] = yp
            oof[name]["score"][te] = score
        if verbose:
            print(f"  pliegue {fold + 1}/{n_splits} listo", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    info = {"pca_dims": pca_dims}
    return records, oof, info


def summarize(df):
    """Media ± desviación por escenario y modelo (TMC: suma sobre pliegues)."""
    agg = {c: ["mean", "std"] for c in METRIC_COLS if c != "tmc"}
    agg["tmc"] = ["sum", "mean"]
    agg["fit_time_s"] = ["mean"]
    agg["infer_ms_per_sample"] = ["mean"]
    if "k_mean" in df:
        agg["k_mean"] = ["mean"]
    s = df.groupby(["scenario", "model"], sort=False).agg(agg)
    s.columns = ["_".join(c) for c in s.columns]
    return s.reset_index()


def format_table(summary, scenario):
    """Tabla comparativa de la Fase III (exactamente las columnas que pide el ABP)."""
    s = summary[summary.scenario == scenario]
    cols = [("Accuracy", "accuracy"), ("Balanced Acc.", "balanced_accuracy"), ("G-Mean", "g_mean"),
            ("F1-Score", "f1"), ("F2-Score", "f2"), ("PR-AUC", "pr_auc")]
    head = ("| Modelo | " + " | ".join(c for c, _ in cols) +
            " | TMC | Entrenamiento (s) | Inferencia por muestra (ms) |")
    lines = [head, "|" + "---|" * (len(cols) + 4)]
    for _, r in s.iterrows():
        vals = [f"{r[k + '_mean']:.3f}±{r[k + '_std']:.3f}" for _, k in cols]
        lines.append(f"| {r.model} | " + " | ".join(vals) +
                     f" | {r.tmc_sum:.1f} | {r.fit_time_s_mean:.4f} | {r.infer_ms_per_sample_mean:.3f} |")
    return "\n".join(lines)


BASIC_METRICS = [("sensibilidad", "sensitivity"), ("especificidad", "specificity"),
                 ("exactitud", "accuracy"), ("precision", "precision")]


def basic_metrics_table(summary):
    """Sensibilidad, especificidad, exactitud y precisión de cada modelo y escenario.

        Sensibilidad  = TP / (TP + FN)          (recall de minas)
        Especificidad = TN / (TN + FP)          (recall de rocas)
        Exactitud     = (TP + TN) / total        (accuracy)
        Precisión     = TP / (TP + FP)           (valor predictivo positivo)

    Media y desviación en los 10 pliegues, con la prueba ponderada a la prevalencia
    operativa (exactitud y precisión dependen de ella; sensibilidad y especificidad no).
    """
    out = summary[["scenario", "model"]].copy()
    for es, en in BASIC_METRICS:
        out[f"{es}_media"] = summary[f"{en}_mean"]
        out[f"{es}_desv"] = summary[f"{en}_std"]
    return out


def format_basic(summary, scenario):
    s = basic_metrics_table(summary)
    s = s[s.scenario == scenario]
    names = ["Sensibilidad", "Especificidad", "Exactitud", "Precisión"]
    lines = ["| Modelo | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for _, r in s.iterrows():
        lines.append(f"| {r.model} | " + " | ".join(
            f"{r[es + '_media']:.3f}±{r[es + '_desv']:.3f}" for es, _ in BASIC_METRICS) + " |")
    return "\n".join(lines)


def format_extra(summary, scenario):
    """Métricas complementarias: sensibilidad, especificidad, precisión y ROC-AUC."""
    s = summary[summary.scenario == scenario]
    cols = [("Sensibilidad", "sensitivity"), ("Especificidad", "specificity"),
            ("Precisión", "precision"), ("ROC-AUC", "roc_auc")]
    lines = ["| Modelo | " + " | ".join(c for c, _ in cols) + " | k medio |",
             "|" + "---|" * (len(cols) + 2)]
    for _, r in s.iterrows():
        k = f"{r.k_mean_mean:.1f}" if "k_mean_mean" in r and not pd.isna(r.k_mean_mean) else "—"
        lines.append(f"| {r.model} | " + " | ".join(f"{r[c + '_mean']:.3f}±{r[c + '_std']:.3f}"
                                                   for _, c in cols) + f" | {k} |")
    return "\n".join(lines)


PCA_PAIRS = [("original", "original+pca"), ("90:10", "90:10+pca"), ("95:5", "95:5+pca")]


def pca_impact_table(summary, models=("Naive Bayes", "KNN (k=5)")):
    """Fase I-4: impacto diferencial de PCA en Naive Bayes vs KNN (Δ = con PCA − sin PCA)."""
    rows = []
    mets = ["accuracy", "balanced_accuracy", "g_mean", "f2", "pr_auc"]
    for a, b in PCA_PAIRS:
        na, nb = SCENARIOS[a].name, SCENARIOS[b].name
        if not {na, nb} <= set(summary.scenario):
            continue
        for m in models:
            ra = summary[(summary.scenario == na) & (summary.model == m)].iloc[0]
            rb = summary[(summary.scenario == nb) & (summary.model == m)].iloc[0]
            row = {"escenario": na, "modelo": m}
            for k in mets:
                row[f"{k}_sin_pca"] = ra[k + "_mean"]
                row[f"{k}_con_pca"] = rb[k + "_mean"]
                row[f"delta_{k}"] = rb[k + "_mean"] - ra[k + "_mean"]
            row["tmc_sin_pca"], row["tmc_con_pca"] = ra.tmc_sum, rb.tmc_sum
            rows.append(row)
    return pd.DataFrame(rows)


def format_pca(pt):
    lines = ["| Escenario | Modelo | Bal. Acc. sin → con PCA | G-Mean sin → con PCA | "
             "F2 sin → con PCA | PR-AUC sin → con PCA | TMC sin → con PCA |",
             "|---|---|---|---|---|---|---|"]
    for _, r in pt.iterrows():
        cell = lambda k: f"{r[k + '_sin_pca']:.3f} → {r[k + '_con_pca']:.3f} ({r['delta_' + k]:+.3f})"
        lines.append(f"| {r.escenario} | {r.modelo} | {cell('balanced_accuracy')} | {cell('g_mean')} | "
                     f"{cell('f2')} | {cell('pr_auc')} | {r.tmc_sin_pca:.1f} → {r.tmc_con_pca:.1f} |")
    return "\n".join(lines)


def confusion_table(oof, y, scenario_name):
    """
    Matriz de confusión de cada algoritmo con conteos reales: predicciones
    fuera de pliegue (cada una de las 208 muestras se predice exactamente una
    vez, por un modelo que no la vio). Positivo = Mina (ω1).
    TMC aquí = 10·FN + 1·FP sobre estos conteos (distribución natural de Sonar).
    """
    rows = []
    for name, o in oof.items():
        c = binary_confusion(y, o["pred"], pos_label=1)
        tp, fn, fp, tn = (int(c[k]) for k in ("tp", "fn", "fp", "tn"))
        rows.append({"scenario": scenario_name, "model": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                     "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "tmc_counts": fn * C_FN + fp * C_FP})
    return pd.DataFrame(rows)


def format_confusion(ct):
    lines = ["| Modelo | TP | FN | FP | TN | Sens. | Espec. | TMC (10·FN + 1·FP) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        lines.append(f"| {r.model} | {r.TP} | {r.FN} | {r.FP} | {r.TN} | {r.sensitivity:.3f} | "
                     f"{r.specificity:.3f} | {r.tmc_counts:.0f} |")
    return "\n".join(lines)


def per_model_report(summary, conf):
    """Reporte por algoritmo: una tabla por modelo con todas sus métricas en cada escenario."""
    out = ["# Métricas por algoritmo\n",
           "Métricas: media ± desviación en 10 pliegues con prevalencia operativa del escenario. "
           "TMC (op.) = suma sobre pliegues con esa prevalencia. TP/FN/FP/TN = conteos reales "
           "fuera de pliegue (208 muestras, positivo = Mina).\n"]
    mets = [("Accuracy", "accuracy"), ("Bal. Acc.", "balanced_accuracy"), ("Sens.", "sensitivity"),
            ("Espec.", "specificity"), ("Precisión", "precision"), ("G-Mean", "g_mean"),
            ("F1", "f1"), ("F2", "f2"), ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc")]
    for model in summary.model.unique():
        out.append(f"\n## {model}\n")
        head = "| Escenario | " + " | ".join(m for m, _ in mets) + " | TMC (op.) | TP | FN | FP | TN |"
        out += [head, "|" + "---|" * (len(mets) + 6)]
        for _, r in summary[summary.model == model].iterrows():
            c = conf[(conf.model == model) & (conf.scenario == r.scenario)].iloc[0]
            vals = [f"{r[k + '_mean']:.3f}±{r[k + '_std']:.3f}" for _, k in mets]
            out.append(f"| {r.scenario} | " + " | ".join(vals) +
                       f" | {r.tmc_sum:.1f} | {c.TP} | {c.FN} | {c.FP} | {c.TN} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Fase V: vectores rebeldes
# ----------------------------------------------------------------------------

def find_rebels(oof, y, exclude=("Trivial (mayoritaria)",)):
    """Índices clasificados mal por TODOS los modelos (fuera de pliegue)."""
    wrong = np.ones(len(y), dtype=bool)
    for name, o in oof.items():
        if name in exclude:
            continue
        wrong &= o["pred"] != y
    return np.where(wrong)[0]


def latent_space(X, var=0.85):
    """Espacio latente de diagnóstico: estandarización + PCA (≥ var) sobre todo Sonar."""
    Z = StandardScaler().fit_transform(X)
    pca = PCA(var).fit(Z)
    return pca.transform(Z), pca


def rebel_diagnostics(X, y, rebels, oof, k=5):
    """
    Para cada muestra: distancia de Mahalanobis (covarianza intra-clase común)
    al centroide de su clase real y al de la clase opuesta (= predicha para los
    rebeldes), razón entre ambas, densidad local k-NN (1 / distancia media a
    sus k vecinos) y proporción de vecinos de su misma clase.
    Regresa (DataFrame por muestra, DataFrame de comparación rebeldes vs resto).
    """
    Z, pca = latent_space(X)
    cents = np.array([Z[y == c].mean(0) for c in (0, 1)])
    Zc = Z - cents[y]
    VI = np.linalg.pinv(Zc.T @ Zc / (len(Z) - 2))
    Dc = pairwise_distances(Z, cents, "mahalanobis", VI=VI)            # (n, 2)
    D = pairwise_distances(Z, Z)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    mean_d = np.take_along_axis(D, nn, axis=1).mean(1)
    n_wrong = sum((o["pred"] != y).astype(int) for name, o in oof.items()
                  if not name.startswith("Trivial"))
    df = pd.DataFrame({
        "idx": np.arange(len(y)),
        "clase_real": np.where(y == 1, "Mina", "Roca"),
        "maha_real": Dc[np.arange(len(y)), y],
        "maha_opuesta": Dc[np.arange(len(y)), 1 - y],
        "densidad_knn": 1.0 / mean_d,
        "dist_media_knn": mean_d,
        "vecinos_misma_clase": (y[nn] == y[:, None]).mean(1),
        "modelos_que_fallan": n_wrong,
        "rebelde": np.isin(np.arange(len(y)), rebels),
    })
    df["razon_maha"] = df.maha_real / df.maha_opuesta
    rows = []
    for col in ["maha_real", "maha_opuesta", "razon_maha", "densidad_knn", "vecinos_misma_clase"]:
        a, b = df.loc[df.rebelde, col], df.loc[~df.rebelde, col]
        p = stats.mannwhitneyu(a, b).pvalue if len(a) > 0 and len(b) > 0 else np.nan
        rows.append({"variable": col, "mediana_rebeldes": a.median(),
                     "mediana_resto": b.median(), "p_valor_mann_whitney": p})
    return df, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--rebels-scenario", default="90:10", choices=list(SCENARIOS))
    ap.add_argument("--quick", action="store_true", help="3 pliegues para probar")
    args = ap.parse_args(argv)

    n_splits = 3 if args.quick else N_SPLITS
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    X, y = load_sonar()
    print(f"Sonar: {X.shape[0]} muestras × {X.shape[1]} bandas | minas={y.sum()} rocas={(y == 0).sum()}")
    print(f"Costos: FN (mina→roca) = C01 = {C01:g}, FP (roca→mina) = C10 = {C10:g} | "
          f"Stratified {n_splits}-Fold | random_state={SEED}\n")

    models = build_models()
    all_records, oofs, pca_dims = [], {}, {}
    for key in args.scenarios:
        sc = SCENARIOS[key]
        print(f"Escenario: {sc.name}")
        t0 = time.perf_counter()
        recs, oof, info = run_scenario(X, y, sc, models, n_splits, SEED)
        all_records += recs
        oofs[key] = oof
        pca_dims[key] = float(np.mean(info["pca_dims"])) if info["pca_dims"] else None
        extra = f" | PCA: {pca_dims[key]:.1f} componentes" if pca_dims[key] else ""
        print(f"  listo en {time.perf_counter() - t0:.1f} s{extra}")
        np.savez(os.path.join(RESULTS, f"oof_{key.replace(':', '-').replace('+', '_')}.npz"),
                 y=y, **{f"{m}|pred": o["pred"] for m, o in oof.items()},
                 **{f"{m}|score": o["score"] for m, o in oof.items()})

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS, "tables", "folds.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(RESULTS, "tables", "summary.csv"), index=False)
    basic_metrics_table(summary).to_csv(os.path.join(RESULTS, "tables", "basic_metrics.csv"),
                                        index=False)

    conf = pd.concat([confusion_table(oofs[k], y, SCENARIOS[k].name) for k in args.scenarios],
                     ignore_index=True)
    conf.to_csv(os.path.join(RESULTS, "tables", "confusion_matrices.csv"), index=False)
    with open(os.path.join(RESULTS, "metrics_by_model.md"), "w", encoding="utf-8") as f:
        f.write(per_model_report(summary, conf))

    md = [f"# Resultados — Sonar, {n_splits}-fold estratificado, semilla {SEED}\n",
          f"Costos: FN (mina→roca) = C01 = {C01:g}, FP (roca→mina) = C10 = {C10:g} "
          "(convención del Bloque B); "
          f"TMC = {C_FN:g}·FN + {C_FP:g}·FP. Métricas de prueba con "
          "prevalencia operativa del escenario (media ± desviación en los pliegues); "
          "TMC = suma sobre los pliegues. Las matrices de confusión usan conteos reales "
          "fuera de pliegue.\n"]
    for key in args.scenarios:
        md.append(f"\n## {SCENARIOS[key].name}\n")
        md.append("### Tabla comparativa (Fase III)\n")
        md.append(format_table(summary, SCENARIOS[key].name))
        md.append("\n\n### Sensibilidad, especificidad, exactitud y precisión\n")
        md.append(format_basic(summary, SCENARIOS[key].name))
        md.append("\n\n### Métricas complementarias\n")
        md.append(format_extra(summary, SCENARIOS[key].name))
        md.append("\n\n### Matrices de confusión (positivo = Mina)\n")
        md.append(format_confusion(conf[conf.scenario == SCENARIOS[key].name]))
        print(f"\n== {SCENARIOS[key].name} ==")
        cols = ["model", "accuracy_mean", "balanced_accuracy_mean", "g_mean_mean", "f1_mean",
                "f2_mean", "pr_auc_mean", "roc_auc_mean", "tmc_sum"]
        print(summary[summary.scenario == SCENARIOS[key].name][cols]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        b = basic_metrics_table(summary)
        print(b[b.scenario == SCENARIOS[key].name].filter(regex="model|_media")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Fase I-4: PCA en Naive Bayes vs KNN
    pt = pca_impact_table(summary)
    if len(pt):
        pt.to_csv(os.path.join(RESULTS, "tables", "pca_impact_nb_vs_knn.csv"), index=False)
        dims = {k: v for k, v in pca_dims.items() if v}
        md.append("\n## Impacto de PCA (≥ 85 % de varianza): Naive Bayes vs KNN\n")
        md.append("Componentes retenidas (media por pliegue): " +
                  ", ".join(f"{SCENARIOS[k].name}: {v:.1f}" for k, v in dims.items()) + "\n")
        md.append(format_pca(pt))
        print("\nImpacto de PCA (Naive Bayes vs KNN):")
        print(pt[["escenario", "modelo", "delta_balanced_accuracy", "delta_g_mean", "delta_pr_auc"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Fase V
    rk = args.rebels_scenario if args.rebels_scenario in oofs else args.scenarios[0]
    rebels = find_rebels(oofs[rk], y)
    diag, comp = rebel_diagnostics(X, y, rebels, oofs[rk])
    diag.to_csv(os.path.join(RESULTS, "tables", "rebels_all_samples.csv"), index=False)
    comp.to_csv(os.path.join(RESULTS, "tables", "rebels_comparison.csv"), index=False)
    md.append(f"\n## Vectores rebeldes ({SCENARIOS[rk].name})\n")
    n_real = sum(not m.startswith("Trivial") for m in oofs[rk])
    md.append(f"{len(rebels)} muestras mal clasificadas por los {n_real} modelos: índices {rebels.tolist()}\n")
    md.append(diag[diag.rebelde].drop(columns=["rebelde"]).to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\nRebeldes vs resto (medianas, prueba de Mann–Whitney):\n")
    md.append(comp.to_markdown(index=False, floatfmt=".4g"))
    print(f"\nVectores rebeldes ({SCENARIOS[rk].name}): {len(rebels)} -> {rebels.tolist()}")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nTablas en {os.path.join(RESULTS, 'tables')} y resumen en results/summary.md")
    return df, summary, oofs


if __name__ == "__main__":
    main()
