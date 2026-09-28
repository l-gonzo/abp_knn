"""
visualization.py — Figuras del ABP (Fase IV y apoyo a Fases III y V)
=====================================================================

Requiere haber corrido antes `python pipeline.py` (lee results/tables y
results/oof_*.npz). Genera en results/figures/ (PNG 200 dpi + PDF vectorial):

    fig_boundaries        Mapas 2D (PC1, PC2) de fronteras de decisión (Fase IV)
    fig_voronoi_cs        Geometría de Distances-CS-KNN: celdas de Voronoi con y sin
                          deformación de la métrica (Bloque B)
    fig_accuracy_fallacy  Accuracy vs G-Mean bajo desbalance (Sección 3)
    fig_metrics_heatmap   Todas las métricas por modelo (90:10 y 95:5)
    fig_basic_metrics     Sensibilidad, especificidad, exactitud y precisión por modelo
                          (original, 90:10 y 95:5; media ± desviación)
    fig_confusion_*       Matriz de confusión (TP/TN/FP/FN) de cada algoritmo
    fig_roc_*             Curvas ROC y ROC-AUC de todos los algoritmos
    fig_pr_*              Curvas Precision-Recall y PR-AUC de todos los algoritmos
    fig_latency           Compromiso latencia de inferencia vs G-Mean
    fig_pca_impact        Impacto de PCA por modelo (Naive Bayes vs KNN)
    fig_rebels            Diagnóstico topológico de vectores rebeldes (Fase V)

Uso:  python visualization.py            (o  --only boundaries rebels ...)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifiers import (StandardScaler, PCA, KNNClassifier, clone, pairwise_distances,
                         binary_cost_matrix, binary_confusion, roc_curve, roc_auc_score,
                         precision_recall_curve, average_precision_score)
from cs_knn import DistancesCSKNN
from pipeline import (load_sonar, induce_imbalance, inject_outliers, build_models,
                      find_rebels, deployment_weights, SCENARIOS, RESULTS, SEED, C_FN, C_FP)

FIG_DIR = os.path.join(RESULTS, "figures")

# ---------------------------------------------------------------------------
# Estilo (paleta categórica validada para daltonismo; tinta neutra para texto)
# ---------------------------------------------------------------------------
ROCA, MINA = "#2a78d6", "#eb6834"                 # identidad de clase (ω0, ω1)
ROCA_BG, MINA_BG = "#e3eefc", "#fde6dc"           # tintes de región
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#f5f9fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "savefig.bbox": "tight",
    "savefig.dpi": 200, "figure.dpi": 110, "font.family": "DejaVu Sans",
})

MODEL_ORDER = ["Mín. Distancia", "Naive Bayes", "KNN (k=5)", "CV-KNN", "Direct-CS-KNN",
               "Direct-CS-KNN (ratio)", "Distances-CS-KNN", "CM-KNN", "CM-KNN (ext.)", "One-Step KNN"]


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def _oof_path(key):
    return os.path.join(RESULTS, f"oof_{key.replace(':', '-').replace('+', '_')}.npz")


def load_oof(key):
    z = np.load(_oof_path(key))
    y = z["y"]
    models = sorted({k.split("|")[0] for k in z.files if "|" in k})
    return y, {m: {"pred": z[f"{m}|pred"], "score": z[f"{m}|score"]} for m in models}


# ---------------------------------------------------------------------------
# Fase IV: fronteras de decisión 2D
# ---------------------------------------------------------------------------

def boundary_training_set(X, y, ratio=0.10, seed=SEED):
    """Entrenamiento 90:10 + 5 % outliers, proyectado a (PC1, PC2) ajustado en él."""
    rng = np.random.default_rng(seed)
    idx = induce_imbalance(np.arange(len(y)), y, ratio, rng)
    scaler = StandardScaler().fit(X[idx])
    Z, yz, out = inject_outliers(scaler.transform(X[idx]), y[idx], 0.05, "gaussian", 3.0, rng)
    pca = PCA(2).fit(Z)
    return pca.transform(Z), yz, out, (lambda A: pca.transform(scaler.transform(A))), pca


def fig_boundaries(X, y, grid_n=90):
    P, yp, out, project, pca = boundary_training_set(X, y)
    ev = pca.explained_variance_ratio_
    rebels = find_rebels(load_oof("90:10")[1], y) if os.path.exists(_oof_path("90:10")) else []
    R = project(X[rebels]) if len(rebels) else np.empty((0, 2))

    allp = np.r_[P, R]                               # incluye outliers y rebeldes
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid_n), np.linspace(lo[1], hi[1], grid_n))
    G = np.c_[gx.ravel(), gy.ravel()]

    base = build_models(include_baseline=False)
    panels = [
        ("Mín. Distancia", base["Mín. Distancia"], "hiperplano rígido"),
        ("Naive Bayes", base["Naive Bayes"], "cuadrática; el prior 90:10 la anula"),
        ("KNN (k=1)", KNNClassifier(1), "Voronoi ruidoso"),
        ("CM-KNN (ext.)", base["CM-KNN (ext.)"], "localidad + w>0 (propia)"),
        ("Direct-CS-KNN", base["Direct-CS-KNN"], "C01=10 (FN), riesgo mínimo"),
        ("Distances-CS-KNN", base["Distances-CS-KNN"], "γ=0.1"),
        ("CM-KNN", base["CM-KNN"], "k_j aprendido (IRLS)"),
        ("One-Step KNN", base["One-Step KNN"], "Group Lasso exclusivo + Θ"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.1), sharex=True, sharey=True)
    for ax, (title, model, sub) in zip(axes.ravel(), panels):
        est = clone(model)
        if "batch_size" in est.get_params():
            # One-Step es transductivo: ‖W̃‖²_{2,1}, los grupos y Θ dependen de todo el
            # lote, y resolver la malla por lotes produce franjas. Cada punto va solo.
            est.set_params(batch_size=1 if title.startswith("One-Step") else 21)
        est.fit(P, yp)
        Zg = est.predict(G).reshape(gx.shape).astype(float)
        ax.contourf(gx, gy, Zg, levels=[-0.5, 0.5, 1.5], colors=[ROCA_BG, MINA_BG])
        ax.contour(gx, gy, Zg, levels=[0.5], colors=INK, linewidths=0.9)
        r, m = (yp == 0) & ~out, (yp == 1) & ~out
        ax.scatter(P[r, 0], P[r, 1], s=9, c=ROCA, edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(P[m, 0], P[m, 1], s=22, c=MINA, marker="^", edgecolors="white",
                   linewidths=0.4, zorder=4)
        ax.scatter(P[out, 0], P[out, 1], s=18, c=INK, marker="x", linewidths=0.9, zorder=5)
        if len(R):
            ax.scatter(R[:, 0], R[:, 1], s=46, facecolors="none", edgecolors=INK,
                       linewidths=0.9, zorder=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_title(title + (f"\n{sub}" if sub else "\n "), color=INK, linespacing=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f} %)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f} %)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=ROCA, mec="white", label="Roca (ω0)"),
        Line2D([], [], ls="", marker="^", ms=5.5, mfc=MINA, mec="white", label="Mina (ω1)"),
        Line2D([], [], ls="", marker="x", ms=5, mec=INK, label="Outlier inyectado"),
        Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=INK, label="Vector rebelde"),
        Patch(fc=ROCA_BG, ec=GRID, label="Región roca"),
        Patch(fc=MINA_BG, ec=GRID, label="Región mina"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    _save(fig, "fig_boundaries")


# ---------------------------------------------------------------------------
# Bloque B: geometría de Distances-CS-KNN
# ---------------------------------------------------------------------------

def fig_voronoi_cs(seed=SEED):
    rng = np.random.default_rng(seed)
    Xr = rng.normal([0.0, 0.0], 1.0, size=(40, 2))
    Xm = rng.normal([1.6, 1.2], 0.6, size=(5, 2))
    Xt = np.r_[Xr, Xm]
    yt = np.r_[np.zeros(40, int), np.ones(5, int)]
    C = binary_cost_matrix(C_FN, C_FP)
    gx, gy = np.meshgrid(np.linspace(-3, 4, 300), np.linspace(-3, 3.5, 300))
    G = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=True)
    for ax, g in zip(axes, (0.0, 0.3, 0.6)):
        est = DistancesCSKNN(1, cost_matrix=C, gamma=g).fit(Xt, yt)
        Zg = est.predict(G).reshape(gx.shape).astype(float)
        idx = est.kneighbors(G, 1, return_distance=False)[:, 0].reshape(gx.shape)
        ax.contourf(gx, gy, Zg, levels=[-0.5, 0.5, 1.5], colors=[ROCA_BG, MINA_BG])
        ax.contour(gx, gy, idx, levels=np.arange(len(Xt)) + 0.5, colors=GRID, linewidths=0.35)
        ax.contour(gx, gy, Zg, levels=[0.5], colors=INK, linewidths=1.0)
        ax.scatter(Xr[:, 0], Xr[:, 1], s=10, c=ROCA, edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(Xm[:, 0], Xm[:, 1], s=26, c=MINA, marker="^", edgecolors="white",
                   linewidths=0.4, zorder=4)
        area = 100 * Zg.mean()
        ax.set_title(f"γ = {g:g}  (factor mina = {C_FN ** g:.2f})\nregión mina: {area:.1f} % del plano")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-3, 4); ax.set_ylim(-3, 3.5)
    fig.suptitle(r"Distances-CS-KNN (k=1): $d_{CS}(x,x_n)=d(x,x_n)\,/\,\omega(c_j)^{\gamma}$ "
                 r"expande las celdas de Voronoi de la clase crítica", y=1.03, fontsize=8.5)
    fig.tight_layout(w_pad=0.5)
    _save(fig, "fig_voronoi_cs")


# ---------------------------------------------------------------------------
# Fase III: métricas
# ---------------------------------------------------------------------------

def _summary():
    return pd.read_csv(os.path.join(RESULTS, "tables", "summary.csv"))


def fig_accuracy_fallacy(scenario="95:5 + ruido"):
    s = _summary()
    s = s[s.scenario == scenario].set_index("model")
    order = ["Trivial (mayoritaria)"] + MODEL_ORDER
    s = s.loc[[m for m in order if m in s.index]][::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    yv = np.arange(len(s))
    ax.hlines(yv, s.g_mean_mean, s.accuracy_mean, color=GRID, lw=2, zorder=1)
    ax.scatter(s.accuracy_mean, yv, s=28, c=MUTED, zorder=3, label="Accuracy")
    ax.scatter(s.g_mean_mean, yv, s=28, c=ROCA, zorder=3, label="G-Mean")
    ax.set_yticks(yv, s.index)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("Media 10-fold, prevalencia 95:5")
    t = list(s.index).index("Trivial (mayoritaria)")
    ax.annotate(f"acc = {s.accuracy_mean.iloc[t]:.2f}, G-Mean = 0", (0.5, t), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.6, 1.04))
    fig.tight_layout()
    _save(fig, "fig_accuracy_fallacy")


def fig_metrics_heatmap(scenarios=("90:10 + ruido", "95:5 + ruido")):
    s = _summary()
    cols = [("Acc.", "accuracy_mean"), ("Bal. Acc.", "balanced_accuracy_mean"),
            ("Sens.", "sensitivity_mean"), ("Espec.", "specificity_mean"),
            ("G-Mean", "g_mean_mean"), ("F1", "f1_mean"), ("F2", "f2_mean"),
            ("PR-AUC", "pr_auc_mean"), ("ROC-AUC", "roc_auc_mean")]
    order = ["Trivial (mayoritaria)"] + MODEL_ORDER
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.16, 3.3), sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        t = s[s.scenario == sc].set_index("model").loc[order]
        M = t[[c for _, c in cols]].to_numpy()
        tmc = t["tmc_sum"].to_numpy()
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if M[i, j] > 0.62 else INK)
            best = np.nanargmin(tmc[1:]) + 1
            ax.text(M.shape[1] - 0.35, i, f"{tmc[i]:.0f}", ha="left", va="center", fontsize=6.3,
                    color=INK, fontweight="bold" if i == best else "normal")
        ax.text(M.shape[1] - 0.35, -0.85, "TMC", ha="left", va="center", fontsize=6.8, color=INK2)
        ax.set_xticks(range(len(cols)), [c for c, _ in cols], rotation=45, ha="right")
        ax.set_yticks(range(len(order)), order)
        ax.set_title(sc, pad=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(-0.5, M.shape[1] + 0.6)
    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig_metrics_heatmap")


def fig_basic_metrics(scenarios=("Original", "90:10 + ruido", "95:5 + ruido")):
    """Sensibilidad, especificidad, exactitud y precisión por modelo (media ± desv., 10 pliegues)."""
    s = _summary()
    mets = [("Sensibilidad", "sensitivity"), ("Especificidad", "specificity"),
            ("Exactitud", "accuracy"), ("Precisión", "precision")]
    order = ["Trivial (mayoritaria)"] + MODEL_ORDER
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.16, 4.4), sharey=True)
    h = 0.2
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        t = s[s.scenario == sc].set_index("model").loc[order][::-1]
        yv = np.arange(len(t))
        for j, (lab, c) in enumerate(mets):
            off = (1.5 - j) * h
            ax.barh(yv + off, t[f"{c}_mean"], height=h * 0.9, color=SERIES[j], label=lab, zorder=2)
            m_, sd = t[f"{c}_mean"].to_numpy(), t[f"{c}_std"].to_numpy()
            err = np.vstack([np.minimum(sd, m_), np.minimum(sd, 1 - m_)])   # recortada a [0, 1]
            ax.errorbar(m_, yv + off, xerr=err, fmt="none",
                        ecolor=INK2, elinewidth=0.5, capsize=0, zorder=3)
        ax.set_yticks(yv, t.index)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        prev = {"Original": "53 % minas", "90:10 + ruido": "prevalencia 10 %",
                "95:5 + ruido": "prevalencia 5 %"}.get(sc, "")
        ax.set_title(f"{sc}\n({prev})" if prev else sc)
        for sep in (0.5, 4.5, 7.5):            # separa grupos: CM/One-Step | CS-KNN | clásicos | trivial
            ax.axhline(len(t) - 1 - sep, color=GRID, lw=0.8, zorder=1)
    h_, l_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    _save(fig, "fig_basic_metrics")


CURVE_GROUPS = [
    ("Clásicos", ["Mín. Distancia", "Naive Bayes", "KNN (k=5)", "CV-KNN"]),
    ("Sensibles al costo y de reconstrucción",
     ["Direct-CS-KNN", "Direct-CS-KNN (ratio)", "Distances-CS-KNN", "CM-KNN", "CM-KNN (ext.)",
      "One-Step KNN"]),
]
SERIES5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]        # codificación secundaria al color


def _key_file(key):
    return key.replace(":", "-").replace("+", "_")


def fig_roc_curves(key="90:10"):
    """Curvas ROC fuera de pliegue de todos los algoritmos (dos paneles)."""
    y, oof = load_oof(key)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6), sharey=True)
    for ax, (title, models) in zip(axes, CURVE_GROUPS):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", label="Azar  AUC = 0.500")
        for c, ls, m in zip(SERIES5, DASHES, models):
            fpr, tpr, _ = roc_curve(y, oof[m]["score"], pos_label=1)
            auc = roc_auc_score(y, oof[m]["score"], pos_label=1)
            ax.plot(fpr, tpr, color=c, ls=ls, lw=1.5, label=f"{m}  AUC = {auc:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  fontsize=6.4, handlelength=2.6, columnspacing=1.0)
        ax.set_title(title)
    axes[0].set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    fig.suptitle(f"Curvas ROC fuera de pliegue — {SCENARIOS[key].name} "
                 "(positivo = Mina; AUC de las 208 predicciones agrupadas,\n"
                 "que difiere del promedio por pliegue de las tablas)", fontsize=8.5, y=1.05)
    fig.tight_layout(w_pad=1.0)
    _save(fig, f"fig_roc_{_key_file(key)}")


def fig_pr_curves(key="90:10"):
    """Curvas Precision-Recall fuera de pliegue con la prevalencia operativa del escenario."""
    y, oof = load_oof(key)
    ratio = SCENARIOS[key].minority_ratio
    w = deployment_weights(y, ratio)
    base = ratio if ratio is not None else y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6), sharey=True)
    for ax, (title, models) in zip(axes, CURVE_GROUPS):
        ax.axhline(base, color=MUTED, lw=0.8, ls="--", label=f"Azar  AP = {base:.3f} (prevalencia)")
        for c, ls, m in zip(SERIES5, DASHES, models):
            prec, rec, _ = precision_recall_curve(y, oof[m]["score"], 1, w)
            ap = average_precision_score(y, oof[m]["score"], 1, w)
            ax.plot(rec, prec, color=c, ls=ls, lw=1.5, drawstyle="steps-post",
                    label=f"{m}  AP = {ap:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall (sensibilidad)")
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  fontsize=6.4, handlelength=2.6, columnspacing=1.0)
        ax.set_title(title)
    axes[0].set_ylabel("Precisión")
    fig.suptitle(f"Curvas Precision-Recall fuera de pliegue — {SCENARIOS[key].name} "
                 "(prevalencia operativa; AP de las 208 predicciones agrupadas)", fontsize=8.5, y=1.02)
    fig.tight_layout(w_pad=1.0)
    _save(fig, f"fig_pr_{_key_file(key)}")


def fig_confusion_matrices(key="90:10"):
    """Matriz de confusión (conteos reales fuera de pliegue) de cada algoritmo."""
    y, oof = load_oof(key)
    models = ["Trivial (mayoritaria)"] + MODEL_ORDER
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.4))
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), models):
        c = binary_confusion(y, oof[m]["pred"], pos_label=1)
        M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])           # filas = real
        R = M / M.sum(1, keepdims=True)
        ax.imshow(R, cmap=cmap, vmin=0, vmax=1)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{M[i, j]:.0f}\n{100 * R[i, j]:.0f} %", ha="center",
                        va="center", fontsize=6.5, linespacing=1.15,
                        color="white" if R[i, j] > 0.6 else INK,
                        fontweight="bold" if (i, j) == (1, 0) else "normal")
        tmc = c["fn"] * C_FN + c["fp"] * C_FP
        ax.set_title(f"{m}\nTMC = {tmc:.0f}", fontsize=7.2, linespacing=1.15)
        ax.set_xticks([0, 1], ["Roca", "Mina"]); ax.set_yticks([0, 1], ["Roca", "Mina"])
        ax.tick_params(length=0, labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.ravel():
        ax.set_xlabel("Predicha", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Real", fontsize=7)
    fig.suptitle(f"Matrices de confusión fuera de pliegue — {SCENARIOS[key].name} "
                 f"(n = {len(y)}; % por fila real; TMC = {C_FN:g}·FN + {C_FP:g}·FP; FN en negritas)",
                 fontsize=8, y=1.01)
    fig.tight_layout(h_pad=0.8, w_pad=0.3)
    _save(fig, f"fig_confusion_{_key_file(key)}")


def fig_latency(scenario="90:10 + ruido"):
    s = _summary()
    s = s[(s.scenario == scenario) & (s.model.isin(MODEL_ORDER))].set_index("model").loc[MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    x, yv = s.infer_ms_per_sample_mean, s.g_mean_mean
    ax.scatter(x, yv, s=30, c=ROCA, edgecolors="white", linewidths=0.5, zorder=3)
    offsets = {"Mín. Distancia": (-5, -1), "Naive Bayes": (-5, -2), "KNN (k=5)": (-5, -1),
               "CV-KNN": (5, -7), "Direct-CS-KNN": (6, -3), "Distances-CS-KNN": (5, -4),
               "CM-KNN": (-6, 7), "CM-KNN (ext.)": (-6, 2), "One-Step KNN": (-6, -6),
               "Direct-CS-KNN (ratio)": (-5, 5)}
    for m in s.index:
        dx, dy = offsets.get(m, (4, 3))
        ax.annotate(m, (x[m], yv[m]), xytext=(dx, dy), textcoords="offset points", fontsize=6.8,
                    color=INK2, ha="left" if dx > 0 else "right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(x.min() / 40, x.max() * 2.5)
    ax.set_xlabel("Inferencia por muestra (ms, escala log)")
    ax.set_ylabel("G-Mean")
    ax.set_ylim(0, 1)
    ax.grid(True, which="major"); ax.set_axisbelow(True)
    ax.set_title(f"Latencia vs desempeño — {scenario}")
    fig.tight_layout()
    _save(fig, "fig_latency")


def fig_pca_impact(pairs=(("Original", "Original + PCA"), ("90:10 + ruido", "90:10 + ruido + PCA"),
                          ("95:5 + ruido", "95:5 + ruido + PCA")),
                   metric="g_mean_mean", label="G-Mean"):
    """Impacto de PCA (≥ 85 % de varianza) en cada modelo: barras horizontales sin/con PCA."""
    s = _summary()
    models = MODEL_ORDER[::-1]
    yv = np.arange(len(models))
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.16, 3.3), sharey=True)
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        va = s[s.scenario == a].set_index("model").loc[models, metric]
        vb = s[s.scenario == b].set_index("model").loc[models, metric]
        ax.barh(yv + 0.2, va, 0.38, color=MUTED, label="Sin PCA (60 bandas)")
        ax.barh(yv - 0.2, vb, 0.38, color=ROCA, label="Con PCA (≥85 % var.)")
        for i, (p, q) in enumerate(zip(va, vb)):
            ax.text(max(p, q) + 0.02, i, f"{q - p:+.2f}", va="center", fontsize=6,
                    color=INK if m_highlight(models[i]) else INK2,
                    fontweight="bold" if m_highlight(models[i]) else "normal")
        ax.set_yticks(yv, models)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.set_title(a.split(" +")[0])
        ax.set_xlabel(label)
        ax.tick_params(axis="y", length=0)
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), frameon=False,
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.04))
    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig_pca_impact")


def m_highlight(name):
    """Naive Bayes y KNN (k=5): la comparación que pide la Fase I-4."""
    return name in ("Naive Bayes", "KNN (k=5)")


# ---------------------------------------------------------------------------
# Fase V: vectores rebeldes
# ---------------------------------------------------------------------------

def fig_rebels():
    df = pd.read_csv(os.path.join(RESULTS, "tables", "rebels_all_samples.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    ax = axes[0]
    rest, reb = df[~df.rebelde], df[df.rebelde]
    for cls, c, mk in (("Roca", ROCA, "o"), ("Mina", MINA, "^")):
        d = rest[rest.clase_real == cls]
        ax.scatter(d.maha_real, d.maha_opuesta, s=9 if mk == "o" else 12, c=c, marker=mk,
                   alpha=0.75, edgecolors="none", label=f"{cls} (resto)")
    ax.scatter(reb.maha_real, reb.maha_opuesta, s=40, facecolors="none", edgecolors=INK,
               linewidths=1.0, label="Rebelde", zorder=4)
    lim = [0, max(df.maha_real.max(), df.maha_opuesta.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.text(lim[1] * 0.62, lim[1] * 0.50, "equidistante", rotation=45, ha="center", va="top",
            fontsize=6.5, color=INK2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Mahalanobis a centroide real"); ax.set_ylabel("Mahalanobis a centroide opuesto")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("Posición relativa a los centroides")

    for ax, col, lab in ((axes[1], "razon_maha", "Razón  d_real / d_opuesto"),
                         (axes[2], "vecinos_misma_clase", "Vecinos (k=5) de su misma clase")):
        data = [rest[col], reb[col]]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED), boxprops=dict(edgecolor=MUTED))
        for patch, c in zip(bp["boxes"], (GRID, "#b7d3f6")):
            patch.set_facecolor(c)
        rng = np.random.default_rng(SEED)
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=6 if i == 1 else 18,
                       c=MUTED if i == 1 else INK, alpha=0.5 if i == 1 else 0.9, zorder=3,
                       edgecolors="none")
        ax.set_xticks([1, 2], [f"Resto (n={len(rest)})", f"Rebeldes (n={len(reb)})"])
        ax.set_ylabel(lab)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[1].axhline(1, color=MUTED, lw=0.8, ls="--")
    comp = pd.read_csv(os.path.join(RESULTS, "tables", "rebels_comparison.csv")).set_index("variable")
    axes[1].set_title(f"Mann–Whitney p = {comp.loc['razon_maha', 'p_valor_mann_whitney']:.1e}")
    axes[2].set_title(f"Mann–Whitney p = {comp.loc['vecinos_misma_clase', 'p_valor_mann_whitney']:.1e}")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_rebels")


CURVE_SCENARIOS = ["original", "90:10", "95:5"]

FIGURES = {
    "boundaries": lambda X, y: fig_boundaries(X, y),
    "voronoi": lambda X, y: fig_voronoi_cs(),
    "fallacy": lambda X, y: fig_accuracy_fallacy(),
    "heatmap": lambda X, y: fig_metrics_heatmap(),
    "basic": lambda X, y: fig_basic_metrics(),
    "confusion": lambda X, y: [fig_confusion_matrices(k) for k in CURVE_SCENARIOS],
    "roc": lambda X, y: [fig_roc_curves(k) for k in CURVE_SCENARIOS],
    "pr": lambda X, y: [fig_pr_curves(k) for k in CURVE_SCENARIOS],
    "latency": lambda X, y: fig_latency(),
    "pca": lambda X, y: fig_pca_impact(),
    "rebels": lambda X, y: fig_rebels(),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args(argv)
    X, y = load_sonar()
    print(f"Figuras en {FIG_DIR}")
    for name in args.only:
        FIGURES[name](X, y)


if __name__ == "__main__":
    main()
