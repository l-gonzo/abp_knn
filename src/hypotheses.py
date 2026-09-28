"""
hypotheses.py — Contraste estadístico de la hipótesis de la memoria
====================================================================

Hipótesis (única, direccional):

    Si la matriz de costo (C01 = 10 por cada falso negativo, C10 = 1 por cada falso
    positivo) se incorpora en la regla de decisión de KNN mediante el umbral de riesgo
    mínimo (Direct-CS-KNN, k = 5), entonces, en cada uno de los escenarios 90:10 y 95:5
    con 5 % de ruido gaussiano en el entrenamiento:
      (a) su TMC será al menos 30 % menor que el de cada método de referencia
          —KNN estándar (k = 5, voto mayoritario), CM-KNN y One-Step KNN—, con
          diferencia significativa por pliegue (Wilcoxon pareado unilateral, p < 0.05); y
      (b) su sensibilidad media a minas será de al menos 0.60.

Definiciones operativas
    TMC_m        = suma sobre los 10 pliegues de (10·FN + 1·FP), con la prueba ponderada a la
                   prevalencia operativa del escenario (10 % o 5 % de minas).
    Reducción    = (TMC_ref − TMC_DCS) / TMC_ref, calculada contra cada referencia.
    Sensibilidad = TP / (TP + FN), media en los 10 pliegues (umbral absoluto, no relativo).
    Condiciones  = mismos 10 pliegues estratificados (semilla 42), mismo submuestreo,
                   estandarización, ruido y ponderación de la prueba; distancia euclidiana;
                   sin PCA.
    Decisión     = se acepta en un escenario si (a) y (b) se cumplen frente a las tres
                   referencias; se acepta en general si se acepta en ambos escenarios.

Además se calculan las pruebas de los análisis complementarios que la memoria reporta
como resultados (no son hipótesis): Spearman entre exactitud y G-Mean, Mann–Whitney de
los vectores rebeldes, impacto de PCA y latencia.

    python hypotheses.py      # escribe results/tables/hypotheses.csv y complementary_tests.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import RESULTS  # noqa: E402

ALPHA = 0.05
TREATMENT = "Direct-CS-KNN"
REFERENCES = ["KNN (k=5)", "CM-KNN", "One-Step KNN"]
SCENARIOS = {"90:10": "90:10 + ruido", "95:5": "95:5 + ruido"}
MIN_REDUCTION = 0.30
MIN_SENSITIVITY = 0.60

TAB = os.path.join(RESULTS, "tables")
F = pd.read_csv(os.path.join(TAB, "folds.csv"))
S = pd.read_csv(os.path.join(TAB, "summary.csv"))
REB = pd.read_csv(os.path.join(TAB, "rebels_all_samples.csv"))


def fold_values(scenario, model, metric):
    d = F[(F.scenario == scenario) & (F.model == model)].sort_values("fold")
    return d[metric].to_numpy()


def summary_row(scenario, model):
    return S[(S.scenario == scenario) & (S.model == model)].iloc[0]


def wilcoxon(a, b, alternative):
    """Wilcoxon pareado a vs b; si todas las diferencias son 0 regresa p = 1."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)


# ---------------------------------------------------------------------------
# Contraste de la hipótesis
# ---------------------------------------------------------------------------
def test_hypothesis():
    rows = []
    for key, sc in SCENARIOS.items():
        t = summary_row(sc, TREATMENT)
        for ref in REFERENCES:
            r = summary_row(sc, ref)
            red = (r.tmc_sum - t.tmc_sum) / r.tmc_sum
            p = wilcoxon(fold_values(sc, TREATMENT, "tmc"), fold_values(sc, ref, "tmc"), "less")
            rows.append({"escenario": key, "criterio": "(a) TMC", "referencia": ref,
                         "valor_referencia": r.tmc_sum, "valor_direct_cs": t.tmc_sum,
                         "reduccion": red, "meta": f"reducción ≥ {MIN_REDUCTION:.0%} y p < {ALPHA}",
                         "p_valor": p, "cumple": bool(red >= MIN_REDUCTION and p < ALPHA)})
        rows.append({"escenario": key, "criterio": "(b) Sensibilidad", "referencia": "umbral absoluto",
                     "valor_referencia": np.nan, "valor_direct_cs": t.sensitivity_mean,
                     "reduccion": np.nan, "meta": f"≥ {MIN_SENSITIVITY:.2f}", "p_valor": np.nan,
                     "cumple": bool(t.sensitivity_mean >= MIN_SENSITIVITY)})
    out = pd.DataFrame(rows)
    dec = out.groupby("escenario", sort=False).cumple.all()
    out["decision_escenario"] = out.escenario.map(lambda k: "Se acepta" if dec[k] else "Se rechaza")
    return out, dec


# ---------------------------------------------------------------------------
# Análisis complementarios (resultados, no hipótesis)
# ---------------------------------------------------------------------------
def complementary_tests():
    rows = []
    for key, sc in SCENARIOS.items():
        s = S[S.scenario == sc]
        rho, p = stats.spearmanr(s.accuracy_mean, s.g_mean_mean)
        rows.append({"analisis": "Falacia de la exactitud", "condicion": f"11 modelos, {key}",
                     "estadistico": f"Spearman rho(accuracy, G-Mean) = {rho:.3f}", "p_valor": float(p),
                     "nota": "p bilateral"})
    reb, rest = REB[REB.rebelde], REB[~REB.rebelde]
    for col, alt, lab in (("razon_maha", "greater", "razón de Mahalanobis real/opuesto"),
                          ("densidad_knn", "less", "densidad k-NN")):
        p = float(stats.mannwhitneyu(reb[col], rest[col], alternative=alt).pvalue)
        rows.append({"analisis": "Vectores rebeldes", "condicion": f"{len(reb)} rebeldes vs {len(rest)}",
                     "estadistico": f"mediana {lab}: {reb[col].median():.3f} vs {rest[col].median():.3f}",
                     "p_valor": p, "nota": f"Mann–Whitney unilateral ({alt})"})
    for key in ("90:10", "orig"):
        sc = SCENARIOS.get(key, "Original")
        o, c, k = (fold_values(sc, m, "infer_ms_per_sample") for m in ("One-Step KNN", "CM-KNN", "KNN (k=5)"))
        rows.append({"analisis": "Latencia", "condicion": f"One-Step vs CM-KNN y KNN, {sc}",
                     "estadistico": f"{o.mean():.3f} vs {c.mean():.3f} vs {k.mean():.4f} ms "
                                    f"(−{100 * (c.mean() - o.mean()) / c.mean():.0f} % vs CM-KNN)",
                     "p_valor": wilcoxon(o, c, "less"), "nota": "Wilcoxon One-Step < CM-KNN"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    h, dec = test_hypothesis()
    h.to_csv(os.path.join(TAB, "hypotheses.csv"), index=False)
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for k, ok in dec.items():
        print(f"  {k}: {'SE ACEPTA' if ok else 'SE RECHAZA'}")
    print(f"  Decisión general: {'SE ACEPTA' if dec.all() else 'SE RECHAZA' if not dec.any() else 'SE ACEPTA SOLO EN ' + ', '.join(k for k, v in dec.items() if v)}")
    c = complementary_tests()
    c.to_csv(os.path.join(TAB, "complementary_tests.csv"), index=False)
    print("\nAnálisis complementarios:")
    print(c.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
