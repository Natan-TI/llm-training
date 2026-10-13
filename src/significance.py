"""Testa se as diferenças de F1 entre experimentos são estatisticamente significativas (bootstrap).

Com 900 exemplos de teste, diferenças pequenas podem ser só ruído da amostra. Para cada experimento,
reamostramos o conjunto de teste com reposição (as MESMAS amostras para os dois modelos) e calculamos
o intervalo de confiança de 95% da diferença de F1 macro em relação à configuração de referência.
Se o intervalo contém 0, a diferença não é significativa (empate estatístico). Se exclui o 0 por
menos de 0,0025 (o limite arredonda para ±0,00), marcamos como limítrofe: o resultado muda conforme a
semente da reamostragem e não deve ser tratado como conclusivo.

Uso:
    python src/significance.py
"""

import json

import numpy as np
from sklearn.metrics import f1_score

from common import LABELS, RECOMMENDED_ADAPTER, RESULTS_DIR

N_BOOT = 2000
BORDERLINE = 0.0025


def load(run):
    rows = [json.loads(l) for l in open(RESULTS_DIR / run / "predictions.jsonl", encoding="utf-8")]
    return np.array([r["label"] for r in rows]), np.array([r["pred"] for r in rows])


def f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    ref = RECOMMENDED_ADAPTER
    y, p_ref = load(ref)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    ref_scores = np.array([f1(y[ix], p_ref[ix]) for ix in boot])

    runs = sorted(d.name for d in RESULTS_DIR.iterdir()
                  if (d / "predictions.jsonl").exists() and d.name != ref)
    lines = [
        f"Referência: **{ref}** (F1 macro = {f1(y, p_ref):.3f}, IC95% "
        f"[{np.percentile(ref_scores, 2.5):.3f}, {np.percentile(ref_scores, 97.5):.3f}]) · "
        f"bootstrap pareado com {N_BOOT} reamostragens do teste\n",
        "| Experimento | F1 macro | Diferença vs referência | IC 95% da diferença | Conclusão |",
        "|---|---|---|---|---|",
    ]
    for run in sorted(runs, key=lambda r: -f1(*load(r))):
        y2, p = load(run)
        assert (y2 == y).all(), f"{run} usa outro conjunto de teste"
        diffs = np.array([f1(y[ix], p[ix]) for ix in boot]) - ref_scores
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        if lo <= 0 <= hi:
            verdict = "empate estatístico"
        elif min(abs(lo), abs(hi)) < BORDERLINE:
            verdict = "limítrofe (IC encosta no 0)"
        else:
            verdict = "**melhor** (significativo)" if lo > 0 else "**pior** (significativo)"
        lines.append(f"| {run} | {f1(y, p):.3f} | {f1(y, p) - f1(y, p_ref):+.3f} "
                     f"| [{lo:+.3f}, {hi:+.3f}] | {verdict} |")

    out = "\n".join(lines) + "\n"
    (RESULTS_DIR / "significance.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
