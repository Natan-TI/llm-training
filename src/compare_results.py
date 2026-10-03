"""Junta os resultados de todos os experimentos em uma tabela Markdown e gráficos.

Uso:
    python src/compare_results.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR


def load_runs():
    runs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        ev = d / "eval_metrics.json"
        if not d.is_dir() or not ev.exists():
            continue
        run = {"name": d.name, "eval": json.loads(ev.read_text(encoding="utf-8"))}
        tr = d / "train_info.json"
        run["train"] = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else None
        lg = d / "train_log.json"
        run["log"] = json.loads(lg.read_text(encoding="utf-8")) if lg.exists() else None
        runs.append(run)
    return runs


def markdown_table(runs):
    head = ("| Experimento | Modelo | LR | LoRA r | Épocas | Treino (s) | VRAM pico (GB) "
            "| Temp. | Acurácia | F1 macro | Inválidas |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in sorted(runs, key=lambda r: -r["eval"]["f1_macro"]):
        e = r["eval"]
        t = r["train"]
        if t is None and e["adapter"]:  # avaliação extra de um adaptador já treinado
            src = RESULTS_DIR / e["adapter"].replace("\\", "/").split("/")[-1] / "train_info.json"
            t = json.loads(src.read_text(encoding="utf-8")) if src.exists() else None
        hp = t["hyperparams"] if t else {}
        lines.append(
            f"| {r['name']} | {e['model'].split('/')[-1]} | {hp.get('lr', '—')} | {hp.get('lora_r', '—')} "
            f"| {hp.get('epochs', '—')} | {t['train_time_s'] if t else '—'} "
            f"| {t['peak_gpu_mem_gb'] if t else '—'} | {e['temperature']} | {e['accuracy']:.3f} "
            f"| {e['f1_macro']:.3f} | {e['invalid_outputs']} |"
        )
    return head + "\n".join(lines) + "\n"


def plot_f1(runs):
    runs = sorted(runs, key=lambda r: r["eval"]["f1_macro"])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(runs) + 1.2))
    vals = [r["eval"]["f1_macro"] for r in runs]
    colors = ["#9aa0a6" if r["eval"]["adapter"] is None else "#2a6fdb" for r in runs]
    ax.barh([r["name"] for r in runs], vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 macro (teste)")
    ax.set_title("Comparação dos experimentos (cinza = sem fine-tuning)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison_f1.png", dpi=120)


def plot_loss(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in runs:
        if not r["log"]:
            continue
        tr = [(x["step"], x["loss"]) for x in r["log"] if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in r["log"] if "eval_loss" in x]
        if tr:
            axes[0].plot(*zip(*tr), label=r["name"], linewidth=1.3)
        if ev:
            axes[1].plot(*zip(*ev), label=r["name"], marker="o", markersize=3, linewidth=1.3)
    for ax, title in zip(axes, ["Loss de treino", "Loss de validação"]):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=120)


def main():
    runs = load_runs()
    table = markdown_table(runs)
    (RESULTS_DIR / "comparison.md").write_text(table, encoding="utf-8")
    plot_f1(runs)
    plot_loss(runs)
    print(table)


if __name__ == "__main__":
    main()
