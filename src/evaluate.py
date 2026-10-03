"""Avalia o modelo base ou um adaptador LoRA no conjunto de teste.

Uso:
    python src/evaluate.py --run-name base                       # modelo sem treino (zero-shot)
    python src/evaluate.py --run-name lr2e-4_r16 --adapter outputs/lr2e-4_r16
    python src/evaluate.py --run-name lr2e-4_r16_temp0.7 --adapter outputs/lr2e-4_r16 --temperature 0.7
"""

import argparse
import json
import time

import common  # noqa: F401  (define variáveis de ambiente do HF antes dos demais imports)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from common import (DATA_DIR, DEFAULT_MODEL, LABELS, RESULTS_DIR, build_messages, generate,
                    load_model, parse_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA_DIR / f"{args.split}.jsonl", encoding="utf-8")]
    rows = rows[: args.limit] if args.limit else rows
    model, tokenizer = load_model(args.model, args.adapter)
    torch.manual_seed(0)

    preds, raws = [], []
    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        outs = generate(model, tokenizer, [build_messages(r["review"]) for r in batch],
                        max_new_tokens=8, temperature=args.temperature)
        raws += outs
        preds += [parse_label(o) for o in outs]
        print(f"\r{min(i + args.batch_size, len(rows))}/{len(rows)}", end="", flush=True)
    elapsed = time.time() - t0
    print()

    y_true = [r["label"] for r in rows]
    y_pred = [p or "inválido" for p in preds]
    invalid = sum(p is None for p in preds)
    metrics = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "n": len(rows),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "invalid_outputs": invalid,
        "eval_time_s": round(elapsed, 1),
        "reviews_per_s": round(len(rows) / elapsed, 1),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                        zero_division=0),
    }

    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r, raw, p in zip(rows, raws, y_pred):
            f.write(json.dumps({"review": r["review"], "label": r["label"], "pred": p, "raw": raw},
                               ensure_ascii=False) + "\n")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues", colorbar=False)
    plt.title(f"{args.run_name}\nacc={metrics['accuracy']:.3f}  f1={metrics['f1_macro']:.3f}")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=120)

    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print(f"Acurácia: {metrics['accuracy']:.4f} | F1 macro: {metrics['f1_macro']:.4f} | "
          f"inválidas: {invalid} | {metrics['reviews_per_s']} reviews/s")


if __name__ == "__main__":
    main()
