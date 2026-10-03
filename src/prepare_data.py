"""Baixa o B2W-Reviews01 e gera splits balanceados (train/val/test) em JSONL.

Uso:
    python src/prepare_data.py --train-per-class 2000 --val-per-class 100 --test-per-class 300
"""

import argparse
import json

import pandas as pd

from common import DATA_DIR, LABELS, rating_to_label

PARQUET_URL = (
    "https://huggingface.co/api/datasets/ruanchaves/b2w-reviews01/parquet/default/train/0.parquet"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=2000)
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=600, help="descarta reviews muito longas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Baixando {PARQUET_URL} ...")
    df = pd.read_parquet(PARQUET_URL)
    print(f"Total bruto: {len(df)} reviews")

    df["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"].str.len() >= 10]
    df["review"] = df.apply(
        lambda r: f"{r.review_title}. {r.review_text}" if r.review_title else r.review_text, axis=1
    )
    df = df[df["review"].str.len() <= args.max_chars]
    df = df.drop_duplicates(subset="review")
    df["label"] = df["overall_rating"].astype(int).map(rating_to_label)

    print("Distribuição após limpeza:")
    print(df["label"].value_counts().to_string())

    n_total = args.train_per_class + args.val_per_class + args.test_per_class
    splits = {"train": [], "val": [], "test": []}
    for label in LABELS:
        sub = df[df["label"] == label].sample(n=n_total, random_state=args.seed)
        a, b = args.train_per_class, args.train_per_class + args.val_per_class
        splits["train"].append(sub.iloc[:a])
        splits["val"].append(sub.iloc[a:b])
        splits["test"].append(sub.iloc[b:])

    DATA_DIR.mkdir(exist_ok=True)
    for name, parts in splits.items():
        part = pd.concat(parts).sample(frac=1, random_state=args.seed)
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in part.itertuples():
                f.write(json.dumps({"review": r.review, "label": r.label,
                                    "rating": int(r.overall_rating)}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} exemplos -> {path}")


if __name__ == "__main__":
    main()
