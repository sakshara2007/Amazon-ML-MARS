import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-per-s1", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.labeled_features, sep="\t")

    # Hard negatives: non-matching candidates with high lexical evidence.
    score_cols = [
        c for c in [
            "tfidf_combined", "tfidf_name", "tfidf_address",
            "bm25_name", "bm25_address"
        ] if c in df.columns
    ]

    if not score_cols:
        raise ValueError("No S lexical score columns found.")

    df["hardness"] = df[score_cols].fillna(0).sum(axis=1)
    negatives = df[df["label"] == 0].copy()

    selected = (
        negatives.sort_values("hardness", ascending=False)
        .groupby("source1_entity_id", group_keys=False)
        .head(args.keep_per_s1)
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, sep="\t", index=False)

if __name__ == "__main__":
    main()
