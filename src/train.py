import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data import (
    read_records, read_candidates, explode_candidates,
    combine_sources, lookup, truth_sets, load_s_features
)
from text import make_name, make_address, make_combined
from embeddings import Embedder
from features import build_features, SEMANTIC_FEATURES, S_FEATURES
from metrics import tune

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--candidate-file", required=True)
    ap.add_argument("--candidate-features", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = read_records(train_dir / "train_source1.tsv")
    s2 = read_records(train_dir / "train_source2.tsv")
    s3 = read_records(train_dir / "train_source3.tsv")
    gt = pd.read_csv(
        train_dir / "train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False
    )

    pairs = explode_candidates(read_candidates(args.candidate_file))
    s23 = combine_sources(s2, s3)

    s1_lookup = lookup(s1)
    s23_lookup = lookup(s23)
    sf = load_s_features(args.candidate_features)

    # Embed only the unique texts that occur in S1/S2/S3 records.
    embedder = Embedder(
        args.embedding_model,
        device=args.device,
        batch_size=args.batch_size
    )

    needed_s1_ids = set(pairs["source1_entity_id"])
    needed_cand_ids = set(pairs["candidate_entity_id"])

    records = (
        [s1_lookup[i] for i in needed_s1_ids if i in s1_lookup]
        + [s23_lookup[i] for i in needed_cand_ids if i in s23_lookup]
    )

    names = [make_name(r) for r in records]
    addresses = [make_address(r) for r in records]
    combined = [make_combined(r) for r in records]

    name_idx, name_emb = embedder.encode_unique(names)
    addr_idx, addr_emb = embedder.encode_unique(addresses)
    comb_idx, comb_emb = embedder.encode_unique(combined)

    feat = build_features(
        pairs,
        s1_lookup,
        s23_lookup,
        name_emb,
        addr_emb,
        comb_emb,
        name_idx,
        addr_idx,
        comb_idx,
        sf,
    )

    truth = truth_sets(gt)

    feat["label"] = [
        int(cid in truth.get(s1, set()))
        for s1, cid in zip(
            feat["source1_entity_id"],
            feat["candidate_entity_id"]
        )
    ]

    feature_cols = SEMANTIC_FEATURES + S_FEATURES

    X = feat[feature_cols].astype(float)
    y = feat["label"].astype(int)
    groups = feat["source1_entity_id"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=42
    )
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Balanced classes via oversampling (MLPClassifier.fit has no sample_weight arg).
    from sklearn.utils import resample

    pos_mask = y_train == 1
    neg_mask = y_train == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos > 0 and n_neg > 0 and n_pos != n_neg:
        X_pos, y_pos = X_train[pos_mask], y_train[pos_mask]
        X_neg, y_neg = X_train[neg_mask], y_train[neg_mask]
        if n_pos < n_neg:
            X_pos, y_pos = resample(X_pos, y_pos, replace=True, n_samples=n_neg, random_state=42)
        else:
            X_neg, y_neg = resample(X_neg, y_neg, replace=True, n_samples=n_pos, random_state=42)
        X_train = pd.concat([X_pos, X_neg])
        y_train = pd.concat([y_pos, y_neg])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        ))
    ])

    mlp.fit(X_train, y_train)

    val_pairs = feat.iloc[val_idx][
        ["source1_entity_id", "candidate_entity_id"]
    ].copy()

    val_pairs["mlp_score"] = mlp.predict_proba(X_val)[:, 1]

    score_df = val_pairs.rename(columns={"mlp_score": "score"})
    threshold = tune(score_df, truth)

    # Save model and metadata.
    joblib.dump(mlp, out_dir / "mlp.joblib")

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "embedding_model": args.embedding_model,
        "embedding_license_check_required": True,
        "device": args.device,
        "feature_columns": feature_cols,
        "train_pairs": int(len(train_idx)),
        "validation_pairs": int(len(val_idx)),
        "positive_train_pairs": int(y_train.sum()),
        "positive_validation_pairs": int(y_val.sum()),
        "best_threshold": threshold,
        "mlp_iterations": int(mlp.named_steps["mlp"].n_iter_),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    val_pairs.to_csv(
        out_dir / "validation_scores.tsv",
        sep="\t",
        index=False
    )

    # Save embedding model name only. The actual pretrained model should be
    # reproducibly downloaded/cached according to the team's environment.
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
