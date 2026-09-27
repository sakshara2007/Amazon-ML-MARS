import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data import read_records, read_candidates, explode_candidates, combine_sources, build_lookup
from features import build_feature_matrix
from labels import ground_truth_sets, add_pair_labels
from metrics import tune_threshold, evaluate_macro

from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier

def load_s_features(path):
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    feature_cols = [
        "exact_name", "token_name_hits", "token_address_hits",
        "tfidf_name", "tfidf_address", "tfidf_combined",
        "bm25_name", "bm25_address"
    ]
    lookup = {}
    for _, r in df.iterrows():
        lookup[(r["source1_entity_id"], r["candidate_entity_id"])] = {
            c: float(r[c]) if str(r[c]).strip() else 0.0
            for c in feature_cols if c in r
        }
    return lookup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--candidate-file", required=True)
    ap.add_argument("--candidate-features", default="")
    ap.add_argument("--output-dir", required=True)
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

    candidates = explode_candidates(read_candidates(args.candidate_file))
    s23 = combine_sources(s2, s3)

    s1_lookup = build_lookup(s1)
    s23_lookup = build_lookup(s23)
    sf = load_s_features(args.candidate_features)
    Xdf = build_feature_matrix(candidates, s1_lookup, s23_lookup, sf)

    truth = ground_truth_sets(gt)
    labeled = add_pair_labels(Xdf, truth)

    feature_cols = [c for c in labeled.columns
                    if c not in ["source1_entity_id", "candidate_entity_id", "label"]]
    X = labeled[feature_cols].astype(float)
    y = labeled["label"].astype(int)

    groups = labeled["source1_entity_id"].values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=42
    )
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Keep class imbalance explicit. The challenge has many more negatives
    # than positive matches in a blocked candidate set.
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    scale_pos = neg / pos

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(100, verbose=False)]
    )

    xgb = XGBClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_base = labeled.iloc[val_idx][
        ["source1_entity_id", "candidate_entity_id"]
    ].copy()

    val_base["lgbm_score"] = lgbm.predict_proba(X_val)[:, 1]
    val_base["xgb_score"] = xgb.predict_proba(X_val)[:, 1]

    # Use the same validation entity IDs and exact competition-style metric.
    lgbm_df = val_base.rename(columns={"lgbm_score": "score"})[
        ["source1_entity_id", "candidate_entity_id", "score"]
    ]
    xgb_df = val_base.rename(columns={"xgb_score": "score"})[
        ["source1_entity_id", "candidate_entity_id", "score"]
    ]

    lgbm_threshold = tune_threshold(lgbm_df, truth)
    xgb_threshold = tune_threshold(xgb_df, truth)

    val_base["blend_score"] = (
        0.70 * val_base["lgbm_score"] +
        0.30 * val_base["xgb_score"]
    )
    blend_df = val_base.rename(columns={"blend_score": "score"})[
        ["source1_entity_id", "candidate_entity_id", "score"]
    ]
    blend_threshold = tune_threshold(blend_df, truth)

    report = {
        "feature_count": len(feature_cols),
        "train_pairs": int(len(train_idx)),
        "validation_pairs": int(len(val_idx)),
        "positive_train_pairs": int(y_train.sum()),
        "positive_validation_pairs": int(y_val.sum()),
        "lgbm_best_iteration": int(getattr(lgbm, "best_iteration_", 0)),
        "lgbm_threshold": lgbm_threshold,
        "xgb_threshold": xgb_threshold,
        "blend_threshold": blend_threshold,
        "blend_weights": {"lightgbm": 0.70, "xgboost": 0.30},
    }

    joblib.dump(lgbm, out_dir / "lightgbm.joblib")
    joblib.dump(xgb, out_dir / "xgboost.joblib")

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(report, f, indent=2)

    val_base.to_csv(
        out_dir / "validation_scores.tsv", sep="\t", index=False
    )

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
