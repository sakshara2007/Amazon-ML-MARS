import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

from data import read_records, read_candidates, explode_candidates, combine_sources, build_lookup
from features import build_feature_matrix

def load_s_features(path):
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    cols = [
        "exact_name", "token_name_hits", "token_address_hits",
        "tfidf_name", "tfidf_address", "tfidf_combined",
        "bm25_name", "bm25_address"
    ]
    out = {}
    for _, r in df.iterrows():
        out[(r["source1_entity_id"], r["candidate_entity_id"])] = {
            c: float(r[c]) if str(r[c]).strip() else 0.0
            for c in cols if c in r
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--candidate-file", required=True)
    ap.add_argument("--candidate-features", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = read_records(test_dir / "test_source1.tsv")
    s2 = read_records(test_dir / "test_source2.tsv")
    s3 = read_records(test_dir / "test_source3.tsv")
    s23 = combine_sources(s2, s3)

    cand = explode_candidates(read_candidates(args.candidate_file))
    s1_lookup = build_lookup(s1)
    s23_lookup = build_lookup(s23)

    sf = load_s_features(args.candidate_features)
    feat = build_feature_matrix(cand, s1_lookup, s23_lookup, sf)

    with open(model_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)

    with open(model_dir / "thresholds.json") as f:
        thresholds = json.load(f)

    lgbm = joblib.load(model_dir / "lightgbm.joblib")
    xgb = joblib.load(model_dir / "xgboost.joblib")

    X = feat[feature_cols].astype(float)
    scored = feat[["source1_entity_id", "candidate_entity_id"]].copy()
    scored["lightgbm_score"] = lgbm.predict_proba(X)[:, 1]
    scored["xgboost_score"] = xgb.predict_proba(X)[:, 1]
    scored["ensemble_score"] = (
        0.70 * scored["lightgbm_score"] +
        0.30 * scored["xgboost_score"]
    )

    threshold = float(thresholds["blend_threshold"]["threshold"])

    results = []
    for s1id in s1["entity_id"]:
        g = scored[scored["source1_entity_id"] == s1id]
        ids = g.loc[
            g["ensemble_score"] >= threshold,
            "candidate_entity_id"
        ].tolist()
        ids = list(dict.fromkeys(ids))
        results.append({
            "source1_entity_id": s1id,
            "matched_entity_ids": ",".join(ids)
        })

    scored.to_csv(out_dir / "ensemble_scores.tsv", sep="\t", index=False)
    scored.to_csv(out_dir / "all_model_scores.tsv", sep="\t", index=False)

    pd.DataFrame(results).to_csv(
        out_dir / "matching_results.tsv",
        sep="\t", index=False
    )

if __name__ == "__main__":
    main()
