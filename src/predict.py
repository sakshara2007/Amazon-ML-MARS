import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data import (
    read_records, read_candidates, explode_candidates,
    combine_sources, lookup, load_s_features
)
from text import make_name, make_address, make_combined
from embeddings import Embedder
from features import build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--candidate-file", required=True)
    ap.add_argument("--candidate-features", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = read_records(test_dir / "test_source1.tsv")
    s2 = read_records(test_dir / "test_source2.tsv")
    s3 = read_records(test_dir / "test_source3.tsv")
    s23 = combine_sources(s2, s3)

    pairs = explode_candidates(read_candidates(args.candidate_file))

    s1_lookup = lookup(s1)
    s23_lookup = lookup(s23)
    sf = load_s_features(args.candidate_features)

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

    with open(model_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    mlp = joblib.load(model_dir / "mlp.joblib")

    X = feat[feature_cols].astype(float)
    scored = feat[
        ["source1_entity_id", "candidate_entity_id"]
    ].copy()

    scored["mlp_score"] = mlp.predict_proba(X)[:, 1]

    scored.to_csv(
        out_dir / "mlp_scores.tsv",
        sep="\t",
        index=False
    )

    # Development-only standalone prediction using R's M-independent
    # semantic threshold. Final team should combine R/M/A centrally.
    threshold = float(
        metadata["best_threshold"]["threshold"]
    )

    results = []
    grouped = scored.groupby("source1_entity_id")
    for s1id, g in grouped:
        ids = g.loc[
            g["mlp_score"] >= threshold,
            "candidate_entity_id"
        ].tolist()

        ids = list(dict.fromkeys(ids))

        results.append({
            "source1_entity_id": s1id,
            "matched_entity_ids": ",".join(ids)
        })

    pd.DataFrame(results).to_csv(
        out_dir / "m_predictions.tsv",
        sep="\t",
        index=False
    )

if __name__ == "__main__":
    main()
