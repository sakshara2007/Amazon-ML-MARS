import pandas as pd

def ground_truth_sets(gt):
    out = {}
    for _, row in gt.iterrows():
        raw = row.get("matched_entity_ids", "")
        out[row["source1_entity_id"]] = {
            x.strip() for x in raw.split(",") if x.strip()
        }
    return out

def add_pair_labels(features, truth):
    out = features.copy()
    out["label"] = [
        int(cid in truth.get(s1, set()))
        for s1, cid in zip(
            out["source1_entity_id"], out["candidate_entity_id"]
        )
    ]
    return out
