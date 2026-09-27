from pathlib import Path
import pandas as pd

REQUIRED_RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def read_records(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[REQUIRED_RECORD_COLUMNS].copy()

def read_ground_truth(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def source_from_id(entity_id):
    if entity_id.startswith("S1-"):
        return "S1"
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"
    return "UNKNOWN"

def write_candidate_pairs(rows, path):
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
