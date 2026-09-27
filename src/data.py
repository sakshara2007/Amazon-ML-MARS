from pathlib import Path
import pandas as pd

def read_records(path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False
    )

def read_candidates(path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False
    )

def explode_candidates(candidate_df):
    rows = []
    for _, r in candidate_df.iterrows():
        ids = [x.strip() for x in r["candidate_entity_ids"].split(",") if x.strip()]
        for cid in ids:
            rows.append({
                "source1_entity_id": r["source1_entity_id"],
                "candidate_entity_id": cid
            })
    return pd.DataFrame(rows)

def combine_sources(s2, s3):
    return pd.concat([s2, s3], ignore_index=True)

def build_lookup(df):
    return df.set_index("entity_id", drop=False).to_dict("index")
