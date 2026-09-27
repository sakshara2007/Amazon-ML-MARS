from pathlib import Path
import pandas as pd

def read_records(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def read_candidates(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def explode_candidates(df):
    rows = []
    for _, r in df.iterrows():
        ids = [x.strip() for x in r["candidate_entity_ids"].split(",") if x.strip()]
        for cid in ids:
            rows.append({
                "source1_entity_id": r["source1_entity_id"],
                "candidate_entity_id": cid,
            })
    return pd.DataFrame(rows)

def combine_sources(s2, s3):
    return pd.concat([s2, s3], ignore_index=True)

def lookup(df):
    return df.set_index("entity_id", drop=False).to_dict("index")

def truth_sets(gt):
    truth = {}
    for _, row in gt.iterrows():
        truth[row["source1_entity_id"]] = {
            x.strip() for x in row.get("matched_entity_ids", "").split(",") if x.strip()
        }
    return truth

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
            c: float(r[c]) if c in r and str(r[c]).strip() else 0.0
            for c in cols
        }
    return out
