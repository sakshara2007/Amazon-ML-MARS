import numpy as np
import pandas as pd

from text import make_name, make_address, make_combined

SEMANTIC_FEATURES = [
    "name_cosine",
    "address_cosine",
    "combined_cosine",
    "name_distance",
    "address_distance",
    "combined_distance",
    "name_address_cosine_gap",
    "source_is_s3",
    "country_match",
]

S_FEATURES = [
    "exact_name",
    "name_no_suffix",
    "name_token_hits",
    "address_token_hits",
    "translit_token_hits",
    "compact_name",
    "ngram_hits",
    "number_hits",
    "name_similarity",
    "address_similarity",
    "name_token_similarity",
    "address_token_similarity",
    "number_match",
    "postal_match",
    "blocking_score",
]
# NOTE: S also produces "country_match" but we deliberately exclude it here
# because M computes its own country_match from source1/2/3 country fields
# in SEMANTIC_FEATURES — including S's version here would silently overwrite
# ours in build_features()'s row dict.

def build_text_maps(s1_lookup, s23_lookup):
    records = list(s1_lookup.values()) + list(s23_lookup.values())
    name = {r["entity_id"]: make_name(r) for r in records}
    address = {r["entity_id"]: make_address(r) for r in records}
    combined = {r["entity_id"]: make_combined(r) for r in records}
    return name, address, combined

def build_features(pairs, s1_lookup, s23_lookup,
                   emb_name, emb_address, emb_combined,
                   name_idx, address_idx, combined_idx,
                   s_features=None):
    rows = []

    for _, p in pairs.iterrows():
        s1id = p["source1_entity_id"]
        cid = p["candidate_entity_id"]

        a = s1_lookup[s1id]
        b = s23_lookup[cid]

        na = name_idx[make_name(a)]
        nb = name_idx[make_name(b)]
        aa = address_idx[make_address(a)]
        ab = address_idx[make_address(b)]
        ca = combined_idx[make_combined(a)]
        cb = combined_idx[make_combined(b)]

        name_cos = float(np.dot(emb_name[na], emb_name[nb]))
        addr_cos = float(np.dot(emb_address[aa], emb_address[ab]))
        comb_cos = float(np.dot(emb_combined[ca], emb_combined[cb]))

        sf = (s_features or {}).get((s1id, cid), {})

        row = {
            "name_cosine": name_cos,
            "address_cosine": addr_cos,
            "combined_cosine": comb_cos,
            "name_distance": 1.0 - name_cos,
            "address_distance": 1.0 - addr_cos,
            "combined_distance": 1.0 - comb_cos,
            "name_address_cosine_gap": abs(name_cos - addr_cos),
            "source_is_s3": float(str(cid).startswith("S3-")),
            "country_match": float(
                str(a.get("country", "")).strip().lower()
                == str(b.get("country", "")).strip().lower()
                and str(a.get("country", "")).strip() != ""
            ),
        }

        for c in S_FEATURES:
            row[c] = float(sf.get(c, 0.0) or 0.0)

        row["source1_entity_id"] = s1id
        row["candidate_entity_id"] = cid
        rows.append(row)

    return pd.DataFrame(rows)
