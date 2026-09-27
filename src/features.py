import numpy as np
import pandas as pd

from text_utils import (
    normalize_simple, tokens, jaccard, overlap_min, containment,
    seq_ratio, length_ratio, number_overlap, postal_overlap
)

BASE_FEATURES = [
    "name_exact",
    "name_suffix_exact",
    "name_seq_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_containment",
    "name_length_ratio",
    "name_length_diff",
    "address_exact",
    "address_seq_ratio",
    "address_token_jaccard",
    "address_token_overlap",
    "address_containment",
    "address_length_ratio",
    "address_length_diff",
    "address_number_overlap",
    "address_postal_overlap",
    "country_match",
    "combined_seq_ratio",
    "combined_token_jaccard",
    "source_is_s3",
]

S_FEATURES = [
    "exact_name",
    "token_name_hits",
    "token_address_hits",
    "tfidf_name",
    "tfidf_address",
    "tfidf_combined",
    "bm25_name",
    "bm25_address",
]

def pair_features(a, b, s_features=None):
    name_a = normalize_simple(a.get("business_name", ""))
    name_b = normalize_simple(b.get("business_name", ""))
    name_as = normalize_simple(a.get("name_no_suffix", a.get("business_name", "")))
    name_bs = normalize_simple(b.get("name_no_suffix", b.get("business_name", "")))

    addr_a = normalize_simple(a.get("business_address", ""))
    addr_b = normalize_simple(b.get("business_address", ""))

    country_a = normalize_simple(a.get("country", ""))
    country_b = normalize_simple(b.get("country", ""))

    nt_a, nt_b = tokens(name_a), tokens(name_b)
    at_a, at_b = tokens(addr_a), tokens(addr_b)

    combined_a = (name_a + " " + addr_a).strip()
    combined_b = (name_b + " " + addr_b).strip()

    out = {
        "name_exact": float(name_a == name_b and name_a != ""),
        "name_suffix_exact": float(name_as == name_bs and name_as != ""),
        "name_seq_ratio": seq_ratio(name_a, name_b),
        "name_token_jaccard": jaccard(nt_a, nt_b),
        "name_token_overlap": overlap_min(nt_a, nt_b),
        "name_containment": containment(nt_a, nt_b),
        "name_length_ratio": length_ratio(name_a, name_b),
        "name_length_diff": abs(len(name_a) - len(name_b)),

        "address_exact": float(addr_a == addr_b and addr_a != ""),
        "address_seq_ratio": seq_ratio(addr_a, addr_b),
        "address_token_jaccard": jaccard(at_a, at_b),
        "address_token_overlap": overlap_min(at_a, at_b),
        "address_containment": containment(at_a, at_b),
        "address_length_ratio": length_ratio(addr_a, addr_b),
        "address_length_diff": abs(len(addr_a) - len(addr_b)),
        "address_number_overlap": number_overlap(addr_a, addr_b),
        "address_postal_overlap": postal_overlap(addr_a, addr_b),

        "country_match": float(
            country_a != "" and country_a == country_b
        ),

        "combined_seq_ratio": seq_ratio(combined_a, combined_b),
        "combined_token_jaccard": jaccard(
            tokens(combined_a), tokens(combined_b)
        ),

        "source_is_s3": float(
            str(b.get("entity_id", "")).startswith("S3-")
        ),
    }

    for col in S_FEATURES:
        out[col] = float((s_features or {}).get(col, 0.0) or 0.0)

    return out

def build_feature_matrix(candidate_pairs, s1_lookup, s23_lookup, s_feature_lookup=None):
    rows = []
    for _, p in candidate_pairs.iterrows():
        s1_id = p["source1_entity_id"]
        cid = p["candidate_entity_id"]
        a = s1_lookup[s1_id]
        b = s23_lookup[cid]
        sf = (s_feature_lookup or {}).get((s1_id, cid), {})
        feats = pair_features(a, b, sf)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cid
        rows.append(feats)

    return pd.DataFrame(rows)
