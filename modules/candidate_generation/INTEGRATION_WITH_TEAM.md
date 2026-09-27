# Integration with R, M and A

S supplies two artifacts:

1. `candidate_pairs.tsv`
2. `candidate_features.tsv`

## candidate_pairs.tsv

This is the official candidate set. R/M/A must score these pairs.

Columns:

- source1_entity_id
- candidate_entity_ids

## candidate_features.tsv

One row per candidate pair:

- source1_entity_id
- candidate_entity_id
- exact_name
- token_name_hits
- token_address_hits
- tfidf_name
- tfidf_address
- tfidf_combined
- bm25_name
- bm25_address

Missing feature values are absent from the row and should be treated as 0.

R can join this table with its richer feature table.
M can add embedding features keyed by `(source1_entity_id, candidate_entity_id)`.
A can rerank only candidates that R flags as ambiguous.

## Important

If R/M/A change the candidate set, the final team must regenerate the official
candidate_pairs.tsv from that final set because the competition requires the
candidate file to represent the exact set passed to the final matching model.
