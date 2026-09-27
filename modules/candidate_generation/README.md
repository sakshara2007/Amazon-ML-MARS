# Teammate S — Candidate Generation & Lexical Retrieval

This module implements the blocking/candidate-generation stage for the Business
Entity Resolution Challenge.

## Responsibility

S owns:

1. TSV loading with explicit `sep="\t"`.
2. Name/address normalization.
3. Exact normalized-name blocking.
4. Token-based blocking.
5. Character TF-IDF retrieval.
6. Address TF-IDF retrieval.
7. BM25 retrieval.
8. Candidate union and deduplication.
9. Candidate-size control.
10. Training-time candidate recall evaluation.
11. Candidate reduction-ratio reporting.
12. Production of `candidate_pairs.tsv`.
13. Export of candidate-level lexical features for teammates R, M and A.

The challenge requires `candidate_pairs.tsv` to contain the final candidate set
actually passed to the matching model, and every final predicted match must be
a member of this set.

## Expected data layout

dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv

## Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run on training data

This evaluates blocking recall and creates training candidate artifacts:

```bash
python src/run_candidates.py \
  --mode train \
  --data-dir dataset/train \
  --output-dir artifacts/train
```

## Run on test data

```bash
python src/run_candidates.py \
  --mode test \
  --data-dir dataset/test \
  --output-dir output
```

The test run creates:

- `output/candidate_pairs.tsv`
- `output/candidate_features.tsv`
- `output/blocking_report.json`

## Candidate-pairs contract

`candidate_pairs.tsv` has exactly:

```text
source1_entity_id    candidate_entity_ids
```

with a TAB between columns and commas only inside the ID list.

Every S1 test entity receives exactly one row. Empty candidate lists are allowed.

Only S2/S3 IDs are emitted.

## Integration with teammates

R, M and A should consume:

- `candidate_pairs.tsv`
- `candidate_features.tsv`

They should NOT silently create a different candidate set for final inference
unless the team agrees to make that new set the official final candidate set.

## Important challenge constraints

Do not use external business databases, APIs, geocoding services, web lookup,
or external identity resolution services. This module uses only the supplied
challenge records.
