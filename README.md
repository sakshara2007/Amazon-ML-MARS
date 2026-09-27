# Teammate M — Semantic Matching + MLP

This module implements the semantic/deep-learning branch of the Business
Entity Resolution system.

## M owns

Model 1:
- Sentence Transformer / Bi-Encoder semantic embeddings.
- Name, address and combined-text embeddings.
- Cosine similarity and distance features.

Model 2:
- MLP pair classifier trained on semantic features plus lexical features
  supplied by Teammate S.

The module does NOT replace S's candidate generation. It scores exactly the
candidate pairs produced by S.

## License

The default embedding model is:

sentence-transformers/all-MiniLM-L6-v2

Its current Hugging Face model card lists Apache-2.0. Verify the model card
and challenge rules again before the final submission.

No external business information is used. The model is used only to encode
the text already present in the challenge files.

## Data

dataset/train/
  train_source1.tsv
  train_source2.tsv
  train_source3.tsv
  train_ground_truth.tsv

dataset/test/
  test_source1.tsv
  test_source2.tsv
  test_source3.tsv

S artifacts:
  artifacts/train/candidate_pairs.tsv
  artifacts/train/candidate_features.tsv
  output/candidate_pairs.tsv
  output/candidate_features.tsv

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Train

```bash
python src/train.py ^
  --train-dir dataset/train ^
  --candidate-file artifacts/train/candidate_pairs.tsv ^
  --candidate-features artifacts/train/candidate_features.tsv ^
  --output-dir artifacts/m
```

## Test inference

```bash
python src/predict.py ^
  --test-dir dataset/test ^
  --candidate-file output/candidate_pairs.tsv ^
  --candidate-features output/candidate_features.tsv ^
  --model-dir artifacts/m ^
  --output-dir output/m
```

## Outputs

Training:
- semantic_model_metadata.json
- mlp.joblib
- feature_columns.json
- thresholds.json
- validation_scores.tsv

Inference:
- semantic_scores.tsv
- mlp_scores.tsv
- m_predictions.tsv

The team should join `semantic_scores.tsv` and `mlp_scores.tsv` with R and A
scores using `(source1_entity_id, candidate_entity_id)`.

## GPU

If CUDA is available, the Sentence Transformer automatically uses GPU when
requested with `--device cuda`.

If no GPU is available, use `--device cpu`.

For a first run, CPU is acceptable but slower.

## Important

Do not use the embedding model to search external businesses or retrieve
external data. Encode only challenge-provided business names and addresses.
