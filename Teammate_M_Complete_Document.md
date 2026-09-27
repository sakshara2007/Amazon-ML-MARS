# Teammate M — Complete Implementation Guide
## Semantic Matching + MLP for Business Entity Resolution

## 1. Role of Teammate M

M owns two complementary models:

### Model 1 — Sentence Transformer / Bi-Encoder

The model converts:

- business name
- business address
- combined name + address

into dense semantic embeddings.

For every S1-candidate pair, M calculates:

- name cosine similarity
- address cosine similarity
- combined cosine similarity
- corresponding distance features
- name/address semantic disagreement

### Model 2 — MLP pair classifier

The MLP consumes:

- semantic similarity features
- S's lexical retrieval features
- country agreement
- S2/S3 source indicator

and predicts:

```text
P(candidate is the same business)
```

M's scores are then handed to the team-level ensemble with R and A.

---

# 2. Where M fits

```text
             Source 1
                 |
                 v
        S candidate generation
                 |
                 v
        candidate_pairs.tsv
                 |
                 v
        ┌───────────────────┐
        │ M semantic branch │
        └─────────┬─────────┘
                  |
        ┌─────────┴──────────┐
        v                    v
 Sentence Transformer       MLP
        |                    |
        v                    v
 semantic features       M score
        |                    |
        └──────────┬─────────┘
                   v
             Team ensemble
              R + M + A
```

The M module does not perform external business lookup.

It only encodes text contained in the challenge's supplied records.

---

# 3. Model selected

The default model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The current Hugging Face model page lists this model as Apache-2.0 licensed,
which is compatible with the challenge's stated requirement that the final
model be MIT/Apache 2.0 licensed. The team should preserve the model-card
license evidence with the final submission and re-check the challenge rules
before submission.

---

# 4. Why use a bi-encoder?

Traditional lexical similarity is strong for:

```text
ABC Technologies Pvt Ltd
ABC Technologies Private Limited
```

but semantic embeddings can provide additional evidence when wording is
different.

M should therefore NOT replace S or R.

Instead:

```text
lexical evidence
+
semantic evidence
```

is the objective.

---

# 5. Three embedding representations

For every business create three texts.

## 5.1 Name

```text
ABC Technologies Pvt Ltd
```

## 5.2 Address

```text
12 Gandhi Road Chennai 600001
```

## 5.3 Combined

```text
ABC Technologies Pvt Ltd [SEP] 12 Gandhi Road Chennai 600001
```

Generate an embedding for each.

---

# 6. Normalize embeddings

The Sentence Transformer is called with:

```python
normalize_embeddings=True
```

Therefore:

```text
dot product ≈ cosine similarity
```

This makes pairwise comparison efficient.

---

# 7. Candidate restriction

M MUST NOT encode all possible S1 × S2/S3 pairs.

S already produces the candidate set.

M only processes:

```text
S1 → S2/S3 candidates
```

This is essential for scalability.

---

# 8. Efficient embedding strategy

Do not encode the same business text repeatedly.

The implementation first creates a list of unique:

```text
names
addresses
combined texts
```

Then encodes each unique text once.

For example:

```text
S1-A name → ABC Technologies
S2-B name → ABC Technologies
S3-C name → ABC Technologies
```

Only one unique embedding is required for the identical text.

---

# 9. GPU configuration

If a CUDA GPU is available:

```bash
python src/train.py \
  --device cuda
```

If no GPU:

```bash
python src/train.py \
  --device cpu
```

The MLP itself is small. The main GPU benefit comes from Sentence Transformer
embedding generation.

---

# 10. Semantic features

For every pair:

```text
S1 ↔ candidate
```

calculate:

```text
name_cosine
address_cosine
combined_cosine

name_distance
address_distance
combined_distance

name_address_cosine_gap
```

Example:

```text
name_cosine = 0.94
address_cosine = 0.82
combined_cosine = 0.91
```

This indicates strong overall similarity.

---

# 11. Semantic disagreement feature

This is useful:

```text
abs(name_cosine - address_cosine)
```

Example A:

```text
name = 0.95
address = 0.91
gap = 0.04
```

Evidence is consistent.

Example B:

```text
name = 0.96
address = 0.22
gap = 0.74
```

This is suspicious.

The MLP can learn this interaction.

---

# 12. Add S's lexical features

M should consume S's:

```text
exact_name
token_name_hits
token_address_hits
tfidf_name
tfidf_address
tfidf_combined
bm25_name
bm25_address
```

This creates a hybrid semantic + lexical classifier.

M is therefore not duplicating R exactly; M gives the team another model whose
representation includes dense semantic evidence.

---

# 13. Add metadata features

M also uses:

```text
country_match
source_is_s3
```

Country is treated as an open string.

Do not create a fixed:

```text
US / India
```

category list because the test data can contain France.

---

# 14. MLP architecture

The implementation uses:

```text
Input
  |
  v
Dense 128
  |
 ReLU
  |
Dense 64
  |
 ReLU
  |
Dense 32
  |
 ReLU
  |
Sigmoid
  |
P(match)
```

Implemented with:

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    ...
)
```

Before the MLP, numerical features are standardized with:

```text
StandardScaler
```

---

# 15. Why an MLP?

R uses gradient-boosted trees.

M's MLP provides a different function class.

For example, it can learn a nonlinear semantic relationship such as:

```text
high name similarity
+
moderate address similarity
+
high combined similarity
+
same country
```

without requiring explicit hand-written rules.

---

# 16. Training labels

Use the same ground truth as R.

For each candidate pair:

```text
label = 1
```

if candidate ID appears in the Source-1 entity's ground-truth match list.

Otherwise:

```text
label = 0
```

Importantly, negatives come from S's candidate set.

These are difficult negatives rather than arbitrary unrelated records.

---

# 17. Validation split

Use a Source-1-level split.

Correct:

```text
80% Source-1 entities → training
20% Source-1 entities → validation
```

Do not randomly split candidate pairs.

Otherwise candidates belonging to the same Source-1 entity can leak into both
training and validation.

---

# 18. Class imbalance

Entity resolution usually has many more negative candidate pairs than
positive pairs.

The implementation computes:

```text
negative / positive
```

and uses that to give positive examples higher sample weight.

This prevents the MLP from learning the trivial rule:

```text
everything = non-match
```

---

# 19. F_0.5 threshold optimization

Do not use:

```python
score >= 0.5
```

automatically.

The competition evaluates macro F_0.5 per Source-1 entity.

Therefore the implementation searches thresholds from:

```text
0.05 → 0.99
```

and selects the validation threshold with the highest macro F_0.5.

---

# 20. Singleton handling

M must allow:

```text
no candidate selected
```

for an S1 entity.

It must never force the top semantic candidate to be a match.

This is especially important because the challenge explicitly includes
singletons in its macro F_0.5 evaluation.

---

# 21. Training command

Windows:

```bat
python src/train.py ^
  --train-dir dataset/train ^
  --candidate-file artifacts/train/candidate_pairs.tsv ^
  --candidate-features artifacts/train/candidate_features.tsv ^
  --output-dir artifacts/m ^
  --device cuda
```

CPU:

```bat
python src/train.py ^
  --train-dir dataset/train ^
  --candidate-file artifacts/train/candidate_pairs.tsv ^
  --candidate-features artifacts/train/candidate_features.tsv ^
  --output-dir artifacts/m ^
  --device cpu
```

Linux/macOS:

```bash
python src/train.py \
  --train-dir dataset/train \
  --candidate-file artifacts/train/candidate_pairs.tsv \
  --candidate-features artifacts/train/candidate_features.tsv \
  --output-dir artifacts/m \
  --device cuda
```

---

# 22. Training outputs

The training stage creates:

```text
artifacts/m/
├── mlp.joblib
├── feature_columns.json
├── metadata.json
└── validation_scores.tsv
```

`metadata.json` records:

- embedding model
- feature list
- train/validation sizes
- positive pair counts
- optimized threshold
- MLP iteration count

---

# 23. Test inference

Run:

```bat
python src/predict.py ^
  --test-dir dataset/test ^
  --candidate-file output/candidate_pairs.tsv ^
  --candidate-features output/candidate_features.tsv ^
  --model-dir artifacts/m ^
  --output-dir output/m ^
  --device cuda
```

CPU:

```bat
python src/predict.py ^
  --test-dir dataset/test ^
  --candidate-file output/candidate_pairs.tsv ^
  --candidate-features output/candidate_features.tsv ^
  --model-dir artifacts/m ^
  --output-dir output/m ^
  --device cpu
```

---

# 24. M's important output

The most important file is:

```text
output/m/mlp_scores.tsv
```

It contains:

```text
source1_entity_id
candidate_entity_id
mlp_score
```

The team should merge this with R's:

```text
lightgbm_score
xgboost_score
```

and A's:

```text
cross_encoder_score
catboost_score
```

---

# 25. M should not own the final ensemble

M produces semantic evidence.

The final team-level decision should be made after all four members' outputs
are available.

Recommended combined table:

```text
source1_entity_id
candidate_entity_id

tfidf_name
tfidf_address
bm25_name

lightgbm_score
xgboost_score

name_cosine
address_cosine
combined_cosine
mlp_score

cross_encoder_score
catboost_score
```

Then optimize the final combination on the common validation set.

---

# 26. Recommended M experiments

Run these in order.

### M1 — semantic only

```text
name cosine
address cosine
combined cosine
```

MLP.

### M2 — semantic + S lexical features

```text
M1
+
TF-IDF
+
BM25
+
exact name
```

This should be the main M model.

### M3 — different text construction

Compare:

```text
name
address
name [SEP] address
```

against:

```text
name + address
```

Keep whichever improves validation F_0.5.

### M4 — embedding dimension/model experiment

Only test alternative pretrained models if their licenses satisfy the
competition.

Do not automatically assume a larger transformer is better.

---

# 27. What M should report

M should give the team:

```text
Embedding model:
Device:
Embedding dimension:

Number of candidate pairs:

MLP:
Validation F0.5:
Best threshold:

Semantic-only performance:
Semantic + lexical performance:

False-positive examples:
False-negative examples:
```

The last two are useful because they help R and A identify complementary
errors.

---

# 28. Error analysis

M should inspect at least:

### False positives

```text
high semantic score
but ground truth = non-match
```

Look for:

- same business name, different address
- generic business names
- chain branches
- common words
- address mismatch

### False negatives

```text
low semantic score
but ground truth = match
```

Look for:

- severe abbreviation
- transliteration
- landmark address
- very short names
- partial address

These examples can motivate feature changes.

---

# 29. Important distinction from R

R's strength:

```text
structured lexical features
+
gradient boosting
```

M's strength:

```text
dense semantic representations
+
neural nonlinear classifier
```

Therefore do not simply copy R's entire feature pipeline and call it M.

M should provide genuinely different evidence.

---

# 30. Team integration

The final team integration should look like:

```text
                    S
           Candidate generation
                    |
                    v
              Candidate pairs
                    |
       ┌────────────┼────────────┐
       v            v            v
       R            M            A
       |            |            |
   LightGBM       MLP       Cross Encoder
   XGBoost      Semantic      CatBoost
       |            |            |
       └────────────┼────────────┘
                    v
              Score table
                    |
                    v
           Ensemble optimizer
                    |
                    v
            F_0.5 threshold
                    |
                    v
         matching_results.tsv
```

---

# 31. Do not use external information

M must not:

- Google the business
- search business registries
- query Google Maps
- use geocoding
- call commercial entity-resolution APIs
- retrieve external business information

The embedding model is only applied to challenge-provided text.

---

# 32. Recommended final M model

My recommended M configuration is:

```text
Sentence Transformer:
all-MiniLM-L6-v2

Representations:
1. business name
2. business address
3. name + address

Semantic features:
1. name cosine
2. address cosine
3. combined cosine
4. three distances
5. name/address cosine gap

Additional features:
S's TF-IDF/BM25 features
country match
source indicator

MLP:
128 → 64 → 32 → 1
ReLU
Adam
StandardScaler
early stopping
class-weighted samples
```

---

# 33. M's final deliverables

```text
teammate_M/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── data.py
    ├── text.py
    ├── embeddings.py
    ├── features.py
    ├── metrics.py
    ├── train.py
    ├── predict.py
    └── cache_embeddings.py
```

Model artifacts:

```text
mlp.joblib
feature_columns.json
metadata.json
validation_scores.tsv
mlp_scores.tsv
```

The team should preserve the pretrained model identifier and license evidence
alongside the submission.
