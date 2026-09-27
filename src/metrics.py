import numpy as np
import pandas as pd

def f05_from_sets(pred, truth):
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0
    if not pred and truth:
        return 0.0
    if pred and not truth:
        return 0.0

    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)

def macro_f05(scored_df, threshold, score_col="score", all_s1=None):
    pred = {}
    for s1, grp in scored_df.groupby("source1_entity_id"):
        pred[s1] = set(
            grp.loc[grp[score_col] >= threshold, "candidate_entity_id"]
        )

    if all_s1 is None:
        all_s1 = sorted(pred)

    truth = getattr(scored_df, "_truth", None)

    # Caller should provide truth through a normal dict using evaluate_macro.
    return pred

def evaluate_macro(scored_df, truth, threshold, score_col="score"):
    grouped = scored_df.groupby("source1_entity_id")
    values = []

    for s1, grp in grouped:
        p = set(
            grp.loc[grp[score_col] >= threshold, "candidate_entity_id"]
        )
        t = truth.get(s1, set())
        values.append(f05_from_sets(p, t))

    # Include S1 entities with no candidates.
    present = set(scored_df["source1_entity_id"])
    for s1 in truth:
        if s1 not in present:
            values.append(f05_from_sets(set(), truth[s1]))

    return float(np.mean(values)) if values else 0.0

def tune_threshold(scored_df, truth, score_col="score",
                   low=0.05, high=0.99, step=0.01):
    best = {"threshold": 0.5, "f05": -1.0}
    t = low
    while t <= high + 1e-9:
        score = evaluate_macro(scored_df, truth, t, score_col)
        if score > best["f05"]:
            best = {"threshold": round(t, 4), "f05": score}
        t += step
    return best
