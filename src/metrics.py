import numpy as np

def f05(pred, truth):
    pred, truth = set(pred), set(truth)

    if not pred and not truth:
        return 1.0
    if not pred:
        return 0.0
    if not truth:
        return 0.0

    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)

    if precision == 0 or recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate(scored, truth, threshold, score_col="score"):
    vals = []

    grouped = scored.groupby("source1_entity_id")

    for s1, g in grouped:
        pred = set(
            g.loc[g[score_col] >= threshold, "candidate_entity_id"]
        )
        vals.append(f05(pred, truth.get(s1, set())))

    present = set(scored["source1_entity_id"])
    for s1, true_ids in truth.items():
        if s1 not in present:
            vals.append(f05(set(), true_ids))

    return float(np.mean(vals)) if vals else 0.0

def tune(scored, truth, start=0.05, stop=0.99, step=0.01):
    best = {"threshold": 0.5, "f05": -1}
    t = start

    while t <= stop + 1e-9:
        score = evaluate(scored, truth, t)
        if score > best["f05"]:
            best = {"threshold": round(t, 4), "f05": score}
        t += step

    return best
