"""AUROC, ECE, HCE, and bootstrap CI computation."""

import math
import random


def auroc(labels, scores):
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(labels, scores, n_bins=10):
    n = len(labels)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores)
               if s >= lo and (s < hi or (b == n_bins - 1 and s <= hi))]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(scores[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def hce(labels, scores, tau=0.8):
    return sum(1 for l, s in zip(labels, scores) if s >= tau and l == 0)


def bootstrap_ci(labels, initial, final, n_boot=1000, seed=17):
    rng = random.Random(seed)
    n = len(labels)
    deltas_auroc, deltas_ece = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bl = [labels[i] for i in idx]
        bi = [initial[i] for i in idx]
        bf = [final[i] for i in idx]
        ia, fa = auroc(bl, bi), auroc(bl, bf)
        if math.isnan(ia) or math.isnan(fa):
            continue
        deltas_auroc.append(fa - ia)
        deltas_ece.append(ece(bl, bf) - ece(bl, bi))
    deltas_auroc.sort()
    deltas_ece.sort()
    def ci(vals):
        if not vals:
            return (float("nan"), float("nan"))
        lo = int(0.025 * len(vals))
        hi = int(0.975 * len(vals))
        return (vals[lo], vals[hi])
    return {"delta_auroc": ci(deltas_auroc), "delta_ece": ci(deltas_ece)}
