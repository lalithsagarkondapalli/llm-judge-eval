"""Agreement metrics with bootstrap confidence intervals."""
import numpy as np
from scipy.stats import spearmanr

LABELS = [1, 2, 3, 4, 5]


def weighted_kappa(a, b, weights="quadratic", labels=LABELS):
    """Cohen's kappa with linear or quadratic weights (matches sklearn.cohen_kappa_score)."""
    a, b = np.asarray(a), np.asarray(b)
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    Wt = np.abs(i - j) if weights == "linear" else (i - j) ** 2
    den = (Wt * E).sum()
    return float("nan") if den == 0 else 1 - (Wt * O).sum() / den


def agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    rho = spearmanr(a, b).correlation if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
    return {
        "qwk": weighted_kappa(a, b),
        "exact": float((a == b).mean()),
        "within_1": float((np.abs(a - b) <= 1).mean()),
        "spearman": float(rho),
        "mean_diff": float((a - b).mean()),  # judge - human: >0 means judge is more lenient
    }


def bootstrap_ci(a, b, stat="qwk", n=2000, seed=0, alpha=0.05):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a))
        v = agreement(a[ix], b[ix])[stat]
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def rate_ci(successes, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, c - h, c + h
