import numpy as np
from sklearn.metrics import cohen_kappa_score

from judge_eval.metrics import agreement, bootstrap_ci, rate_ci, weighted_kappa


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 6, 300)
    b = np.clip(a + rng.integers(-1, 2, 300), 1, 5)
    for w in ["linear", "quadratic"]:
        assert abs(weighted_kappa(a, b, w) - cohen_kappa_score(a, b, weights=w, labels=[1, 2, 3, 4, 5])) < 1e-9


def test_perfect_agreement():
    a = [1, 2, 3, 4, 5, 3]
    s = agreement(a, a)
    assert s["qwk"] == 1 and s["exact"] == 1 and s["mean_diff"] == 0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    a = rng.integers(1, 6, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 1, 5)
    lo, hi = bootstrap_ci(a, b, n=500)
    assert lo <= weighted_kappa(a, b) <= hi


def test_wilson():
    p, lo, hi = rate_ci(45, 50)
    assert lo < 0.9 < hi and hi <= 1
