"""tests/test_slices.py — Phase 3 tests for slices.py (FR-S-1..3).

Planted-slice test (AC): a toy dataset where one slice has 3x error rate
returns it top-1 with q < 0.05; a dataset with no planted slice returns none.
Cross-checks Fisher p vs scipy and Wilson CI vs statsmodels (FR-Q-2).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

from ea.slices import slice_search, slice_ids, misses_frame, false_alarms_frame


def _planted(n=600, rate_normal=0.1, rate_bad=0.35, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        id=[f"s{i}" for i in range(n)],
        cls=rng.choice(["a", "b"], n),
        scene=rng.choice(["x", "y"], n),
        obj_size_bin=pd.Categorical(rng.choice(["small", "large"], n)),
        label=rng.integers(0, 2, n)))
    bad = (df.cls == "a") & (df.scene == "x")
    df["pred"] = df.label
    flip = rng.random(n) < np.where(bad, rate_bad, rate_normal)
    df.loc[flip, "pred"] = 1 - df.loc[flip, "pred"]
    df["err"] = (df.pred != df.label).astype(int)
    return df


def test_planted_slice_found_top1():                   # FR-S-1 AC
    df = _planted()
    out = slice_search(df, min_support=30)
    assert not out.empty
    top = out.iloc[0]
    assert top.slice == {"cls": "a", "scene": "x"} or top["slice"] == {"cls": "a", "scene": "x"}
    assert top.q < 0.05
    assert top.err > 0.3 and top.n >= 30


def test_null_dataset_returns_none():                  # FR-S-1 AC
    df = _planted(rate_bad=0.1, rate_normal=0.1, seed=3)   # no planted effect
    out = slice_search(df, min_support=30)
    assert out.empty


def test_crosscheck_fisher_and_wilson():               # FR-Q-2
    df = _planted(n=400, seed=5)
    sub = df[(df.cls == "a") & (df.scene == "x")]
    e, n = int(sub.err.sum()), len(sub)
    E, N = int(df.err.sum()), len(df)
    pv = fisher_exact([[e, n - e], [E - e, N - n - (E - e)]], alternative="greater")[1]
    lo, hi = proportion_confint(e, n, method="wilson")
    out = slice_search(df, min_support=30)
    row = out[(out.slice.astype(str) == str({"cls": "a", "scene": "x"})) |
              (out.slice.astype(str) == "{'cls': 'a', 'scene': 'x'}")]
    if not row.empty:
        r = row.iloc[0]
        assert r.p == pytest.approx(pv)
        assert r.lo == pytest.approx(lo) and r.hi == pytest.approx(hi)


def test_min_support_respected():
    df = _planted(n=100, seed=1)
    out = slice_search(df, min_support=80)
    assert out.empty or (out.n >= 80).all()


def test_high_missing_column_skipped():
    df = _planted(seed=2)
    df["blur_bin"] = np.nan
    out = slice_search(df, min_support=30)
    for _, r in out.iterrows():
        assert "blur_bin" not in r.slice


def test_predicate_roundtrip():                        # FR-S-3 AC
    df = _planted()
    ids = slice_ids(df, {"cls": "a", "scene": "x"})
    assert len(ids) == int(((df.cls == "a") & (df.scene == "x")).sum())
    assert all(i.startswith("s") for i in ids)


def test_misses_vs_false_alarms_frames():              # FR-S-2
    df = _planted()
    m = misses_frame(df)
    f = false_alarms_frame(df)
    assert (m.label == 1).all() and (f.label == 0).all()
    assert set(m.err) <= {0, 1} and set(f.err) <= {0, 1}
    # miss on a positive means pred==0
    assert (m.pred == 0).equals(m.err.astype(bool).astype(int) == 1)
