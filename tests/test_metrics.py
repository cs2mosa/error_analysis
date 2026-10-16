"""tests/test_metrics.py — Phase 3 unit tests for metrics.py (FR-B-*).

Design §7.4 checklist + FR-Q-2 independent cross-checks: Wilson/bootstrap/BH
(vs statsmodels), ROC (vs sklearn), ECE vs a naive loop, known-answer scores,
cluster bootstrap resamples sample ids, flip/pair/polarity semantics, AURC
oracle minimum, visual necessity, margin shift sign.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import ea.metrics as M


# ------------------------------------------------------------ FR-Q-2 cross-checks

def test_boot_ci_crosscheck_and_shrink():             # FR-B-0 + FR-Q-2
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        for p in ("d0", "d1"):                        # 2 rows per sample: must resample ids
            rows.append(dict(sample_id=f"s{i}", prompt_id=p, s=rng.normal(0.3, 1),
                             pred=1, gold=1, correct=int(rng.random() < 0.7), arm="base",
                             valid=True, group="D"))
    df = pd.DataFrame(rows)
    lo, hi = M.boot_ci(lambda d: d.correct.mean(), df, n=400, seed=1)
    ids = df.sample_id.unique()
    g = dict(tuple(df.groupby("sample_id")))
    vals = [pd.concat([g[i] for i in rng.choice(ids, len(ids))]).correct.mean() for _ in range(400)]
    nlo, nhi = np.percentile(vals, [2.5, 97.5])
    assert abs(lo - nlo) < 0.05 and abs(hi - nhi) < 0.05   # naive loop cross-check
    df_small = df[df.sample_id.isin(ids[:80])]        # fewer clusters -> wider CI
    lo2, hi2 = M.boot_ci(lambda d: d.correct.mean(), df_small, n=400, seed=1)
    assert (hi - lo) < (hi2 - lo2)
    # width is set by the number of CLUSTERS, not rows: 3x duplicated rows ~ same width
    df_tri = pd.concat([df, df, df], ignore_index=True)
    lo3, hi3 = M.boot_ci(lambda d: d.correct.mean(), df_tri, n=400, seed=1)
    assert abs((hi3 - lo3) - (hi - lo)) < 0.02


def test_wilson_bh_crosscheck():                     # FR-Q-2 (used by slices/escalate)
    for k, n in ((0, 10), (10, 10), (5, 10)):
        lo, hi = proportion_confint(k, n, method="wilson")
        assert 0 <= lo <= hi <= 1
    ps = np.array([0.001, 0.008, 0.02, 0.04, 0.4])
    q = multipletests(ps, method="fdr_bh")[1]
    assert np.all(np.diff(q) >= 0)                   # BH monotone


# ------------------------------------------------------------ operating points (FR-B-2)

def test_roc_metrics_known_answers():
    y = np.array([1, 1, 1, 0, 0, 0])
    perfect = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    assert M.recall_at_fpr(perfect, y, 0.0) == pytest.approx(1.0)
    assert M.fpr_at_recall(perfect, y, 1.0) == pytest.approx(0.0)
    random = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
    f, t, _ = roc_curve(y, random)
    assert 0.4 < np.interp(0.5, f, t) < 0.8          # random scorer near chance
    # cross-check: AUC of perfect == 1 via sklearn on the same inputs
    assert roc_auc_score(y, perfect) == pytest.approx(1.0)


def test_cost_and_threshold():                       # FR-B-3
    y = np.array([1, 1, 0, 0])
    s = np.array([2.0, -1.0, 1.0, -2.0])             # 1 miss, 1 fa
    pred = (s > 0).astype(int)
    assert M.expected_cost(pred, y, 10, 1) == pytest.approx((10 * 0.25 + 1 * 0.25))
    c, t = M.min_cost_threshold(s, y, 10, 1)
    assert c <= M.expected_cost((s > 0).astype(int), y, 10, 1)


# ------------------------------------------------------------ pairs (FR-B-5)

def _pair_df(order):
    # two pairs; order=True -> s_viol > s_safe in both; inverted flips; random ~0.5
    if order == "perfect":
        vals = {("p1", 1): 2.0, ("p1", 0): -2.0, ("p2", 1): 1.5, ("p2", 0): -1.0}
        corr = {("p1", 1): 1, ("p1", 0): 1, ("p2", 1): 1, ("p2", 0): 1}
    elif order == "inverted":
        vals = {("p1", 1): -2.0, ("p1", 0): 2.0, ("p2", 1): -1.5, ("p2", 0): 1.0}
        corr = {("p1", 1): 0, ("p1", 0): 0, ("p2", 1): 0, ("p2", 0): 0}
    else:
        vals = {("p1", 1): 0.3, ("p1", 0): 0.4, ("p2", 1): -0.2, ("p2", 0): 0.1}
        corr = {("p1", 1): 0, ("p1", 0): 1, ("p2", 1): 1, ("p2", 0): 0}
    rows = []
    for (pid, gold), s in vals.items():
        rows.append(dict(sample_id=f"{pid}_{gold}", pair_id=pid, gold=gold, s=s,
                         pred=int(s > 0), correct=corr[(pid, gold)], arm="base", valid=True))
    return pd.DataFrame(rows)


def test_pair_metrics_extremes():
    assert M.pair_acc(_pair_df("perfect")) == pytest.approx(1.0)
    assert M.pair_rank_acc(_pair_df("perfect")) == pytest.approx(1.0)
    assert M.pair_rank_acc(_pair_df("inverted")) == pytest.approx(0.0)
    assert M.pair_acc(_pair_df("inverted")) == pytest.approx(0.0)
    # random ~ 0.5 on rank; pair_acc <= min(item acc) trivially holds here
    r = M.pair_rank_acc(_pair_df("random"))
    assert r in (0.0, 0.5, 1.0)


def test_pair_acc_le_min_item_acc():
    df = _pair_df("random")
    item = df.groupby("gold").correct.mean()
    assert M.pair_acc(df) <= item.min() + 1e-9 or M.pair_acc(df) == item.min()


# ------------------------------------------------------------ consistency (FR-B-6/7/9)

def test_always_yes_polarity_inconsistent():
    # An always-Yes model: pol_pos (pos=No) -> pred=0; pol_neg/pol_miss (pos=Yes) -> pred=1
    rows = []
    for i in range(20):
        for pid, pred in (("pol_pos", 0), ("pol_neg", 1), ("pol_miss", 1)):
            rows.append(dict(sample_id=f"s{i}", prompt_id=pid, s=(1.0 if pred else -1.0),
                             pred=pred, gold=1, correct=0, arm="base", valid=True))
    df = pd.DataFrame(rows)
    yr = M.yes_rate(df, {"pol_pos": False, "pol_neg": True, "pol_miss": True})
    assert (yr.yes_rate == 1.0).all()                 # always answers Yes (affirmation bias)
    assert M.polarity_consistency(df) == pytest.approx(0.0)   # AC: consistency 0
    df2 = df.copy()
    df2["pred"] = 1                                    # asserts the positive class on all prompts
    assert M.polarity_consistency(df2) == pytest.approx(1.0)


def test_flip_rate_and_blank_report():
    rows = []
    for i in range(10):
        preds = [1, 1, 0] if i < 4 else [1, 1, 1]      # 4 samples flip
        for p, pr in zip(("d0", "d1", "d2"), preds):
            rows.append(dict(sample_id=f"s{i}", prompt_id=p, s=float(pr) or -1,
                             pred=pr, gold=1, correct=0, arm="base", valid=True, group="D"))
            rows.append(dict(sample_id=f"s{i}", prompt_id=p, s=-1.0, pred=0, gold=1,
                             correct=0, arm="O3.blank", valid=True, group="D"))
    df = pd.DataFrame(rows)
    assert M.flip_rate(df[(df.arm == "base")]) == pytest.approx(0.4)
    rep = M.flip_report(df)
    assert rep["flip_real"] == pytest.approx(0.4) and rep["flip_blank"] == pytest.approx(0.0)
    with pytest.raises(ValueError):                    # FR-B-7: refuse without blank
        M.flip_report(df[df.arm == "base"])


def test_framing_shift():
    rows = []
    for i in range(10):
        for pid, s in (("frm_neu", 0.0), ("frm_pos", 1.0), ("frm_neg", -0.5)):
            rows.append(dict(sample_id=f"s{i}", prompt_id=pid, s=s, pred=int(s > 0),
                             gold=1, correct=0, arm="base", valid=True))
    assert M.framing_shift(pd.DataFrame(rows)) == pytest.approx(dict(pos=1.0, neg=-0.5))


# ------------------------------------------------------------ necessity + calibration

def test_visual_necessity_identical_is_zero():        # FR-B-8 AC
    assert M.visual_necessity([1.0, -2.0, 0.5], [1.0, -2.0, 0.5]) == pytest.approx(0.0)
    assert M.visual_necessity([2.0, 2.0], [-2.0, -2.0]) > 1.0


def test_ece_brier_calibration():                     # FR-B-12 AC
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.02, 0.98, 6000)               # calibrated by construction
    y = rng.binomial(1, prob)
    assert M.ece(prob, y) < 0.02
    y2 = rng.integers(0, 2, 6000)                      # confident, half wrong
    assert M.ece(np.full(6000, 0.9), y2) > 0.3
    # naive loop cross-check
    prob2 = np.clip(rng.uniform(0, 1, 500), 0.01, 0.99)
    y3 = rng.integers(0, 2, 500)
    idx = np.minimum((prob2 * 15).astype(int), 14)
    naive = sum((idx == b).mean() * abs(prob2[idx == b].mean() - y3[idx == b].mean())
                for b in range(15) if (idx == b).any())
    assert M.ece(prob2, y3) == pytest.approx(naive)
    rel = M.reliability(prob2, y3)
    assert set(rel.columns) == {"bin", "conf", "acc", "n"} and rel.n.sum() == 500
    assert 0 <= M.brier(prob2, y3) <= 1


# ------------------------------------------------------------ selective prediction (FR-B-13)

def test_aurc_perfect_ranking_is_oracle():
    err = np.array([0, 0, 0, 0, 1, 1])
    conf = np.array([5, 4, 3, 2, 1, 0.5])              # errors ranked last
    c, rk = M.risk_coverage(conf, err)
    assert rk[-1] == pytest.approx(2 / 6)
    assert M.aurc(conf, err) == pytest.approx(np.mean([0, 0, 0, 0, 1 / 5, 2 / 6]))
    assert M.coverage_at_risk(conf, err, 0.0) == pytest.approx(4 / 6)  # risk-free prefix
    assert M.coverage_at_risk(conf, err, 1 / 3) == pytest.approx(1.0)  # full coverage holds


def test_aurc_worse_when_errors_ranked_high():
    conf = np.array([5, 4, 1, 0.5])
    bad = np.array([1, 1, 0, 0])                       # errors ranked FIRST
    good = np.array([0, 0, 1, 1])                      # errors ranked last
    assert M.aurc(conf, bad) == pytest.approx(np.mean([1, 1, 2 / 3, 2 / 4]))
    assert M.aurc(conf, bad) > M.aurc(conf, good)


def test_coverage_at_recall_all_prefixes():
    # accepted-set recall by prefix (conf order idx0, idx2, idx1, idx3): 1, 1, 1, 1
    conf = np.array([.9, .2, .8, .1])
    pred = np.array([1, 1, 0, 0])                      # positives: idx0, idx1
    y = np.array([1, 1, 0, 0])
    assert M.coverage_at_recall(conf, pred, y, 1.0) == pytest.approx(1.0)
    # idx1 (an accepted positive) predicted negative: recall 1, 1, .5, .5 -> only k<=2 meet 1.0
    missed = np.array([1, 0, 0, 0])
    assert M.coverage_at_recall(conf, missed, y, 1.0) == pytest.approx(0.5)
    assert M.coverage_at_recall(conf, missed, y, 0.5) == pytest.approx(1.0)
    assert M.coverage_at_recall(conf, np.zeros(4), y, 0.5) == 0.0   # no positive prediction at all


# ------------------------------------------------------------ recovery / margin shift (FR-B-15/18)

def _runs_arm(arm, svals, golds):
    """Arm rows + base rows. Base is confidently WRONG (|s| = 2, pred != gold):
    these samples are errors by definition (§2 'Error'), as recovery tests need."""
    rows = []
    for i, (s, g) in enumerate(zip(svals, golds)):
        rows.append(dict(sample_id=f"s{i}", prompt_id="d0", group="D", arm=arm, knob=None,
                         value=None, s=float(s), pred=int(s > 0), gold=int(g),
                         correct=int((s > 0) == g), valid=True, n_vis_tokens=10,
                         latency_ms=1.0, t=None, extra="{}"))
        sb = 2 - 4 * int(g)                            # gold=1 -> s_base=-2; gold=0 -> +2
        rows.append(dict(sample_id=f"s{i}", prompt_id="d0", group="D", arm="base", knob=None,
                         value=None, s=float(sb), pred=int(sb > 0), gold=int(g),
                         correct=int((sb > 0) == g), valid=True, n_vis_tokens=10,
                         latency_ms=1.0, t=None, extra="{}"))
    return pd.DataFrame(rows)


def test_arm_recovery_and_breakage():
    # 10 confident errors (|s_base|=2); the arm fixes 3 of them
    df = _runs_arm("O1.gtcrop", [-3, -3, -3, -3, -3, -3, 3, -3, 3, 3], [1] * 10)
    ids = [f"s{i}" for i in range(10)]
    k, n = M.arm_recovery(df, ids, "O1.gtcrop")
    assert (k, n) == (3, 10)
    df2 = _runs_arm("O1.gtcrop", [-3] * 4 + [3] * 6, [1] * 10)
    k2, n2 = M.arm_recovery(df2, ids, "O1.gtcrop")
    assert (k2, n2) == (6, 10)
    # matched corrects (gold=0) break when the arm flips them to violation
    df3 = _runs_arm("O1.gtcrop", [3] * 4 + [-3] * 6, [0] * 10)
    bk, bn = M.arm_breakage(df3, ids, "O1.gtcrop")
    assert (bk, bn) == (4, 10)


def test_margin_shift_signed_toward_gold():            # FR-B-18
    # gold=1: dshift = s_arm - s_base; gold=0: s_base - s_arm
    df = _runs_arm("O1.gtcrop", [-1.0, 1.0], [1, 0])   # base: gold1 s=-2, gold0 s=+2
    ids = ["s0", "s1"]
    mean, (lo, hi), n = M.margin_shift(df, ids, "O1.gtcrop")
    # s0: (-1) - (-2) = +1 toward gold; s1: (+2) - (+1) = +1 toward gold
    assert mean == pytest.approx(1.0)
    assert n == 2 and lo <= mean <= hi


def test_invalid_rows_excluded():                      # FR-R-7 / C16
    df = _runs_arm("O1.gtcrop", [3.0, 3.0], [1, 1])
    df.loc[df.arm == "O1.gtcrop", "valid"] = [True, False]
    k, n = M.arm_recovery(df, ["s0", "s1"], "O1.gtcrop")
    assert (k, n) == (1, 1)


# ------------------------------------------------------------ curves (FR-B-16/17)

def test_quality_curve_and_temporal_gains():
    df = pd.DataFrame([dict(sample_id=f"s{i}", kind="blur", sev=sv, correct=int(i % 2),
                            s=0.1 * sv, arm="O6.degrade", valid=True, pred=1, gold=1)
                       for i, sv in zip(range(12), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2])])
    q = M.quality_curve(df)
    assert set(q.columns) == {"kind", "sev", "acc", "mean_abs_s", "n"}
    tg = M.temporal_gains(dict(full=0.9, single=0.7, shuffle=0.8, reverse=0.6))
    assert tg == pytest.approx(dict(multi_frame=0.2, order=0.1, reverse=0.3))


def test_by_knob_index():                              # FR-B-14
    df = pd.DataFrame([dict(sample_id=f"s{i}", knob="quant", value=v, s=1.0, pred=1,
                            gold=1, correct=1, valid=True)
                       for i, v in zip(range(6), [None, None, "8bit", "8bit", "4bit", "4bit"])])
    df.loc[df.knob.isna(), ["knob", "value"]] = None
    out = M.by_knob(df, dict(acc=lambda g: g.correct.mean()))
    assert list(out.index.names) == ["knob", "value"] and len(out) == 2
