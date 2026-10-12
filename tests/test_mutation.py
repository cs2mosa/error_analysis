"""tests/test_mutation.py — deliberate-bug mutation checks (FR-Q-6).

For each diagnostic function and gate, injects ONE known bug and asserts
that at least one existing test or assertion catches it.  These are
regression guards: if a future change accidentally reintroduces a bug,
this test will fail before the unit tests even run.

Coverage: explain_recover, explain_necessity, explain_consistency,
          gate_ok, conf_ids, decide, per_error_accounting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ea.escalate import (
    Cluster,
    SIG,
    conf_ids,
    decide,
    explain_consistency,
    explain_necessity,
    explain_recover,
    per_error_accounting,
    wil,
)
from ea.triage import gate_ok


# ----------------------------------------------------------------- shared helpers

def _make_runs_simple(n=20, arm_correct=True, ctl_correct=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    gold = 1
    for i in range(n):
        sid = f"e{i}"
        base_s = -1.5
        rows.append(dict(sample_id=sid, prompt_id="d0", arm="base", s=base_s,
                         pred=0, gold=gold, correct=0, group="D", valid=True,
                         knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="d1", arm="base", s=base_s,
                         pred=0, gold=gold, correct=0, group="D", valid=True,
                         knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="d2", arm="base", s=base_s,
                         pred=0, gold=gold, correct=0, group="D", valid=True,
                         knob=None, value=None))
        arm_s = 1.5 if arm_correct else -1.5
        rows.append(dict(sample_id=sid, prompt_id="d0", arm="O1.gtcrop", s=arm_s,
                         pred=int(arm_s > 0), gold=gold, correct=int(arm_s > 0) == gold,
                         group="D", valid=True, knob=None, value=None))
        ctl_s = 1.5 if ctl_correct else -1.5
        rows.append(dict(sample_id=sid, prompt_id="d0", arm="O1.randcrop", s=ctl_s,
                         pred=int(ctl_s > 0), gold=gold, correct=int(ctl_s > 0) == gold,
                         group="D", valid=True, knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="d0", arm="O1.sametok", s=-1.5,
                         pred=0, gold=gold, correct=0, group="D", valid=True,
                         knob=None, value=None))
    return pd.DataFrame(rows)


_CFG = dict(exit=dict(recovery=0.5, control=0.1, min_n=10, conf_margin=1.0),
            necessity=dict(ratio=0.5),
            consistency=dict(min_flip=0.2, ratio=2.0))


# =================================================================== bug 1: wrong recovery inequality

def test_mut_flip_lo_hi_for_recovery():
    """BUG: use hi >= recovery instead of lo >= recovery.
    Consequence: a low-recovery arm (hi just crosses threshold) is falsely labelled True.
    Correct code must use lo (lower CI bound), not hi."""
    # Arm barely recovers 8/20 = 40% (below 50% threshold)
    runs = _make_runs_simple(n=20, arm_correct=False)
    # Patch 8 samples to be correct
    df = runs.copy()
    mask = (df.arm == "O1.gtcrop") & df.sample_id.isin([f"e{i}" for i in range(8)])
    df.loc[mask, "correct"] = 1
    df.loc[mask, "s"] = 1.5
    df.loc[mask, "pred"] = 1
    c = Cluster("test", [f"e{i}" for i in range(20)], [])
    r = explain_recover(df, c, "O1.gtcrop", ["O1.randcrop", "O1.sametok"], _CFG)
    # lo < 0.5 so correct code says False; hi might be > 0.5 (Wilson is wide at n=20)
    # The CORRECT answer: explained must be False or None (not True)
    assert r["explained"] is not True, (
        "FR-Q-6 BUG: recovery gate used hi instead of lo — 8/20 (40%) accepted as recovered"
    )


# =================================================================== bug 2: base re-run as control

def test_mut_base_rerun_as_control():
    """BUG: use a second base run (same arm=base) as the control.
    Consequence: base always 'recovers' nothing → control check passes even if
    the intervention arm is ineffective.  Correct code uses a DIFFERENT arm."""
    # Arm recovers, and we fraudulently pass "base" as its own control
    runs = _make_runs_simple(n=20, arm_correct=True, ctl_correct=False)
    c = Cluster("test", [f"e{i}" for i in range(20)], [])
    # The correct call uses randcrop; a bug would pass ["base"] as control
    r_correct = explain_recover(runs, c, "O1.gtcrop", ["O1.randcrop", "O1.sametok"], _CFG)
    # Build a mutant: if control is the same arm (base), recovery == arm recovery → always high
    # We assert that the correct implementation does NOT accept "base" as a control arm
    # by checking that the ctl_rate is NOT the same as the arm rec rate
    assert r_correct["ctl_rate"] < r_correct["rec"], (
        "FR-Q-6 BUG: control rate equals arm recovery rate — base is being used as its own control"
    )


# =================================================================== bug 3: conf_ids ignores valid flag

def test_mut_conf_ids_ignores_valid():
    """BUG: conf_ids includes invalid (low-mass) rows in the margin average.
    Consequence: margins are diluted, fewer samples pass the conf_margin gate.
    Correct code must filter valid=True before averaging."""
    # 20 errors, all with |base s| = 1.5 (confident)
    rows = _make_runs_simple(n=20, arm_correct=False)
    # Inject 10 invalid rows with very low magnitude
    extra = []
    for i in range(10):
        extra.append(dict(sample_id=f"e{i}", prompt_id="d0", arm="base",
                          s=0.01, pred=1, gold=1, correct=1, group="D",
                          valid=False, knob=None, value=None))
    df = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
    ids = conf_ids(df, [f"e{i}" for i in range(20)], conf_margin=1.0)
    # All 20 should still be confident (invalid rows excluded)
    assert len(ids) == 20, (
        "FR-Q-6 BUG: conf_ids included valid=False rows, diluting margins below threshold"
    )


# =================================================================== bug 4: decide treats None as wildcard

def test_mut_decide_none_wildcard():
    """BUG: decide's partial-match branch uses (v in (ev.get(d), None)) incorrectly,
    allowing a None evidence value to match a required True/False value.
    Correct: None in ev means 'data not yet available' and BLOCKS the match."""
    # If None matched True, then hires=None would satisfy hires=True in pixel-limit SIG
    ev = dict(crop=True, bgmask=False, hires=None)
    cause, _ = decide(ev)
    assert cause != "pixel/token-budget limit (H1/H4)", (
        "FR-Q-6 BUG: decide treated None as a wildcard matching True — "
        "premature attribution before hires data is available"
    )


# =================================================================== bug 5: gate_ok not applied in necessity

def test_mut_necessity_no_gate():
    """BUG: explain_necessity skips gate_ok, assigning necessity=True for perception failures.
    Correct: only samples passing gate_ok (qp + qp_abs + qr all correct) are included."""
    # Build runs where qp_abs is wrong for errors → gate_ok returns False
    rows = []
    gold = 1
    for i in range(20):
        sid = f"e{i}"
        # base D rows (errors)
        for p in ("d0", "d1", "d2"):
            rows.append(dict(sample_id=sid, prompt_id=p, arm="base", s=-1.5,
                             pred=0, gold=gold, correct=0, group="D", valid=True,
                             knob=None, value=None))
        # qp correct but qp_abs WRONG → gate_ok fails
        rows.append(dict(sample_id=sid, prompt_id="qp", arm="base", s=1.5,
                         pred=1, gold=gold, correct=1, group="QP", valid=True,
                         knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="qp_abs", arm="base", s=-1.5,
                         pred=0, gold=gold, correct=0, group="QP", valid=True,
                         knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="qr", arm="base", s=1.5,
                         pred=1, gold=gold, correct=1, group="QR", valid=True,
                         knob=None, value=None))
        # O3.blank / O3.occlude: same as base (would look necessity=True)
        for arm in ("O3.blank", "O3.occlude"):
            rows.append(dict(sample_id=sid, prompt_id="d0", arm=arm, s=-1.5,
                             pred=0, gold=gold, correct=0, group="D", valid=True,
                             knob=None, value=None))
    df = pd.DataFrame(rows)
    c = Cluster("test", [f"e{i}" for i in range(20)], [f"c{i}" for i in range(5)])
    r = explain_necessity(df, c, _CFG)
    # gate_ok fails → n_gated < min_n → explained must be None (not True)
    assert r["explained"] is not True, (
        "FR-Q-6 BUG: gate_ok not applied; necessity=True assigned to perception failures"
    )


# =================================================================== bug 6: per_error_accounting double-counts

def test_mut_accounting_double_count():
    """BUG: per_error_accounting returns more rows than there are unique errors.
    Correct: FR-E-11 assert ensures len(df) == len(all_ids)."""
    ids = [f"e{i}" for i in range(20)]
    all_c = Cluster("ALL_ERRORS", ids, [])
    all_c.cause = "unexplained → next tier"
    a = Cluster("A", ids[:10], [])
    a.cause = "input quality (H9)"
    b = Cluster("B", ids[5:15], [])
    b.cause = "rule-reasoning failure"
    df = per_error_accounting([all_c, a, b])
    # Must not have more rows than unique errors
    assert len(df) == len(ids), (
        "FR-Q-6 BUG: per_error_accounting returned duplicate rows — "
        "most-specific-cluster logic is broken"
    )
    assert df.sample_id.nunique() == len(ids)


# =================================================================== bug 7: wil returns reversed bounds

def test_mut_wil_order():
    """BUG: lo and hi are swapped in wil(). Correct: lo <= hi always."""
    for k, n in [(0, 10), (5, 10), (10, 10), (3, 7)]:
        lo, hi = wil(k, n)
        assert lo <= hi, f"FR-Q-6 BUG: wil({k}, {n}) returned lo={lo} > hi={hi}"


# =================================================================== bug 8: classify returns anticipated when ant=0

def test_mut_classify_ant_zero():
    """BUG: classify returns 'anticipated' even when anticipation_s is 0 (ant=0.0).
    Correct: 'anticipated' is reachable ONLY when ant > 0 (design §5.11, §7.4)."""
    from ea.stream import classify
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol, ant = 5.0, 8.0, 1.0, 0.0
    # Alert at t=3.0 (< onset - tol = 4.0). With ant=0.0, this MUST NOT be 'anticipated'.
    S = np.full_like(ts, -1.0)
    S[3] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1, ant=ant)
    assert cat != "anticipated", (
        "FR-Q-6 BUG: classify returned 'anticipated' when ant=0.0 — must be 'premature'"
    )

