"""tests/test_stream.py — tests for streaming track (FR-ST-1..8).

Coverage:
  - FR-ST-1: causal online step planning (plan_stream).
  - FR-ST-2: k-of-n persistence offline trigger alerts.
  - FR-ST-3: timing metrics (onset error, warning latency, asymmetric score, spam rate).
  - FR-ST-4: decomposition (ok, anticipated, silent, premature, policy_late, perception_late, spam).
  - FR-ST-5: window-level vs stream-level metric gap (H11 test).
  - FR-ST-6: persistence/recovery after degraded frame.
  - FR-ST-7: latency profile (p50/p95 forward time).
  - FR-ST-8: protocol disclosure block header.
  - Causality test (perturbing future frames leaves earlier steps unchanged).
"""
import numpy as np
import pandas as pd
import pytest

from ea.data import Sample
from ea.run import plan_stream
from ea.stream import (
    alerts,
    asymmetric_score,
    classify,
    evaluate_stream,
    onset_error,
    protocol_header,
    recovery_steps,
    spam_rate,
    stream_latency_profile,
    warning_latency,
    window_vs_stream_gap,
)


# ----------------------------------------------------------------- FR-ST-2: alerts (k-of-n)

def test_alerts_k_of_n():
    """Verify k-of-n persistence and rising-edge detection."""
    # k=2, n=3: requires 2 positives in a 3-window
    S = [-1.0, -1.0, 1.5, 1.5, -1.0, -1.0, 1.5, 1.5, 1.5]
    # Hits at indices: 2, 3, 6, 7, 8
    # Window sums (n=3):
    # i=0: [0] -> 0
    # i=1: [0, 0] -> 0
    # i=2: [0, 0, 1] -> 1
    # i=3: [0, 1, 1] -> 2 >= 2 -> ON (rising edge!)
    # i=4: [1, 1, 0] -> 2 >= 2 -> ON (sustained, not rising edge)
    # i=5: [1, 0, 0] -> 1 -> OFF
    # i=6: [0, 0, 1] -> 1 -> OFF
    # i=7: [0, 1, 1] -> 2 >= 2 -> ON (rising edge!)
    # i=8: [1, 1, 1] -> 3 >= 2 -> ON (sustained)
    al = alerts(S, tau=0.0, k=2, n=3)
    assert al[3] == True
    assert al[4] == False  # sustained, not rising edge
    assert al[7] == True
    assert al[8] == False
    assert al.sum() == 2


def test_alerts_empty():
    assert len(alerts([], tau=0.0, k=2, n=3)) == 0


# ----------------------------------------------------------------- FR-ST-4: decomposition

def test_classify_ok():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol = 5.0, 8.0, 1.0
    # Alert at t=5.0 (within [onset - tol, onset + tol])
    S = np.full_like(ts, -1.0)
    S[5] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1)
    assert cat == "ok"


def test_classify_anticipated():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol, ant = 5.0, 8.0, 1.0, 2.0
    # Valid early window: [onset - tol - ant, onset - tol) = [2.0, 4.0)
    S = np.full_like(ts, -1.0)
    S[3] = 1.0  # t=3.0 is inside [2.0, 4.0)
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1, ant=ant)
    assert cat == "anticipated"


def test_classify_anticipated_disabled_when_ant_zero():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol, ant = 5.0, 8.0, 1.0, 0.0
    # Alert at t=3.0 (< onset - tol = 4.0). With ant=0.0, this is premature!
    S = np.full_like(ts, -1.0)
    S[3] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1, ant=ant)
    assert cat == "premature"


def test_classify_silent():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol = 5.0, 8.0, 1.0
    S = np.full_like(ts, -1.0)  # no alerts ever
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1)
    assert cat == "silent"


def test_classify_premature():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol = 8.0, 10.0, 1.0
    # Alert at t=2.0 (way early), none in [7.0, 10.0]
    S = np.full_like(ts, -1.0)
    S[2] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1, ant=0.0)
    assert cat == "premature"


def test_classify_policy_late():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol = 4.0, 10.0, 1.0
    # Model detects positive at onset (t=4, 5, 6), but k=3 persistence delays alert to t=6
    # first alert = 6.0 > onset + tol = 5.0
    # win_pos.min() = 4.0 < first - tol = 5.0 => policy_late
    S = np.full_like(ts, -1.0)
    S[4:10] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=3, n=3)
    assert cat == "policy_late"


def test_classify_perception_late():
    ts = np.arange(0.0, 11.0, 1.0)
    onset, end, tol = 4.0, 10.0, 1.0
    # Model stays negative until t=8.0 (perception is late)
    S = np.full_like(ts, -1.0)
    S[8:] = 1.0
    cat = classify(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1)
    assert cat == "perception_late"


def test_classify_spam():
    ts = np.arange(0.0, 11.0, 1.0)
    onset = None  # safe / no-risk clip
    S = np.full_like(ts, -1.0)
    S[4] = 1.0  # false alarm alert
    cat = classify(ts, S, tau=0.0, onset=onset, end=None, tol=1.0, k=1, n=1)
    assert cat == "spam"


def test_classify_no_risk_ok():
    ts = np.arange(0.0, 11.0, 1.0)
    onset = None
    S = np.full_like(ts, -1.0)
    cat = classify(ts, S, tau=0.0, onset=onset, end=None, tol=1.0, k=1, n=1)
    assert cat == "ok"


# ----------------------------------------------------------------- FR-ST-1: causality

def test_causality():
    """Online evaluation at step T is causal: future steps cannot alter past alerts."""
    ts_short = np.arange(0.0, 6.0, 1.0)
    S_short = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    al_short = alerts(S_short, tau=0.0, k=2, n=3)

    # Extend stream with future steps
    ts_long = np.arange(0.0, 12.0, 1.0)
    S_long = np.concatenate([S_short, np.array([2.0, 2.0, -2.0, -2.0, 3.0, 3.0])])
    al_long = alerts(S_long, tau=0.0, k=2, n=3)

    # First len(S_short) steps must be identical
    np.testing.assert_array_equal(al_short, al_long[:len(S_short)])


# ----------------------------------------------------------------- FR-ST-3: metrics

def test_timing_metrics():
    ts = np.arange(0.0, 10.0, 1.0)
    onset = 4.0
    S = np.full_like(ts, -1.0)
    S[6] = 2.0  # alert at t=6.0

    # onset error = 6.0 - 4.0 = 2.0
    assert onset_error(ts, S, tau=0.0, onset=onset, k=1, n=1) == pytest.approx(2.0)
    # warning latency = 6.0 - 4.0 = 2.0
    assert warning_latency(ts, S, tau=0.0, onset=onset, k=1, n=1) == pytest.approx(2.0)

    # asymmetric score for late alert
    # on-time/late: exp(-(6.0 - 4.0) / 5.0) = exp(-0.4)
    sc = asymmetric_score(ts, S, tau=0.0, onset=onset, end=8.0, tol=1.0, k=1, n=1, decay_s=5.0)
    assert sc == pytest.approx(np.exp(-2.0 / 5.0))

    # silent clip returns -c_miss
    sc_silent = asymmetric_score(ts, np.full_like(ts, -1.0), tau=0.0, onset=onset, end=8.0,
                                 tol=1.0, k=1, n=1, c_miss=1.5)
    assert sc_silent == pytest.approx(-1.5)


def test_spam_rate():
    ts = np.arange(0.0, 61.0, 1.0)  # 60s total = 1 minute
    onset = None  # no risk
    S = np.full_like(ts, -1.0)
    S[10] = 2.0
    S[30] = 2.0  # 2 alerts in 1 minute
    sr = spam_rate(ts, S, tau=0.0, onset=onset, end=None, tol=1.0, k=1, n=1)
    assert sr == pytest.approx(2.0)


def test_recovery_steps():
    S = [0.5, -0.5, -0.8, -0.2, 0.4, 0.6]
    # degradation at index 1 (-0.5). Next > 0 is at index 4 (0.4)
    # steps = 4 - 1 = 3
    rec = recovery_steps(S, tau=0.0, degrade_idx=1)
    assert rec == 3


# ----------------------------------------------------------------- FR-ST-8: protocol header

def test_protocol_header():
    hdr = protocol_header(window_n=4, fps=2.0, k=2, n=3)
    assert hdr.startswith("# evidence boundary: frames <= t;")
    assert "history: window-recompute (4,2.0)" in hdr
    assert "trigger: k-of-n (2-of-3)" in hdr


# ----------------------------------------------------------------- FR-ST-5 & FR-ST-7: gap & latency

def test_window_vs_stream_gap_h11():
    cfg = dict(stream=dict(tolerance_s=1.0, k=3, n=3, step_s=1.0))
    # Create sample video
    s = Sample(id="v1", bench="toy", modality="video", media="v1.mp4", label=1, cls="ppe",
               video=dict(duration_s=10.0, fps=2.0, onset_s=3.0, end_s=6.0))
    # Runs where window-level s > 0 on onset..end (high window recall),
    # but k=3 with only 2 positive steps delays alert so stream-level is late or silent
    rows = []
    for t in range(1, 10):
        # only t=3, 4 are positive (window-level TP=2, FN=1 for [3,4,5,6]),
        # but k=3 requires 3 consecutive hits, so alert NEVER fires! (stream recall = 0)
        s_val = 1.5 if t in (3, 4) else -1.5
        rows.append(dict(sample_id="v1", t=float(t), s=s_val, latency_ms=50.0))

    runs_df = pd.DataFrame(rows)
    gap = window_vs_stream_gap(runs_df, [s], cfg, tau=0.0)
    assert gap["window_recall"] > 0.0
    assert gap["stream_recall"] == 0.0
    assert gap["gap"] > 0.2
    assert gap["stream_policy"] == True


def test_stream_latency_profile():
    runs_df = pd.DataFrame([
        dict(t=1.0, knob="max_pixels", value="high", latency_ms=40.0),
        dict(t=2.0, knob="max_pixels", value="high", latency_ms=60.0),
        dict(t=1.0, knob="max_pixels", value="low", latency_ms=20.0),
    ])
    prof = stream_latency_profile(runs_df)
    assert len(prof) == 2
    row_hi = prof[prof.value == "high"].iloc[0]
    assert row_hi["p50_ms"] == pytest.approx(50.0)


def test_evaluate_stream_pipeline(tmp_path):
    cfg = dict(stream=dict(tolerance_s=1.0, k=1, n=1, step_s=1.0, c_miss=1.0, c_early=0.3, decay_s=5.0))
    s1 = Sample(id="v1", bench="toy", modality="video", media="v1.mp4", label=1, cls="ppe",
                video=dict(duration_s=5.0, fps=2.0, onset_s=2.0, end_s=4.0))
    s2 = Sample(id="v2", bench="toy", modality="video", media="v2.mp4", label=0, cls="ppe",
                video=dict(duration_s=5.0, fps=2.0))

    rows = []
    for t in range(1, 6):
        rows.append(dict(sample_id="v1", t=float(t), s=1.5 if t >= 2 else -1.5, latency_ms=45.0))
        rows.append(dict(sample_id="v2", t=float(t), s=-1.5, latency_ms=40.0))

    runs_df = pd.DataFrame(rows)
    res = evaluate_stream(runs_df, [s1, s2], cfg, tau=0.0, out_dir=tmp_path)

    assert (tmp_path / "stream_summary.csv").exists()
    assert (tmp_path / "stream_latency.csv").exists()

    # Check header exists
    content = (tmp_path / "stream_summary.csv").read_text(encoding="utf-8")
    assert content.startswith("# evidence boundary:")


# ----------------------------------------------------------------- plan_stream

def test_plan_stream():
    s_vid = Sample(id="v1", bench="toy", modality="video", media="v1.mp4", label=1, cls="ppe",
                   video=dict(duration_s=4.0, fps=2.0))
    s_img = Sample(id="img1", bench="toy", modality="image", media="i.png", label=1, cls="ppe")
    cfg = dict(stream=dict(step_s=1.0, window=4))

    jobs = plan_stream(cfg, [s_vid, s_img])
    assert len(jobs) == 4  # t=1.0, 2.0, 3.0, 4.0
    assert all(j.sid == "v1" for j in jobs)
    assert [j.params["t_end"] for j in jobs] == [1.0, 2.0, 3.0, 4.0]
