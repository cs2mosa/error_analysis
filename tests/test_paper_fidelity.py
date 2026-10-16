"""tests/test_paper_fidelity.py — tests for research paper fidelity and logical fixes.

Validates:
  1. ROAD (arXiv:2202.00449): Explainer validity gate evaluated relative to model prediction sign.
  2. Zhang & Nanda (arXiv:2309.16042): Activation patching multi-corruption requirement.
  3. Information Horizon (arXiv:2512.07580): Signed visual effect retention across depth.
  4. SPOT-Bench / TRACE (arXiv:2604.24317, 2609.30670): Dead-time spam rate on non-overlapping clips.
  5. ReCoVERR (arXiv:2402.15610): Selective recall and non-trivial coverage on accepted set.
  6. VLind-Bench (arXiv:2406.08702): Adaptive perception/knowledge gating for non-item samples.
  7. Seeing is not Enough (arXiv:2609.22588, 2510.17771): Behavioral readout/process failure signature.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ea.attrib import explainer_validity_gate
from ea.mech import horizon, is_patch_causal
from ea.metrics import coverage_at_recall
from ea.stream import spam_rate
from ea.triage import gate_ok
from ea.escalate import decide, SIG


# ----------------------------------------------------------------- 1. ROAD (arXiv:2202.00449)

def test_road_explainer_validity_model_sign():
    """ROAD faithfulness evaluates drop in model prediction confidence, not ground truth on errors."""
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    # Target feature in top-left
    d.rectangle([0, 0, 16, 16], fill=(255, 0, 0))

    # Error sample scenario: model predicts negative class (s < 0) when feature is present
    # Feature present: s = -3.0; feature deleted (inpainting): s rises to -0.5
    def score_fn(im):
        arr = np.asarray(im, dtype=float)
        red_mass = (arr[:16, :16, 0].mean() - arr[:16, :16, 1].mean()) / 50.0
        return float(-3.0 * red_mass)

    amap = np.zeros((4, 4))
    amap[0, 0] = 5.0
    amap[0, 1] = 5.0

    s_clean = score_fn(img)
    assert s_clean < 0  # model prediction is negative

    # If evaluated with ground-truth sign (+1), deletion increases score and fails
    res_gt = explainer_validity_gate(score_fn, img, amap, steps=4, sign=+1)
    assert res_gt["del_beats_rand"] is False

    # Under ROAD model-decision sign (sign = -1), deletion reduces model confidence and passes
    model_sign = 1 if s_clean >= 0 else -1
    res_model = explainer_validity_gate(score_fn, img, amap, steps=4, sign=model_sign)
    assert res_model["del_beats_rand"] is True


# ----------------------------------------------------------------- 2. Zhang & Nanda (arXiv:2309.16042)

def test_activation_patching_multi_corruption_fidelity():
    """Causal circuit claim requires distinct corruption methods (e.g. pair and boxnoise)."""
    p_res_pair = {("vis_all", 3): {"recovery": 0.85, "delta_s": 1.4}}
    p_res_boxnoise = {("vis_all", 3): {"recovery": 0.78, "delta_s": 1.2}}

    # Single corruption must not pass
    assert is_patch_causal([p_res_pair]) is False

    # Two genuine, distinct corruptions pass
    assert is_patch_causal([p_res_pair, p_res_boxnoise]) is True


# ----------------------------------------------------------------- 3. Information Horizon (arXiv:2512.07580)

def test_information_horizon_negative_shift():
    """Information horizon must handle visual evidence that reduces margin (s_full < s_blind)."""
    class MockVLM:
        def __init__(self):
            self.m = type("M", (), {"model": type("LM", (), {"language_model": type("L", (), {"layers": [None] * 36})()})()})()
        def score_inp(self, inp, pos, neg):
            return -2.5
        def prep(self, *a, **k):
            return {}

    # Mock knockout scores: s_full = -2.5, s_blind = -0.5 (delta_vis = -2.0)
    # Layers 0..9 retain little visual effect: s_l = -0.6
    # Layer 10+ retains >= 95% of visual effect: s_l = -2.45
    class MockHorizonContext:
        def __init__(self, spec, layers):
            self.layers = list(layers)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    import ea.model
    orig_knockout = ea.model.knockout
    try:
        def mock_knockout(vlm, inp, spec, layers):
            layers_list = list(layers)
            if len(layers_list) == 36:
                vlm.score_inp = lambda *a, **k: -0.5
            else:
                l_start = layers_list[0] if layers_list else 36
                vlm.score_inp = lambda *a, **k: -2.45 if l_start >= 10 else -0.6
            return MockHorizonContext(spec, layers)

        ea.model.knockout = mock_knockout
        vlm = MockVLM()
        vlm.score_inp = lambda *a, **k: -2.5  # s_full

        h_layer = horizon(vlm, {}, L=36, frac=0.95)
        assert h_layer == 10
    finally:
        ea.model.knockout = orig_knockout


# ----------------------------------------------------------------- 4. SPOT-Bench / TRACE (arXiv:2604.24317, 2609.30670)

def test_spam_rate_clip_outside_risk_window():
    """Dead time must accurately measure clip duration outside hazard window without overcounting."""
    # Video clip runs from t=20.0 to t=30.0 (10s duration = 0.1667 min)
    ts = np.arange(20.0, 31.0, 1.0)
    # Hazard window was t=2.0 to t=5.0 with tol=1.0 (ended at t=6.0)
    onset, end, tol = 2.0, 5.0, 1.0

    # 1 alert at t=25.0
    S = np.full_like(ts, -1.0)
    S[5] = 2.0  # ts[5] = 25.0

    # True dead time in clip = 10s = 0.16667 min -> 1 / (10/60) = 6.0 alerts/min
    sr = spam_rate(ts, S, tau=0.0, onset=onset, end=end, tol=tol, k=1, n=1)
    assert sr == pytest.approx(6.0, rel=1e-2)


# ----------------------------------------------------------------- 5. ReCoVERR (arXiv:2402.15610)

def test_selective_prediction_coverage_at_recall():
    """Coverage at target recall evaluates max accepted fraction over all prefixes (FR-B-13)."""
    # 4 samples: confidences [.9, .8, .2, .1]
    # Positive items at idx0, idx1. Predictions: idx0, idx1 positive.
    conf = np.array([0.9, 0.2, 0.8, 0.1])
    pred = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([1, 1, 0, 0])

    # For reachable recall rec=1.0, coverage recovers to max prefix (1.0)
    assert coverage_at_recall(conf, pred, y, rec=1.0) == pytest.approx(1.0)
    assert coverage_at_recall(conf, pred, y, rec=0.5) == pytest.approx(1.0)

    # For unreachable recall (target 1.0 when model achieves 0.5), coverage is 0.0
    unreachable = np.array([1.0, 0.0, 0.0, 0.0])
    assert coverage_at_recall(conf, unreachable, y, rec=1.0) == 0.0


# ----------------------------------------------------------------- 6. VLind-Bench (arXiv:2406.08702)

def test_vlind_bench_gating_safeguard():
    """Perception/knowledge gate prevents misattributing perception failure as a prior."""
    # When all prompts (qp, qp_abs, qr) are correct and valid, gate opens
    runs_ok = pd.DataFrame([
        dict(sample_id="s1", arm="base", prompt_id="qp", valid=True, correct=True),
        dict(sample_id="s1", arm="base", prompt_id="qp_abs", valid=True, correct=True),
        dict(sample_id="s1", arm="base", prompt_id="qr", valid=True, correct=True),
    ])
    assert gate_ok(runs_ok, "s1") is True

    # If any gating prompt is missing, gate closes to prevent false prior attribution
    runs_missing = runs_ok[runs_ok.prompt_id != "qr"]
    assert gate_ok(runs_missing, "s1") is False


# ----------------------------------------------------------------- 7. Seeing is not Enough (arXiv:2609.22588)

def test_readout_process_tier_escalation():
    """Behavioral readout evidence triggers T3/T4 escalation and confirms at T4."""
    # At T1/T2: behavioral diags present, T4 pending -> inconclusive with readout candidate
    ev_t1 = dict(qp_ok=True, qr_ok=True, crop=False, hires=False, loc="loc_equal_high")
    cause_t1, candidates_t1 = decide(ev_t1)
    assert cause_t1 == "inconclusive: more data"
    assert "readout/process (H3)" in candidates_t1

    # At T4: probe decodability and knockout causality confirmed -> assigned cause
    ev_t4 = dict(qp_ok=True, qr_ok=True, crop=False, hires=False, loc="loc_equal_high",
                 t4="H3", causal=True)
    cause_t4, candidates_t4 = decide(ev_t4)
    assert cause_t4 == "readout/process (H3)"
    assert candidates_t4 == []
