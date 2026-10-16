"""tests/test_fidelity_fixes.py — regression tests for defects found when the code was
audited against the cited papers.

Each test fails on the pre-fix code:
  1. metrics.coverage_at_recall   — was degenerate (only 0.0 or 1.0).
  2. mech.horizon                 — noise floor guard (reconciliation item 3 was a no-op).
  3. attrib gate (FR-T3-6)        — ties passed, no Wilcoxon, fill check was magnitude-based,
                                    no randomization / robustness / internal-agreement checks.
  4. attrib.nli                   — 20 relaxation sweeps instead of the exact sparse solve (ROAD §5.2).
  5. mech.is_patch_causal         — "any cell in each arm" instead of the same site in >= 2 arms.
  6. model.knockout / randomized_layers — support for the gate's internal-agreement and
                                    randomization checks.
"""
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import ea.metrics as M
from ea.attrib import (explainer_validity_gate, gate_aggregate, map_spearman, nli)
from ea.mech import horizon, is_patch_causal


# ----------------------------------------------------------------- 1. coverage_at_recall

def test_coverage_at_recall_recall_dips_and_recovers():
    conf = np.array([.9, .8, .7, .6])
    y = np.ones(4, int)
    s = np.array([1.0, -1.0, 1.0, 1.0])
    # recall of the accepted set by prefix: 1, .5, .667, .75
    assert M.coverage_at_recall(conf, s, y, 0.7) == pytest.approx(1.0)     # recovers at k=4
    assert M.coverage_at_recall(conf, s, y, 0.8) == pytest.approx(0.25)    # only k=1 qualifies
    assert M.coverage_at_recall(conf, s, y, 1.0) == pytest.approx(0.25)
    assert M.coverage_at_recall(conf, -np.ones(4), y, 0.5) == 0.0          # never reachable


def test_coverage_at_recall_is_not_binary_and_is_monotone_in_target():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        n = 120
        y = (rng.random(n) < 0.3).astype(int)
        s = np.where(y == 1, rng.normal(1, 2, n), rng.normal(-1, 2, n))
        conf = np.abs(s) + rng.random(n) * 0.01
        covs = [M.coverage_at_recall(conf, s, y, r) for r in (0.5, 0.7, 0.9)]
        assert covs[0] >= covs[1] >= covs[2]
        seen.update(round(c, 3) for c in covs)
    assert len(seen) > 5, "coverage@recall collapsed to a handful of values"


# ----------------------------------------------------------------- 2. horizon noise floor

def test_horizon_abstains_when_visual_effect_is_below_noise_floor():
    import ea.model

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _VLM:
        def __init__(self):
            self.score_inp = lambda *a, **k: -2.50

    orig = ea.model.knockout
    try:
        def mock_knockout(vlm, inp, spec, layers):
            vlm.score_inp = lambda *a, **k: -2.47      # |s_full - s_blind| = 0.03 < bf16 floor
            return _Ctx()
        ea.model.knockout = mock_knockout
        assert horizon(_VLM(), {}, L=36, frac=0.95) is None
    finally:
        ea.model.knockout = orig


# ----------------------------------------------------------------- 3. explainer-validity gate

def _object_scene():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    ImageDraw.Draw(img).rectangle([16, 16, 31, 31], fill=(255, 0, 0))

    def score_fn(im):
        a = np.asarray(im, dtype=float)
        return float((a[16:32, 16:32, 0].mean() - 128.0) / 20.0)
    return img, score_fn


def _maps():
    good = np.full((8, 8), 0.01)
    good[2:4, 2:4] = 1.0
    bad = np.full((8, 8), 1.0)
    bad[2:4, 2:4] = 0.0
    return good, bad


def test_gate_accepts_faithful_map_and_rejects_inverted_map():
    img, fn = _object_scene()
    good, bad = _maps()
    r_good = explainer_validity_gate(fn, img, good, steps=4, sign=1)
    r_bad = explainer_validity_gate(fn, img, bad, steps=4, sign=1)
    assert r_good["valid"] is True
    assert r_bad["valid"] is False
    assert set(r_good["del_beats_rand_by_fill"]) == {"nli", "gray", "blur"}


def test_gate_does_not_pass_a_model_that_ignores_the_image():
    img, _ = _object_scene()
    good, _ = _maps()
    res = explainer_validity_gate(lambda im: 0.3, img, good, steps=4, sign=1)
    assert res["informative"] is False
    assert res["valid"] is False                      # flat curves are ties, not evidence


def _detail(d_e, d_r, i_e, i_r, valid=True):
    return dict(valid=valid, informative=True, del_expl=d_e, del_rand=d_r,
                ins_expl=i_e, ins_rand=i_r, del_base=None, ins_base=None)


def test_gate_aggregate_uses_paired_wilcoxon_and_requires_power():
    ok = [_detail(0.1 + 0.01 * i, 1.0 + 0.1 * i, 1.0 + 0.1 * i, 0.1 + 0.01 * i) for i in range(8)]
    res = gate_aggregate(ok, alpha=0.05, min_pass_rate=0.7)
    assert res["valid"] is True and res["p_del_rand"] < 0.05 and res["p_ins_rand"] < 0.05

    assert gate_aggregate(ok[:3], alpha=0.05, min_pass_rate=0.7)["valid"] is False   # under-powered

    mixed = [_detail(0.1, 1.0, 1.0, 0.1)] * 4 + [_detail(1.0, 0.1, 0.1, 1.0, valid=False)] * 4
    assert gate_aggregate(mixed, alpha=0.05, min_pass_rate=0.7)["valid"] is False


def test_gate_aggregate_randomization_and_robustness_checks():
    ok = [_detail(0.1 + 0.01 * i, 1.0 + 0.1 * i, 1.0 + 0.1 * i, 0.1 + 0.01 * i) for i in range(8)]
    assert gate_aggregate(ok, randomization_rho=[0.1, 0.2], robustness_rho=[0.9, 0.8])["valid"] is True
    assert gate_aggregate(ok, randomization_rho=[0.9, 0.8])["valid"] is False         # map ignores weights
    assert gate_aggregate(ok, robustness_rho=[0.1, 0.2])["valid"] is False            # unstable under noise
    res = gate_aggregate(ok, randomization_rho=None)
    assert "randomization" in res["skipped"]


def test_map_spearman():
    a = np.arange(16.0).reshape(4, 4)
    assert map_spearman(a, a) == pytest.approx(1.0)
    assert map_spearman(a, -a) == pytest.approx(-1.0)
    assert map_spearman(a, np.ones((4, 4))) == 0.0     # constant map carries no ranking


# ----------------------------------------------------------------- 4. nli exact solve (ROAD §5.2)

def _structured_image(h=96, w=96):
    rng = np.random.default_rng(0)
    base = np.kron(rng.normal(0, 1, (h // 8, w // 8, 3)), np.ones((8, 8, 1)))
    return (128 + 40 * base + 20 * np.sin(np.arange(w) / 3.0)[None, :, None]).clip(0, 255)


def test_nli_masked_pixels_satisfy_the_road_stencil():
    img = _structured_image()
    mask = np.zeros(img.shape[:2], bool)
    mask[24:72, 24:72] = True                            # interior block: no border effects
    out = nli(img.copy(), mask, noise_std=0.0, seed=0)
    ys, xs = np.nonzero(mask)
    p = out
    direct = (p[ys - 1, xs] + p[ys + 1, xs] + p[ys, xs - 1] + p[ys, xs + 1]) / 6.0
    diag = (p[ys - 1, xs - 1] + p[ys - 1, xs + 1] + p[ys + 1, xs - 1] + p[ys + 1, xs + 1]) / 12.0
    resid = np.abs(p[ys, xs] - (direct + diag))
    assert resid.max() < 1e-3, f"stencil residual {resid.max():.3f}: not the exact ROAD solution"
    np.testing.assert_array_equal(out[~mask], img[~mask])


def test_nli_fill_obeys_maximum_principle_when_mask_touches_border():
    img = _structured_image()
    mask = np.zeros(img.shape[:2], bool)
    mask[:, :48] = True
    out = nli(img.copy(), mask, noise_std=0.0, seed=0)
    known = img[~mask]
    assert np.isfinite(out).all()
    assert out[mask].min() >= known.min() - 1e-6 and out[mask].max() <= known.max() + 1e-6


# ----------------------------------------------------------------- 5. patching: same site in >= 2 arms

def test_patch_causal_requires_the_same_site_to_hold_in_both_corruptions():
    strong = {"recovery": 0.9, "delta_s": 2.0}
    none = {"recovery": 0.0, "delta_s": 0.0}
    a = {("vis_all", 10): strong, ("q", 10): none}
    b = {("vis_all", 10): none, ("q", 10): strong}
    assert is_patch_causal([a, b]) is False
    assert is_patch_causal([a, a]) is True


# ----------------------------------------------------------------- 6. model hooks used by the gate

class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, h, attention_mask=None):
        self.seen = attention_mask
        return h


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttn()
        self.lin = torch.nn.Linear(4, 4)


def _fake_vlm(n_layers=4):
    layers = torch.nn.ModuleList([_FakeLayer() for _ in range(n_layers)])
    lm = type("LM", (), {})()
    lm.layers = layers
    inner = type("Inner", (), {})()
    inner.language_model = lm
    top = torch.nn.Module()
    top.model = type("W", (), {"language_model": lm})()
    top.layers = layers
    vlm = type("V", (), {})()
    vlm.m = top
    vlm.layers = layers
    vlm.groups = lambda inp: dict(vis=torch.tensor([1, 2, 3]), q=torch.tensor([4]),
                                  last=torch.tensor([5]), grid=None)
    return vlm


def test_knockout_accepts_explicit_token_subsets():
    from ea.model import knockout
    vlm = _fake_vlm()
    T = 6
    mask = torch.zeros(1, 1, T, T)
    subset = torch.tensor([2])                           # only visual token 2
    with knockout(vlm, {}, [(subset, "last")], range(4)):
        vlm.layers[0].self_attn(torch.zeros(1, T, 4), attention_mask=mask)
    seen = vlm.layers[0].self_attn.seen
    assert seen[0, 0, 5, 2] < -1e30                      # last row cannot see token 2
    assert seen[0, 0, 5, 1] == 0 and seen[0, 0, 5, 3] == 0


def test_randomized_layers_changes_top_half_and_restores_exactly():
    from ea.model import randomized_layers
    vlm = _fake_vlm(4)
    before = [[p.detach().clone() for p in L.parameters()] for L in vlm.layers]
    with randomized_layers(vlm, frac=0.5, seed=0):
        for i, L in enumerate(vlm.layers):
            changed = any(not torch.equal(p, b) for p, b in zip(L.parameters(), before[i]))
            assert changed == (i >= 2)
    for i, L in enumerate(vlm.layers):
        assert all(torch.equal(p, b) for p, b in zip(L.parameters(), before[i]))


# ----------------------------------------------------------------- 7. cli gate glue (fake model)

class _GateVLM:
    """Margin depends only on the object box; attention is a function of the layer weights
    (set `weight_free` for an explainer that ignores them)."""

    def __init__(self, weight_free=False):
        base = _fake_vlm(4)
        self.m, self.layers = base.m, base.layers
        self.weight_free = weight_free
        self._ref = self.layers[-1].lin.weight.detach().clone()

    def prep(self, text, images=None, frames=None):
        return dict(img=images[0])

    def score_inp(self, inp, pos, neg):
        a = np.asarray(inp["img"], dtype=float)
        return float((a[16:32, 16:32, 0].mean() - 128.0) / 20.0)

    def groups(self, inp):
        return dict(vis=torch.arange(64), q=torch.tensor([64]), last=torch.tensor([65]), grid=(1, 8, 8))

    def randomized(self):
        return not torch.equal(self.layers[-1].lin.weight, self._ref)


def _run_gate(monkeypatch, vlm, n=8):
    import ea.attrib as A
    from ea.cli import _explainer_gate
    good, _ = _maps()
    rng = np.random.default_rng(0)

    def fake_map(v, inp, band=None, ratio=10.0):
        if v.randomized() or ratio is None:
            return rng.uniform(0, 1, (8, 8))
        if v.weight_free:                      # fixed map regardless of weights
            return good.copy()
        return good * (1.0 + 0.001 * np.arange(64).reshape(8, 8))

    monkeypatch.setattr(A, "map_attn", fake_map)
    img, _ = _object_scene()
    smap = {i: i for i in range(n)}
    return _explainer_gate(
        vlm, list(range(n)), smap, lambda s: "q", lambda s: img.copy(),
        lambda s: (lambda im: vlm.score_inp(vlm.prep("q", images=[im]), None, None)),
        dict(gate_steps=4), 10.0)


def test_cli_gate_runs_end_to_end_and_restores_weights(monkeypatch):
    vlm = _GateVLM()
    gate = _run_gate(monkeypatch, vlm)
    assert gate["randomization"]["ok"] is True          # map changes when weights are destroyed
    assert gate["robustness"]["ok"] in (True, False)    # noise check ran
    assert gate["skipped"] == []
    assert not vlm.randomized()                         # weights restored after the check
    assert all("internal_agreement" in d for d in gate["details"])
    assert gate["valid"] is True, gate["status"]        # faithful map, weight-dependent, stable
    assert gate["baseline_not_worse"] is True           # ties with raw attention do not block
    assert gate["p_del_rand"] < 0.05 and gate["p_ins_rand"] < 0.05


def test_cli_gate_rejects_an_explainer_that_ignores_the_weights(monkeypatch):
    # An explainer whose map is unchanged when weights are destroyed cannot be trusted.
    import ea.attrib as A
    vlm = _GateVLM(weight_free=True)
    good, _ = _maps()
    from ea.cli import _explainer_gate
    monkeypatch.setattr(A, "map_attn", lambda v, inp, band=None, ratio=10.0: good.copy())
    img, _ = _object_scene()
    gate = _explainer_gate(vlm, list(range(8)), {i: i for i in range(8)}, lambda s: "q",
                           lambda s: img.copy(),
                           lambda s: (lambda im: vlm.score_inp(vlm.prep("q", images=[im]), None, None)),
                           dict(gate_steps=4), 10.0)
    assert gate["randomization"]["ok"] is False
    assert gate["valid"] is False and gate["tag"] == "UNVALIDATED"


def test_gate_aggregate_blocks_an_explainer_significantly_worse_than_raw_attention():
    worse = [dict(_detail(0.1 + 0.01 * i, 1.0 + 0.1 * i, 1.0 + 0.1 * i, 0.1 + 0.01 * i),
                  del_base=0.01 * i, ins_base=2.0 + 0.1 * i) for i in range(8)]
    res = gate_aggregate(worse)
    assert res["baseline_not_worse"] is False and res["valid"] is False
