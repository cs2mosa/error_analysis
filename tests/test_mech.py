"""tests/test_mech.py — tests for T4 mechanistic analysis (FR-T4-1..10).

Coverage:
  - FR-T4-2: probe accuracy and selectivity on linearly separable synthetic data.
  - FR-T4-4: probe_verdict decision rule covering all branches (encoder limit, H1, H2, H3).
  - FR-T4-8: is_patch_causal dual-metric dual-corruption robustness (C13; 2309.16042).
  - FR-T4-9: visual token similarity across depth.
  - FR-T4-10: evidence rule (>= 2 independent methods: decodability + causal).
"""
import numpy as np
import pytest
import torch

from ea.mech import (
    internal_label,
    is_patch_causal,
    probe,
    probe_verdict,
    t4_tag,
    visual_token_similarity,
)


# ----------------------------------------------------------------- FR-T4-2: probes

def test_probe_linearly_separable():
    """Linearly separable synthetic features must yield accuracy >= 0.95."""
    rng = np.random.default_rng(0)
    # 60 samples of class 0 around (-3, -3), 60 of class 1 around (3, 3)
    X0 = rng.normal(loc=-3.0, scale=0.5, size=(60, 4))
    X1 = rng.normal(loc=3.0, scale=0.5, size=(60, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)

    pred, acc, ctl_acc = probe(X, y, seed=0)
    assert acc >= 0.95
    # Selectivity (acc - ctl_acc) must be large
    assert (acc - ctl_acc) > 0.3


# ----------------------------------------------------------------- FR-T4-4: probe_verdict

def test_probe_verdict_all_branches():
    # 1. encoder limit: vit=False, dino=True
    assert probe_verdict({"vit": False, "dino": True}) == "encoder/resolution limit"

    # 2. check data: vit=False, dino=False
    assert probe_verdict({"vit": False, "dino": False}) == "label/too-small: check data"

    # 3. merger bottleneck H1: vit=True, merger=False
    assert probe_verdict({"vit": True, "merger": False}) == "H1 merger bottleneck"

    # 4. depth fading H2: vit=True, merger=True, lost from layer 10 onward
    layers = {l: (l < 10) for l in range(36)}
    res_h2 = probe_verdict({"vit": True, "merger": True, "layers": layers})
    assert res_h2 == "H2 depth fading (lost at layer 10)"

    # 5. readout H3: decodable throughout all layers
    layers_all = {l: True for l in range(36)}
    assert probe_verdict({"vit": True, "merger": True, "layers": layers_all}) == "H3 decodable throughout -> readout"


def test_t4_tag():
    assert t4_tag("H1 merger bottleneck") == "H1"
    assert t4_tag("H2 depth fading (lost at layer 10)") == "H2"
    assert t4_tag("H3 decodable throughout -> readout") == "H3"
    assert t4_tag("encoder/resolution limit") == "enc"
    assert t4_tag("something else") == "unresolved"


# ----------------------------------------------------------------- FR-T4-10: evidence rule

def test_evidence_rule_internal_label():
    """Internal root cause emitted ONLY when >= 2 methods agree (decodability + causal)."""
    # Case 1: single method (decodability only) -> unresolved
    ev_single = [dict(method="probe", kind="decodability", supports="H1")]
    assert internal_label(ev_single) == "unresolved (single method)"

    # Case 2: single method (causal only) -> unresolved
    ev_causal = [dict(method="knockout", kind="causal", supports="H1")]
    assert internal_label(ev_causal) == "unresolved (single method)"

    # Case 3: two methods of the same kind (two decodability) -> unresolved
    ev_two_dec = [
        dict(method="probe", kind="decodability", supports="H1"),
        dict(method="lens", kind="decodability", supports="H1"),
    ]
    assert internal_label(ev_two_dec) == "unresolved (single method)"

    # Case 4: decodability + causal agree on H1 -> H1
    ev_agree = [
        dict(method="probe", kind="decodability", supports="H1"),
        dict(method="knockout", kind="causal", supports="H1"),
    ]
    assert internal_label(ev_agree) == "H1"

    # Case 5: decodability and causal conflict -> unresolved
    ev_conflict = [
        dict(method="probe", kind="decodability", supports="H1"),
        dict(method="patching", kind="causal", supports="H2"),
    ]
    assert internal_label(ev_conflict) == "unresolved (single method)"


# ----------------------------------------------------------------- FR-T4-8: is_patch_causal (C13)

def test_is_patch_causal_requires_two_corruptions_and_both_metrics():
    """C13 / FR-T4-8: causal only if >= 2 corruptions pass BOTH recovery >= 0.3 AND |delta_s| >= 0.5."""
    # Case 1: only 1 corruption provided -> False regardless of values
    strong = {("vis_all", 3): {"recovery": 0.9, "delta_s": 1.5}}
    assert is_patch_causal([strong]) is False

    # Case 2: 2 corruptions, both pass both thresholds -> True
    assert is_patch_causal([strong, strong]) is True

    # Case 3: 2 corruptions, one fails on delta_s -> only 1 passes -> False
    weak_delta = {("vis_all", 3): {"recovery": 0.9, "delta_s": 0.1}}
    assert is_patch_causal([strong, weak_delta]) is False

    # Case 4: 2 corruptions, one fails on recovery -> only 1 passes -> False
    weak_rec = {("vis_all", 3): {"recovery": 0.1, "delta_s": 1.5}}
    assert is_patch_causal([strong, weak_rec]) is False

    # Case 5: 2 corruptions, both weak -> False
    assert is_patch_causal([weak_delta, weak_rec]) is False


# ----------------------------------------------------------------- FR-T4-9: visual token similarity

def test_visual_token_similarity():
    vis_idx = torch.tensor([0, 1, 2, 3])
    # Layer 0: all visual tokens identical -> cosine similarity 1.0
    h0 = torch.ones((1, 4, 16))
    # Layer 1: orthogonal vectors -> cosine similarity 0.0
    h1 = torch.zeros((1, 4, 16))
    for i in range(4):
        h1[0, i, i] = 1.0

    sims = visual_token_similarity({0: h0, 1: h1}, vis_idx)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)
