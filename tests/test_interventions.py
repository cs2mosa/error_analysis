"""tests/test_interventions.py — Phase 4 tests for interventions.py.

FR-Q-5 property tests + FR-T2-1 registry checks, all on make_toy images:
bgmask inside bit-identical / outside constant; occlude is its complement;
same-token control matches the crop's area; random controls same size with
IoU = 0 vs the evidence box; N0.null differs from base yet PSNR >= 40 dB;
degrade PSNR monotone in severity; injections construct what they claim.
"""
import io

import numpy as np
import pytest
from PIL import Image

from ea.data import make_toy
from ea.interventions import INTERV, degrade, o1_gtcrop, o1_bgmask, o3_occlude


@pytest.fixture(scope="module")
def toy():
    import tempfile
    from pathlib import Path
    td = tempfile.mkdtemp()
    S = make_toy(30, root=Path(td) / "toy", seed=0)
    return S, td


def psnr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def viol_box(s):
    return next(b["xyxy"] for b in s.boxes if b["role"] == "violator")


def sample_with_square(S):
    return next(s for s in S if viol_box(s) is not None and s.meta.get("vtype") != "absence")


# ---------------------------------------------------------------- registry (FR-T2-1 AC)

def test_registry_lookup_returns_fn_and_control():
    for name in ("O1.gtcrop", "O1.bgmask", "O3.occlude", "O2.facts", "N0.null",
                 "O5.personcrop", "O6.degrade", "O6.sharpen"):
        spec = INTERV[name]
        assert callable(spec["fn"]) and "control" in spec and "kind" in spec
    assert INTERV["O1.gtcrop"]["control"] == "O1.randcrop"
    assert INTERV["O1.bgmask"]["control"] == "O1.randmask"
    assert INTERV["O3.occlude"]["control"] == "O3.randoccl"
    assert INTERV["O2.facts"]["control"] == "O2.wrongfacts"
    assert INTERV["O6.degrade"]["control"] == "N0.null"


def test_injections_not_in_t2_plans(toy):
    from ea.run import plan_t2
    S, td = toy
    jobs = plan_t2({"knobs": {"quality": {"kinds": ["blur"], "severities": [1]}}}, S, {s.id for s in S[:4]})
    assert not any(j.arm.startswith("INJ.") for j in jobs)


# ---------------------------------------------------------------- O1 crop ladder

def test_gtcrop_box_exact(toy):
    S, td = toy
    s = sample_with_square(S)
    img = Image.open(f"{td}/toy/{s.id}.png").convert("RGB")
    imgs, _ = o1_gtcrop(s, root=td)
    b = viol_box(s)
    m = 0.25
    w, h = b[2] - b[0], b[3] - b[1]
    exp = (int(max(0, b[0] - m * w) * img.size[0]), int(max(0, b[1] - m * h) * img.size[1]),
           int(min(1, b[2] + m * w) * img.size[0]), int(min(1, b[3] + m * h) * img.size[1]))
    assert imgs[0].size == (exp[2] - exp[0], exp[3] - exp[1])


def test_randcrop_same_size_deterministic(toy):          # pitfall 5
    S, td = toy
    s = sample_with_square(S)
    (a,), _ = INTERV["O1.randcrop"]["fn"](s, root=td)
    (b,), _ = INTERV["O1.randcrop"]["fn"](s, root=td)
    (c,), _ = o1_gtcrop(s, root=td)
    assert a.size == c.size                              # same size as the crop
    assert np.array_equal(np.asarray(a), np.asarray(b))  # seeded by sample id


def test_sametok_matches_crop_area(toy):                 # FR-T2-5 / FR-Q-5
    S, td = toy
    s = sample_with_square(S)
    (crop,), _ = o1_gtcrop(s, root=td)
    (same,), _ = INTERV["O1.sametok"]["fn"](s, root=td)
    a_crop = crop.size[0] * crop.size[1]
    a_same = same.size[0] * same.size[1]
    assert abs(a_same - a_crop) / a_crop < 0.15          # area-matched (same-token proxy)


# ---------------------------------------------------------------- bgmask / occlude (FR-Q-5)

def _region_mask(shape, box):
    """box_mask convention: floor on low edges, ceil on high edges (design §5.1)."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    m[int(np.floor(box[1] * h)):int(np.ceil(box[3] * h)),
      int(np.floor(box[0] * w)):int(np.ceil(box[2] * w))] = True
    return m


def test_bgmask_inside_identical_outside_constant(toy):  # FR-Q-5
    S, td = toy
    s = sample_with_square(S)
    img = np.asarray(Image.open(f"{td}/toy/{s.id}.png").convert("RGB"))
    (masked,), _ = o1_bgmask(s, root=td)
    mm = np.asarray(masked)
    b = viol_box(s)
    # expand by margin like the arm does
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    eb = [max(0, x0 - .25 * w), max(0, y0 - .25 * h), min(1, x1 + .25 * w), min(1, y1 + .25 * h)]
    inside = _region_mask(img.shape, eb)
    assert np.array_equal(img[inside], mm[inside])       # inside pixels bit-identical
    assert (mm[~inside] == 128).all()                    # outside constant gray


def test_occlude_is_complement_of_bgmask(toy):           # FR-Q-5
    S, td = toy
    s = sample_with_square(S)
    (occ,), _ = o3_occlude(s, root=td)
    om = np.asarray(occ)
    img = np.asarray(Image.open(f"{td}/toy/{s.id}.png").convert("RGB"))
    b = viol_box(s)
    inside = _region_mask(img.shape, b)
    assert (om[inside] == 128).all()                     # evidence box grayed
    assert np.array_equal(img[~inside], om[~inside])     # rest untouched


def test_randmask_randoccl_iou0_and_size(toy):           # FR-Q-5
    S, td = toy
    s = sample_with_square(S)
    orig = Image.open(f"{td}/toy/{s.id}.png").convert("RGB")
    (rm,), _ = INTERV["O1.randmask"]["fn"](s, root=td)
    (ro,), _ = INTERV["O3.randoccl"]["fn"](s, root=td)
    assert rm.size == ro.size == orig.size               # native resolution kept
    a = np.asarray(rm)
    o = np.asarray(ro)
    assert (a != 128).any() and (o == 128).any()         # randmask keeps real pixels; randoccl grays
    # geometric IoU check: recompute the kept/random box with the same seed
    from ea.interventions import _rand_box, _seed
    vb = viol_box(s)
    for extra in ("randmask", "randoccl"):
        rb = _rand_box(s, np.random.default_rng(_seed(s, extra)), 0)
        inter = max(0, min(rb[2], vb[2]) - max(rb[0], vb[0])) * max(0, min(rb[3], vb[3]) - max(rb[1], vb[1]))
        union = ((rb[2] - rb[0]) * (rb[3] - rb[1]) + (vb[2] - vb[0]) * (vb[3] - vb[1]) - inter)
        assert inter / union == pytest.approx(0.0), extra   # IoU = 0 with the evidence box


# ---------------------------------------------------------------- N0.null (FR-Q-5)

def test_null_differs_but_psnr_high(toy):
    S, td = toy
    s = sample_with_square(S)
    (nl,), _ = INTERV["N0.null"]["fn"](s, root=td, jpeg_q=95, shift_px=1)
    img = Image.open(f"{td}/toy/{s.id}.png").convert("RGB")
    assert not np.array_equal(np.asarray(nl), np.asarray(img))     # differs from base
    assert psnr(nl, img) >= 40                                     # yet imperceptible (psnr_null_min_db)


# ---------------------------------------------------------------- O6 degrade (FR-Q-5)

@pytest.mark.parametrize("kind", ["blur", "lowlight", "noise", "haze", "jpeg", "resample"])
def test_degrade_psnr_monotone(toy, kind):
    S, td = toy
    s = sample_with_square(S)
    img = Image.open(f"{td}/toy/{s.id}.png").convert("RGB")
    ps = [psnr(degrade(img, kind, sev), img) for sev in (1, 3, 5)]
    assert ps[0] >= ps[1] >= ps[2], (kind, ps)           # monotone in severity
    assert ps[0] > 10 and ps[2] < ps[0]                  # sev1 mild, sev5 harsh


def test_o6_sharpen_changes_image(toy):
    S, td = toy
    s = sample_with_square(S)
    (sh,), _ = INTERV["O6.sharpen"]["fn"](s, root=td)
    img = Image.open(f"{td}/toy/{s.id}.png").convert("RGB")
    assert np.abs(np.asarray(sh, float) - np.asarray(img, float)).mean() > 0.5


# ---------------------------------------------------------------- O2 / O5 / O3

def test_wrongfacts_flips_facts_and_keeps_gold(toy):
    S, td = toy
    s = sample_with_square(S)
    _, s2 = INTERV["O2.wrongfacts"]["fn"](s, root=td)
    assert set(s2.facts) == set(s.facts)                 # same keys
    for k, v in s2.facts.items():                        # every present/absent flipped
        if s.facts[k] == "present":
            assert v == "absent"
        elif s.facts[k] == "absent":
            assert v == "present"
        else:
            assert v == s.facts[k]                       # not_visible passes through
    assert s2.label == s.label and s2.id == s.id         # gold untouched (pitfall 2)


def test_o2_facts_returns_text_only(toy):
    S, td = toy
    s = sample_with_square(S)
    imgs, s2 = INTERV["O2.facts"]["fn"](s, root=td)
    assert imgs is None and s2 is s                      # qr prompt, sample unchanged


def test_personcrop_uses_person_box(toy):
    S, td = toy
    s = next(x for x in S if any(b["role"] == "person" for b in x.boxes))
    (c0,), _ = INTERV["O5.personcrop"]["fn"](s, root=td, idx=0)
    (c1,), _ = INTERV["O5.personcrop"]["fn"](s, root=td, idx=1) if \
        sum(b["role"] == "person" for b in s.boxes) > 1 else ((c0,), None)
    assert c0.size[0] > 8                                # a real crop came back


def test_o3_blank_noise_swap(toy):
    S, td = toy
    s = sample_with_square(S)
    (bl,), _ = INTERV["O3.blank"]["fn"](s, root=td)
    assert np.asarray(bl).min() == np.asarray(bl).max() == 128
    (nz,), _ = INTERV["O3.noise"]["fn"](s, root=td)
    assert np.asarray(nz).std() > 30
    other = next(x for x in S if x.label != s.label)
    (sw,), _ = INTERV["O3.swap"]["fn"](s, root=td, other=f"{td}/toy/{other.id}.png")
    assert np.asarray(sw).shape[:2] == np.asarray(Image.open(f"{td}/toy/{other.id}.png")).shape[:2]


# ---------------------------------------------------------------- injections (§5.8b)

def test_inj_small_scales_canvas_and_boxes(toy):
    S, td = toy
    s = sample_with_square(S)
    (canvas,), s2 = INTERV["INJ.small"]["fn"](s, root=td, k=3)
    img = Image.open(f"{td}/toy/{s.id}.png")
    assert canvas.size == (img.size[0] * 3, img.size[1] * 3)
    b2 = viol_box(s2)
    b1 = viol_box(s)
    assert all(abs(x2 - (x1 + 1) / 3) < 1e-9 for x1, x2 in zip(b1, b2))   # (x + (k-1)/2)/k


def test_inj_clutter_disjoint_from_violator(toy):
    S, td = toy
    s = sample_with_square(S)
    (img,), s2 = INTERV["INJ.clutter"]["fn"](s, root=td, m=3)
    vb = viol_box(s)
    added = [b["xyxy"] for b in s2.boxes if b not in s.boxes]
    assert len(added) == 3
    for b in added:
        assert b[2] < vb[0] or b[0] > vb[2] or b[3] < vb[1] or b[1] > vb[3]


def test_inj_degrade_and_lead(toy):
    S, td = toy
    s = sample_with_square(S)
    (d1,), _ = INTERV["INJ.degrade"]["fn"](s, root=td, kind="blur", sev=4)
    assert psnr(d1, Image.open(f"{td}/toy/{s.id}.png")) < 30
    _, s2 = INTERV["INJ.lead"]["fn"](s, root=td)
    if s.label == 1:
        assert "excellent safety record" in s2.rule_text  # pushed toward "safe" on violations
    _, s3 = INTERV["INJ.label"]["fn"](s, root=td)
    assert s3 is None                                    # pass-through; validate flips gold itself
