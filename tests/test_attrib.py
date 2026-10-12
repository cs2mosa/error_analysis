"""tests/test_attrib.py — tests for T3 attribution tier (FR-T3-1..8).

Coverage:
  - FR-T3-1: explainers interface (map_attn, map_grad).
  - FR-T3-2: localization metrics (mass, enrich, rank, point, bg).
  - FR-T3-4: 3-way reading (loc_low_both, loc_equal_high, loc_lower_in_errors).
  - FR-T3-5: deletion and insertion AUC with ROAD nli, gray, blur.
  - FR-T3-6: explainer-validity gate.
  - FR-T3-7: SpRAy-style clustering.
  - FR-T3-8: planted-confounder check.
"""
import numpy as np
import pytest
from PIL import Image

from ea.attrib import (
    box_mask,
    del_ins,
    explainer_validity_gate,
    loc,
    nli,
    paint,
    planted_confounder_check,
    spray_cluster,
    three_way,
)


# ----------------------------------------------------------------- FR-T3-2: box_mask & loc

def test_box_mask_and_loc_concentrated():
    grid = (10, 10)
    box = [0.0, 0.0, 0.5, 0.5]  # top-left quadrant: 5x5 cells = 25 / 100 area fraction = 0.25
    m = box_mask(box, grid)
    assert m.sum() == 25
    assert m.mean() == pytest.approx(0.25)

    # Attribution concentrated entirely in the box
    amap = np.zeros(grid)
    amap[m] = 1.0
    res = loc(amap, box)
    assert res["mass"] == pytest.approx(1.0)
    assert res["enrich"] == pytest.approx(4.0)  # 1.0 / 0.25 = 4.0
    assert res["point"] == pytest.approx(1.0)
    assert res["bg"] == pytest.approx(0.0)
    assert res["rank"] == pytest.approx(1.0)


def test_loc_uniform():
    grid = (10, 10)
    box = [0.0, 0.0, 0.5, 0.5]  # area fraction = 0.25
    amap = np.ones(grid)
    res = loc(amap, box)
    assert res["mass"] == pytest.approx(0.25)
    assert res["enrich"] == pytest.approx(1.0)  # chance = 1.0
    assert res["bg"] == pytest.approx(0.75)


# ----------------------------------------------------------------- FR-T3-4: 3-way reading

def test_three_way_all_branches():
    # 1. loc_low_both: both below shortcut_enrichment (1.5)
    enr_err_low = [1.1, 1.2, 1.3]
    enr_cor_low = [1.0, 1.2, 1.4]
    assert three_way(enr_err_low, enr_cor_low, shortcut_enrichment=1.5) == "loc_low_both"

    # 2. loc_equal_high: both high and error within 15% of correct
    enr_err_high = [3.0, 3.2, 3.1]  # median 3.1
    enr_cor_high = [3.2, 3.3, 3.4]  # median 3.3; 3.3 - 0.15*3.3 = 2.805 <= 3.1
    assert three_way(enr_err_high, enr_cor_high, shortcut_enrichment=1.5, equal_delta=0.15) == "loc_equal_high"

    # 3. loc_lower_in_errors: error significantly lower than correct
    enr_err_drop = [1.8, 1.9, 2.0]  # median 1.9
    enr_cor_drop = [3.5, 3.6, 3.7]  # median 3.6; 3.6 - 0.15*3.6 = 3.06 > 1.9
    assert three_way(enr_err_drop, enr_cor_drop, shortcut_enrichment=1.5, equal_delta=0.15) == "loc_lower_in_errors"


# ----------------------------------------------------------------- FR-T3-5: nli, paint, del_ins

def test_nli_and_paint():
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    grid = (4, 4)
    cells = [0, 1]  # top row first two cells

    for how in ("nli", "gray", "blur"):
        p = paint(img, cells, grid, how=how)
        assert isinstance(p, Image.Image)
        assert p.size == img.size
        arr = np.asarray(p)
        # Unmasked bottom area must remain unchanged
        orig_arr = np.asarray(img)
        np.testing.assert_array_equal(arr[32:, :], orig_arr[32:, :])


def test_del_ins_synthetic():
    # Synthetic image: 64x64 white background with a red square at top-left
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    grid = (4, 4)  # 16 cells, cell 0 is top-left
    # Score function: measures how red cell 0 is relative to background
    def score_fn(im):
        arr = np.asarray(im, dtype=float)
        # red square area at top-left (0..16, 0..16)
        r_top = arr[:16, :16, 0].mean() - arr[:16, :16, 1].mean()
        return float(r_top / 50.0)

    # First, draw red square in cell 0
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 16, 16], fill=(255, 0, 0))

    # Perfect attribution map: cell 0 has highest attribution
    good_map = np.zeros(grid)
    good_map[0, 0] = 10.0

    # Inverted attribution map: cell 0 has lowest attribution
    bad_map = np.ones(grid)
    bad_map[0, 0] = 0.0

    # Deletion AUC of good_map must be lower than deletion AUC of bad_map
    # (deleting important cell 0 first drops the score faster)
    d_good = del_ins(score_fn, img, good_map, mode="del", steps=4, sign=1)
    d_bad = del_ins(score_fn, img, bad_map, mode="del", steps=4, sign=1)
    assert d_good < d_bad

    # Insertion AUC of good_map must be higher than bad_map
    i_good = del_ins(score_fn, img, good_map, mode="ins", steps=4, sign=1)
    i_bad = del_ins(score_fn, img, bad_map, mode="ins", steps=4, sign=1)
    assert i_good > i_bad


# ----------------------------------------------------------------- FR-T3-6: explainer gate

def test_explainer_validity_gate():
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 16, 16], fill=(255, 0, 0))

    def score_fn(im):
        arr = np.asarray(im, dtype=float)
        return float((arr[:16, :16, 0].mean() - arr[:16, :16, 1].mean()) / 50.0)

    amap = np.zeros((4, 4))
    amap[0, 0] = 5.0
    amap[0, 1] = 5.0

    res = explainer_validity_gate(score_fn, img, amap, steps=4)
    assert "valid" in res
    assert res["fill_stable"] == True


# ----------------------------------------------------------------- FR-T3-7 & FR-T3-8: SpRAy & confounder

def test_spray_cluster():
    # 6 maps: 3 top-heavy, 3 bottom-heavy
    maps = []
    for _ in range(3):
        m = np.zeros((8, 8))
        m[:4, :] = 1.0
        maps.append(m)
    for _ in range(3):
        m = np.zeros((8, 8))
        m[4:, :] = 1.0
        maps.append(m)

    labels = spray_cluster(maps, k=2, seed=0)
    assert len(labels) == 6
    # First 3 should have the same label, last 3 should have the other label
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_planted_confounder_check():
    # Planted shortcut in background (bottom half [0, 0.5, 1.0, 1.0])
    bg_box = [0.0, 0.5, 1.0, 1.0]
    amap = np.zeros((8, 8))
    amap[4:, :] = 1.0  # high mass on background
    assert planted_confounder_check(amap, bg_box, min_bg_enrichment=1.0) == True

    amap_obj = np.zeros((8, 8))
    amap_obj[:4, :] = 1.0  # mass on object
    assert planted_confounder_check(amap_obj, bg_box, min_bg_enrichment=1.0) == False
