"""tests/test_data.py — Phase 1 tests for data.py (FR-D-*).

Known-answer / property tests per design §7.4: make_toy validity, speed and
determinism; load round trip; missing metadata robustness; binning; box_mask
geometry; causal video frame sampling; adapter stubs.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ea.data import (Sample, load, make_toy, bin_meta, box_mask, frames,
                     to_df, media_path, adapter_construction_site10k)


# ---------------------------------------------------------------- FR-D-4 make_toy

def test_make_toy_valid_and_fast(tmp_path):                     # FR-D-4
    t0 = time.perf_counter()
    S = make_toy(200, root=tmp_path / "toy", seed=0)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"make_toy(200) took {dt:.1f}s (AC: < 10 s)"
    assert len(S) == 200
    jf = tmp_path / "toy" / "samples.jsonl"
    assert jf.exists()
    rows = [json.loads(l) for l in open(jf, encoding="utf-8")]
    assert len(rows) == 200
    need = {"id", "bench", "modality", "media", "label", "cls"}
    assert all(need <= set(r) for r in rows)
    for s in S:                                                 # every PNG exists and decodes
        p = media_path(s, tmp_path)
        assert p.exists()
        im = Image.open(p); im.verify()


def test_make_toy_deterministic(tmp_path):                      # NFR-1 (fixed seed)
    a = make_toy(60, root=tmp_path / "a", seed=0)
    b = make_toy(60, root=tmp_path / "b", seed=0)
    # byte-identical apart from the media root, which differs by construction
    ja = (tmp_path / "a" / "samples.jsonl").read_bytes().replace((tmp_path / "a").as_posix().encode(), b"ROOT")
    jb = (tmp_path / "b" / "samples.jsonl").read_bytes().replace((tmp_path / "b").as_posix().encode(), b"ROOT")
    assert ja == jb
    assert [as_dict_wo_media(s) for s in a] == [as_dict_wo_media(s) for s in b]


def as_dict_wo_media(s):
    d = s.__dict__.copy(); d["media"] = Path(s.media).name
    return d


def test_make_toy_slices_present(tmp_path):                     # design §7.3 clusters
    S = make_toy(400, root=tmp_path / "toy", seed=0)
    df = to_df(S)
    for cls in ("prior_cls", "rand_slice", "label_noise"):
        assert (df.cls == cls).sum() >= 5, cls
    assert (df.vtype == "absence").sum() >= 20
    assert df.clutter.fillna(False).sum() >= 20
    assert set(df.obj_size) == {"small", "medium", "large"}
    pairs = df.dropna(subset=["pair_id"])
    assert pairs.groupby("pair_id").label.nunique().eq(2).all()  # pairs are 1 violation + 1 safe


def test_make_toy_facts_match_image(tmp_path):                  # toy semantics (module docstring)
    from ea.data import media_path
    S = make_toy(200, root=tmp_path / "toy", seed=0)
    for s in S:
        if s.item is None or not s.facts:
            continue
        p = s.facts[s.item]
        assert p in ("present", "absent", "not_visible")
        # red/green square scenes: absent iff violation, except absence scenes
        if s.cls != "label_noise":
            im = np.asarray(Image.open(media_path(s, tmp_path)).convert("RGB"))
            red = ((im[..., 0] > 150) & (im[..., 1] < 90) & (im[..., 2] < 90)).sum()
            green = ((im[..., 1] > 110) & (im[..., 0] < 90)).sum()
            if s.meta.get("vtype") != "absence" and (red > 50 or green > 50):
                assert s.facts[s.item] == ("absent" if red > green else "present")


# ---------------------------------------------------------------- FR-D-1 load / schema

def test_load_roundtrip(tmp_path):                              # FR-D-1
    S = make_toy(40, root=tmp_path / "toy", seed=1)
    S2 = load("toy", root=tmp_path)
    assert [s.id for s in S2] == [s.id for s in S]
    assert all(isinstance(s, Sample) for s in S2)
    assert S2[0].label == S[0].label and S2[0].cls == S[0].cls


def test_load_missing_raises_actionable(tmp_path):              # FR-D-1
    with pytest.raises(FileNotFoundError, match="make_toy"):
        load("nope", root=tmp_path)


def test_missing_metadata_no_crash(tmp_path):                   # FR-D-3
    rows = [dict(id=f"x{i}", bench="toy", modality="image", media=f"m{i}.png",
                 label=i % 2, cls="r", rule_text=None, item=None, facts=None,
                 boxes=None, pair_id=None, meta=None, video=None) for i in range(5)]
    d = tmp_path / "nulls"; d.mkdir()
    (d / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    S = load("nulls", root=tmp_path)
    df = to_df(S)                                               # must not raise
    assert len(df) == 5
    assert "scene" not in df.columns or df["scene"].isna().all()  # absent metadata -> NaN/absent
    df2 = bin_meta(df, ["obj_size", "dist"])                    # binning skips non-numeric/absent
    assert "dist_bin" not in df2.columns


def test_unknown_extra_keys_ignored(tmp_path):                  # FR-D-3 (schema is forward-tolerant)
    d = tmp_path / "extra"; d.mkdir()
    row = dict(id="x", bench="toy", modality="image", media="m.png", label=0, cls="r",
               future_field="whatever")
    (d / "samples.jsonl").write_text(json.dumps(row), encoding="utf-8")
    s = load("extra", root=tmp_path)[0]
    assert s.label == 0 and not hasattr(s, "future_field")


# ---------------------------------------------------------------- FR-D-5 bin_meta

def test_bin_meta_quantiles_and_nan():                          # FR-D-5
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(dist=rng.uniform(0, 1, 300), scene=["a"] * 300))
    df.loc[5, "dist"] = np.nan
    out = bin_meta(df, ["dist", "scene"], q=3)
    assert "dist_bin" in out.columns
    assert str(out["dist_bin"].dtype) == "category"
    assert out["dist_bin"].isna().sum() == 1                    # NaN stays NaN, no leak
    assert out["dist_bin"].notna().nunique() <= 3
    # quantile bins are roughly balanced
    counts = out["dist_bin"].value_counts()
    assert all(c >= 90 for c in counts)


# ---------------------------------------------------------------- geometry invariants

def test_box_mask_image():                                      # design §5.1
    m = box_mask([0.0, 0.0, 0.5, 0.5], (10, 10))
    assert m.shape == (100,) and m.sum() == 25                  # floor/ceil rounding on exact halves
    m2 = box_mask([0.25, 0.25, 0.75, 0.75], (8, 8))
    assert m2.sum() == 16


def test_box_mask_video_tiles_time_major():                      # design §5.1
    m = box_mask([0.0, 0.0, 1.0, 1.0], (3, 4, 6))
    assert m.shape == (72,) and m.all()                         # full box on every frame
    part = box_mask([0.0, 0.0, 0.5, 0.5], (2, 4, 6))
    assert part.shape == (48,)
    assert part.reshape(2, 24)[0].sum() == 6 and part.reshape(2, 24)[1].sum() == 6


# ---------------------------------------------------------------- FR-D-6 frames

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """10 s video, 2 fps, frame k is solid gray level 10*k -> timestamps readable."""
    import cv2
    p = tmp_path_factory.mktemp("vid") / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32))
    assert w.isOpened()
    for k in range(20):
        w.write(np.full((32, 32, 3), 10 * k, dtype=np.uint8))
    w.release()
    return p


def test_frames_causal(tiny_video):                             # FR-D-6 AC
    _, ts = frames(tiny_video, fps=1, n=4, t_end=8.0)
    assert len(ts) == 4 and (ts <= 8.0).all()                   # no timestamp exceeds t_end
    assert ts[-1] == 8.0 and ts[0] == pytest.approx(5.0)
    _, ts2 = frames(tiny_video, fps=2, n=6, t_end=0.5)          # clip at 0
    assert (ts2 >= 0).all() and (ts2 <= 0.5).all()


def test_frames_single_and_odd_pad(tiny_video):                 # FR-D-6, pitfall 10
    imgs, ts = frames(tiny_video, fps=1, n=3, t_end=4.0, single=True)
    assert len(imgs) == 1 and len(ts) == 1 and ts[0] == 4.0
    imgs, ts = frames(tiny_video, fps=1, n=3, t_end=4.0)        # odd -> repeat last (temporal patch 2)
    assert len(imgs) == 4 and len(ts) == 4
    assert np.array_equal(np.asarray(imgs[-1]), np.asarray(imgs[-2]))


def test_frames_orders(tiny_video):                             # FR-D-6 orders
    f_fwd, ts_fwd = frames(tiny_video, fps=1, n=4, t_end=8.0)
    f_rev, _ = frames(tiny_video, fps=1, n=4, t_end=8.0, order="reverse")
    f_shuf, _ = frames(tiny_video, fps=1, n=4, t_end=8.0, order="shuffle", seed=0)
    f_shuf2, _ = frames(tiny_video, fps=1, n=4, t_end=8.0, order="shuffle", seed=0)
    assert [np.asarray(f).mean() for f in f_rev] == [np.asarray(f).mean() for f in f_fwd][::-1]
    assert [np.asarray(f).mean() for f in f_shuf] == [np.asarray(f).mean() for f in f_shuf2]
    means_fwd = {round(float(np.asarray(f).mean()), 3) for f in f_fwd}
    means_shuf = {round(float(np.asarray(f).mean()), 3) for f in f_shuf}
    assert means_fwd == means_shuf                              # same multiset, different order
    with pytest.raises(ValueError):
        frames(tiny_video, fps=1, n=2, t_end=2.0, order="bogus")


# ---------------------------------------------------------------- FR-D-2 adapters

@pytest.mark.parametrize("ad", ["adapter_construction_site10k", "adapter_isafetybench",
                                "adapter_fieldworkarena", "adapter_pasbench_video",
                                "adapter_ovo_bench_rt"])
def test_adapters_are_stubs(ad):                                # FR-D-2
    import ea.data as D
    with pytest.raises(NotImplementedError, match="VERIFY"):
        getattr(D, ad)()
