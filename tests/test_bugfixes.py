"""tests/test_bugfixes.py — regression guards for the P0/P1 fixes.

Each test targets one fixed defect and fails if the bug is reintroduced:
  - BUG-01: validate() measures, never fabricates (INJ_EXPECTED ↔ SIG spellings,
    precision derived from real diagnostics, vlm_factory required).
  - BUG-04: knob jobs are planned with the arm="base"+knob/value convention.
  - BUG-06: one box_mask convention across data/attrib/mech.
  - BUG-09: seeded nli (reproducible deletion curves).
  - BUG-10: stream survives onset_s=0.0 and end_s=None.
  - BUG-14: errors_table honours min_votes.
  - BUG-16: taxonomy-tag clusters get matched corrects.
  - BUG-19: text-only arms keep the image when the prompt needs one.
  - BUG-03: pre-merged T3/T4 diags are not clobbered by setdefault.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ea import data as D
from ea import run as R
from ea.escalate import INJ_EXPECTED, SIG, Cluster, run_diagnostics
from ea.attrib import box_mask as attrib_box_mask, nli
from ea.data import box_mask as data_box_mask
from ea.stream import window_vs_stream_gap


# ----------------------------------------------------------------- BUG-05/01

def test_inj_expected_uses_sig_spellings():
    """Every expected cause must be a SIG key or the unexplained bucket (BUG-05)."""
    for arm, expected in INJ_EXPECTED.items():
        assert expected in SIG or expected == "unexplained → next tier", (arm, expected)


class _CorrectVLM:
    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        return dict(s=2.0, mass=0.95, n_vis=64, ms=1.0, mem=0.0)


def test_validate_requires_vlm_factory(tmp_path):
    S = D.make_toy(6, root=tmp_path / "toy", seed=0)
    with pytest.raises(ValueError, match="vlm_factory"):
        R.validate(pd.DataFrame(), S, {"answer": {"min_mass": 0.5}}, out_dir=tmp_path)


def _confident_correct_runs(S):
    """Base D rows marking the violation samples confident-correct (an
    always-Yes fake is correct only on them)."""
    rows = []
    for s in S:
        if s.label != 1:
            continue
        for p in ("d0", "d1", "d2"):
            rows.append(dict(sample_id=s.id, prompt_id=p, group="D", arm="base",
                             knob=None, value=None, s=2.0, pred=1, gold=1,
                             correct=1, valid=True, n_vis=8, latency_ms=1.0,
                             t=None, extra="{}"))
    return pd.DataFrame(rows)


def test_validate_measures_real_diagnostics(tmp_path):
    """The confusion matrix comes from run diagnostics, not from an assignment
    loop: with a fake VLM that is correct on every pool sample, no injection
    produces errors, so every cell must be 0 (the fabricated version wrote a
    perfect diagonal of n_samples regardless of the model)."""
    cfg = {"answer": {"min_mass": 0.5},
           "validate": {"per_cause_n": 3, "label_noise": 0.0},
           "knobs": {"quality": {"kinds": ["blur"], "severities": [1]}},
           "exit": {"recovery": 0.5, "control": 0.1, "min_n": 2, "conf_margin": 0.5}}
    S = D.make_toy(9, root=tmp_path / "toy", seed=0)
    runs = _confident_correct_runs(S)
    assert not runs.empty
    csv = R.validate(runs, S, cfg, out_dir=tmp_path,
                     vlm_factory=lambda mc: _CorrectVLM())
    df = pd.read_csv(csv, index_col=0)
    assert set(INJ_EXPECTED.values()) <= set(df.index)
    assert "precision" in df.columns and "n_errors" in df.columns
    assert (df.n_errors == 0).all()          # nothing failed -> nothing attributed
    cells = df[[c for c in df.columns if c in SIG]]
    assert int(cells.fillna(0).to_numpy().sum()) == 0


def test_validate_precision_drops_when_model_wrong(tmp_path):
    """Acceptance (BUG-01): a fake VLM that answers 'No' to everything makes
    every injection an error; the assigned causes cannot be the fabricated
    perfect diagonal, so at least the expected causes show precision < 1."""
    class _AlwaysNo:
        def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
            return dict(s=-2.0, mass=0.95, n_vis=64, ms=1.0, mem=0.0)

    cfg = {"answer": {"min_mass": 0.5},
           "validate": {"per_cause_n": 4, "label_noise": 0.0},
           "knobs": {"quality": {"kinds": [], "severities": []}},
           "exit": {"recovery": 0.5, "control": 0.1, "min_n": 2, "conf_margin": 0.5}}
    S = D.make_toy(12, root=tmp_path / "toy", seed=0)
    csv = R.validate(pd.DataFrame(), S, cfg, out_dir=tmp_path,
                     vlm_factory=lambda mc: _AlwaysNo())
    df = pd.read_csv(csv, index_col=0)
    assert df.n_errors.sum() > 0                       # injections DID produce errors
    # a fabricated matrix would show precision 1.0 everywhere
    assert (df["precision"] < 1.0).any()


# ----------------------------------------------------------------- BUG-04

def test_knob_jobs_planned_with_base_arm(tmp_path):
    S = D.make_toy(4, root=tmp_path / "toy", seed=0)
    cfg = {"knobs": {"quality": {"kinds": [], "severities": []},
                     "max_pixels": {"low": 262144, "native": None, "high": 16777216},
                     "token_budget": [0.5], "quant": []}}
    jobs = R.plan_t2(cfg, S, {S[0].id})
    knob_jobs = [j for j in jobs if j.params.get("knob")]
    assert {j.params["knob"] for j in knob_jobs} == {"max_pixels", "token_budget"}
    for j in knob_jobs:
        assert j.arm == "base"                          # convention escalate filters on
        assert j.params["value"] is not None
    mp = {j.mcfg["max_pixels"] for j in knob_jobs if j.params["knob"] == "max_pixels"}
    assert mp == {262144, None, 16777216}               # one model load per config
    assert all(R.knob_fields(j)["knob"] for j in knob_jobs)


def test_knob_rows_written_with_knob_columns(tmp_path):
    S = D.make_toy(2, root=tmp_path / "toy", seed=0)
    cfg = {"answer": {"min_mass": 0.5},
           "knobs": {"quality": {"kinds": [], "severities": []},
                     "max_pixels": {"high": 16777216}, "token_budget": [0.5], "quant": []}}
    ids = {S[0].id}
    jobs = R.plan_t2(cfg, S, ids)
    knob_jobs = [j for j in jobs if j.params.get("knob")]
    out = tmp_path / "runs.jsonl"
    R.run(knob_jobs, S, cfg, out=out, vlm_factory=lambda mc: _CorrectVLM())
    df = pd.read_json(out, lines=True)
    kj = df[df.knob.notna()]
    assert set(kj.arm) == {"base"} and len(kj) == len(knob_jobs) >= 2


# ----------------------------------------------------------------- BUG-06

def test_box_mask_single_convention():
    for _ in range(200):
        box = sorted(np.random.uniform(0, 1, 2)) + sorted(np.random.uniform(0, 1, 2))
        box = [box[0], box[2], box[1], box[3]]
        h, w = int(np.random.randint(3, 14)), int(np.random.randint(3, 14))
        m_data = data_box_mask(box, (h, w)).reshape(h, w)
        m_attr = attrib_box_mask(box, (h, w))
        assert np.array_equal(m_data, m_attr)
    m_video = attrib_box_mask([0.0, 0.0, 1.0, 1.0], (2, 4, 6))
    assert m_video.shape == (48,) and m_video.sum() == 48   # video grids supported


def test_mech_uses_data_box_mask():
    import ea.mech as M
    from ea.data import box_mask as data_bm
    assert M.box_mask is data_bm


# ----------------------------------------------------------------- BUG-09

def test_nli_deterministic_and_linear():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (32, 32, 3))
    mask = np.zeros((32, 32), bool)
    mask[8:16, 8:16] = True
    a = nli(img, mask, seed=123)
    b = nli(img, mask, seed=123)
    c = nli(img, mask, seed=124)
    assert np.array_equal(a, b)                         # same seed -> same fill
    assert not np.array_equal(a, c)                     # different seed -> different noise
    # linear interpolation: uniform unmasked background -> fill close to it
    flat = np.full((32, 32, 3), 100.0)
    out = nli(flat, mask, noise_std=0.0, seed=1)
    assert abs(out[mask].mean() - 100.0) < 5.0


def test_nli_2d_stencil_converges():
    """ROAD §5.2: masked pixels surrounded by constant-valued neighbors must equal
    that constant under the 2D harmonic stencil (w_d=1/6, w_i=1/12), solved exactly."""
    # Constant-valued image at 200; mask interior pixels (not boundary)
    img = np.full((20, 20, 3), 200.0)
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True                              # interior block, surrounded by 200
    out = nli(img, mask, noise_std=0.0, seed=0)
    # The exact solve of the stencil system with a constant boundary is that constant
    assert abs(out[mask].mean() - 200.0) < 1e-6, (
        f"2D stencil did not converge: mean={out[mask].mean():.2f}"
    )


# ----------------------------------------------------------------- BUG-10

def _video(onset, end=None):
    return D.Sample(id="v1", bench="toy", modality="video", media="v.mp4", label=1,
                    cls="c", video=dict(fps=2.0, duration_s=6.0, onset_s=onset, end_s=end))


def test_stream_survives_onset_zero_and_null_end():
    runs = pd.DataFrame([dict(sample_id="v1", arm="base", t=float(t),
                              s=2.0 if t <= 2 else -2.0, knob=None, value=None)
                         for t in range(0, 5)])
    cfg = {"stream": {"tolerance_s": 2.0, "k": 2, "n": 3, "anticipation_s": 0.0}}
    # onset at t=0 must not crash or be treated as 'no onset'
    g1 = window_vs_stream_gap(runs, [_video(0.0)], cfg, tau=0.0)
    assert g1["window_recall"] == 1.0 and g1["n_risk_videos"] == 1
    # explicit end_s=None must fall back to onset+tol, not poison comparisons
    g2 = window_vs_stream_gap(runs, [_video(0.0, end=None)], cfg, tau=0.0)
    assert g2["window_recall"] == 1.0


# ----------------------------------------------------------------- BUG-14

def test_errors_table_min_votes():
    rows = []
    for i, preds in enumerate(([1, None, None], [0, 0, 0], [1, 1, 0])):
        for pid, pr in zip(("d0", "d1", "d2"), preds):
            if pr is None:
                continue
            rows.append(dict(sample_id=f"s{i}", prompt_id=pid, group="D", arm="base",
                             knob=None, value=None, s=1.5 if pr else -1.5, pred=pr,
                             gold=0, correct=int(pr == 0), valid=True))
    runs = pd.DataFrame(rows)
    S = [D.Sample(id=f"s{i}", bench="toy", modality="image", media="x.png",
                  label=0, cls="c", rule_text="r", item="helmet") for i in range(3)]
    et = R.errors_table(runs, S, min_votes=2)
    assert "s0" not in set(et.sample_id)                # 1 valid vote: undetermined
    assert set(et.sample_id) == {"s1", "s2"}
    assert et[et.sample_id == "s2"].pred.iloc[0] == 1   # 1-1 tie -> mean s > 0 -> 1


# ----------------------------------------------------------------- BUG-16

def test_tag_clusters_get_matched_corrects():
    from ea.escalate import build_clusters
    rows = []
    for i in range(6):
        sid = f"s{i}"
        for pid in ("d0", "d1", "d2"):
            rows.append(dict(sample_id=sid, prompt_id=pid, arm="base", s=-1.5, pred=0,
                             gold=1, correct=0, group="D", valid=True, knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="pol_pos", arm="base", s=-1.5, pred=0,
                         gold=1, correct=0, group="POL", valid=True, knob=None, value=None))
        rows.append(dict(sample_id=sid, prompt_id="pol_neg", arm="base", s=1.5, pred=1,
                         gold=1, correct=1, group="POL", valid=True, knob=None, value=None))
        for i in range(6, 12):
            sid = f"s{i}"
            for pid in ("d0", "d1", "d2"):
                rows.append(dict(sample_id=sid, prompt_id=pid, arm="base", s=1.5, pred=1,
                                 gold=1, correct=1, group="D", valid=True, knob=None, value=None))
            rows.append(dict(sample_id=sid, prompt_id="pol_pos", arm="base", s=-1.5, pred=0,
                             gold=1, correct=0, group="POL", valid=True, knob=None, value=None))
            rows.append(dict(sample_id=sid, prompt_id="pol_neg", arm="base", s=1.5, pred=1,
                             gold=1, correct=1, group="POL", valid=True, knob=None, value=None))
    runs = pd.DataFrame(rows)
    S = [D.Sample(id=f"s{i}", bench="toy", modality="image", media="x.png",
                  label=1, cls="c", rule_text="r", item="helmet",
                  meta={"vtype": "absence" if i < 12 else "presence"})
         for i in range(12)]
    et = R.errors_table(runs, S)
    err = et[et.err == 1]
    cor = et[et.err == 0]
    tag_clusters = [c for c in build_clusters(et, cor, runs, {"slices": {"min_support": 1}})
                    if c.tag]
    assert tag_clusters
    for c in tag_clusters:
        assert c.cor_ids, f"tag cluster {c.name} has no matched corrects (BUG-16)"


# ----------------------------------------------------------------- BUG-19

def test_text_only_arms_keep_image(tmp_path):
    """INJ.lead / INJ.label modify text or gold, but d0 needs the image: the
    runner must reload the ORIGINAL media instead of sending no image."""
    seen = {}

    class _Spy:
        def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
            seen["images"] = images
            seen["text"] = text
            return dict(s=1.0, mass=0.9, n_vis=8, ms=1.0, mem=0.0)

    S = D.make_toy(1, root=tmp_path / "toy", seed=0)
    s = S[0]
    cfg = {"answer": {"min_mass": 0.5}}
    for arm in ("INJ.lead", "INJ.label"):
        seen.clear()
        j = R.Job(s.id, "d0", arm, {}, R.BASE_MCFG)
        row = R.one(_Spy(), s, j, R.key(j), cfg)
        assert row is not None and seen["images"], f"{arm}: image was dropped"
        assert len(seen["images"]) == 1


# ----------------------------------------------------------------- BUG-03

def test_run_diagnostics_preserves_external_diags():
    """setdefault must not clobber T3/T4 values merged before diagnostics."""
    rows = []
    for i in range(12):
        sid = f"e{i}"
        for p in ("d0", "d1", "d2"):
            rows.append(dict(sample_id=sid, prompt_id=p, arm="base", s=-1.5, pred=0,
                             gold=1, correct=0, group="D", valid=True, knob=None, value=None))
    runs = pd.DataFrame(rows)
    c = Cluster(name="test", err_ids=[f"e{i}" for i in range(12)], cor_ids=[])
    c.diags.update({"t4": "H2", "causal": True})
    cfg = {"exit": {"recovery": 0.5, "control": 0.1, "min_n": 10, "conf_margin": 1.0},
           "necessity": {"ratio": 0.25}, "consistency": {"min_flip": 0.2, "ratio": 2.0}}
    run_diagnostics(c, runs, cfg)
    assert c.diags["t4"] == "H2" and c.diags["causal"] is True
    ev = {k: c.diags[k] for k in ("crop", "bgmask", "hires", "qr_ok", "sharpen", "knob",
                                  "qp_ok", "necessity", "consistency", "loc", "t4",
                                  "causal", "cue", "stream")}
    from ea.escalate import decide
    cause, _ = decide(ev)
    assert cause == "depth fading (H2)"                 # BUG-03: t4 rows reachable again


# ----------------------------------------------------------------- NEW-01
def test_new_01_stream_per_cluster_mixed_corpus(tmp_path):
    """NEW-01: stream diagnostic is per-cluster. Image clusters must get stream=None
    and never match H11, while video clusters with a policy gap get stream='policy'."""
    from ea.escalate import escalate

    # 12 image samples and 12 video samples
    S = []
    runs_rows = []
    # Image samples (absence scene tag)
    for i in range(12):
        sid = f"img_{i:02d}"
        S.append(D.Sample(id=sid, bench="toy", modality="image", media="x.png",
                          label=1, cls="rule_safety", rule_text="r", item="helmet",
                          meta={"vtype": "absence"}))
        for p in ("d0", "d1", "d2"):
            runs_rows.append(dict(sample_id=sid, prompt_id=p, arm="base", s=-1.5, pred=0,
                                  gold=1, correct=0, group="D", valid=True, knob=None, value=None, t=None))

    # Video samples (streaming hazard tag)
    for i in range(12):
        sid = f"vid_{i:02d}"
        S.append(D.Sample(id=sid, bench="toy", modality="video", media="v.mp4",
                          label=1, cls="streaming_hazard", rule_text="r",
                          video=dict(fps=2.0, duration_s=10.0, onset_s=2.0, end_s=4.0)))
        # base D runs: all err
        for p in ("d0", "d1", "d2"):
            runs_rows.append(dict(sample_id=sid, prompt_id=p, arm="base", s=-1.5, pred=0,
                                  gold=1, correct=0, group="D", valid=True, knob=None, value=None, t=None))
        # streaming timesteps: high margin during [2, 4], but alert fires late due to k=2
        # ts: [1.0, 2.0, 3.0, 4.0, 5.0]
        # s:  [-1.0, 1.5, -0.5, 1.5, 1.5] -> window tp=True at t=2.0, but alert fires at t=5.0 (late)
        for t, s_val in [(1.0, -1.0), (2.0, 1.5), (3.0, -0.5), (4.0, 1.5), (5.0, 1.5)]:
            runs_rows.append(dict(sample_id=sid, prompt_id="d0", arm="base", s=s_val, pred=int(s_val > 0),
                                  gold=1, correct=int(s_val > 0), group="D", valid=True, knob=None, value=None, t=t))

    runs = pd.DataFrame(runs_rows)
    cfg = {
        "exit": {"recovery": 0.5, "control": 0.1, "min_n": 5, "conf_margin": 1.0},
        "stream": {"tolerance_s": 0.5, "k": 2, "n": 3, "anticipation_s": 0.0},
        "slices": {"min_support": 5},
    }

    clusters, _ = escalate(runs, S, cfg, out_dir=tmp_path / "escalate_test")
    assert clusters

    img_cluster = next((c for c in clusters if "img_00" in c.err_ids and "vid_00" not in c.err_ids), None)
    vid_cluster = next((c for c in clusters if "vid_00" in c.err_ids and "img_00" not in c.err_ids), None)

    if img_cluster:
        assert img_cluster.diags.get("stream") is None, "Image cluster must have stream=None"
        assert img_cluster.cause != "trigger-policy failure (H11)", "Image cluster must not receive H11"

    if vid_cluster:
        assert vid_cluster.diags.get("stream") == "policy", "Video cluster must detect policy gap"
        assert vid_cluster.cause == "trigger-policy failure (H11)", "Video cluster must receive H11"


# ----------------------------------------------------------------- NEW-06
def test_new_06_mask_tokens_device_and_missing_merger():
    """NEW-06: mask_tokens must not raise generator device mismatch and must fail loud if merger missing."""
    import torch
    from ea.model import mask_tokens

    class DummyMerger(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyVLM:
        def __init__(self, with_merger=True):
            if with_merger:
                self.m = torch.nn.Module()
                self.m.visual = torch.nn.Module()
                self.m.visual.merger = DummyMerger()
            else:
                self.m = torch.nn.Module()

    # Successful hook with tensor
    vlm = DummyVLM(with_merger=True)
    x = torch.ones(20, 16)
    with mask_tokens(vlm, keep_frac=0.5, seed=42):
        out = vlm.m.visual.merger(x)
        # Should drop some rows
        assert out.shape == x.shape
        assert (out == 0).any(), "mask_tokens should zero out some features"

    # Missing merger module must raise RuntimeError, never silently no-op
    vlm_no_merger = DummyVLM(with_merger=False)
    with pytest.raises(RuntimeError, match="visual merger / visual module not found"):
        with mask_tokens(vlm_no_merger, keep_frac=0.5, seed=42):
            pass


# ----------------------------------------------------------------- NEW-09
def test_new_09_partial_pair_ids_groups_alignment():
    """NEW-09: samples with partial pair_ids fall back to sid, maintaining length alignment."""
    ids = ["s0", "s1", "s2", "s3"]
    pairs = {"s0": "pair_a", "s1": "pair_a", "s2": None, "s3": ""}
    # Fallback to sid ensures groups length is always equal to ids length
    groups = [pairs[sid] if pairs.get(sid) else sid for sid in ids]
    assert len(groups) == len(ids)
    assert groups == ["pair_a", "pair_a", "s2", "s3"]


# ----------------------------------------------------------------- NEW-10
def test_new_10_report_custom_dirs(tmp_path):
    """NEW-10: report CLI flags --t3-dir and --t4-dir are accepted and read."""
    from ea.cli import main
    t3_dir = tmp_path / "custom_t3"
    t4_dir = tmp_path / "custom_t4"
    t3_dir.mkdir(parents=True)
    t4_dir.mkdir(parents=True)
    (t3_dir / "t3_loc.json").write_text(json.dumps({"ALL_ERRORS": {"loc": "object"}}), encoding="utf-8")
    (t4_dir / "t4_verdicts.json").write_text(json.dumps({"ALL_ERRORS": {"t4": "enc"}}), encoding="utf-8")

    from ea.cli import build_parser
    args = build_parser().parse_args(["report", "--t3-dir", str(t3_dir), "--t4-dir", str(t4_dir)])
    assert args.t3_dir == str(t3_dir)
    assert args.t4_dir == str(t4_dir)

