"""tests/test_hazard.py — latent-hazard track (design §5.14, FR-H-1..7).

Verification cards per FR-Q-1:
- FR-H-1: hazard sample fields + track skip with a counted reason.
- FR-H-2: secured twins in make_toy (pixel-identical mechanism, flipped lid).
- FR-H-3: hz_* prompt rendering, skips and gold/polarity conventions.
- FR-H-4: explain_cue three-valued table (incl. pitfall 24: a cue that pushes
  twins to "Yes" is bias ⇒ False); hz Q_P conjunction; hz_r exclusion test;
  signature-table assignment of "uncued-hazard under-reporting".
- FR-H-5: hazard_metrics recall + twin false-alarm + pair headline.
- FR-H-6: report caps the hazard cause at 'candidate' below min_agreement.
- FR-H-7: hz_report checklist scoring.
- End-to-end: a scripted FakeVLM that under-reports uncued hazards (and never
  fires on twins) is labelled through decide() via the full pipeline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import ea.report as RP
from ea.data import Sample, hazard_track_samples, is_hazard, make_toy, media_path
from ea.escalate import (Cluster, decide, explain_cue, hz_qp_ok_rate,
                         hz_qr_ok_rate, per_error_accounting)
from ea.metrics import hazard_metrics, hz_checklist
from ea.prompts import PROMPTS, render
from ea.run import errors_table, match, plan_hazard
from ea import run as R
from ea.escalate import escalate
from ea.report import write_report


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    S = make_toy(120, root=tmp_path_factory.mktemp("toy") / "toy", seed=3)
    assert sum(is_hazard(s) for s in S) >= 6, "toy must contain hazard samples"
    return S


def _pixel_count(img_path, color, tol=12):
    a = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    return int((np.abs(a - np.array(color)).sum(-1) <= tol).sum())


# ----------------------------------------------------------------- FR-H-1/2: toy twins + track validation

def test_hazard_toy_samples_and_twins(toy, tmp_path):            # FR-H-1/2
    S = toy
    hz = [s for s in S if is_hazard(s)]
    for s in hz:
        assert s.meta["vtype"] == "latent_hazard"
        assert s.meta["trigger"] and s.meta["consequence"]
        assert s.meta["hazard_gold"] == s.label
        assert s.meta.get("agreement") is not None               # FR-H-6 input
        cs = s.facts["conditions"]
        assert cs and all(set(c) >= {"obj", "state", "present"} for c in cs)
        twins = [t for t in S if t.pair_id == s.pair_id and t.id != s.id]
        assert twins, "every hazard sample has a secured twin (FR-H-2)"
        for t in twins:
            assert t.label != s.label and is_hazard(t)
            assert [(c["obj"], c["state"]) for c in t.facts["conditions"]] == \
                   [(c["obj"], c["state"]) for c in cs]          # same questions...
            assert all(c["present"] != p["present"]              # ...different truth
                       for c, p in zip(cs, t.facts["conditions"]))


def test_hazard_toy_pixels_read_hazard(toy):                     # FR-H-2 mechanism
    for s in toy:
        if not is_hazard(s):
            continue
        p = media_path(s, ".")
        open_px = _pixel_count(p, (230, 140, 30))
        closed_px = _pixel_count(p, (0, 150, 150))
        if s.label == 1:
            assert open_px > 20 and closed_px == 0               # open (orange) lid
        else:
            assert closed_px > 20 and open_px == 0               # closed (teal) lid


def test_hazard_track_skips_incomplete_with_reason():            # FR-H-1 AC
    good = dict(id="g", bench="toy", modality="image", media="g.png", label=1, cls="hz",
                facts={"conditions": [{"obj": "container", "state": "lid open",
                                       "present": "present"}]},
                meta={"vtype": "latent_hazard", "trigger": "wind",
                      "consequence": "the container falling"})
    no_trigger = dict(good, id="b1", meta={"vtype": "latent_hazard",
                                           "consequence": "the container falling"})
    no_conds = dict(good, id="b2", facts={})
    plain = dict(id="p", bench="toy", modality="image", media="p.png", label=0, cls="r")
    ok, skip = hazard_track_samples([Sample.from_json(d) for d in
                                     (good, no_trigger, no_conds, plain)])
    assert [s.id for s in ok] == ["g"]
    assert {i for i, _ in skip} == {"b1", "b2"}                  # 'plain' is not track material
    assert all(reason for _, reason in skip)                     # never silent (NFR-5)


# ----------------------------------------------------------------- FR-H-3: prompts

def test_hz_prompts_render_gold_and_skip(toy):                   # FR-H-3
    hz = next(s for s in toy if is_hazard(s) and s.label == 1)
    twin = next(t for t in toy if t.pair_id == hz.pair_id and t.id != hz.id)

    cue = render(PROMPTS["hz_cue"], hz)
    assert hz.meta["consequence"] in cue and hz.meta["trigger"] in cue
    r = render(PROMPTS["hz_r"], hz)
    assert "Facts:" in r and hz.meta["trigger"] in r
    s0 = render(PROMPTS["hz_s0"], hz)
    c0 = hz.facts["conditions"][0]
    assert c0["obj"] in s0 and c0["state"] in s0
    assert "NOT" in render(PROMPTS["hz_s0_abs"], hz)             # C7 second polarity

    # gold: conditions hold on the hazard sample, not on the twin; abs mirrored
    assert PROMPTS["hz_s0"]["gold"](hz) == 1 and PROMPTS["hz_s0"]["gold"](twin) == 0
    assert PROMPTS["hz_s0_abs"]["gold"](hz) == 0 and PROMPTS["hz_s0_abs"]["gold"](twin) == 1
    assert PROMPTS["hz_s0"]["gold"](hz) + PROMPTS["hz_s0_abs"]["gold"](hz) == 1
    # decision prompts: hazard_gold = label (FR-H-1)
    assert PROMPTS["hz_d"]["gold"](hz) == 1 and PROMPTS["hz_d"]["gold"](twin) == 0
    assert PROMPTS["hz_cue"]["gold"](hz) == 1 and PROMPTS["hz_cue"]["gold"](twin) == 0
    # hz_r derives gold from the TEXT facts, not the label (exclusion test)
    assert PROMPTS["hz_r"]["gold"](hz) == 1 and PROMPTS["hz_r"]["gold"](twin) == 0

    # plain samples: field-bearing hz prompts skip (FR-P-3); hz_d is static text
    plain = next(s for s in toy if not is_hazard(s))
    assert render(PROMPTS["hz_cue"], plain) is None
    assert render(PROMPTS["hz_r"], plain) is None
    assert render(PROMPTS["hz_s0"], plain) is None


# ----------------------------------------------------------------- FR-H-4: cue diagnostic + signature

CFG = dict(exit={"recovery": 0.5, "control": 0.1, "min_n": 10, "conf_margin": 1.0})


def _hz_runs(n_err=12, n_twin=40, cue_recovers=True, twin_yes=0.0,
             recover_half=False, seed=0):
    """Synthetic hazard runs: confident errors (base D margins -1.5), correct
    hz_s/hz_r rows, cued rows per scenario, twin HZ.cue_twin rows. n_twin=40
    so that 0 false alarms has Wilson upper ≈0.09 ≤ exit.control=0.1."""
    rng = np.random.default_rng(seed)
    rows = []

    def row(sid, prompt, arm, s, gold, group="HZ"):
        rows.append(dict(sample_id=sid, prompt_id=prompt, arm=arm, s=float(s),
                         pred=int(s > 0), gold=int(gold), correct=int((s > 0) == gold),
                         group=group, valid=True, knob=None, value=None))

    errs = [f"e{i}" for i in range(n_err)]
    twins = [f"t{i}" for i in range(n_twin)]
    for sid in errs:
        for p in ("d0", "d1", "d2"):
            s = -1.5 + rng.normal(0, 0.05)
            row(sid, p, "base", s, 1, "D")
        for p in ("hz_s0", "hz_s0_abs", "hz_s1", "hz_s1_abs", "hz_r"):
            row(sid, p, "base", 2.0, 1)
        cued = 3.0 if (not recover_half or rng.random() < 0.5) else -3.0
        row(sid, "hz_cue", "HZ.cue", cued if cue_recovers else -3.0, 1)
    for sid in twins:
        row(sid, "hz_cue", "HZ.cue_twin", 3.0 if rng.random() < twin_yes else -3.0, 0)
    return pd.DataFrame(rows), errs


def _cue_case(**kw):
    runs, errs = _hz_runs(**kw)
    c = Cluster(name="hz", err_ids=errs, cor_ids=[])
    return explain_cue(runs, c, CFG)["explained"]


def test_explain_cue_true_when_recovered_and_twins_silent():     # FR-H-4
    assert _cue_case(cue_recovers=True, twin_yes=0.0) is True


def test_explain_cue_false_when_not_recovered():                 # FR-H-4
    assert _cue_case(cue_recovers=False, twin_yes=0.0) is False


def test_explain_cue_false_when_cue_fires_on_twins():            # FR-H-4, pitfall 24
    # the cue pushes "Yes" on the SECURED twins: prompt bias, not awareness
    assert _cue_case(cue_recovers=True, twin_yes=1.0) is False


def test_explain_cue_none_when_underpowered_or_straddling():     # FR-H-4
    assert _cue_case(cue_recovers=True, twin_yes=0.0, n_err=4, n_twin=4) is None
    assert _cue_case(cue_recovers=True, twin_yes=0.0, recover_half=True) is None


def test_hz_qp_ok_is_a_conjunction_of_both_polarities():         # FR-H-4 (§5.14)
    runs, errs = _hz_runs(n_err=12)
    c = Cluster(name="hz", err_ids=errs, cor_ids=[])
    assert hz_qp_ok_rate(runs, errs, CFG)["explained"] is True
    # break ONE polarity row on 10 of 12 samples → share 2/12 → hi < 0.5 ⇒ False
    bad = errs[:10]
    m = (runs.prompt_id == "hz_s0_abs") & runs.sample_id.isin(bad)
    runs.loc[m, "correct"] = 0
    assert hz_qp_ok_rate(runs, errs, CFG)["explained"] is False


def test_hz_qr_ok_exclusion_test():                              # FR-H-4, FR-T2-6
    runs, errs = _hz_runs(n_err=12)
    c = Cluster(name="hz", err_ids=errs, cor_ids=[])
    assert hz_qr_ok_rate(runs, errs, CFG)["explained"] is True   # True is uninformative...
    m = runs.prompt_id == "hz_r"
    runs.loc[m & runs.sample_id.isin(errs[:10]), "correct"] = 0
    assert hz_qr_ok_rate(runs, errs, CFG)["explained"] is False  # ...only False is evidence


def test_signature_uncued_hazard_unique():                       # FR-H-4 AC (table-driven)
    ev = dict(crop=False, bgmask=False, hires=None, sharpen=False, knob=None,
              qr_ok=True, qp_ok=True, cue=True, necessity=None,
              consistency=None, loc=None, t4=None, causal=None, stream=None)
    cause, cands = decide(ev)
    assert (cause, cands) == ("uncued-hazard under-reporting", [])
    # and never without the cue
    cause2, _ = decide({**ev, "cue": None})
    assert cause2 != "uncued-hazard under-reporting"


# ----------------------------------------------------------------- FR-H-5: metrics

def test_hazard_metrics_recall_with_twin_fa_and_pairs():         # FR-H-5
    rows = []
    for i in range(10):
        rows.append(dict(sample_id=f"h{i}", prompt_id="hz_d", arm="base",
                         s=2.0, gold=1, pred=1, correct=1, pair_id=f"p{i}", valid=True))
        rows.append(dict(sample_id=f"t{i}", prompt_id="hz_d", arm="base",
                         s=-2.0, gold=0, pred=0, correct=1, pair_id=f"p{i}", valid=True))
    df = pd.DataFrame(rows)
    m = hazard_metrics(df, tau=0.0)
    assert m["recall"] == 1.0 and m["twin_fa"] == 0.0            # reported TOGETHER (FR-H-2)
    assert m["pair_acc"] == 1.0 and m["pair_rank_acc"] == 1.0
    assert m["cued_recall"] is np.nan or np.isnan(m["cued_recall"])
    inv = df.copy()
    inv["s"], inv["pred"], inv["correct"] = -inv["s"], 1 - inv["pred"], 0
    m2 = hazard_metrics(inv, tau=0.0)
    assert m2["recall"] == 0.0 and m2["pair_rank_acc"] == 0.0


# ----------------------------------------------------------------- FR-H-7: checklist

def test_hz_checklist_lenient_keywords():                        # FR-H-7 AC
    s = Sample.from_json(dict(id="x", bench="toy", modality="image", media="x.png",
                              label=1, cls="hz",
                              facts={"conditions": [{"obj": "container", "state": "lid open",
                                                     "present": "present"}]},
                              meta={"consequence": "the container falling"}))
    good = hz_checklist("I can see an open container lid; it could fall in the wind.",
                        s)
    assert good["object"] and good["condition"] and good["consequence"]
    assert good["complete"] is True
    bad = hz_checklist("The weather looks fine today.", s)
    assert not bad["complete"] and not bad["object"]


# ----------------------------------------------------------------- FR-H-6: agreement cap in the report

def _report_inputs(tmp_path, agreement):
    clusters = [dict(name="vtype=latent_hazard", n_errors=12, n_corrects=12,
                     cause="uncued-hazard under-reporting", cause_candidates=[],
                     tag=None, pred={"vtype": "latent_hazard"}, err_ids=[], cor_ids=[],
                     hazard=True, agreement=agreement,
                     cue=dict(test="HZ.cue", explained=True, twin_fa=0.0))]
    (tmp_path / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    pd.DataFrame(dict(sample_id=[f"e{i}" for i in range(12)],
                      cause=["uncued-hazard under-reporting"] * 12,
                      cluster_name=["vtype=latent_hazard"] * 12,
                      cluster_size=[12] * 12)).to_parquet(tmp_path / "diag_vector.parquet")
    # validated precision 0.95 would be 'decisive' — the agreement cap must win
    conf = pd.DataFrame({"precision": [0.95], "prec_lo": [0.7], "prec_hi": [1.0]},
                        index=["uncued-hazard under-reporting"])
    conf.to_csv(tmp_path / "confusion.csv")
    return tmp_path / "clusters.json", tmp_path / "diag_vector.parquet", tmp_path / "confusion.csv"


def test_report_caps_low_agreement_hazard_cause(tmp_path):       # FR-H-6
    cfg = {"validate": {"min_precision": 0.8}, "hazard": {"min_agreement": 0.6}}
    cp, dp, conf = _report_inputs(tmp_path, agreement=0.4)
    md, _ = write_report(cp, dp, tmp_path, validate_csv=conf, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "candidate (annotator agreement 0.40" in text
    assert "decisive" not in text.split("## Cluster details")[1].split("## Accounting")[0]
    assert "trigger is not visible" in text                      # FR-H-6 caveat


def test_report_high_agreement_hazard_not_capped(tmp_path):      # FR-H-6 boundary
    cfg = {"validate": {"min_precision": 0.8}, "hazard": {"min_agreement": 0.6}}
    cp, dp, conf = _report_inputs(tmp_path, agreement=0.9)
    md, _ = write_report(cp, dp, tmp_path, validate_csv=conf, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "annotator agreement 0.90" not in text


# ----------------------------------------------------------------- end-to-end (§5.14)

class HazardFakeVLM:
    """Scripted VLM (§7.2 style) with a planted uncued-hazard mechanism.

    Reads toy pixels: orange raised lid = hazard, teal flat lid = secured
    twin; no marker = ordinary PPE scene (red/blue = violation, green = ok).
    Mechanism: hazards are reported only when the consequence is named (cued);
    twins never fire; everything else answers the square semantics correctly.
    """

    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        img = (images or [None])[0]
        hz = _hazard_state(img)
        t = text.lower()
        base = dict(mass=0.95, n_vis=64, ms=1.0, mem=0.0)
        if "foreseeably lead to an incident" in t:               # hz_d: uncued → under-report
            return dict(s=-3.0, **base)
        if "happen here" in t:                                   # hz_cue: recovered, twins silent
            return dict(s=(3.0 if hz == 1 else -3.0), **base)
        if "facts:" in t and "could" in t:                       # hz_r: oracle reasoning, correct
            return dict(s=(-2.0 if "absent" in t else 2.0), **base)
        if hz is not None:
            q = t.split("answer")[0]
            if " lid " in q or " unsupported" in q:              # hz_s<k>(_abs): reads the lid
                holds, negated = (hz == 1), " not " in q
                return dict(s=(2.0 if holds != negated else -2.0), **base)
            return dict(s=-3.0, **base)                          # hazard scenes under-report uncued
        if img is None:                                          # text-only rows (qr)
            return dict(s=2.0, **base)
        return dict(s=_square_margin(img), **base)


def _hazard_state(img):
    if img is None:
        return None
    a = np.asarray(img.convert("RGB")).astype(int)
    if (np.abs(a - np.array([230, 140, 30])).sum(-1) <= 30).any():
        return 1
    if (np.abs(a - np.array([0, 150, 150])).sum(-1) <= 30).any():
        return 0
    return None


def _square_margin(img):
    a = np.asarray(img.convert("RGB")).astype(int)

    def px(c):
        return int((np.abs(a - np.array(c)).sum(-1) <= 40).sum())

    if px((200, 30, 30)) > 20 or px((30, 60, 200)) > 20:         # red evidence / blue absence border
        return 2.0
    return -2.0                                                  # green square / blank


def test_end_to_end_hazard_under_reporting(tmp_path):
    S = make_toy(120, root=tmp_path / "toy", seed=3)
    cfg = {"model": {"name": "fake"},
           "answer": {"min_mass": 0.5},
           # control=0.35: with only ~8 twins, the Wilson upper bound of 0 false
           # alarms is ≈0.32 — cue=True is otherwise unreachable at this n. The
           # strict 0.1 behaviour is covered by the table-driven tests above.
           "exit": {"recovery": 0.5, "control": 0.35, "min_n": 5, "conf_margin": 1.0},
           "slices": {"max_order": 1, "min_support": 8, "fdr": 0.05},
           "audit": {"k_per_cluster": 5},
           "validate": {"min_precision": 0.8, "per_cause_n": 10},
           "knobs": {"quality": {"kinds": ["blur"], "severities": [1]}}}

    def factory(mcfg):
        return HazardFakeVLM()

    runs_path = tmp_path / "runs.jsonl"
    R.run(R.plan_t0(cfg, S), S, cfg, out=runs_path, vlm_factory=factory)
    runs = R.load_runs(runs_path)
    et = errors_table(runs, S)
    err, cor = et[et.err == 1], et[et.err == 0]
    assert (err.vtype == "latent_hazard").sum() >= 5             # hazard errors exist
    matched, _ = match(err, cor)
    E = set(err.sample_id) | set(matched)

    R.run(R.plan_t1(cfg, S, E), S, cfg, out=runs_path, vlm_factory=factory)
    R.run(R.plan_t2(cfg, S, E), S, cfg, out=runs_path, vlm_factory=factory)
    R.run(plan_hazard(cfg, S, E), S, cfg, out=runs_path, vlm_factory=factory)
    runs = R.load_runs(runs_path)

    # FR-H-5 metrics on the raw runs: recall 0 uncued, 1 cued, twins silent.
    # The under-reporting model ties hazards with their twins, so the headline
    # pair ranking is 0 (hazards never ranked above their secured twins).
    m = hazard_metrics(runs, S, tau=0.0)
    assert m["recall"] == 0.0 and m["cued_recall"] == 1.0
    assert m["twin_fa"] == 0.0
    assert m["pair_rank_acc"] == 0.0 and m["pair_acc"] == 0.0

    clusters, diag_vec = escalate(runs, S, cfg, out_dir=tmp_path / "esc")
    hz_clusters = [c for c in clusters if c.hazard]
    assert hz_clusters, "a latent-hazard cluster must be built"
    for c in hz_clusters:
        assert c.diags["cue"] is True, f"cue must be True for {c.name}"
        assert c.cause == "uncued-hazard under-reporting"
        assert c.cause_candidates == []

    # per-error accounting: every hazard error carries the hazard cause
    hz_err_ids = set(err[err.vtype == "latent_hazard"].sample_id)
    got = diag_vec[diag_vec.sample_id.isin(hz_err_ids)]
    assert (got.cause == "uncued-hazard under-reporting").mean() > 0.5
    assert len(diag_vec) == len(err)                             # FR-E-11 identity

    md, _ = write_report(clusters_path=tmp_path / "esc" / "clusters.json",
                         diag_vec_path=tmp_path / "esc" / "diag_vector.parquet",
                         out_dir=tmp_path / "esc", cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "uncued-hazard under-reporting" in text
    assert "trigger is not visible" in text
