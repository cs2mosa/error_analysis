"""escalate.py — cluster builder, three-valued exit rule, signature table, per-error accounting.

Purpose:  FR-E-1..12: build error clusters from slice predicates and taxonomy
          tags; compute three-valued diagnostics (explain_recover,
          explain_necessity, explain_consistency, qp_ok_rate, decide); assign
          cause via the literal signature table (SIG); perform per-error
          accounting (C17); generate the audit sample. Latent-hazard clusters
          (§5.14) use the hz_* diagnostics (hz_qp_ok_rate, hz_qr_ok_rate,
          explain_cue with the secured-twin false-alarm gate, FR-H-4).
Implements plan sections: §5.8, §5.14, §7; C1, C4, C5, C6, C17, C19, C20.
Inputs:   runs DataFrame (runs.jsonl), list[Sample], config dict.
Outputs:  list[Cluster] annotated with diagnostics and cause;
          diag_vector.parquet (one row per error); clusters.json;
          audit_sample.csv.

Measures: cause via matched-control recovery, necessity KL, consistency flips.
Does not measure: T3/T4 diagnostics (loc, t4, causal, stream) — set
          externally; until then they are absent from ev, so SIG rows that
          require them stay inconclusive rather than falsely matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from ea.metrics import boot_ci, flip_rate, margin_shift, visual_necessity
from ea.slices import COLUMNS, slice_search
from ea.triage import gate_ok


# ----------------------------------------------------------------- data types

@dataclass
class Cluster:
    """A set of error samples sharing a predicate or taxonomy tag (FR-E-1).

    err_ids: sample_ids whose majority-vote D-group decision is wrong.
    cor_ids: matched correct sample_ids (same cls/obj_size_bin, not the error).
    pred:    column→value predicate (empty for ALL_ERRORS or tag clusters).
    tag:     taxonomy tag string or None for slice-based clusters.
    hazard:  True when every error in the cluster is a latent-hazard sample
             (§5.14) — switches Q_P/Q_R to the hz_* prompts and adds `cue`.
    agreement: mean hazard annotator agreement of the errors (FR-H-6) or None.
    diags:   diagnostic results filled by run_diagnostics(); keyed by test name.
    cause:   final assigned cause (from decide()); starts as unexplained.
    cause_candidates: partial matches returned by decide() for ambiguous cases.
    """
    name: str
    err_ids: list[str]
    cor_ids: list[str]
    pred: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None
    hazard: bool = False
    agreement: float | None = None
    triage_counts: dict[str, int] = field(default_factory=dict)
    diags: dict[str, Any] = field(default_factory=dict)
    cause: str = "unexplained → next tier"
    cause_candidates: list[str] = field(default_factory=list)


# ----------------------------------------------------------------- config helpers

def _exit(cfg: dict):
    e = cfg.get("exit", {})
    return (
        float(e.get("recovery", 0.5)),
        float(e.get("control", 0.1)),
        int(e.get("min_n", 10)),
        float(e.get("conf_margin", 1.0)),
    )


def _nec_ratio(cfg):
    return float(cfg.get("necessity", {}).get("ratio", 0.25))


def _cons(cfg):
    c = cfg.get("consistency", {})
    return float(c.get("min_flip", 0.2)), float(c.get("ratio", 2.0))


# ----------------------------------------------------------------- Wilson CI + base helpers

def wil(k: int, n: int) -> tuple[float, float]:
    """Wilson CI for k successes in n trials (returns (0, 1) when n=0)."""
    if n == 0:
        return 0.0, 1.0
    lo, hi = proportion_confint(k, n, method="wilson")
    return float(lo), float(hi)


def conf_ids(runs: pd.DataFrame, ids: list[str], conf_margin: float) -> list[str]:
    """Confident error IDs: |mean s over base D rows| ≥ conf_margin (C5).

    Measures: samples whose base-arm margin magnitude is large enough to treat
    as confident predictions (not borderline fence-sitters).
    Does not measure: whether the prediction is correct.
    """
    b = runs[(runs.arm == "base") & (runs.group == "D") & runs.sample_id.isin(ids)
             & runs.valid.fillna(True)].groupby("sample_id").s.mean()
    return list(b.index[b.abs() >= conf_margin])


def _recovery_mask(runs: pd.DataFrame, ids: list[str], arm) -> pd.Series:
    """Per-sample majority-correct under arm; arm may be str or (knob, value)."""
    if isinstance(arm, tuple):
        knob, value = arm
        r = runs[(runs.arm == "base") & (runs.knob == knob) & (runs.value == value)
                 & runs.sample_id.isin(ids) & runs.valid.fillna(True)]
    else:
        r = runs[(runs.arm == arm) & runs.sample_id.isin(ids) & runs.valid.fillna(True)]
    return r.groupby("sample_id").correct.mean().ge(0.5)


def recovery(runs: pd.DataFrame, ids: list[str], arm) -> tuple[int, int]:
    """(k, n): samples in ids that are majority-correct under arm."""
    v = _recovery_mask(runs, ids, arm)
    return int(v.sum()), len(v)


def breakage(runs: pd.DataFrame, ids: list[str], arm) -> tuple[int, int]:
    """Matched-correct breakage: n - recovered (applied to cor_ids)."""
    k, n = recovery(runs, ids, arm)
    return n - k, n


# ----------------------------------------------------------------- explain_recover (FR-E-3/4)

# Below this visual-necessity KL (nats) the corrects' margin barely moves under
# the arm, so VN_err/VN_cor is a 0/0-style ratio; it must not be trusted.
_VN_FLOOR = 1e-3

def explain_recover(
    runs: pd.DataFrame,
    c: Cluster,
    arm,
    ctls: list,
    cfg: dict,
) -> dict:
    """Three-valued exit rule for one intervention arm (FR-E-3).

    Measures: whether arm recovers confident errors beyond the worst control.
    Does not measure: whether the recovery holds across all sample subgroups.

    Returns dict with keys: test, rec, ci, ctl_rate, brk, dshift, n, explained.
      explained = True  → arm recovers (lo ≥ exit.recovery ∧ ctl hi ≤ exit.control)
      explained = False → arm does not recover (hi < exit.recovery ∨ ctl lo > exit.control)
      explained = None  → inconclusive (n < min_n or CIs straddle thresholds)
    """
    rec_thr, ctl_thr, min_n, conf_margin = _exit(cfg)
    ids = conf_ids(runs, c.err_ids, conf_margin)
    k, n = recovery(runs, ids, arm)
    lo, hi = wil(k, n)

    cs = [recovery(runs, ids, a) for a in ctls]
    # worst control = highest recovery rate (most lenient baseline)
    ctl_rates = [x[0] / max(x[1], 1) for x in cs]
    ctl_lo = max(wil(*x)[0] for x in cs) if cs else 0.0
    ctl_hi = max(wil(*x)[1] for x in cs) if cs else 0.0

    if n < min_n or (cs and min(x[1] for x in cs) < min_n):
        explained: bool | None = None
    elif lo >= rec_thr and ctl_hi <= ctl_thr:
        explained = True
    elif hi < rec_thr or ctl_lo > ctl_thr:
        explained = False
    else:
        explained = None

    ms, _, _ = margin_shift(runs, ids, arm, ci=False)

    brk = breakage(runs, c.cor_ids, arm)
    return dict(
        test=str(arm), rec=k / max(n, 1), ci=(lo, hi),
        ctl_rate=max(ctl_rates) if ctl_rates else np.nan,
        brk=brk, dshift=ms, n=n, explained=explained,
    )


# ----------------------------------------------------------------- explain_necessity (FR-E-4)

def _vn(runs: pd.DataFrame, ids: list[str], arm) -> float:
    """Visual necessity KL between base and arm margin distributions."""
    b = (runs[(runs.arm == "base") & runs.sample_id.isin(ids) & runs.valid.fillna(True)]
         .groupby("sample_id").s.mean())
    if isinstance(arm, tuple):
        knob, value = arm
        a = (runs[(runs.arm == "base") & (runs.knob == knob) & (runs.value == value)
                  & runs.sample_id.isin(ids) & runs.valid.fillna(True)]
             .groupby("sample_id").s.mean())
    else:
        a = (runs[(runs.arm == arm) & runs.sample_id.isin(ids) & runs.valid.fillna(True)]
             .groupby("sample_id").s.mean())
    j = b.rename("s_b").to_frame().join(a.rename("s_a")).dropna()
    return visual_necessity(j.s_b.values, j.s_a.values) if not j.empty else np.nan


def explain_necessity(runs: pd.DataFrame, c: Cluster, cfg: dict) -> dict:
    """O3: image-insensitive errors ⟹ prior-driven (FR-E-4).

    Measures: whether errors are equally wrong with and without the image
    (blanked and evidence-occluded) relative to matched corrects — gated by
    the perception/knowledge gate so perception failures are not mislabelled.
    Does not measure: the specific prior content; see T1 framing arms for that.
    """
    _, _, min_n, conf_margin = _exit(cfg)
    nec_ratio = _nec_ratio(cfg)
    gated = [i for i in conf_ids(runs, c.err_ids, conf_margin) if gate_ok(runs, i)]
    if len(gated) < min_n:
        return dict(test="O3", n=len(gated), explained=None)

    arms = ("O3.blank", "O3.occlude")
    sub = runs[runs.sample_id.isin(set(gated) | set(c.cor_ids))]

    ci: dict[str, tuple] = {}
    for arm in arms:
        def _ratio(d, _arm=arm, _gated=gated, _cor_ids=c.cor_ids):
            vn_e = _vn(d, _gated, _arm)
            vn_c = _vn(d, _cor_ids, _arm)
            if not np.isfinite(vn_c) or vn_c < _VN_FLOOR:
                return np.nan       # corrects' visual necessity ~ 0: ratio undefined -> None
            return vn_e / vn_c
        ci[arm] = boot_ci(_ratio, sub)

    # necessity = True when VN_err/VN_cor ≤ ratio for both arms (both CI upper bounds)
    if all(hi <= nec_ratio for _, hi in ci.values()):
        explained: bool | None = True
    elif any(lo > nec_ratio for lo, _ in ci.values()):
        explained = False
    else:
        explained = None

    return dict(test="O3.necessity", ci=ci, n=len(gated), explained=explained)


# ----------------------------------------------------------------- explain_consistency (FR-E-4)

def explain_consistency(runs: pd.DataFrame, c: Cluster, cfg: dict) -> dict:
    """O4: polarity-flip gap larger than paraphrase noise ⟹ framing bias (FR-E-4).

    Measures: whether errors flip more often under polarity/framing prompts
    than matched corrects, beyond the paraphrase noise floor.
    Does not measure: which direction the bias runs; see framing_shift() for that.
    """
    _, _, min_n, _ = _exit(cfg)
    min_flip, cons_ratio = _cons(cfg)

    def _pol(d, ids):
        return d[(d.arm == "base") & d.sample_id.isin(ids) & (d.group == "POL") & d.valid.fillna(True)]

    def _par(d, ids):
        return d[(d.arm == "base") & d.sample_id.isin(ids) & (d.group == "D") & d.valid.fillna(True)]

    f_e = flip_rate(_pol(runs, c.err_ids))
    f_c = flip_rate(_pol(runs, c.cor_ids))
    f_par = flip_rate(_par(runs, c.err_ids))

    if pd.isna(f_e) or pd.isna(f_par) or len(c.err_ids) < min_n:
        return dict(test="O4.flip", f_err=f_e, f_cor=f_c, f_par=f_par,
                    gap_ci=(np.nan, np.nan), n=len(c.err_ids), explained=None)

    sub = runs[runs.sample_id.isin(set(c.err_ids) | set(c.cor_ids))]

    def _gap(d):
        return flip_rate(_pol(d, c.err_ids)) - flip_rate(_pol(d, c.cor_ids))

    lo, hi = boot_ci(_gap, sub)
    ok = f_e >= min_flip and f_e >= cons_ratio * max(f_par, 1e-9)
    # Truth table (BUG-17 fix): True only with ok ∧ CI gap > 0; False only
    # when NOT ok (too few flips, or the gap CI says errors flip no more than
    # corrects); an ok=True evidence vector with a straddling CI stays None —
    # it must never be reported as False.
    if ok and lo > 0:
        explained: bool | None = True
    elif not ok and (f_e < min_flip or hi <= 0):
        explained = False
    else:
        explained = None

    return dict(test="O4.flip", f_err=f_e, f_cor=f_c, f_par=f_par,
                gap_ci=(lo, hi), n=len(c.err_ids), explained=explained)


# ----------------------------------------------------------------- qp_ok rate (C7)

def qp_ok_rate(runs: pd.DataFrame, ids: list[str], cfg: dict) -> dict:
    """Share of confident errors with qp ∧ qp_abs correct; Wilson CI vs 0.5.

    Measures: whether the error cluster's perception (PPE-item identification
    and its absence counterpart) is intact.
    Does not measure: whether Q_R (rule reasoning) is correct.

    Returns dict with rate, ci, n, explained (True / False / None).
    """
    _, _, min_n, conf_margin = _exit(cfg)
    cids = conf_ids(runs, ids, conf_margin)
    if not cids:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=0, explained=None)

    base = runs[(runs.arm == "base") & runs.sample_id.isin(cids) & runs.valid.fillna(True)]
    qp = base[base.prompt_id == "qp"].groupby("sample_id").correct.mean()
    qp_abs = base[base.prompt_id == "qp_abs"].groupby("sample_id").correct.mean()
    both = qp.align(qp_abs, join="inner")
    ok_mask = (both[0] >= 0.5) & (both[1] >= 0.5)
    k, n = int(ok_mask.sum()), len(ok_mask)

    if n < min_n:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=n, explained=None)

    lo, hi = wil(k, n)
    rate = k / max(n, 1)
    if lo > 0.5:
        explained: bool | None = True
    elif hi < 0.5:
        explained = False
    else:
        explained = None
    return dict(rate=rate, ci=(lo, hi), n=n, explained=explained)


# ----------------------------------------------------------------- hazard diagnostics (§5.14, FR-H-4)

def hz_qp_ok_rate(runs: pd.DataFrame, ids: list[str], cfg: dict) -> dict:
    """Q_P for latent-hazard clusters: the share of confident errors with ALL
    condition prompts correct in BOTH polarities (hz_s<k> ∧ hz_s<k>_abs for
    every k — a conjunction, so binding errors surface; §5.14). Wilson CI vs
    0.5, three-valued like qp_ok_rate.

    Measures: whether the hazard cluster perceives the *conditions*.
    Does not measure: hazard reasoning (hz_r) or the decision (hz_d).
    """
    from ea.prompts import HZ_CONDITION_SLOTS

    _, _, min_n, conf_margin = _exit(cfg)
    cids = conf_ids(runs, ids, conf_margin)
    hz_pids = [p for k in range(HZ_CONDITION_SLOTS) for p in (f"hz_s{k}", f"hz_s{k}_abs")]
    if not cids:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=0, explained=None)

    base = runs[(runs.arm == "base") & runs.sample_id.isin(cids)
                & runs.prompt_id.isin(hz_pids) & runs.valid.fillna(True)]
    ok = base.groupby("sample_id").correct.min().eq(1)         # EVERY rendered prompt correct
    have = base.groupby("sample_id").prompt_id.nunique()
    ok = ok[have >= 2]                                         # ≥1 condition, both polarities
    k, n = int(ok.sum()), len(ok)

    if n < min_n:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=n, explained=None)
    lo, hi = wil(k, n)
    rate = k / max(n, 1)
    if lo > 0.5:
        explained: bool | None = True
    elif hi < 0.5:
        explained = False
    else:
        explained = None
    return dict(rate=rate, ci=(lo, hi), n=n, explained=explained)


def hz_qr_ok_rate(runs: pd.DataFrame, ids: list[str], cfg: dict) -> dict:
    """Q_R for latent-hazard clusters from the text-only `hz_r` prompt (§5.14):
    physical/hazard reasoning from oracle conditions. EXCLUSION test only
    (FR-T2-6): qr_ok=True says nothing about perception; only False (⇒
    hazard/physics reasoning failure) is informative. Three-valued via Wilson
    CI vs 0.5 on the hz_r-correct share of confident errors.

    Measures: whether the model can chain conditions+trigger→consequence in
    text. Does not measure: perception (hz_s*) or volunteering (hz_d/HZ.cue).
    """
    _, _, min_n, conf_margin = _exit(cfg)
    cids = conf_ids(runs, ids, conf_margin)
    if not cids:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=0, explained=None)
    r = (runs[(runs.arm == "base") & (runs.prompt_id == "hz_r")
              & runs.sample_id.isin(cids) & runs.valid.fillna(True)]
         .groupby("sample_id").correct.mean())
    k, n = int(r.ge(0.5).sum()), len(r)
    if n < min_n:
        return dict(rate=np.nan, ci=(np.nan, np.nan), n=n, explained=None)
    lo, hi = wil(k, n)
    rate = k / max(n, 1)
    if lo > 0.5:
        explained: bool | None = True
    elif hi < 0.5:
        explained = False
    else:
        explained = None
    return dict(rate=rate, ci=(lo, hi), n=n, explained=explained)


def explain_cue(runs: pd.DataFrame, c: Cluster, cfg: dict) -> dict:
    """The `cue` diagnostic (FR-H-4, §5.14): cued recovery on the hazard error
    cluster passing the exit rule AND the secured twins' false-alarm rate
    staying ≤ exit.control. A cue that also raises "Yes" on the safe twins is
    prompt bias, not hazard awareness (pitfall 24) ⇒ cue=False.

    Twins are the samples with HZ.cue_twin rows (plan_hazard links them by
    pair_id). Measures: cannot-infer vs does-not-volunteer separation.
    Does not measure: uncued sensitivity (that is the hz_d error itself).
    """
    rec_thr, ctl_thr, min_n, conf_margin = _exit(cfg)
    ids = conf_ids(runs, c.err_ids, conf_margin)
    k, n = recovery(runs, ids, "HZ.cue")
    lo, hi = wil(k, n)

    tw = (runs[(runs.arm == "HZ.cue_twin") & runs.valid.fillna(True)]
          .groupby("sample_id").pred.mean())
    fa_k, fa_n = int(tw.ge(0.5).sum()), len(tw)               # pred=1 = "Yes" on a safe twin
    fa = fa_k / max(fa_n, 1)
    fa_lo, fa_hi = wil(fa_k, fa_n)

    if n < min_n or fa_n < min_n:
        explained: bool | None = None
    elif lo >= rec_thr and fa_hi <= ctl_thr:
        explained = True
    elif hi < rec_thr or fa_lo > ctl_thr:
        explained = False
    else:
        explained = None
    return dict(test="HZ.cue", rec=k / max(n, 1), ci=(lo, hi), n=n,
                twin_fa=fa, twin_ci=(fa_lo, fa_hi), twin_n=fa_n, explained=explained)


# ----------------------------------------------------------------- signature table (FR-E-5, C1)

# Cause → required diagnostic outcomes.  Compared with ==; None never matches.
# T3/T4 keys (loc, t4, causal, stream) default to None (absent) until those
# tiers have run, so those SIG rows stay inconclusive instead of false-matching.
SIG: dict[str, dict] = {
    "pixel/token-budget limit (H1/H4)":  dict(crop=True, hires=True, bgmask=False),
    "clutter/binding/context (H8/H5)":   dict(crop=True, bgmask=True, hires=False),
    "prior-driven (H7)":                 dict(crop=False, bgmask=False, hires=False,
                                              qp_ok=True, qr_ok=True, necessity=True),
    "polarity/framing bias (H6/H7)":     dict(consistency=True, qp_ok=True),
    "rule-reasoning failure":            dict(qr_ok=False),
    "uncued-hazard under-reporting":     dict(cue=True, qp_ok=True, qr_ok=True),
    "input quality (H9)":                dict(sharpen=True),
    "deployment knob (H4)":              dict(knob=True),
    "scene shortcut (H5)":               dict(loc="loc_low_both", necessity=True),
    "readout/process (H3)":              dict(qp_ok=True, qr_ok=True, crop=False,
                                              hires=False, loc="loc_equal_high",
                                              t4="H3", causal=True),
    "merger bottleneck (H1)":            dict(t4="H1", causal=True),
    "depth fading (H2)":                 dict(t4="H2", causal=True),
    "encoder/perception limit":          dict(qp_ok=False, crop=False, hires=False,
                                              t4="enc"),
    "trigger-policy failure (H11)":      dict(stream="policy"),
}

# Cause → actionable implication printed in the report.
IMPL: dict[str, str] = {
    "pixel/token-budget limit (H1/H4)":  "higher resolution/tiling, crop-then-ask, less merger compression",
    "clutter/binding/context (H8/H5)":   "person-level cropping/tracking before the VLM; context-matched hard negatives",
    "prior-driven (H7)":                 "counterfactual data; neutral prompt template",
    "polarity/framing bias (H6/H7)":     "balanced polarity prompts/data; fixed template",
    "rule-reasoning failure":            "prompt/rule representation; for latent hazards: add hazard/physics knowledge to the prompt",
    "uncued-hazard under-reporting":     "hazard-checklist prompting (name triggers/consequences explicitly); calibrated threshold on the uncued margin",
    "input quality (H9)":                "image-quality gates; degradation augmentation",
    "deployment knob (H4)":              "adjust serving configuration",
    "scene shortcut (H5)":               "rebalance; hard negatives; scene-matched contrast sets",
    "readout/process (H3)":              "auxiliary grounding loss; perceive-then-decide prompting; training-free evidence highlighting (cf. 2510.17771)",
    "merger bottleneck (H1)":            "unfreeze/redesign merger; detail-preserving tokens",
    "depth fading (H2)":                 "re-inject features (DeepStack-style/IRI); layer-targeted tuning",
    "encoder/perception limit":          "better encoder/resolution; crop-then-ask upstream",
    "trigger-policy failure (H11)":      "trigger head / calibrated thresholds",
}

# Known-cause injection -> expected cause (design §5.8b, FR-V-1). Single source
# of truth shared by run.validate and report (BUG-05 fix): every expected cause
# is a SIG key except INJ.label, whose expectation is the unexplained bucket
# itself (FR-V-2: any confident cause assigned to label noise is a false
# attribution and must show up off-diagonal in the confusion matrix).
INJ_EXPECTED: dict[str, str] = {
    "INJ.small":   "pixel/token-budget limit (H1/H4)",
    "INJ.clutter": "clutter/binding/context (H8/H5)",
    "INJ.degrade": "input quality (H9)",
    "INJ.lead":    "polarity/framing bias (H6/H7)",
    "INJ.label":   "unexplained → next tier",
}


def decide(ev: dict) -> tuple[str, list[str]]:
    """Signature-table lookup (FR-E-5, C1).

    ev: {diag_key: True | False | None | str}.  Keys absent from ev act like
    None — they block any SIG row that requires a specific non-None value.

    Returns (cause, candidates):
      - unique match  → (cause_name, [])
      - multiple hits → ("ambiguous", list_of_hit_names)
      - none but partial → ("inconclusive: more data", partial_list)
      - nothing at all  → ("unexplained → next tier", [])
    """
    hit = [k for k, req in SIG.items()
           if all(ev.get(d) == v for d, v in req.items())]
    if len(hit) == 1:
        return hit[0], []
    if len(hit) > 1:
        return "ambiguous", hit

    # partial matches: None-blocked (data still pending) or value mismatch
    part = [k for k, req in SIG.items()
            if all(ev.get(d) in (v, None) for d, v in req.items())]
    return ("inconclusive: more data" if part else "unexplained → next tier"), part


# ----------------------------------------------------------------- cluster builder (FR-E-1)

def _taxonomy_tags(table: pd.DataFrame, runs: pd.DataFrame) -> dict[str, list[str]]:
    """FR-E-2: assign plan §5 failure types automatically from metadata.

    Computed over the FULL table (errors AND corrects — BUG-16 fix): tag
    clusters look up matched corrects sharing the tag, which was impossible
    when tags were built for error ids only (every `tag:` cluster had
    cor_ids == [] and degenerate error-vs-correct comparisons).

    Automatable rules (others require human/T3-T4 input):
      - type_2a_absence: vtype == 'absence'
      - type_6_polarity_flip: sample flips pred across POL group prompts
    """
    tags: dict[str, list[str]] = {sid: [] for sid in table.sample_id}

    # Type 2a: absence violations (from sample metadata)
    if "vtype" in table.columns:
        for sid in table[table.vtype == "absence"].sample_id:
            if sid in tags:
                tags[sid].append("type_2a_absence")

    # Type 6: polarity/framing inconsistency per sample
    pol = runs[(runs.arm == "base") & (runs.group == "POL") & runs.valid.fillna(True)]
    if not pol.empty:
        flippers = pol.groupby("sample_id").pred.nunique()
        for sid in flippers[flippers > 1].index:
            if sid in tags:
                tags[sid].append("type_6_polarity_flip")

    return tags


def build_clusters(
    table: pd.DataFrame,
    cor_df: pd.DataFrame,
    runs: pd.DataFrame,
    cfg: dict,
) -> list[Cluster]:
    """FR-E-1/10: build slice-based and taxonomy-tag clusters plus ALL_ERRORS.

    table: output of errors_table() over ALL samples (errors AND corrects) —
    slice search needs the per-sample error column at the population base
    rate (design §5.7 "one row per sample"); on an errors-only frame every
    row has err=1, the base rate is 1.0 and no slice can reach significance.
    cor_df: matched correct samples (from run.match()).
    """
    from ea.slices import COLUMNS, slice_search, misses_frame, false_alarms_frame

    err_df = table[table.err == 1]
    all_ids = list(err_df.sample_id)
    cor_ids = list(cor_df.sample_id if "sample_id" in cor_df.columns else cor_df.index)
    clusters: list[Cluster] = []

    def _hz_flags(eids: list[str]):
        rows = err_df[err_df.sample_id.isin(eids)]
        hazard = bool("vtype" in rows.columns and len(rows)
                      and (rows.vtype == "latent_hazard").all())
        agr = (float(rows.agreement.mean())
               if "agreement" in rows.columns and rows.agreement.notna().any() else None)
        return hazard, agr

    # FR-E-10: ALL_ERRORS is always built first
    hz_all, agr_all = _hz_flags(all_ids)
    clusters.append(Cluster(name="ALL_ERRORS", err_ids=all_ids, cor_ids=cor_ids,
                            hazard=hz_all, agreement=agr_all))

    # Slice-based clusters (FR-E-1)
    min_support = int(cfg.get("slices", {}).get("min_support", 30))
    fdr = float(cfg.get("slices", {}).get("fdr", 0.05))
    max_order = int(cfg.get("slices", {}).get("max_order", 2))
    avail_cols = [c for c in COLUMNS if c in table.columns]

    seen: set[frozenset] = set()

    def _add_slice_cluster(pred: dict):
        key = frozenset(pred.items())
        if key in seen or not pred:
            return
        seen.add(key)
        mask = pd.Series(True, index=err_df.index)
        for col, val in pred.items():
            mask &= err_df[col] == val
        eids = list(err_df[mask].sample_id)
        if not eids:
            return
        # vectorised over the whole correct pool once (BUG-15 fix: the old
        # per-id set_index().loc() lookup was O(n^2) and silently produced []
        # when cor_df had no id column, leaving clusters with no matched
        # corrects and no warning).
        id_col = "sample_id" if "sample_id" in cor_df.columns else cor_df.index.name
        if id_col:
            cmask = pd.Series(True, index=cor_df.index)
            for col, val in pred.items():
                if col in cor_df.columns:
                    cmask &= cor_df[col] == val
            cids = list(cor_df.loc[cmask, id_col]) if id_col in cor_df.columns \
                else list(cor_df.index[cmask])
        else:
            cids = []
        if not cids:
            print(f"[build_clusters] WARNING: cluster '{key}' has no matched corrects "
                  "(under-powered error-vs-correct diagnostics, NFR-5)")
        hz, agr = _hz_flags(eids)
        clusters.append(Cluster(
            name=", ".join(f"{k}={v}" for k, v in sorted(pred.items())),
            err_ids=eids, cor_ids=cids, pred=dict(pred), hazard=hz, agreement=agr,
        ))

    if avail_cols:
        miss_f = misses_frame(table)
        fa_f = false_alarms_frame(table)
        combined = pd.concat([miss_f, fa_f], ignore_index=True)
        if not combined.empty:
            sig = slice_search(combined, cols=avail_cols, min_support=min_support,
                               alpha=fdr, max_order=max_order)
            for _, row in sig.iterrows():
                pred = row.get("slice") if isinstance(row.get("slice"), dict) else {
                    c: row[c] for c in avail_cols if pd.notna(row.get(c))
                }
                if isinstance(pred, dict):
                    _add_slice_cluster(pred)

    # Latent-hazard track cluster (§5.14): first-class — always built when
    # hazard errors exist, even below the slice min_support, because the
    # hazard diagnostics run on it (FR-H-4).
    if "vtype" in err_df.columns and (err_df.vtype == "latent_hazard").any():
        _add_slice_cluster({"vtype": "latent_hazard"})

    # Taxonomy-tag clusters (FR-E-2): tags computed over the FULL table so
    # matched corrects sharing the tag exist (BUG-16 fix)
    tags = _taxonomy_tags(table, runs)
    tag_groups: dict[str, list[str]] = {}
    for sid, ts in tags.items():
        for t in ts:
            tag_groups.setdefault(t, []).append(sid)
    err_set = set(err_df.sample_id)
    for tag, tagged_ids in tag_groups.items():
        eids = [i for i in tagged_ids if i in err_set]
        if len(eids) < min_support:
            continue
        cids = [i for i in tagged_ids if i not in err_set]
        clusters.append(Cluster(name=f"tag:{tag}", err_ids=eids, cor_ids=cids, tag=tag))

    return clusters


# ----------------------------------------------------------------- per-cluster diagnostics

def run_diagnostics(c: Cluster, runs: pd.DataFrame, cfg: dict) -> None:
    """Compute all cheap (T1/T2) diagnostics for cluster c (design §5.8).

    All diagnostics are computed for every cluster; decide() runs once after.
    Latent-hazard clusters (§5.14) swap Q_P/Q_R for the hz_* conjunction /
    hz_r exclusion test and add the `cue` diagnostic (FR-H-4); the generic
    T2 arms (crop/bgmask/hires/…) stay because §5.14 row 1 (perception/binding
    failure) escalates through them.
    T3/T4 diagnostics (loc, t4, causal, stream) are absent until those tiers
    run — their absence leaves the corresponding SIG rows inconclusive.
    Results stored in-place in c.diags.
    """
    d = c.diags

    # O1.gtcrop recovery; controls: random-crop and same-token crop
    d["crop_result"] = explain_recover(runs, c, "O1.gtcrop",
                                       ["O1.randcrop", "O1.sametok"], cfg)
    d["crop"] = d["crop_result"]["explained"]

    # O1.bgmask recovery; control: random mask
    d["bgmask_result"] = explain_recover(runs, c, "O1.bgmask", ["O1.randmask"], cfg)
    d["bgmask"] = d["bgmask_result"]["explained"]

    # hires: max_pixels=high knob; control: N0.null
    d["hires_result"] = explain_recover(runs, c, ("max_pixels", "high"), ["N0.null"], cfg)
    d["hires"] = d["hires_result"]["explained"]

    # O6.sharpen: image sharpening; control: N0.null
    d["sharpen_result"] = explain_recover(runs, c, "O6.sharpen", ["N0.null"], cfg)
    d["sharpen"] = d["sharpen_result"]["explained"]

    # best knob recovery (max_pixels, token_budget); control: N0.null
    knob_arms = [("max_pixels", v) for v in ("high", "native")]
    best_knob = None
    for ka in knob_arms:
        r = explain_recover(runs, c, ka, ["N0.null"], cfg)
        if r["explained"] is True:
            best_knob = r
            break
    d["knob_result"] = best_knob
    d["knob"] = (best_knob["explained"] if best_knob else None)

    if c.hazard:
        # §5.14: Q_P = all conditions, both polarities (conjunction); Q_R from
        # the text-only hz_r prompt (exclusion only); cue = cued recovery AND
        # twin false-alarm gate. The generic qp/qr prompts never render for
        # hazard samples (no item), so the O2 arm path has nothing to say and
        # the necessity gate is failed for lack of qp rows (prior-driven is
        # not in the §5.14 reading table).
        d["qp_ok_result"] = hz_qp_ok_rate(runs, c.err_ids, cfg)
        d["qp_ok"] = d["qp_ok_result"]["explained"]
        d["qr_ok_result"] = hz_qr_ok_rate(runs, c.err_ids, cfg)
        d["qr_ok"] = d["qr_ok_result"]["explained"]
        d["cue_result"] = explain_cue(runs, c, cfg)
        d["cue"] = d["cue_result"]["explained"]
    else:
        # O2.facts: text-only oracle facts (qr_ok diagnostic; exclusion-only: FR-T2-6)
        d["qr_ok_result"] = explain_recover(runs, c, "O2.facts", ["O2.wrongfacts"], cfg)
        d["qr_ok"] = d["qr_ok_result"]["explained"]
        # qp_ok rate (C7)
        d["qp_ok_result"] = qp_ok_rate(runs, c.err_ids, cfg)
        d["qp_ok"] = d["qp_ok_result"]["explained"]

    # O3 necessity (prior-driven gate)
    d["necessity_result"] = explain_necessity(runs, c, cfg)
    d["necessity"] = d["necessity_result"]["explained"]

    # O4 consistency (polarity/framing bias)
    d["consistency_result"] = explain_consistency(runs, c, cfg)
    d["consistency"] = d["consistency_result"]["explained"]

    # T3/T4 keys: None until external tiers supply them
    for key in ("loc", "t4", "causal", "cue", "stream"):
        d.setdefault(key, None)


# ----------------------------------------------------------------- per-error accounting (C17)

def per_error_accounting(clusters: list[Cluster]) -> pd.DataFrame:
    """C17: each error gets the cause of the MOST SPECIFIC cluster containing it
    that has a unique cause.  Ties or no unique cause → 'ambiguous' / 'ALL_ERRORS'.

    Returns a DataFrame with one row per error sample_id and columns:
      sample_id, cause, cluster_name, cluster_size.
    Invariant (FR-E-11): every error appears exactly once; counts sum to total.
    """
    all_ids = set(clusters[0].err_ids)  # ALL_ERRORS is always first

    named: list[Cluster] = []
    for c in clusters:
        cause_is_specific = c.cause not in (
            "unexplained → next tier", "ambiguous", "inconclusive: more data"
        )
        if cause_is_specific:
            named.append(c)

    # Sort ascending by size so most-specific (smallest) cluster wins
    named_sorted = sorted(named, key=lambda c: len(c.err_ids))

    rows = {}
    for c in named_sorted:
        for sid in c.err_ids:
            # Only assign if not yet claimed: smallest cluster was processed first → it wins
            if sid in all_ids and sid not in rows:
                rows[sid] = dict(
                    sample_id=sid,
                    cause=c.cause,
                    cluster_name=c.name,
                    cluster_size=len(c.err_ids),
                )

    # Remaining errors inherit the ALL_ERRORS cluster's cause
    fallback_cause = clusters[0].cause
    for sid in all_ids:
        if sid not in rows:
            rows[sid] = dict(
                sample_id=sid,
                cause=fallback_cause,
                cluster_name="ALL_ERRORS",
                cluster_size=len(clusters[0].err_ids),
            )

    df = pd.DataFrame(list(rows.values())).sort_values("sample_id").reset_index(drop=True)
    assert len(df) == len(all_ids), "FR-E-11: per-error count mismatch"
    return df


# ----------------------------------------------------------------- audit sample (FR-E-6)

def audit_sample(clusters: list[Cluster], k: int = 20, seed: int = 0) -> pd.DataFrame:
    """FR-E-6: draw k errors per cluster for the human audit sample.

    Mandatory for clusters with hazard agreement < hazard.min_agreement
    (FR-H-6); returned as a DataFrame with sample_id, cluster_name, cause.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in clusters:
        chosen = list(rng.choice(c.err_ids, min(k, len(c.err_ids)), replace=False))
        for sid in chosen:
            rows.append(dict(sample_id=sid, cluster_name=c.name, cause=c.cause))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main entry point

def escalate(
    runs: pd.DataFrame,
    S: list,
    cfg: dict,
    out_dir: str | Path = "outputs/escalate",
    external_diags: dict[str, dict] | None = None,
) -> tuple[list[Cluster], pd.DataFrame]:
    """Full escalation pipeline (FR-E-1..12).

    Steps:
      1. Build error table via majority-vote D rows.
      2. Build matched-correct pool.
      3. Build clusters (ALL_ERRORS + slices + taxonomy tags).
      4. Run T1/T2 diagnostics on each cluster.
      5. Call decide() to assign cause.
      6. Per-error accounting.
      7. Write clusters.json, diag_vector.parquet, audit_sample.csv.

    external_diags: cluster name -> {diag: value} with the T3/T4 results
    (loc, t4, causal) produced by `ea t3` / `ea t4` (BUG-03 fix: merged into
    c.diags BEFORE decide() so SIG rows requiring them can finally match).
    The `stream` diagnostic is computed here from the stored margin series
    (offline trigger re-analysis, FR-ST-2 — no new model runs needed).

    Returns (clusters, diag_vector DataFrame).
    NFR-5: prints counts of skipped/underpowered clusters.
    """
    from ea.run import errors_table, match

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: errors
    et = errors_table(runs, S)
    if et.empty:
        print("[escalate] WARNING: no valid base D rows — nothing to escalate.")
        return [], pd.DataFrame()

    err_df = et[et.err == 1].copy()
    cor_df = et[et.err == 0].copy()

    n_errs = len(err_df)
    print(f"[escalate] {n_errs} errors, {len(cor_df)} corrects")

    # Step 2: matched corrects
    match_keys = tuple(cfg.get("match", {}).get("keys", ("cls", "obj_size_bin")))
    cor_ids, fallbacks = match(err_df, cor_df, keys=match_keys)
    matched_cor = cor_df[cor_df.sample_id.isin(cor_ids)]
    if fallbacks:
        print(f"[escalate] {fallbacks} errors used cls-only fallback matching (NFR-5)")

    # Step 3: clusters (full table: slice search needs the population base rate)
    clusters = build_clusters(et, matched_cor, runs, cfg)
    print(f"[escalate] {len(clusters)} clusters built")

    # Merge T3/T4 diagnostics supplied by the attribution/mechanistic tiers
    # (BUG-03 fix). Unknown cluster names are ignored so stale t3/t4 outputs
    # from a different run cannot fabricate evidence.
    if external_diags:
        merged = 0
        for c in clusters:
            extra = external_diags.get(c.name)
            if extra:
                c.diags.update(extra)
                merged += 1
        print(f"[escalate] merged external T3/T4 diagnostics for {merged} clusters")

    # `stream` diagnostic (H11, FR-ST-5): offline window-vs-stream re-analysis
    # of the stored margin series — trigger policies never need new model runs.
    # "policy" when the trigger policy (not perception) dominates the lateness.
    from ea.stream import window_vs_stream_gap
    gap = window_vs_stream_gap(runs, S, cfg, tau=0.0)
    stream_diag = "policy" if gap.get("stream_policy") else False
    if gap.get("n_risk_videos"):
        print(f"[escalate] stream gap: window={gap.get('window_recall', 0):.2f} "
              f"stream={gap.get('stream_recall', 0):.2f} policy={gap.get('stream_policy')}")

    # Step 4 + 5: diagnostics + decide
    underpowered = 0
    for c in clusters:
        run_diagnostics(c, runs, cfg)
        c.diags.setdefault("stream", stream_diag)      # H11; external merge wins if present
        ev = {k: c.diags[k] for k in (
            "crop", "bgmask", "hires", "qr_ok", "sharpen", "knob",
            "qp_ok", "necessity", "consistency", "loc", "t4", "causal", "cue", "stream",
        )}
        c.cause, c.cause_candidates = decide(ev)
        if c.cause in ("unexplained → next tier", "inconclusive: more data"):
            underpowered += 1

    print(f"[escalate] {underpowered} clusters unexplained/inconclusive → queued for T3/T4")

    # Step 6: per-error accounting
    diag_vec = per_error_accounting(clusters)
    diag_vec.to_parquet(out_dir / "diag_vector.parquet", index=False)

    # Step 7: write outputs
    cluster_records = []
    for c in clusters:
        rec = dict(
            name=c.name,
            n_errors=len(c.err_ids),
            n_corrects=len(c.cor_ids),
            cause=c.cause,
            cause_candidates=c.cause_candidates,
            tag=c.tag,
            pred=c.pred,
            err_ids=c.err_ids,
            cor_ids=c.cor_ids,
        )
        if c.hazard:
            rec["hazard"] = True
            rec["agreement"] = c.agreement
            rec["cue"] = c.diags.get("cue")
        cluster_records.append(rec)
    (out_dir / "clusters.json").write_text(
        json.dumps(cluster_records, indent=2, default=str), encoding="utf-8"
    )

    audit = audit_sample(clusters, k=int(cfg.get("audit", {}).get("k_per_cluster", 20)))
    audit.to_csv(out_dir / "audit_sample.csv", index=False)

    print(f"[escalate] wrote clusters.json, diag_vector.parquet, audit_sample.csv → {out_dir}")
    return clusters, diag_vec
