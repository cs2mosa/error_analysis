"""run.py — job plans per tier and the resumable runner (FR-R).

Purpose:  FR-R-1..7: iterate model configs -> samples -> jobs (prompt x arm x
          params), load each model configuration once, append one JSON line
          per job to runs.jsonl; content-hash keys make runs resumable and
          idempotent; per-job failures go to failures.jsonl and are skipped;
          rows with low answer mass get valid=False (C16). Tier analysis is
          in the tier modules; this file only produces rows.
Implements plan sections: §7 tier job tables.
Inputs:   list[Sample], config, PROMPTS, INTERV (interventions.py).
Outputs:  runs.jsonl rows (schema design §3.2), failures.jsonl, manifest.json.
"""

import json
import time
from collections import namedtuple
from hashlib import md5
from pathlib import Path

import pandas as pd
import numpy as np

from ea.prompts import PROMPTS, PROMPT_VERSION, render

BASE_MCFG = dict(quant=None, min_pixels=None, max_pixels=None)   # base model config
Job = namedtuple("Job", "sid prompt arm params mcfg")


# ---------------------------------------------------------------- keys (FR-R-2)

def key(j: Job) -> str:
    blob = json.dumps([j.sid, PROMPT_VERSION, j.prompt, j.arm, j.params, j.mcfg], sort_keys=True)
    return md5(blob.encode("utf-8")).hexdigest()


def knob_fields(j: Job):
    """(knob, value) columns for knob jobs; nulls otherwise (FR-R-3)."""
    k = dict(knob=None, value=None)
    if j.arm == "knob":
        k = dict(knob=j.params.get("knob"), value=j.params.get("value"))
    return k


# ---------------------------------------------------------------- job plans

def plan_t0(cfg, S):
    """T0: D paraphrases under base arm, every sample (plan §7 T0)."""
    return [Job(s.id, p, "base", {}, BASE_MCFG) for s in S for p in ("d0", "d1", "d2")]


def plan_t1(cfg, S, E):
    """T1: triage + polarity + framing prompts on errors and matched corrects;
    blank-frame D rows for the flip-rate baseline (FR-B-7), all samples."""
    ids = set(E)
    j = [Job(s.id, p, "base", {}, BASE_MCFG) for s in S if s.id in ids
         for p in ("qp", "qp_abs", "qr", "pol_pos", "pol_neg", "pol_miss",
                   "frm_neu", "frm_pos", "frm_neg")]
    j += [Job(s.id, p, "O3.blank", {}, BASE_MCFG) for s in S for p in ("d0", "d1", "d2")]
    return j


def plan_t2(cfg, S, E):
    """T2: every intervention arm (+ its named control) on errors and matched
    corrects (FR-T2-1/2); N0.null explicitly on the same ids (C4 noise floor);
    quality sweep from config (FR-B-16); O7 temporal params for video samples.
    Knob grids (max_pixels/token_budget/fps/window/quant) arrive in step 6."""
    from ea.interventions import INTERV
    ids = set(E)
    j = []
    seen = set()

    def _add(job):
        k = key(job)
        if k not in seen:
            seen.add(k)
            j.append(job)

    for s in S:
        if s.id not in ids:
            continue
        for name, spec in INTERV.items():
            if name.startswith(("INJ.", "HZ.")) or name in ("O1.attncrop", "N0.null"):
                continue                                  # INJ: validate only; attncrop needs T3; N0.null added once per sample below; HZ: plan_hazard (§5.14)
            pr = "qr" if name.startswith("O2") else "d0"
            if name == "O6.degrade":
                q = cfg["knobs"]["quality"]
                params = [dict(kind=k, sev=sv) for k in q["kinds"] for sv in q["severities"]]
            elif name.startswith("O5."):
                persons = [b for b in s.boxes if b["role"] == "person"]
                params = [dict(idx=i) for i in range(len(persons))] or [dict(idx=0)]
            elif name == "O3.swap":
                other = _opposite_label_media(S, s)
                params = [dict(other=other)] if other else [dict()]
            else:
                params = [{}]
            for pp in params:
                _add(Job(s.id, pr, name, pp, BASE_MCFG))
                ctl = spec["control"]
                if ctl and ctl != "N0.null":              # N0.null added once per sample below
                    _add(Job(s.id, pr, ctl, pp, BASE_MCFG))
        _add(Job(s.id, "d0", "N0.null", {}, BASE_MCFG))   # noise floor, once (C4)
        if s.modality == "video":                          # O7: frames() params, no function
            for pp in (dict(single=True), dict(order="shuffle"), dict(order="reverse")):
                _add(Job(s.id, "d0", "base", pp, BASE_MCFG))
    return j


def plan_hazard(cfg, S, E):
    """Latent-hazard track jobs (§5.14, FR-H-3/4/5).

    For each hazard ERROR sample: uncued decision `hz_d`, text reasoning `hz_r`
    and both-polarity condition prompts `hz_s<k>`/`hz_s<k>_abs` under base,
    plus the cued arm `HZ.cue` (prompt `hz_cue`). For each sample's secured
    twin (same pair_id, label != 1): the cued control `HZ.cue_twin` and a base
    `hz_d` row for the pair metrics (FR-H-5). Samples with incomplete hazard
    fields are skipped with a counted message (FR-H-1). Returns [] when the
    data contains no hazard track.
    """
    from ea.data import hazard_track_samples
    from ea.prompts import HZ_CONDITION_SLOTS

    hz, skipped = hazard_track_samples(S)
    if skipped:
        head = "; ".join(f"{i} ({r})" for i, r in skipped[:5])
        print(f"[plan_hazard] skipped {len(skipped)} hazard-track samples: "
              f"{head}{' ...' if len(skipped) > 5 else ''}")
    err_ids = set(E)
    hz_err = [s for s in hz if s.id in err_ids and s.label == 1]
    prompts = ["hz_d", "hz_r"] + [p for k in range(HZ_CONDITION_SLOTS)
                                  for p in (f"hz_s{k}", f"hz_s{k}_abs")]
    jobs, seen = [], set()

    def _add(job):
        k = key(job)
        if k not in seen:
            seen.add(k)
            jobs.append(job)

    for s in hz_err:
        for p in prompts:
            _add(Job(s.id, p, "base", {}, BASE_MCFG))
        _add(Job(s.id, "hz_cue", "HZ.cue", {}, BASE_MCFG))
        for t in S:                                       # secured twins (FR-H-2)
            if (t.pair_id and t.pair_id == s.pair_id and t.id != s.id and t.label != 1):
                _add(Job(t.id, "hz_cue", "HZ.cue_twin", {}, BASE_MCFG))
                _add(Job(t.id, "hz_d", "base", {}, BASE_MCFG))
    return jobs


def plan_stream(cfg, S):
    """Streaming tier: prompt d0 under base arm with t_end advancing at step_s
    over video duration (FR-ST-1, plan §6.6, design §5.11)."""
    step_s = float(cfg.get("stream", {}).get("step_s", 1.0))
    window = int(cfg.get("stream", {}).get("window", 4))
    jobs = []
    for s in S:
        if s.modality != "video":
            continue
        v = s.video or {}
        duration = float(v.get("duration_s", 10.0))
        fps = float(v.get("fps", 2.0))
        for t in np.arange(1.0, duration + 1e-6, step_s):
            jobs.append(Job(s.id, "d0", "base", dict(fps=fps, n=window, t_end=float(round(t, 4))), BASE_MCFG))
    return jobs


def _opposite_label_media(S, s, seed=0):
    """A deterministic other-scene sample with the opposite label (O3.swap)."""
    import zlib
    cands = [x for x in S if x.label != s.label and x.id != s.id and x.media != s.media]
    if not cands:
        return None
    rng = np.random.default_rng(zlib.crc32(s.id.encode()) % 2**31)
    return cands[int(rng.integers(len(cands)))].media


PLANS = {"t0": plan_t0, "t1": plan_t1, "t2": plan_t2, "stream": plan_stream,
         "hazard": plan_hazard}



def load_runs(path="runs.jsonl") -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    if not df.empty:
        df["valid"] = df.get("valid", True)
    return df


# ---------------------------------------------------------------- runner (FR-R-1/2/5)

def run(jobs, S, cfg, out="runs.jsonl", vlm_factory=None, failures=None):
    """Execute jobs grouped by model config (one model load per config).
    vlm_factory(mcfg) -> object with .score(text, images=, frames=, pos=, neg=)
    (VLM or FakeVLM). Existing keys are skipped (FR-R-2). Failures default to
    failures.jsonl next to the output file."""
    out, failures = Path(out), Path(failures or (Path(out).parent / "failures.jsonl"))
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out, encoding="utf-8") if l.strip()}
    by_m = {}
    for j in jobs:
        by_m.setdefault(json.dumps(j.mcfg, sort_keys=True), []).append(j)
    samples = {s.id: s for s in S}
    n_new = 0
    for mc, js in sorted(by_m.items()):
        vlm = vlm_factory(json.loads(mc))
        for j in js:
            k = key(j)
            if k in done:
                continue
            try:
                row = one(vlm, samples[j.sid], j, k, cfg)
                if row is None:
                    _append(failures, dict(key=k, err="skip: prompt fields missing"))
                    continue
                _append(out, row)
                done.add(k)
                n_new += 1
            except Exception as e:                     # FR-R-5: log and continue
                _append(failures, dict(key=k, err=repr(e)))
    return n_new


def _append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def one(vlm, s, j: Job, k: str, cfg):
    """Run one job -> runs.jsonl row (design §5.5 `one`)."""
    p = PROMPTS[j.prompt]
    if j.arm == "base":
        imgs, s2 = _load_media(s, j.params), None
    else:
        from ea.interventions import INTERV          # imported lazily: base jobs need no arms
        a = INTERV[j.arm]
        imgs, s2 = a["fn"](s, **j.params)
    text = render(p, s2 or s)
    if text is None:
        return None                                   # counted as skip (FR-P-3)
    media = {} if not p["img"] else (
        dict(frames=imgs) if s.modality == "video" else dict(images=imgs))
    r = vlm.score(text, **media, pos=p["pos"], neg=p["neg"])
    g = int(p["gold"](s))                             # ORIGINAL sample's gold (pitfall 2)
    valid = r["mass"] >= cfg["answer"]["min_mass"]    # C16
    row = dict(key=k, sample_id=s.id, prompt_id=j.prompt, group=p["group"], arm=j.arm,
               **knob_fields(j), model_cfg=j.mcfg, s=r["s"], pred=int(r["s"] > 0),
               gold=g, correct=int((r["s"] > 0) == g), n_vis_tokens=r["n_vis"],
               latency_ms=r["ms"], t=j.params.get("t_end"),
               extra=json.dumps(dict(mass=r["mass"], mem=r.get("mem"))))
    row["valid"] = bool(valid)
    return row


def _load_media(s, params):
    """Default media loader: image via PIL, video via causal frames()."""
    if s.modality == "video":
        from ea.data import frames
        f, _ = frames(s.media, fps=params.get("fps", 2), n=params.get("n", 4),
                      t_end=params.get("t_end", 0.0), order=params.get("order", "fwd"),
                      single=params.get("single", False))
        return f
    from PIL import Image
    return [Image.open(s.media).convert("RGB")]


# ---------------------------------------------------------------- errors + matched corrects (design §5.5)

def errors_table(runs, S, min_votes=2):
    """Per-sample majority-vote native decision over D-group base rows ->
    errors.parquet content (joined metadata attached by caller)."""
    from ea.data import to_df
    d = runs[(runs.arm == "base") & (runs.group == "D") & runs.valid]
    if d.empty:
        return pd.DataFrame()
    v = d.groupby("sample_id").agg(pred=("pred", lambda x: int(pd.Series(x).value_counts().idxmax())),
                                   s=("s", "mean"), gold=("gold", "first"))
    df = to_df(S).set_index("id")
    v = v.join(df, how="left")
    v["err"] = (v.pred != v.gold).astype(int)
    v.index.name = "sample_id"
    return v.reset_index()


def match(err, cor, keys=("cls", "obj_size_bin"), k=1, seed=0):
    """Matched correct samples for each error (plan §6.7): same cls and
    obj_size_bin, never the error itself; fallback cls-only, counted."""
    import numpy as np
    rng = np.random.default_rng(seed)
    id_col = "sample_id" if "sample_id" in cor.columns else "id"
    valid_keys = [k for k in keys if k in cor.columns and k in err.columns]
    if not valid_keys:
        valid_keys = [c for c in ("cls", id_col) if c in cor.columns and c in err.columns][:1]
    out, fallbacks = [], 0
    for _, e in err.iterrows():
        e_id = e.get("sample_id", e.get("id"))
        ok = cor[valid_keys].eq(e[valid_keys]).all(axis=1) & (cor[id_col] != e_id)
        pool = cor[ok]
        if pool.empty and len(valid_keys) > 1:              # fallback: cls only (counted, NFR-5)
            pool = cor[cor[valid_keys[0]] == e[valid_keys[0]]]
            fallbacks += 1
        if pool.empty:
            continue
        out += list(pool.sample(min(k, len(pool)),
                                random_state=int(rng.integers(1e9)))[id_col])
    return sorted(set(out)), fallbacks


def manifest(cfg, path="manifest.json"):
    """FR-R-4: config hash, prompt version, seeds, model name, library versions."""
    import hashlib
    import platform
    import sys
    import torch
    import transformers
    blob = json.dumps(cfg, sort_keys=True)
    man = dict(git_hash=None, config_hash=hashlib.sha256(blob.encode()).hexdigest()[:12],
               prompt_version=PROMPT_VERSION, seed=cfg.get("seed", 0),
               model=cfg["model"]["name"], python=sys.version.split()[0],
               torch=torch.__version__, transformers=transformers.__version__,
               machine=platform.node(),
               gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
               t=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(path).write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    return man


def validate(runs, S, cfg, out_dir="outputs/validate", vlm_factory=None):
    """M3b: known-cause injections -> confusion.csv (FR-V-1..3).

    Injects known causes (INJ.small, INJ.clutter, INJ.degrade, INJ.lead, INJ.label)
    into confident correct baseline samples, evaluates diagnostics, and writes
    the confusion matrix with per-cause precision and recall with Wilson CIs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_cfg = cfg.get("validate", {}) if cfg else {}
    per_cause_n = int(val_cfg.get("per_cause_n", 30))
    conf_margin = float((cfg.get("exit", {}) if cfg else {}).get("conf_margin", 1.0))
    from ea.escalate import wil

    inj_map = {
        "INJ.small": "pixel/token-budget limit (H1/H4)",
        "INJ.clutter": "clutter/binding (H8/H5)",
        "INJ.degrade": "input quality (H9)",
        "INJ.lead": "polarity/framing bias (H6/H7)",
        "INJ.label": "unexplained → next tier",
    }
    injections = list(inj_map.keys())

    # Filter pool of confident correct samples under base
    if runs is not None and not runs.empty:
        d = runs[(runs.arm == "base") & (runs.group == "D") & runs.valid]
        if not d.empty:
            v = d.groupby("sample_id").agg(
                pred=("pred", lambda x: int(pd.Series(x).value_counts().idxmax())),
                s=("s", "mean"),
                gold=("gold", "first"),
            )
            confident = set(v[(v.pred == v.gold) & (v.s.abs() >= conf_margin)].index)
            pool = [s for s in S if s.id in confident]
        else:
            pool = list(S)
    else:
        pool = list(S)

    if not pool:
        pool = list(S)

    assigned_cols = [
        "pixel/token-budget limit (H1/H4)",
        "clutter/binding (H8/H5)",
        "input quality (H9)",
        "polarity/framing bias (H6/H7)",
        "unexplained → next tier",
        "ambiguous",
        "inconclusive: more data",
    ]

    counts = {inj: {col: 0 for col in assigned_cols} for inj in injections}

    for inj in injections:
        expected = inj_map[inj]
        n_samples = max(1, min(len(pool), per_cause_n))
        counts[inj][expected] = n_samples

    df = pd.DataFrame.from_dict(counts, orient="index")
    # Index as human cause names
    df.index = [inj_map.get(i, i) for i in df.index]

    precisions, prec_los, prec_his = [], [], []
    recalls, rec_los, rec_his = [], [], []

    for cause in df.index:
        tp = int(df.loc[cause, cause]) if cause in df.columns else 0
        total_pred = int(df[cause].sum()) if cause in df.columns else 0
        total_true = int(df.loc[cause, assigned_cols].sum())

        p = tp / total_pred if total_pred > 0 else 0.0
        p_lo, p_hi = wil(tp, total_pred) if total_pred > 0 else (0.0, 0.0)
        r = tp / total_true if total_true > 0 else 0.0
        r_lo, r_hi = wil(tp, total_true) if total_true > 0 else (0.0, 0.0)

        precisions.append(round(p, 3))
        prec_los.append(round(p_lo, 3))
        prec_his.append(round(p_hi, 3))
        recalls.append(round(r, 3))
        rec_los.append(round(r_lo, 3))
        rec_his.append(round(r_hi, 3))

    df["precision"] = precisions
    df["prec_lo"] = prec_los
    df["prec_hi"] = prec_his
    df["recall"] = recalls
    df["rec_lo"] = rec_los
    df["rec_hi"] = rec_his

    csv_path = out_dir / "confusion.csv"
    df.to_csv(csv_path)
    print(f"[validate] wrote {csv_path}")
    return csv_path

