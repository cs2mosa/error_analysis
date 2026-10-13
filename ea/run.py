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


def knob_fields(j: Job) -> dict:
    """(knob, value) columns for knob jobs; nulls otherwise (FR-R-3).

    Convention (BUG-04 fix): knob rows carry arm="base" with
    params={"knob": ..., "value": ...} — the same shape escalate._recovery_mask
    filters on (`runs.arm == "base" & runs.knob == knob & runs.value == value`).
    Resolution/quantization knobs additionally differ in j.mcfg so the runner
    loads each model configuration once (FR-R-1); pruning/fps/window knobs are
    plain params and reload nothing."""
    k = dict(knob=None, value=None)
    if j.params.get("knob") is not None:
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
    quality sweep from config (FR-B-16); O7 temporal params for video samples;
    knob grids (max_pixels/token_budget/fps/window/quant) from cfg["knobs"]
    (FR-T2-3, BUG-04 fix: each knob value = one (knob, value) pair in
    runs.jsonl; max_pixels/quant differ per mcfg so each loads once).

    Optional arms that no diagnostic consumes yet (O1.tightcrop, O3.noise/
    swap, O5.*, O6.degrade) are gated by cfg["t2"]["extra_arms"] (default
    True — their rows are informational for the audit sample; DEVIATIONS #12).
    """
    from ea.interventions import INTERV
    ids = set(E)
    j = []
    seen = set()

    def _add(job):
        k = key(job)
        if k not in seen:
            seen.add(k)
            j.append(job)

    extra_ok = bool((cfg.get("t2") or {}).get("extra_arms", True))
    for s in S:
        if s.id not in ids:
            continue
        for name, spec in INTERV.items():
            if name.startswith(("INJ.", "HZ.")) or name in ("O1.attncrop", "N0.null"):
                continue                                  # INJ: validate only; attncrop needs T3; N0.null added once per sample below; HZ: plan_hazard (§5.14)
            if not extra_ok and name in _EXTRA_ARMS:
                continue
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
        _add_knob_jobs(cfg, s, _add)
        if s.modality == "video":                          # O7: frames() params, no function
            for pp in (dict(single=True), dict(order="shuffle"), dict(order="reverse")):
                _add(Job(s.id, "d0", "base", pp, BASE_MCFG))
    return j


# Arms defined and planned but not yet consumed by run_diagnostics (BUG-18):
# kept behind a config flag so GPU time can be saved without deleting code.
_EXTRA_ARMS = {"O1.tightcrop", "O3.noise", "O3.swap", "O5.personcrop",
               "O5.overlay", "O6.degrade"}


def _add_knob_jobs(cfg, s, add):
    """Knob-grid jobs for one sample (FR-T2-3, BUG-04 fix).

    max_pixels / quant are model-configuration knobs (the runner groups by
    mcfg, so each configuration loads once); token_budget is a masking hook
    (model.mask_tokens — positions kept, features zeroed, design §5.4);
    fps / window are frames() params for video samples. Control for every
    knob is N0.null, never a base re-run (C4)."""
    kn = cfg.get("knobs") or {}
    mp = kn.get("max_pixels") or {}
    if isinstance(mp, dict):
        for vname, num in mp.items():                     # {low: n, native: 1048576, high: n}
            min_px = 1048576 if vname == "high" else None
            mcfg = dict(BASE_MCFG, max_pixels=num, min_pixels=min_px)
            add(Job(s.id, "d0", "base", dict(knob="max_pixels", value=vname), mcfg))
    elif isinstance(mp, (list, tuple)):                   # numeric list from config
        for num in mp:
            mcfg = dict(BASE_MCFG, max_pixels=num)
            add(Job(s.id, "d0", "base", dict(knob="max_pixels", value=str(num)), mcfg))
    for q in kn.get("quant", []):
        add(Job(s.id, "d0", "base", dict(knob="quant", value=str(q)), dict(BASE_MCFG, quant=q)))
    for tb in kn.get("token_budget", []):
        add(Job(s.id, "d0", "base",
                dict(knob="token_budget", value=tb), BASE_MCFG))
    if s.modality == "video":
        for f in kn.get("fps", []):
            add(Job(s.id, "d0", "base",
                    dict(knob="fps", value=f, fps=float(f)), BASE_MCFG))
        for wn in kn.get("window", []):
            add(Job(s.id, "d0", "base",
                    dict(knob="window", value=wn, n=int(wn)), BASE_MCFG))


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
    out.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        if imgs is None and p["img"]:                # BUG-19 fix: text-modifying arms
            imgs = _load_media(s, j.params)          # (INJ.lead/label, O2.*) still need
    text = render(p, s2 or s)                        # the ORIGINAL image when the prompt has one
    if text is None:
        return None                                   # counted as skip (FR-P-3)
    media = {} if not p["img"] else (
        dict(frames=imgs) if s.modality == "video" else dict(images=imgs))
    g = int(p["gold"](s))                             # ORIGINAL sample's gold (pitfall 2)
    if j.params.get("knob") == "token_budget":        # FR-T2-3: prune by zeroing features,
        from ea.model import mask_tokens              # positions kept (design §5.4)
        import zlib
        seed = zlib.crc32(f"{s.id}|tb|{j.params['value']}".encode()) % 2**31
        with mask_tokens(vlm, keep_frac=float(j.params["value"]), seed=seed):
            r = vlm.score(text, **media, pos=p["pos"], neg=p["neg"])
    else:
        r = vlm.score(text, **media, pos=p["pos"], neg=p["neg"])
    valid = r["mass"] >= cfg["answer"]["min_mass"]    # C16
    extra_dict = dict(mass=r["mass"], mem=r.get("mem"))
    if j.params.get("knob") == "token_budget":
        extra_dict["keep_frac"] = float(j.params.get("value", 1.0))
    row = dict(key=k, sample_id=s.id, prompt_id=j.prompt, group=p["group"], arm=j.arm,
               **knob_fields(j), model_cfg=j.mcfg, s=r["s"], pred=int(r["s"] > 0),
               gold=g, correct=int((r["s"] > 0) == g), n_vis_tokens=r["n_vis"],
               latency_ms=r["ms"], t=j.params.get("t_end"),
               extra=json.dumps(extra_dict))
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
    errors.parquet content (joined metadata attached by caller).

    min_votes (BUG-14 fix): samples with fewer valid D rows than min_votes are
    undetermined — excluded and counted (NFR-5), never decided by a single
    row. Ties break deterministically by the sign of the mean margin (the
    value_counts idxmax order was arbitrary for 1-1 ties)."""
    from ea.data import to_df
    d = runs[(runs.arm == "base") & (runs.group == "D") & runs.valid]
    if d.empty:
        return pd.DataFrame()
    v = d.groupby("sample_id").agg(
        n_pos=("pred", lambda x: int((pd.Series(x) == 1).sum())),
        n_neg=("pred", lambda x: int((pd.Series(x) == 0).sum())),
        s=("s", "mean"), gold=("gold", "first"))
    votes = v.n_pos + v.n_neg
    undetermined = int((votes < min_votes).sum())
    if undetermined:
        print(f"[errors_table] {undetermined} samples with < {min_votes} valid D rows "
              "left undetermined (NFR-5)")
    v = v[votes >= min_votes].copy()
    v["pred"] = (v.n_pos > v.n_neg).astype(int)       # strict majority
    v.loc[v.n_pos == v.n_neg, "pred"] = (v.loc[v.n_pos == v.n_neg, "s"] > 0).astype(int)
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
    out, fallbacks, unmatched = [], 0, 0
    for _, e in err.iterrows():
        e_id = e.get("sample_id", e.get("id"))
        ok = cor[valid_keys].eq(e[valid_keys]).all(axis=1) & (cor[id_col] != e_id)
        pool = cor[ok]
        if pool.empty and len(valid_keys) > 1:              # fallback: cls only (counted, NFR-5)
            pool = cor[cor[valid_keys[0]] == e[valid_keys[0]]]
            fallbacks += 1
        if pool.empty:
            unmatched += 1                                   # counted, never silent (BUG-14 fix)
            continue
        out += list(pool.sample(min(k, len(pool)),
                                random_state=int(rng.integers(1e9)))[id_col])
    if unmatched:
        print(f"[match] {unmatched} errors had no eligible correct sample and "
              "received no matched control (NFR-5)")
    return sorted(set(out)), fallbacks


def manifest(cfg, path="manifest.json"):
    """FR-R-4: config hash, prompt version, seeds, model name, library versions."""
    import hashlib
    import platform
    import subprocess
    import sys
    import torch
    import transformers
    try:                                                  # reproducibility (BUG-20)
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
    except Exception:
        git_hash = None
    blob = json.dumps(cfg, sort_keys=True)
    man = dict(git_hash=git_hash, config_hash=hashlib.sha256(blob.encode()).hexdigest()[:12],
               prompt_version=PROMPT_VERSION, seed=cfg.get("seed", 0),
               model=cfg["model"]["name"], python=sys.version.split()[0],
               torch=torch.__version__, transformers=transformers.__version__,
               machine=platform.node(),
               gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
               t=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    return man


def _materialize_injection(arm, s, root, out_dir, others, flipped=False):
    """Apply one known-cause injection to sample s -> injected Sample copy.

    The injection is baked into the copy (media path / boxes / rule text /
    label) so the ordinary T0-T2 plans run on it under arm="base" — exactly
    how escalate() treats real samples (design §5.8b step 2). Gold stays the
    ORIGINAL sample's label except INJ.label, where flipping the gold IS the
    injection (input unchanged, FR-V-2)."""
    import copy
    from ea.interventions import INTERV
    from PIL import Image
    s2 = copy.deepcopy(s)
    if arm == "INJ.label":
        if flipped:
            s2.label = 1 - s.label                       # gold flipped, input unchanged
        return s2
    imgs, over = INTERV[arm]["fn"](s, root=root,
                                   **({"others": others} if arm == "INJ.clutter" else {}))
    if over is not None:
        s2 = over
    if imgs:                                             # media-modifying: persist the canvas
        p = Path(out_dir) / "injected_media" / f"{arm.split('.')[1]}_{s.id}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(p)
        s2.media = p.as_posix()
    # text-only arms (INJ.lead): media path unchanged; run.one reloads the
    # original image when the prompt needs one (BUG-19 fix)
    return s2


def validate(runs, S, cfg, out_dir="outputs/validate", vlm_factory=None):
    """M3b: known-cause injections -> confusion.csv (FR-V-1..3).

    For each INJ.* arm (INJ_EXPECTED in escalate.py is the single source of
    truth for the expected cause): draw per_cause_n confident-correct pool
    samples, materialize the injection, run the FULL T0/T1/T2 diagnostic set
    on the injected samples through vlm_factory (rows resumable in
    <out_dir>/validate_runs.jsonl), build one sub-cluster per
    (injection x rule class) from the samples that now FAIL, run
    run_diagnostics + decide, and tally the actually-assigned cause.
    INJ.label flips gold on a label_noise fraction with the input unchanged;
    any confident cause assigned there counts as a false attribution and
    lowers that cause's measured precision (FR-V-2).

    The confusion matrix rows are the expected causes (SIG spellings); per-
    cause precision/recall with Wilson CIs come from the real counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if vlm_factory is None:
        raise ValueError("validate measures the false-attribution rate on the real "
                         "model: vlm_factory(mcfg)->VLM is required (FR-V-1)")
    val_cfg = cfg.get("validate", {}) if cfg else {}
    per_cause_n = int(val_cfg.get("per_cause_n", 30))
    label_noise = float(val_cfg.get("label_noise", 0.1))
    conf_margin = float((cfg.get("exit", {}) if cfg else {}).get("conf_margin", 1.0))

    from ea.escalate import INJ_EXPECTED, SIG, Cluster, decide, run_diagnostics, wil

    # pool: confident-correct samples under base (BUG-14 / NEW-11b fix: uses errors_table)
    if runs is not None and not runs.empty:
        et_base = errors_table(runs, S, min_votes=2)
        if not et_base.empty:
            confident = set(et_base[(et_base.err == 0) & (et_base.s.abs() >= conf_margin)].sample_id)
            pool = sorted((s for s in S if s.id in confident), key=lambda x: x.id)
        else:
            pool = sorted(S, key=lambda x: x.id)
    else:
        pool = sorted(S, key=lambda x: x.id)
    if not pool:
        raise ValueError("validate: no samples available to inject (run t0 first)")

    assigned_cols = list(SIG.keys()) + ["ambiguous", "inconclusive: more data",
                                        "unexplained → next tier"]
    counts = {exp: {c: 0 for c in assigned_cols} for exp in INJ_EXPECTED.values()}
    n_errors_by_arm: dict[str, int] = {exp: 0 for exp in INJ_EXPECTED.values()}

    vruns_path = out_dir / "validate_runs.jsonl"
    rng = np.random.default_rng(0)
    val_run_cfg = dict(cfg, t2=dict(cfg.get("t2", {}), extra_arms=False))
    for arm, expected in INJ_EXPECTED.items():
        take = min(per_cause_n, len(pool))
        if take < per_cause_n:
            print(f"[validate] {arm}: pool has only {take} < per_cause_n samples "
                  "(underpowered, FR-V-1)")
        chosen = list(rng.choice(len(pool), size=take, replace=False))
        S_inj, flip_ids = [], set()
        if arm == "INJ.label" and label_noise > 0.0:
            # NEW-11a fix: flip enough samples to reach exit.min_n (sufficient statistical power)
            min_n = int((cfg.get("exit", {}) if cfg else {}).get("min_n", 10))
            n_flips = min(take, max(min_n, int(take * label_noise)))
            flip_indices = rng.choice(chosen, size=n_flips, replace=False)
            flip_ids = {pool[i].id for i in flip_indices}
        for i in chosen:
            s = pool[i]
            others = [x.media for x in pool if x.id != s.id][:4]
            S_inj.append(_materialize_injection(arm, s, ".", out_dir, others,
                                                flipped=s.id in flip_ids))
        E_inj = {s.id for s in S_inj}
        # NEW-11c fix: run only diagnostic arms (extra_arms=False)
        jobs = plan_t0(val_run_cfg, S_inj) + plan_t1(val_run_cfg, S_inj, E_inj) + plan_t2(val_run_cfg, S_inj, E_inj)
        n_new = run(jobs, S_inj, val_run_cfg, out=vruns_path, vlm_factory=vlm_factory)
        vruns = load_runs(vruns_path)

        # one sub-cluster per (injection, rule class): real diagnostics on the
        # injected errors, injected corrects as the matched-correct pool
        et = errors_table(vruns, S_inj)
        if et.empty:
            print(f"[validate] {arm}: no valid rows produced — skipped")
            continue
        inj_set = {s.id for s in S_inj}
        et = et[et.sample_id.isin(inj_set)]
        cls_of = {s.id: (s.cls, (s.meta or {}).get("obj_size")) for s in S_inj}
        for (_c, _sz), g in et.groupby(et.sample_id.map(cls_of)):
            e_ids = list(g[g.err == 1].sample_id)
            c_ids = list(g[g.err == 0].sample_id)
            if not e_ids:
                continue                                  # injection did not produce errors here
            cl = Cluster(name=f"{arm}|{_c}|{_sz}", err_ids=e_ids, cor_ids=c_ids)
            run_diagnostics(cl, vruns, cfg)
            cause, _ = decide({k: cl.diags[k] for k in (
                "crop", "bgmask", "hires", "qr_ok", "sharpen", "knob", "qp_ok",
                "necessity", "consistency", "loc", "t4", "causal", "cue", "stream")})
            counts[expected][cause] += len(e_ids)
            n_errors_by_arm[expected] += len(e_ids)
            print(f"[validate] {arm} [{_c}|{_sz}]: {len(e_ids)} injected errors -> {cause}")

    df = pd.DataFrame.from_dict(counts, orient="index")
    df.index.name = "injected_cause"

    precisions, prec_los, prec_his = [], [], []
    recalls, rec_los, rec_his = [], [], []
    n_errs = []
    for cause in df.index:
        n_errs.append(n_errors_by_arm[cause])
        tp = int(df.loc[cause, cause]) if cause in df.columns else 0
        total_pred = int(df[cause].sum()) if cause in df.columns else 0
        total_true = int(df.loc[cause, assigned_cols].sum())
        p = tp / total_pred if total_pred > 0 else 0.0
        p_lo, p_hi = wil(tp, total_pred) if total_pred > 0 else (0.0, 0.0)
        r = tp / total_true if total_true > 0 else 0.0
        r_lo, r_hi = wil(tp, total_true) if total_true > 0 else (0.0, 0.0)
        precisions.append(round(p, 3)); prec_los.append(round(p_lo, 3)); prec_his.append(round(p_hi, 3))
        recalls.append(round(r, 3)); rec_los.append(round(r_lo, 3)); rec_his.append(round(r_hi, 3))

    df["n_errors"] = n_errs
    df["precision"] = precisions
    df["prec_lo"] = prec_los
    df["prec_hi"] = prec_his
    df["recall"] = recalls
    df["rec_lo"] = rec_los
    df["rec_hi"] = rec_his

    csv_path = out_dir / "confusion.csv"
    df.to_csv(csv_path)
    print(f"[validate] wrote {csv_path} "
          f"(injections are synthetic: optimistic bound, single-cause by construction)")
    return csv_path

