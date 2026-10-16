"""cli.py — argparse subcommands (design §5.13): selfcheck, t0, t1, t2, hazard,
stream, t3, t4, report, validate (FR-R-6; hazard per §5.14/FR-H)."""

import argparse
import sys


def build_parser():
    ap = argparse.ArgumentParser(prog="ea")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # selfcheck — verify [VERIFY] items (NFR-3)
    sc = sub.add_parser("selfcheck", help="assert the [VERIFY] items (requirements §9)")
    sc.add_argument("--load", action="store_true",
                    help="also run forward-pass checks (needs the weights)")

    # t0, t1, t2 — tier runners (FR-R-1)
    _add_run_args(sub.add_parser("t0", help="run T0 baseline (FR-R-1)"))
    _add_run_args(sub.add_parser("t1", help="run T1 triage prompts (FR-R-1)"))
    _add_run_args(sub.add_parser("t2", help="run T2 interventions (FR-R-1)"))

    # hazard — latent-hazard track runner (§5.14, FR-H-3/4/5)
    _add_run_args(sub.add_parser("hazard", help="run latent-hazard prompts/arms (FR-H)"))

    # report — read runs.jsonl → root_causes.md + .csv (FR-E-7)
    sr = sub.add_parser("report", help="generate root-cause report (FR-E-7)")
    sr.add_argument("--runs", default="runs.jsonl", metavar="PATH")
    sr.add_argument("--data", default="data", metavar="DIR")
    sr.add_argument("--out", default="outputs/escalate", metavar="DIR")
    sr.add_argument("--validate-csv", default=None, metavar="PATH")
    sr.add_argument("--cfg", default="ea/config.yaml", metavar="PATH")
    sr.add_argument("--t3-dir", default="outputs/t3", metavar="DIR", help="directory containing t3_loc.json")
    sr.add_argument("--t4-dir", default="outputs/t4", metavar="DIR", help="directory containing t4_verdicts.json")

    # validate — known-cause injections → confusion.csv (M3b)
    sv = sub.add_parser("validate", help="measure false-attribution rate (FR-V)")
    sv.add_argument("--runs", default="runs.jsonl")
    sv.add_argument("--data", default="data")
    sv.add_argument("--out", default="outputs/validate")
    sv.add_argument("--cfg", default="ea/config.yaml")
    sv.add_argument("--model", default=None, metavar="PATH_OR_REPO")

    # stream — streaming track runner & evaluation (FR-ST-1..8)
    sst = sub.add_parser("stream", help="run streaming track / offline trigger evaluation (FR-ST-1..8)")
    _add_run_args(sst)
    sst.add_argument("--eval-only", action="store_true", help="skip model jobs, evaluate existing runs")
    sst.add_argument("--out", default="outputs/stream", metavar="DIR")
    sst.add_argument("--tau", type=float, default=0.0)

    # t3 — attribution tier (FR-T3)
    s3 = sub.add_parser("t3", help="run T3 attribution tier (FR-T3)")
    s3.add_argument("--runs", default="runs.jsonl")
    s3.add_argument("--data", default="data")
    s3.add_argument("--out", default="outputs/t3")
    s3.add_argument("--cfg", default="ea/config.yaml")
    s3.add_argument("--model", default=None, metavar="PATH_OR_REPO")

    # t4 — mechanistic tier (FR-T4)
    s4 = sub.add_parser("t4", help="run T4 mechanistic tier (FR-T4)")
    s4.add_argument("--runs", default="runs.jsonl")
    s4.add_argument("--data", default="data")
    s4.add_argument("--out", default="outputs/t4")
    s4.add_argument("--cfg", default="ea/config.yaml")
    s4.add_argument("--model", default=None, metavar="PATH_OR_REPO")

    return ap


def main(argv=None):
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)


    if args.cmd == "selfcheck":
        from ea.selfcheck import main as selfcheck_main
        return selfcheck_main(["--load"] if args.load else rest)

    if args.cmd in ("t0", "t1", "t2", "hazard"):
        return _run_tier(args)

    if args.cmd == "report":
        return _run_report(args)

    if args.cmd == "validate":
        return _run_validate(args)

    if args.cmd == "stream":
        return _run_stream(args)

    if args.cmd == "t3":
        return _run_t3(args)

    if args.cmd == "t4":
        return _run_t4(args)

    ap.error(f"unknown command {args.cmd!r}")
    return 2




def _add_run_args(p):
    p.add_argument("--cfg", default="ea/config.yaml")
    p.add_argument("--runs", default="runs.jsonl")
    p.add_argument("--data", default="data")
    p.add_argument("--model", default=None)


def _load_cfg(path):
    from ea import load_cfg
    return load_cfg(path)


def _resolve_runs(path, for_write=False):
    """Resolve runs.jsonl path.
    If an explicit path was passed (different from the default 'runs.jsonl'), use it.
    Otherwise:
      - If 'outputs/runs.jsonl' already exists, use it.
      - If 'runs.jsonl' exists in current dir, use it.
      - If writing a new run and 'outputs/' exists, write to 'outputs/runs.jsonl'.
      - Fallback: 'outputs/runs.jsonl' if 'outputs/' exists else 'runs.jsonl'.
    """
    from pathlib import Path
    if path is not None and path not in ("runs.jsonl", "outputs/runs.jsonl"):
        return str(path)
    if Path("outputs/runs.jsonl").exists():
        return "outputs/runs.jsonl"
    if Path("runs.jsonl").exists():
        return "runs.jsonl"
    if Path("outputs").is_dir():
        return "outputs/runs.jsonl"
    return "runs.jsonl"


def _load_samples(data_dir, cfg, cmd=None):
    """Load samples.jsonl:
      1. Explicit path directly pointing to a folder with samples.jsonl.
      2. Root directory containing existing bench sub-folders -> load cached bench.
      3. Explicit known bench name (toy, cs10k, pasbench, ...) -> auto-download/generate.
      4. No dataset passed within CLI / empty data dir:
         Mirrors model weight resolution (_model_ref): auto-resolve default benchmark
         (ENV -> cfg -> cs10k/pasbench), auto-download and persist locally (FR-D-2).
    """
    import os
    from pathlib import Path
    from ea.data import load, download_bench, _BENCH_REGISTRY

    # Check environment variable overrides (e.g. DATA_PATH or BENCH)
    env_data = os.environ.get("DATA_PATH") or os.environ.get("DATA_DIR")
    if (not data_dir or data_dir == "data") and env_data:
        data_dir = env_data

    p = Path(data_dir) if data_dir else Path("data")

    # Case 1: --data points directly at a bench folder with samples.jsonl
    if (p / "samples.jsonl").exists():
        return load(p.name, root=str(p.parent))

    # Case 2: --data is a root containing bench sub-folders
    if p.is_dir():
        children = sorted(p.glob("*/samples.jsonl"))
        if children:
            target_child = children[0]
            if cmd == "stream":
                for c in children:
                    if c.parent.name in ("pasbench", "ovo_bench_rt"):
                        target_child = c
                        break
            print(f"[data] using local cached benchmark -> {target_child.parent}")
            return load(target_child.parent.name, root=str(p))

    # Case 3 / 4: treat data_dir as a bench name (either known registry or a path basename)
    bench_name = p.name  # e.g. "toy", "cs10k", "pasbench"
    root_dir = p.parent if p.parent != Path(".") and p.parent != p else Path("data")

    # If data_dir itself looks like a known bench name (no path separators), resolve root
    if str(data_dir) == bench_name or not p.parts[:-1]:
        root_dir = Path("data")

    if bench_name in _BENCH_REGISTRY or bench_name == "toy":
        return download_bench(bench_name, root=root_dir)

    # Case 4: No dataset passed within the CLI or data directory is empty.
    # Mirrors the VLM model resolution (_model_ref): download once, persist locally.
    if data_dir in (None, "data", ""):
        default_bench = os.environ.get("BENCH") or os.environ.get("DATASET")
        if not default_bench and cfg:
            default_bench = (cfg.get("data", {}).get("bench") or
                             cfg.get("data", {}).get("name") or
                             cfg.get("bench"))
        if not default_bench:
            default_bench = "pasbench" if cmd == "stream" else "cs10k"

        print(f"[data] no dataset passed within CLI; auto-downloading '{default_bench}' -> {root_dir / default_bench}")
        return download_bench(default_bench, root=root_dir)

    raise FileNotFoundError(
        f"No samples.jsonl found under '{data_dir}'. "
        f"Pass a bench folder path, a root containing bench folders, "
        f"or a known bench name ({', '.join(sorted(_BENCH_REGISTRY))})."
    )



def _run_report(args):
    from pathlib import Path
    import json
    from ea.run import load_runs
    from ea.escalate import escalate
    from ea.report import write_report

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs_path = _resolve_runs(args.runs)
    runs = load_runs(runs_path)
    out = Path(args.out)

    # BUG-03 / NEW-10: merge T3/T4 diagnostics from configurable directories
    t3_dir = Path(getattr(args, "t3_dir", "outputs/t3") or "outputs/t3")
    t4_dir = Path(getattr(args, "t4_dir", "outputs/t4") or "outputs/t4")
    external = {}
    for p_dir, fname in ((t3_dir, "t3_loc.json"), (t4_dir, "t4_verdicts.json")):
        p = p_dir / fname
        if p.exists():
            try:
                for cluster, diags in json.loads(p.read_text(encoding="utf-8")).items():
                    external.setdefault(cluster, {}).update(diags)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[report] ignoring unreadable {p}: {e!r}")
        else:
            print(f"[report] note: {p} not found; run '{fname.split('_')[0]}' tier if that evidence is desired.")
    if external:
        print(f"[report] merged T3/T4 diagnostics for {len(external)} clusters")

    clusters, _ = escalate(runs, S, cfg, out_dir=out, external_diags=external or None)
    if not clusters:
        print("[report] escalation produced no clusters.")
        return 1

    md, csv = write_report(
        clusters_path=out / "clusters.json",
        diag_vec_path=out / "diag_vector.parquet",
        out_dir=out,
        validate_csv=getattr(args, "validate_csv", None),
        cfg=cfg,
    )
    print(f"[report] {md}")
    return 0


def _run_tier(args):
    """Execute T0, T1, T2 or hazard-track tier jobs."""
    from ea.run import (plan_t0, plan_t1, plan_t2, plan_hazard, run,
                        load_runs, errors_table, match)

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs_path = _resolve_runs(args.runs, for_write=True)

    if args.cmd == "t0":
        jobs = plan_t0(cfg, S)
    elif args.cmd in ("t1", "t2", "hazard"):
        runs = load_runs(runs_path)
        et = errors_table(runs, S)
        if et.empty or "err" not in et.columns:
            print(f"[{args.cmd}] no error samples found in {runs_path}. Run t0 first.")
            return 1
        err = et[et.err == 1]
        cor = et[et.err == 0]
        matched, _ = match(err, cor)
        E = set(err.sample_id).union(matched)
        if args.cmd == "t1":
            jobs = plan_t1(cfg, S, E)
        elif args.cmd == "t2":
            jobs = plan_t2(cfg, S, E)
        else:
            jobs = plan_hazard(cfg, S, E)
    else:
        return 1

    from ea.model import VLM
    from ea import model_ref                 # BUG-12: single resolver, no torch import
    def factory(mcfg):
        target = model_ref(cfg, override=args.model)
        return VLM(target, **mcfg)

    if not jobs:
        print(f"[{args.cmd}] no jobs to run (no hazard-track samples/errors?)")
        return 0
    n = run(jobs, S, cfg, out=runs_path, vlm_factory=factory)
    print(f"[{args.cmd}] executed {n} new jobs -> {runs_path}")
    return 0


def _run_validate(args):
    """M3b: inject known causes, run diagnostics, write confusion.csv."""
    from pathlib import Path
    from ea.run import load_runs, validate

    cfg = _load_cfg(args.cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs_path = _resolve_runs(args.runs)
    runs = load_runs(runs_path)
    from ea.model import VLM
    from ea import model_ref
    def factory(mcfg):
        return VLM(model_ref(cfg, override=args.model), **mcfg)
    validate(runs, S, cfg, out_dir=out, vlm_factory=factory)
    return 0


def _run_stream(args):
    """Execute streaming track jobs and/or evaluate offline policies (FR-ST-1..8)."""
    from pathlib import Path
    from ea.run import plan_stream, run, load_runs
    from ea.stream import evaluate_stream

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs_path = _resolve_runs(args.runs, for_write=not getattr(args, "eval_only", False))

    if not getattr(args, "eval_only", False):
        jobs = plan_stream(cfg, S)
        from ea.model import VLM
        from ea import model_ref
        def factory(mcfg):
            target = model_ref(cfg, override=args.model)
            return VLM(target, **mcfg)
        n = run(jobs, S, cfg, out=runs_path, vlm_factory=factory)
        print(f"[stream] executed {n} streaming jobs -> {runs_path}")

    runs = load_runs(runs_path)
    out = Path(args.out)
    evaluate_stream(runs, S, cfg, tau=getattr(args, "tau", 0.0), out_dir=out)
    print(f"[stream] wrote summary and latency profile to {out}")
    return 0


def _common_context(args, for_write=False):
    """cfg / samples / runs / error table / matched corrects shared by the
    t3 / t4 / report / validate drivers."""
    from ea.run import load_runs, errors_table, match
    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs = load_runs(_resolve_runs(args.runs, for_write=for_write))
    if runs.empty:
        raise SystemExit(f"[{args.cmd}] no runs found — run t0 first (FR-R-6)")
    et = errors_table(runs, S)
    if et.empty:
        raise SystemExit(f"[{args.cmd}] no valid base D rows in runs — run t0 first")
    err, cor = et[et.err == 1], et[et.err == 0]
    matched, _ = match(err, cor)
    matched_cor = cor[cor.sample_id.isin(matched)] if len(cor) else cor
    return cfg, S, runs, err, matched_cor


def _make_eager_vlm(cfg, args):
    """T3/T4 read model internals: hooks need attn='eager' (design §5.3)."""
    from ea.model import VLM
    from ea import model_ref
    return VLM(model_ref(cfg, override=args.model), attn="eager")


def _explainer_gate(vlm, calib_ids, smap, prompt_text, open_img, score_fn_factory, t3_cfg, ratio):
    """Run the FR-T3-6 explainer-validity gate on calibration samples and aggregate it:
    per-sample deletion/insertion vs random and raw-attention maps (debiased fill, rank
    stability across fills, internal agreement), then paired Wilcoxon, the layer
    re-initialization sanity check and the noise-robustness check (gate_aggregate)."""
    import numpy as np
    import torch
    from PIL import Image
    from ea.prompts import PROMPTS
    from ea.attrib import (map_attn, map_spearman, explainer_validity_gate, gate_aggregate)
    from ea.model import knockout, randomized_layers

    gate_steps = int(t3_cfg.get("gate_steps", 4))
    pos, neg = PROMPTS["d0"]["pos"], PROMPTS["d0"]["neg"]
    lm = getattr(getattr(vlm.m, "model", vlm.m), "language_model", None)
    n_layers = len(getattr(lm, "layers", []))
    sigma = float(t3_cfg.get("gate_noise_sigma", 0.02)) * 255.0

    gate_details, cal, noise_rho, skipped_reason = [], [], [], {}
    nrng = np.random.default_rng(0)
    for sid in calib_ids:
        s = smap[sid]
        img = open_img(s)
        inp = vlm.prep(prompt_text(s), images=[img])
        amap = map_attn(vlm, inp, ratio=ratio)
        base_map = map_attn(vlm, inp, ratio=float("inf"))        # raw attention, sinks kept
        s_base = score_fn_factory(s)(img)
        g = vlm.groups(inp)

        def ko_fn(idx, inp=inp, g=g, s_base=s_base):
            sub = g["vis"][torch.as_tensor(np.asarray(idx), device=g["vis"].device)]
            with knockout(vlm, inp, [(sub, "last")], range(n_layers)):
                return abs(s_base) - abs(vlm.score_inp(inp, pos, neg))

        res = explainer_validity_gate(score_fn_factory(s), img, amap, steps=gate_steps,
                                      sign=1 if s_base >= 0 else -1,
                                      ko_fn=ko_fn if n_layers else None, base_map=base_map)
        gate_details.append(dict(sample_id=sid, **res))
        cal.append((inp, amap))

        noisy = Image.fromarray((np.asarray(img, float) + nrng.normal(0, sigma, (img.height, img.width, 3))
                                 ).clip(0, 255).astype(np.uint8))
        noise_rho.append(map_spearman(amap, map_attn(
            vlm, vlm.prep(prompt_text(s), images=[noisy]), ratio=ratio)))

    rand_rho = None
    try:
        with randomized_layers(vlm, frac=float(t3_cfg.get("gate_rand_frac", 0.5)), seed=0):
            rand_rho = [map_spearman(a0, map_attn(vlm, i0, ratio=ratio)) for i0, a0 in cal]
    except RuntimeError as e:
        skipped_reason["randomization"] = str(e)

    gate = gate_aggregate(
        gate_details, alpha=float(t3_cfg.get("gate_alpha", 0.05)),
        min_pass_rate=float(t3_cfg.get("gate_pass_rate", 0.7)),
        randomization_rho=rand_rho, robustness_rho=noise_rho or None,
        rand_rho_max=float(t3_cfg.get("gate_rand_rho_max", 0.5)),
        noise_rho_min=float(t3_cfg.get("gate_noise_rho_min", 0.6)))
    gate.update(tag=None if gate["valid"] else "UNVALIDATED",
                skipped_reason=skipped_reason, details=gate_details)
    return gate


def _run_t3(args):
    """T3 attribution tier (FR-T3): attention/gradient maps on the token grid,
    localization vs the GT box, the 3-way `loc` reading per cluster, the
    computed explainer-validity gate (never a hard-coded valid:true — BUG-02),
    and SpRAy clustering of the error maps (exploratory, C11)."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    from ea.prompts import PROMPTS, render
    from ea.attrib import (map_attn, map_grad, loc, three_way,
                           explainer_validity_gate, spray_cluster)
    from ea.escalate import build_clusters
    from ea.run import errors_table

    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    cfg, S, runs, err, matched_cor = _common_context(args)
    vlm = _make_eager_vlm(cfg, args)
    smap = {s.id: s for s in S}
    ratio = float(cfg.get("sink", {}).get("ratio", 10))

    def prompt_text(s):
        return render(PROMPTS["d0"], s)

    def open_img(s):
        return Image.open(s.media).convert("RGB")

    def score_fn_factory(s):
        text, pos, neg = prompt_text(s), PROMPTS["d0"]["pos"], PROMPTS["d0"]["neg"]
        return lambda im: vlm.score_inp(vlm.prep(text, images=[im]), pos, neg)

    # --- explainer-validity gate (FR-T3-6): per-sample checks, then a dataset-level test
    t3_cfg = cfg.get("t3", {})
    calib_ids = list(matched_cor.sample_id)[:int(t3_cfg.get("gate_samples", 10))] \
        or list(err.sample_id)[:int(t3_cfg.get("gate_samples", 10))]
    gate = _explainer_gate(vlm, calib_ids, smap, prompt_text, open_img, score_fn_factory,
                           t3_cfg, ratio)
    (out / "t3_gate.json").write_text(json.dumps(gate, indent=2, default=float), encoding="utf-8")
    print(f"[t3] explainer gate: valid={gate['valid']} ({gate['status']}; pass_rate={gate['pass_rate']}, "
          f"p_del={gate['p_del_rand']}, p_ins={gate['p_ins_rand']}, skipped={gate['skipped']})")

    # --- maps + localization for errors and matched corrects (O(1) dictionary lookup)
    rows, enr_map = [], {}
    for ids, kind in ((list(err.sample_id), "err"), (list(matched_cor.sample_id), "cor")):
        for sid in ids:
            s = smap[sid]
            inp = vlm.prep(prompt_text(s), images=[open_img(s)])
            amap = map_attn(vlm, inp, ratio=ratio)
            gmap = map_grad(vlm, inp)
            np.savez(out / "maps" / f"{sid}.npz", attn=amap, grad=gmap)
            vbox = next((b["xyxy"] for b in s.boxes if b["role"] == "violator"), None)
            a_res = loc(amap, vbox) if vbox else {}
            g_res = loc(gmap, vbox) if vbox else {}
            enr_map[(kind, sid)] = a_res.get("enrich", np.nan)
            rows.append(dict(sample_id=sid, kind=kind, explainer="attn", **a_res))
            rows.append(dict(sample_id=sid, kind=kind, explainer="grad", **g_res))
    pd.DataFrame(rows).to_csv(out / "t3_loc.csv", index=False)

    # --- SpRAy over error maps (exploratory, C11)
    spray_k = int(t3_cfg.get("spray_k", 4))
    err_maps = [np.load(out / "maps" / f"{sid}.npz")["attn"] for sid in err.sample_id]
    spray_ids = spray_cluster(err_maps, k=spray_k, seed=0).tolist() if err_maps else []
    pd.DataFrame(dict(sample_id=list(err.sample_id), spray=spray_ids)).to_csv(
        out / "t3_spray.csv", index=False)

    # --- per-cluster 3-way reading (FR-T3-4, C8) — names identical to escalate()
    clusters = build_clusters(errors_table(runs, S), matched_cor, runs, cfg)
    loc_cfg = cfg.get("loc", {})
    min_samples = int(loc_cfg.get("min_samples", 3))
    verdicts = {}
    for c in clusters:
        box_err = [enr_map[("err", i)] for i in c.err_ids if ("err", i) in enr_map]
        box_cor = [enr_map[("cor", i)] for i in c.cor_ids if ("cor", i) in enr_map]
        # Filter NaNs from absence scenes without violator boxes (NEW-08a)
        box_err = [x for x in box_err if np.isfinite(x)]
        box_cor = [x for x in box_cor if np.isfinite(x)]
        if not gate["valid"]:
            verdicts[c.name] = {"loc": None, "t3_gate": "UNVALIDATED"}
        elif len(box_err) < min_samples or len(box_cor) < min_samples:
            verdicts[c.name] = {"loc": None}          # under-powered / absence scenes: abstain
        else:
            verdicts[c.name] = {"loc": three_way(
                box_err, box_cor,
                shortcut_enrichment=float(loc_cfg.get("shortcut_enrichment", 1.5)),
                equal_delta=float(loc_cfg.get("equal_delta", 0.15)))}
    (out / "t3_loc.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(f"[t3] wrote gate, maps, localization and {len(verdicts)} cluster verdicts -> {out}")
    return 0


def _run_t4(args):
    """T4 mechanistic tier (FR-T4): pooled site-feature cache, linear probes
    with selectivity + the differential matched-correct guard (C12), probe
    verdict -> t4 tag, knockout causal evidence, evidence rule (FR-T4-10).
    Writes t4_verdicts.json for escalate/report to merge (BUG-03)."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    from ea.prompts import PROMPTS, render
    from ea.mech import (cache_sample, probe, probe_verdict, t4_tag,
                         knock_effect, internal_label, patch_scan, is_patch_causal)
    from ea.escalate import build_clusters, wil
    from ea.run import errors_table

    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    cfg, S, runs, err, matched_cor = _common_context(args)
    vlm = _make_eager_vlm(cfg, args)
    smap = {s.id: s for s in S}

    def prompt_text(s):
        return render(PROMPTS["d0"], s)

    def viol_box(s):
        v = next((b["xyxy"] for b in s.boxes if b["role"] == "violator"), None)
        if v is None and (s.meta or {}).get("vtype") == "absence":
            return None  # absence scenes have no violator object: abstain rather than whole-image fallback
        return v or [0.0, 0.0, 1.0, 1.0]

    # --- feature cache (FR-T4-1): stratified subset, errors + matched corrects
    t4_cfg = cfg.get("t4") or {}
    cap = int(t4_cfg.get("max_samples", 40))
    dino_model = str(t4_cfg.get("dino_model", "facebook/dinov2-base"))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    err_list = list(err.sample_id)
    cor_list = list(matched_cor.sample_id)
    k_err = min(len(err_list), cap // 2)
    k_cor = min(len(cor_list), cap // 2)
    ids_err = list(rng.choice(err_list, size=k_err, replace=False)) if k_err else []
    ids_cor = list(rng.choice(cor_list, size=k_cor, replace=False)) if k_cor else []

    from ea.mech import dinov2_pool
    for sid in ids_err + ids_cor:
        s = smap[sid]
        vbox = viol_box(s)
        cache_sample(vlm, s, prompt_text(s), vbox, out / "cache")
        dino = dinov2_pool(Image.open(s.media), vbox, model_name=dino_model)   # FR-T4-3 ceiling
        if dino is not None:
            cache_p = out / "cache" / f"{sid}.npz"
            with np.load(cache_p) as prev:
                merged_feats = {**prev, "dino": dino}
            np.savez(cache_p, **merged_feats)
    if len(ids_err) + len(ids_cor) < 4:
        print("[t4] too few cached samples — writing empty verdicts")
        (out / "t4_verdicts.json").write_text("{}", encoding="utf-8")
        return 0

    # --- probes per site/layer (FR-T4-2)
    feats = {}
    for sid in ids_err + ids_cor:
        with np.load(out / "cache" / f"{sid}.npz") as z:
            feats[sid] = {k: z[k].astype(np.float32) for k in z.files}
    ids = list(feats)
    y = np.array([int(smap[sid].label) for sid in ids])
    is_err = np.array([sid in set(ids_err) for sid in ids])
    pairs = {sid: smap[sid].pair_id for sid in ids}
    # NEW-09(a): fall back to sid so groups is never shorter than ids and y
    groups = [pairs[sid] if pairs.get(sid) else sid for sid in ids]
    dec_acc = float(cfg.get("probe", {}).get("decodable_acc", 0.70))
    min_sel = float(cfg.get("probe", {}).get("selectivity", 0.15))

    def _decodable(site):
        """(decodable_err, acc_err, acc_cor) for one cached site (C12)."""
        if any(site not in feats[sid] for sid in ids):
            return None, np.nan, np.nan
        X = np.stack([feats[sid][site] for sid in ids])
        if len(np.unique(y)) < 2 or len(y) < 8:
            return None, np.nan, np.nan
        pred, acc, ctl = probe(X, y, groups=groups, seed=0)
        e_idx, c_idx = np.where(is_err)[0], np.where(~is_err)[0]
        acc_err = float((pred[e_idx] == y[e_idx]).mean()) if len(e_idx) and \
            len(np.unique(y[e_idx])) == 2 else np.nan
        acc_cor = float((pred[c_idx] == y[c_idx]).mean()) if len(c_idx) and \
            len(np.unique(y[c_idx])) == 2 else np.nan
        k_err = int((pred[e_idx] == y[e_idx]).sum()) if len(e_idx) else 0
        lo, _ = wil(k_err, len(e_idx))
        dec = (not np.isnan(acc_err) and lo >= dec_acc and (acc - ctl) >= min_sel)
        return dec, acc_err, acc_cor

    sites = sorted({k for f in feats.values() for k in f})
    probe_rows, dec_err, acc_cor = [], {}, {}
    for site in sites:
        d_e, a_e, a_c = _decodable(site)
        dec_err[site], acc_cor[site] = d_e, a_c
        probe_rows.append(dict(site=site, decodable=d_e, acc_err=a_e, acc_cor=a_c))
    pd.DataFrame(probe_rows).to_csv(out / "t4_probes.csv", index=False)

    layer_ids = sorted({int(k[1:].split("_")[0]) for k in sites if k.startswith("L")})

    def _site_true(site):
        d = dec_err.get(site)
        return d is True

    vit = _site_true("vit")
    merger = _site_true("merger")
    dino = dec_err.get("dino")
    layers = {}
    for l in layer_ids:
        any_d = any(_site_true(k) for k in (f"L{l}_vis", f"L{l}_q", f"L{l}_last"))
        # C12 differential guard: a per-layer loss counts only if matched
        # corrects REMAIN decodable there; otherwise it is the generic horizon.
        cor_accs = [acc_cor.get(k) for k in (f"L{l}_vis", f"L{l}_q", f"L{l}_last")]
        cor_decodable = any(a is not None and not np.isnan(a) and a >= dec_acc
                            for a in cor_accs)
        layers[l] = bool(any_d or not cor_decodable)
    d = {"dino": bool(dino), "vit": bool(vit), "merger": bool(merger), "layers": layers}
    verdict = probe_verdict(d)
    label = t4_tag(verdict)

    # --- knockout causal evidence: vis→q / vis→last in the upper half layers
    half = len(vlm.m.model.language_model.layers) // 2
    ko_err, ko_cor = [], []
    for ids_k, bucket in ((ids_err[:6], ko_err), (ids_cor[:6], ko_cor)):
        for sid in ids_k:
            s = smap[sid]
            inp = vlm.prep(prompt_text(s), images=[Image.open(s.media).convert("RGB")])
            eff = knock_effect(vlm, inp, PROMPTS["d0"]["pos"], PROMPTS["d0"]["neg"],
                               spec=(("vis", "q"), ("vis", "last")),
                               window=(half, half * 2))
            bucket.append(abs(float(eff)))
    ko_err_v = float(np.median(ko_err)) if ko_err else 0.0
    ko_cor_v = float(np.median(ko_cor)) if ko_cor else 0.0
    causal_min_effect = float(t4_cfg.get("causal_min_effect", 0.05))
    causal_ratio = float(t4_cfg.get("causal_ratio", 2.0))
    causal = bool(ko_err_v > causal_min_effect and ko_err_v >= causal_ratio * max(ko_cor_v, 1e-9))
    pd.DataFrame([dict(median_knockout_effect_err=ko_err_v,
                       median_knockout_effect_cor=ko_cor_v, causal=causal)]).to_csv(
        out / "t4_knockout.csv", index=False)

    # --- activation patching causal evidence (FR-T4-8 / C13; 2309.16042)
    patch_causal = False
    if t4_cfg.get("run_patching", False) and ids_err and ids_cor:
        try:
            s_clean = smap[ids_cor[0]]
            s_corr = smap[ids_err[0]]
            c_img = Image.open(s_clean.media).convert("RGB")
            c_inp = vlm.prep(prompt_text(s_clean), images=[c_img])
            k_inp = vlm.prep(prompt_text(s_corr), images=[Image.open(s_corr.media).convert("RGB")])
            patch_results = []
            if c_inp["input_ids"].shape == k_inp["input_ids"].shape:
                p_res_pair = patch_scan(vlm, c_inp, k_inp, PROMPTS["d0"]["pos"], PROMPTS["d0"]["neg"],
                                        return_raw=True)
                patch_results.append(p_res_pair)
                p_rows = [dict(corruption="pair", group=pg, layer=l, recovery=v["recovery"], delta_s=v["delta_s"])
                          for (pg, l), v in p_res_pair.items()]

                # Second corruption: box Gaussian noise per config.yaml (Zhang & Nanda 2023)
                m_arr = np.asarray(c_img).copy()
                H, W = m_arr.shape[:2]
                box = s_clean.boxes[0] if getattr(s_clean, "boxes", None) else [0.25, 0.25, 0.75, 0.75]
                x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
                m_arr[y0:y1, x0:x1] = np.random.default_rng(0).normal(128, 40, size=(max(1, y1 - y0), max(1, x1 - x0), 3)).clip(0, 255)
                noise_inp = vlm.prep(prompt_text(s_clean), images=[Image.fromarray(m_arr.astype(np.uint8))])
                if c_inp["input_ids"].shape == noise_inp["input_ids"].shape:
                    p_res_box = patch_scan(vlm, c_inp, noise_inp, PROMPTS["d0"]["pos"], PROMPTS["d0"]["neg"],
                                           return_raw=True)
                    patch_results.append(p_res_box)
                    p_rows.extend([dict(corruption="boxnoise", group=pg, layer=l, recovery=v["recovery"], delta_s=v["delta_s"])
                                   for (pg, l), v in p_res_box.items()])

                pd.DataFrame(p_rows).to_csv(out / "t4_patch.csv", index=False)
                patch_causal = is_patch_causal(patch_results)
        except Exception as e:
            print(f"[t4] warning: patch_scan skipped ({e})")

    # --- evidence rule (FR-T4-10): label only when decodability + causal agree
    evidence = []
    if label != "unresolved":
        evidence.append(dict(method="probe", kind="decodability", supports=label))
        if causal:
            evidence.append(dict(method="knockout", kind="causal", supports=label))
        if patch_causal:
            evidence.append(dict(method="patching", kind="causal", supports=label))
    resolved = internal_label(evidence) if evidence else "unresolved (single method)"

    # attach the verdict ONLY to clusters whose errors were actually probed
    clusters = build_clusters(errors_table(runs, S), matched_cor, runs, cfg)
    probed = set(ids_err)
    verdicts = {}
    for c in clusters:
        c_probed_errs = [sid for sid in c.err_ids if sid in probed]
        if c_probed_errs:
            has_causal = causal or patch_causal
            verdicts[c.name] = dict(
                t4=(label if resolved == label else None),
                causal=has_causal, verdict=verdict,
                pooled=True,
                n_probed_errors=len(c_probed_errs),
                evidence=((f"probe+causal ({resolved})" if has_causal
                           else f"unresolved (single method; verdict={verdict})") if resolved == label
                          else f"unresolved (single method; verdict={verdict})"))
    (out / "t4_verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(f"[t4] probe verdict: {verdict} | causal={causal} | resolved={resolved}")
    print(f"[t4] wrote probes/knockout/verdicts for {len(verdicts)} clusters -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


