"""cli.py — argparse subcommands (design §5.13): selfcheck, t0, t1, t2, hazard,
stream, t3, t4, report, validate (FR-R-6; hazard per §5.14/FR-H)."""

import argparse
import sys


def main(argv=None):
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
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # BUG-13: machine-specific overrides (model.path, GPU limits) live in
    # config.local.yaml (git-ignored); config.yaml stays portable.
    local = Path(path).parent / "config.local.yaml"
    if local.exists():
        over = yaml.safe_load(local.read_text(encoding="utf-8")) or {}

        def _merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    _merge(a[k], v)
                else:
                    a[k] = v
        _merge(cfg, over)
    return cfg


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

    # BUG-03: merge the T3/T4 diagnostics written by `ea t3` / `ea t4` so the
    # SIG rows requiring loc/t4/causal can finally match.
    external = {}
    for name, fname in (("outputs/t3", "t3_loc.json"), ("outputs/t4", "t4_verdicts.json")):
        p = Path(name) / fname
        if p.exists():
            try:
                for cluster, diags in json.loads(p.read_text(encoding="utf-8")).items():
                    external.setdefault(cluster, {}).update(diags)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[report] ignoring unreadable {p}: {e!r}")
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

    # --- explainer-validity gate (FR-T3-6): computed on real deletion curves
    gate_ok_all, gate_details = True, []
    for sid in list(err.sample_id)[:5]:
        s = smap[sid]
        inp = vlm.prep(prompt_text(s), images=[open_img(s)])
        amap = map_attn(vlm, inp, ratio=ratio)
        res = explainer_validity_gate(score_fn_factory(s), open_img(s), amap,
                                      steps=4, sign=int(2 * s.label - 1))
        gate_details.append(dict(sample_id=sid, **res))
        gate_ok_all &= bool(res["valid"])
    gate = dict(valid=bool(gate_ok_all and gate_details), n_samples=len(gate_details),
                status="computed (FR-T3-6)" if gate_details else "no error samples",
                tag=None if gate_ok_all else "UNVALIDATED", details=gate_details)
    (out / "t3_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"[t3] explainer gate: valid={gate['valid']} ({len(gate_details)} samples)")

    # --- maps + localization for errors and matched corrects
    rows, enr = [], {}
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
            enr.setdefault(kind, []).append(a_res.get("enrich", np.nan))
            rows.append(dict(sample_id=sid, kind=kind, explainer="attn", **a_res))
            rows.append(dict(sample_id=sid, kind=kind, explainer="grad", **g_res))
    pd.DataFrame(rows).to_csv(out / "t3_loc.csv", index=False)

    # --- SpRAy over error maps (exploratory, C11)
    err_maps = [np.load(out / "maps" / f"{sid}.npz")["attn"] for sid in err.sample_id]
    spray_ids = spray_cluster(err_maps, k=4, seed=0).tolist() if err_maps else []
    pd.DataFrame(dict(sample_id=list(err.sample_id), spray=spray_ids)).to_csv(
        out / "t3_spray.csv", index=False)

    # --- per-cluster 3-way reading (FR-T3-4, C8) — names identical to escalate()
    clusters = build_clusters(errors_table(runs, S), matched_cor, runs, cfg)
    err_ids, cor_ids = list(err.sample_id), list(matched_cor.sample_id)
    loc_cfg = cfg.get("loc", {})
    verdicts = {}
    for c in clusters:
        box_err = [enr["err"][err_ids.index(i)] for i in c.err_ids if i in err_ids]
        box_cor = [enr["cor"][cor_ids.index(i)] for i in c.cor_ids if i in cor_ids]
        if not gate["valid"]:
            verdicts[c.name] = {"loc": None, "t3_gate": "UNVALIDATED"}
        elif not box_err or not box_cor:
            verdicts[c.name] = {"loc": None}          # under-powered: abstain
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
                         knock_effect, internal_label)
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
        return next((b["xyxy"] for b in s.boxes if b["role"] == "violator"),
                    [0.0, 0.0, 1.0, 1.0])

    # --- feature cache (FR-T4-1): stratified subset, errors + matched corrects
    cap = int((cfg.get("t4") or {}).get("max_samples", 40))
    ids_err = list(err.sample_id)[:cap // 2]
    ids_cor = list(matched_cor.sample_id)[:cap // 2]
    from ea.mech import dinov2_pool
    for sid in ids_err + ids_cor:
        s = smap[sid]
        cache_sample(vlm, s, prompt_text(s), viol_box(s), out / "cache")
        dino = dinov2_pool(Image.open(s.media), viol_box(s))   # FR-T4-3 ceiling
        if dino is not None:
            np.savez(out / "cache" / f"{sid}.npz",
                     **{**np.load(out / "cache" / f"{sid}.npz"), "dino": dino})
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
    groups = [pairs[sid] for sid in ids if pairs[sid]] or None
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
    causal = bool(ko_err_v > 0.05 and ko_err_v >= 2 * max(ko_cor_v, 1e-9))
    pd.DataFrame([dict(median_knockout_effect_err=ko_err_v,
                       median_knockout_effect_cor=ko_cor_v, causal=causal)]).to_csv(
        out / "t4_knockout.csv", index=False)

    # --- evidence rule (FR-T4-10): label only when decodability + causal agree
    evidence = []
    if label != "unresolved":
        evidence.append(dict(method="probe", kind="decodability", supports=label))
        if causal:
            evidence.append(dict(method="knockout", kind="causal", supports=label))
    resolved = internal_label(evidence) if evidence else "unresolved (single method)"

    # attach the verdict ONLY to clusters whose errors were actually probed
    clusters = build_clusters(errors_table(runs, S), matched_cor, runs, cfg)
    probed = set(ids_err)
    verdicts = {}
    for c in clusters:
        if probed & set(c.err_ids):
            verdicts[c.name] = dict(
                t4=(label if resolved == label else None),
                causal=causal, verdict=verdict,
                evidence=(f"probe+knockout ({resolved})" if resolved == label
                          else f"unresolved (single method; verdict={verdict})"))
    (out / "t4_verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(f"[t4] probe verdict: {verdict} | causal={causal} | resolved={resolved}")
    print(f"[t4] wrote probes/knockout/verdicts for {len(verdicts)} clusters -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


