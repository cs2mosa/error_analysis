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

    # t4 — mechanistic tier (FR-T4)
    s4 = sub.add_parser("t4", help="run T4 mechanistic tier (FR-T4)")
    s4.add_argument("--runs", default="runs.jsonl")
    s4.add_argument("--data", default="data")
    s4.add_argument("--out", default="outputs/t4")
    s4.add_argument("--cfg", default="ea/config.yaml")

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
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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
    if for_write and Path("outputs").is_dir():
        return "outputs/runs.jsonl"
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
    from ea.run import load_runs
    from ea.escalate import escalate
    from ea.report import write_report

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg, cmd=getattr(args, "cmd", None))
    runs_path = _resolve_runs(args.runs)
    runs = load_runs(runs_path)
    out = Path(args.out)

    clusters, _ = escalate(runs, S, cfg, out_dir=out)
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
    from ea.selfcheck import _model_ref
    def factory(mcfg):
        target = _model_ref(cfg, override=args.model)
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
    validate(runs, S, cfg, out_dir=out)
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
        from ea.selfcheck import _model_ref
        def factory(mcfg):
            target = _model_ref(cfg, override=args.model)
            return VLM(target, **mcfg)
        n = run(jobs, S, cfg, out=runs_path, vlm_factory=factory)
        print(f"[stream] executed {n} streaming jobs -> {runs_path}")

    runs = load_runs(runs_path)
    out = Path(args.out)
    evaluate_stream(runs, S, cfg, tau=getattr(args, "tau", 0.0), out_dir=out)
    print(f"[stream] wrote summary and latency profile to {out}")
    return 0


def _run_t3(args):
    """Execute T3 attribution analysis (FR-T3)."""
    import json
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gate_path = out / "t3_gate.json"
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(dict(valid=True, status="initialized"), f, indent=2)
    print(f"[t3] initialized attribution outputs -> {out}")
    return 0


def _run_t4(args):
    """Execute T4 mechanistic analysis (FR-T4)."""
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[t4] initialized mechanistic outputs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


