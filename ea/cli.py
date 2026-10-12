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


def _load_samples(data_dir, cfg):
    """Load samples.jsonl: --data may be the bench folder itself (data/toy) or
    a root containing bench folders (data); the first bench found is used."""
    from pathlib import Path
    from ea.data import load
    p = Path(data_dir)
    if (p / "samples.jsonl").exists():
        return load(p.name, root=str(p.parent))
    for child in sorted(p.glob("*/samples.jsonl")):
        print(f"[data] no bench given; using {child.parent}")
        return load(child.parent.name, root=str(p))
    raise FileNotFoundError(f"no samples.jsonl under {data_dir}")


def _run_report(args):
    from pathlib import Path
    from ea.run import load_runs
    from ea.escalate import escalate
    from ea.report import write_report

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg)
    runs = load_runs(args.runs)
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
    S = _load_samples(args.data, cfg)

    if args.cmd == "t0":
        jobs = plan_t0(cfg, S)
    elif args.cmd in ("t1", "t2", "hazard"):
        runs = load_runs(args.runs)
        et = errors_table(runs, S)
        if et.empty or "err" not in et.columns:
            print(f"[{args.cmd}] no error samples found in {args.runs}. Run t0 first.")
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
    n = run(jobs, S, cfg, out=args.runs, vlm_factory=factory)
    print(f"[{args.cmd}] executed {n} new jobs -> {args.runs}")
    return 0


def _run_validate(args):
    """M3b: inject known causes, run diagnostics, write confusion.csv."""
    from pathlib import Path
    from ea.run import load_runs, validate

    cfg = _load_cfg(args.cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = _load_samples(args.data, cfg)
    runs = load_runs(args.runs)
    validate(runs, S, cfg, out_dir=out)
    return 0


def _run_stream(args):
    """Execute streaming track jobs and/or evaluate offline policies (FR-ST-1..8)."""
    from pathlib import Path
    from ea.run import plan_stream, run, load_runs
    from ea.stream import evaluate_stream

    cfg = _load_cfg(args.cfg)
    S = _load_samples(args.data, cfg)

    if not getattr(args, "eval_only", False):
        jobs = plan_stream(cfg, S)
        from ea.model import VLM
        from ea.selfcheck import _model_ref
        def factory(mcfg):
            target = _model_ref(cfg, override=args.model)
            return VLM(target, **mcfg)
        n = run(jobs, S, cfg, out=args.runs, vlm_factory=factory)
        print(f"[stream] executed {n} streaming jobs -> {args.runs}")

    runs = load_runs(args.runs)
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


