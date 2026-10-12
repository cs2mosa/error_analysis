#!/bin/bash
# =============================================================================
# Docker entrypoint — Qwen3-VL Error Analysis Engine (ea)
# =============================================================================
set -e

CMD="$1"
shift || true

case "$CMD" in
    # System & GPU startup assertions
    selfcheck)
        echo "[ea-docker] Running selfcheck..."
        exec python -m ea selfcheck "$@"
        ;;

    # Tier T0: Baseline evaluation
    t0)
        echo "[ea-docker] Running Tier 0 (baseline evaluation)..."
        exec python -m ea t0 "$@"
        ;;

    # Tier T1: Triage (perception, process, rule reasoning)
    t1)
        echo "[ea-docker] Running Tier 1 (triage)..."
        exec python -m ea t1 "$@"
        ;;

    # Tier T2: Interventions & Knobs
    t2)
        echo "[ea-docker] Running Tier 2 (interventions & knobs)..."
        exec python -m ea t2 "$@"
        ;;

    # Latent-hazard track
    hazard)
        echo "[ea-docker] Running latent-hazard track..."
        exec python -m ea hazard "$@"
        ;;

    # Tier T3: Attribution & Saliency
    t3)
        echo "[ea-docker] Running Tier 3 (attribution)..."
        exec python -m ea t3 "$@"
        ;;

    # Tier T4: Mechanistic Probing & Patching
    t4)
        echo "[ea-docker] Running Tier 4 (mechanistic analysis)..."
        exec python -m ea t4 "$@"
        ;;

    # Streaming evaluation
    stream)
        echo "[ea-docker] Running streaming evaluation..."
        exec python -m ea stream "$@"
        ;;

    # Generate root cause report
    report)
        echo "[ea-docker] Generating root cause report..."
        exec python -m ea report "$@"
        ;;

    # Run validation engine (false-attribution measurement)
    validate)
        echo "[ea-docker] Running validation engine..."
        exec python -m ea validate "$@"
        ;;

    # Run pytest test suite
    test|pytest)
        echo "[ea-docker] Running pytest..."
        exec pytest "$@"
        ;;

    # Interactive shell
    bash|sh)
        exec bash
        ;;

    # Help message
    --help|help|"")
        cat <<'EOF'
Qwen3-VL Error Analysis Engine Container
==========================================
Usage: docker run [docker-opts] qwen3vl-error-analysis <COMMAND> [args...]

Available Commands:
  selfcheck     Verify model architecture & weights (e.g. selfcheck --load)
  t0            Run Tier 0 baseline evaluation
  t1            Run Tier 1 triage (Q_P, Q_R, Q_D)
  t2            Run Tier 2 interventions (O1-O7, bgmask, knobs)
  t3            Run Tier 3 attribution (saliency, attention maps, ROAD deletion/insertion)
  t4            Run Tier 4 mechanistic analysis (probes, logit lens, patching)
  stream        Run streaming video evaluation
  report        Generate root_causes.md and root_causes.csv
  validate      Run validation engine with synthetic cause injections
  test          Run pytest test suite (e.g. test -v or test -m gpu)
  bash          Open an interactive shell inside the container

Volume Mounts:
  /workspace/.hf_cache   HuggingFace cache directory (model snapshots)
  /workspace/data        Benchmark datasets (samples.jsonl + media)
  /workspace/outputs     Run artifacts, CSVs, Parquet, and reports
EOF
        ;;

    # Pass-through for arbitrary commands
    *)
        exec "$CMD" "$@"
        ;;
esac
