"""tests/test_integration.py — end-to-end pipeline integration test (M3 gate).

Exercises the full pipeline on synthetic toy data with a scripted model:
1. make_toy data generation
2. T0 plan execution -> runs.jsonl
3. Error table & matched controls extraction
4. T1 & T2 plan execution -> runs.jsonl
5. validate runner -> confusion.csv
6. escalate engine -> clusters.json, diag_vector.parquet, audit_sample.csv
7. report generator -> root_causes.md, root_causes.csv
8. Accounting identity and report assertions
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from ea import data as D
from ea import run as R
from ea.escalate import escalate
from ea.report import write_report


class EndToEndFakeVLM:
    """Scripted VLM producing predictable errors on toy data."""

    def __init__(self, error_ids=None):
        self.error_ids = set(error_ids or ["toy_0000", "toy_0001", "toy_0002"])

    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        # Default correct margin
        s = 2.0
        mass = 0.95
        # We can inspect the text or context if needed, but simple heuristics suffice:
        if "absent" in text.lower():
            s = -2.0
        return dict(s=s, mass=mass, n_vis=64, ms=1.0, mem=0.0)


def test_end_to_end_pipeline(tmp_path):
    # 1. Generate toy dataset
    data_dir = tmp_path / "toy"
    S = D.make_toy(15, root=data_dir, seed=42)
    assert len(S) == 15

    cfg = {
        "model": {"name": "fake-vlm"},
        "answer": {"min_mass": 0.5},
        "exit": {"recovery": 0.5, "control": 0.1, "min_n": 2, "conf_margin": 0.5},
        "slices": {"max_order: 1": 1, "min_support": 2, "fdr": 0.05},
        "validate": {"min_precision": 0.8, "per_cause_n": 10, "label_noise": 0.1},
        "audit": {"k_per_cluster": 5},
        "knobs": {"quality": {"kinds": ["blur"], "severities": [1]}},
    }

    # 2. T0 execution
    runs_path = tmp_path / "runs.jsonl"
    t0_jobs = R.plan_t0(cfg, S)
    assert len(t0_jobs) == 45  # 15 * 3 prompts

    # Custom factory returning scripted responses
    def vlm_factory(mc):
        class ScriptedVLM:
            def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
                # Flip prediction for specific sample IDs to create known errors
                s = 2.0
                return dict(s=s, mass=0.95, n_vis=64, ms=1.0, mem=0.0)

        return ScriptedVLM()

    n0 = R.run(t0_jobs, S, cfg, out=runs_path, vlm_factory=vlm_factory)
    assert n0 == 45
    assert runs_path.exists()

    runs = R.load_runs(runs_path)
    assert len(runs) == 45

    # 3. Extract errors & matched controls
    # Manually ensure at least 2 errors in runs for pipeline test
    runs.loc[runs.sample_id.isin(["toy_0000", "toy_0001"]), "pred"] = 1 - runs.loc[
        runs.sample_id.isin(["toy_0000", "toy_0001"]), "gold"
    ]
    # Re-save runs
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in runs.to_dict(orient="records"):
            f.write(json.dumps(r) + "\n")

    runs = R.load_runs(runs_path)
    et = R.errors_table(runs, S)
    err = et[et.err == 1]
    cor = et[et.err == 0]
    assert len(err) >= 2

    matched, _ = R.match(err, cor)
    E = set(err.sample_id).union(matched)

    # 4. T1 & T2 execution
    t1_jobs = R.plan_t1(cfg, S, E)
    R.run(t1_jobs, S, cfg, out=runs_path, vlm_factory=vlm_factory)

    t2_jobs = R.plan_t2(cfg, S, E)
    R.run(t2_jobs, S, cfg, out=runs_path, vlm_factory=vlm_factory)

    runs = R.load_runs(runs_path)

    # 5. Validate runner
    val_dir = tmp_path / "outputs" / "validate"
    confusion_csv = R.validate(runs, S, cfg, out_dir=val_dir)
    assert confusion_csv.exists()
    conf_df = pd.read_csv(confusion_csv, index_col=0)
    assert "precision" in conf_df.columns
    assert "recall" in conf_df.columns

    # 6. Escalate engine
    esc_dir = tmp_path / "outputs" / "escalate"
    clusters, diag_vec = escalate(runs, S, cfg, out_dir=esc_dir)
    assert len(clusters) >= 1
    assert any(c.name == "ALL_ERRORS" for c in clusters)
    assert (esc_dir / "clusters.json").exists()
    assert (esc_dir / "diag_vector.parquet").exists()
    assert (esc_dir / "audit_sample.csv").exists()

    # 7. Report generator
    md_path, csv_path = write_report(
        clusters_path=esc_dir / "clusters.json",
        diag_vec_path=esc_dir / "diag_vector.parquet",
        out_dir=esc_dir,
        validate_csv=confusion_csv,
        cfg=cfg,
    )
    assert md_path.exists()
    assert csv_path.exists()

    # 8. Report content verification
    report_text = md_path.read_text(encoding="utf-8")
    assert "Root-cause report" in report_text
    assert "Accounting identity (FR-E-11)" in report_text
    assert "Scope of evidence (FR-E-12)" in report_text
    assert "Deployment vs architecture split" in report_text

    # Accounting identity
    diag_df = pd.read_parquet(esc_dir / "diag_vector.parquet")
    csv_df = pd.read_csv(csv_path)
    assert csv_df["n"].sum() == len(diag_df)
    assert len(diag_df) == len(err)
