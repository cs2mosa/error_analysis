"""tests/test_report.py — tests for ea/report.py (FR-E-7/8/11/12, FR-V-3).

Checks:
- root_causes.md and root_causes.csv are generated.
- Accounting identity holds: total errors matches diag_vector.parquet rows.
- Scope of evidence section is present.
- Deployment vs architecture split is correct.
- Without validate: causes are marked candidate.
- With validate: causes with precision >= min_precision are marked decisive.
- With low precision: causes are marked candidate.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from ea.report import write_report


@pytest.fixture
def sample_report_data(tmp_path):
    # Create synthetic clusters.json
    clusters = [
        {
            "name": "ALL_ERRORS",
            "n_errors": 10,
            "cause": "deployment knob (H4)",
            "tag": "deployment",
            "pred": {},
            "cause_candidates": [],
        },
        {
            "name": "cluster_clutter",
            "n_errors": 6,
            "cause": "clutter/binding (H8/H5)",
            "tag": "perception",
            "pred": {"clutter": 1},
            "cause_candidates": [],
        },
        {
            "name": "cluster_unexplained",
            "n_errors": 4,
            "cause": "unexplained → next tier",
            "tag": "unknown",
            "pred": {},
            "cause_candidates": ["merger bottleneck (H1)"],
        },
    ]
    clusters_path = tmp_path / "clusters.json"
    clusters_path.write_text(json.dumps(clusters), encoding="utf-8")

    # Create synthetic diag_vector.parquet
    diag_rows = []
    # 5 deployment knob (H4)
    for i in range(5):
        diag_rows.append({"sample_id": f"s_{i}", "cause": "deployment knob (H4)"})
    # 3 clutter/binding (H8/H5)
    for i in range(5, 8):
        diag_rows.append({"sample_id": f"s_{i}", "cause": "clutter/binding (H8/H5)"})
    # 2 unexplained
    for i in range(8, 10):
        diag_rows.append({"sample_id": f"s_{i}", "cause": "unexplained → next tier"})

    diag_df = pd.DataFrame(diag_rows)
    diag_path = tmp_path / "diag_vector.parquet"
    diag_df.to_parquet(diag_path)

    return clusters_path, diag_path


def test_report_creates_md_and_csv(tmp_path, sample_report_data):
    clusters_path, diag_path = sample_report_data
    out_dir = tmp_path / "out"
    md_path, csv_path = write_report(clusters_path, diag_path, out_dir)

    assert md_path.exists()
    assert csv_path.exists()
    assert md_path.name == "root_causes.md"
    assert csv_path.name == "root_causes.csv"

    # Check CSV contents
    df = pd.read_csv(csv_path)
    assert "cause" in df.columns
    assert "n" in df.columns
    assert "pct" in df.columns
    assert "implication" in df.columns
    assert "side" in df.columns
    assert "verdict" in df.columns
    assert df["n"].sum() == 10


def test_report_accounting_identity_and_scope(tmp_path, sample_report_data):
    clusters_path, diag_path = sample_report_data
    out_dir = tmp_path / "out"
    md_path, _ = write_report(clusters_path, diag_path, out_dir)
    text = md_path.read_text(encoding="utf-8")

    # FR-E-11 accounting identity
    assert "Accounting identity (FR-E-11)" in text
    assert "Total errors: **10**" in text

    # FR-E-12 scope of evidence
    assert "Scope of evidence (FR-E-12)" in text
    assert "Injections are synthetic" in text
    assert "Controls are matched but not randomized" in text


def test_report_deployment_vs_architecture_split(tmp_path, sample_report_data):
    clusters_path, diag_path = sample_report_data
    out_dir = tmp_path / "out"
    md_path, csv_path = write_report(clusters_path, diag_path, out_dir)
    text = md_path.read_text(encoding="utf-8")

    # deployment knob (H4) is deployment: 5 errors = 50%
    assert "Deployment-side causes: **5** (50.0%)" in text
    # clutter/binding + unexplained are architecture: 5 errors = 50%
    assert "Architecture-side causes: **5** (50.0%)" in text


def test_report_candidate_without_validate(tmp_path, sample_report_data):
    clusters_path, diag_path = sample_report_data
    out_dir = tmp_path / "out"
    md_path, csv_path = write_report(clusters_path, diag_path, out_dir, validate_csv=None)
    text = md_path.read_text(encoding="utf-8")

    assert "All causes labelled **candidate**" in text
    df = pd.read_csv(csv_path)
    assert all("candidate" in v for v in df["verdict"])


def test_report_decisive_with_validate(tmp_path, sample_report_data):
    clusters_path, diag_path = sample_report_data
    out_dir = tmp_path / "out"

    # Create mock confusion.csv with high precision for deployment knob
    conf_df = pd.DataFrame(
        {
            "precision": [0.90, 0.60],
            "prec_lo": [0.80, 0.40],
            "prec_hi": [0.95, 0.75],
        },
        index=["deployment knob (H4)", "clutter/binding (H8/H5)"],
    )
    conf_path = tmp_path / "confusion.csv"
    conf_df.to_csv(conf_path)

    cfg = {"validate": {"min_precision": 0.8}}
    md_path, csv_path = write_report(
        clusters_path, diag_path, out_dir, validate_csv=conf_path, cfg=cfg
    )

    df = pd.read_csv(csv_path)
    verdicts = dict(zip(df["cause"], df["verdict"]))
    assert verdicts["deployment knob (H4)"] == "decisive"
    assert "candidate" in verdicts["clutter/binding (H8/H5)"]
