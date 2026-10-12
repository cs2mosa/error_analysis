"""report.py — root-cause report generator (FR-E-7/8/9/12, FR-V-3).

Purpose:  read clusters.json + diag_vector.parquet (+ optional validate/
          confusion.csv) and write root_causes.md + root_causes.csv.
          Labels a cause 'decisive' only when validated precision ≥
          validate.min_precision (FR-V-3); all others are 'candidate'.
          Always prints the accounting identity table (FR-E-11) and the
          scope-of-evidence section (FR-E-12).
Implements plan sections: §9, deliverable 7.
Inputs:   outputs/escalate/clusters.json, diag_vector.parquet;
          optionally outputs/validate/confusion.csv.
Outputs:  root_causes.md, root_causes.csv (both in out_dir).

Measures: aggregated per-error cause distribution and implication.
Does not measure: new diagnostics; this is a pure reader/formatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ea.escalate import IMPL, SIG, wil

# Causes that can only be 'candidate' unless T4 evidence-rule passes (FR-V-3).
_T4_ONLY_CAUSES = frozenset({
    "readout/process (H3)", "merger bottleneck (H1)", "depth fading (H2)",
})

# Validated precision loaded from confusion.csv (may be empty).
_PRECISION: dict[str, tuple[float, float, float]] = {}  # cause → (prec, lo, hi)

# Scope-of-evidence protocol disclosure (FR-E-12).
_SCOPE_LIMITS = """
### Scope of evidence

- **Injections are synthetic**: the confusion matrix is an optimistic bound on
  the false-attribution rate; real errors are mixtures of causes.
- **Controls are matched but not randomized**: matched-correct pools balance
  cls and obj_size_bin; uncontrolled confounders (lighting, occlusion) remain.
- **Recall and false-positive rate are tightly coupled** for latent hazards on
  look-alike scenes (2606.02443); hazard recall is not reported without twins.
- **Latent-hazard labels (FR-H-6)**: the trigger is not visible in the image,
  so the label rests on an assumed foreseeable condition; likelihood/severity
  are subjective, and clusters below `hazard.min_agreement` are capped at
  'candidate' with a mandatory audit sample.
- **T3/T4 diagnostics**: loc, t4, causal are present only for clusters that
  ran the full attribution + mechanistic tiers.
- **Per-error accounting**: each error is assigned to the *most specific*
  cluster with a unique cause; ties or no unique cause → 'ambiguous' /
  'ALL_ERRORS' cause.
""".strip()


# ----------------------------------------------------------------- helpers

def _load_precision(validate_csv: Path | None) -> None:
    """Load per-cause precision from confusion.csv if available."""
    global _PRECISION
    _PRECISION = {}
    if validate_csv is None or not validate_csv.exists():
        return
    df = pd.read_csv(validate_csv, index_col=0)
    for cause in df.index:
        if "precision" in df.columns:
            p = float(df.loc[cause, "precision"])
            lo = float(df.loc[cause, "prec_lo"]) if "prec_lo" in df.columns else np.nan
            hi = float(df.loc[cause, "prec_hi"]) if "prec_hi" in df.columns else np.nan
            _PRECISION[str(cause)] = (p, lo, hi)


def _decisive(cause: str, min_prec: float, agreement: float | None = None,
              min_agreement: float = 0.6) -> str:
    """Return 'decisive' or 'candidate (...)' label (FR-V-3).

    FR-H-6: the hazard cause is additionally capped at 'candidate' when the
    cluster's annotator agreement < hazard.min_agreement (subjective labels).
    """
    if cause == "uncued-hazard under-reporting" and agreement is not None \
            and agreement < min_agreement:
        return f"candidate (annotator agreement {agreement:.2f} < hazard.min_agreement {min_agreement})"
    if cause in _T4_ONLY_CAUSES:
        # T4-only causes need the evidence rule to pass; always candidate otherwise
        if cause in _PRECISION:
            p, lo, hi = _PRECISION[cause]
            if p >= min_prec:
                return "decisive"
        return "candidate (evidence rule not yet verified)"
    if cause in _PRECISION:
        p, lo, hi = _PRECISION[cause]
        label = f"validated precision {p:.2f} [{lo:.2f}, {hi:.2f}]"
        if p >= min_prec:
            return "decisive"
        return f"candidate ({label})"
    return "candidate (not validated)"


def _cause_table(diag_vec: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-error cause distribution; include n and percentage."""
    total = len(diag_vec)
    rows = (
        diag_vec.groupby("cause")
        .agg(n=("sample_id", "count"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    rows["pct"] = (rows.n / total * 100).round(1)
    return rows


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate."""
    headers = list(df.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header_row, sep_row] + data_rows)


def _is_deployment(cause: str) -> bool:
    """FR-E-8: deployment-side vs architecture-side split."""
    deployment = {
        "deployment knob (H4)",
        "trigger-policy failure (H11)",
        "input quality (H9)",
        "rule-reasoning failure",
        "polarity/framing bias (H6/H7)",
        "prior-driven (H7)",
    }
    return cause in deployment


# ----------------------------------------------------------------- markdown sections

def _cluster_section(c: dict, min_prec: float, validate_ok: bool,
                     min_agreement: float = 0.6) -> str:
    """One markdown section per cluster."""
    name = c["name"]
    n = c["n_errors"]
    cause = c["cause"]
    implication = IMPL.get(cause, "—")
    tag = c.get("tag") or ""
    pred = c.get("pred") or {}
    candidates = c.get("cause_candidates") or []
    hazard = bool(c.get("hazard"))
    agreement = c.get("agreement")

    dec = _decisive(cause, min_prec, agreement=agreement,
                    min_agreement=min_agreement) if validate_ok \
        else "candidate (validate not run)"
    side = "deployment" if _is_deployment(cause) else "architecture"

    lines = [
        f"#### {name}",
        f"- **Errors:** {n}  |  **Cause:** {cause}  |  **Verdict:** {dec}  |  **Side:** {side}",
    ]
    if hazard:
        lines.append(f"- **Latent-hazard cluster (§5.14)**  |  cue diagnostic: {c.get('cue')}"
                     f"  |  annotator agreement: {agreement if agreement is not None else 'n/a'}")
    if tag:
        lines.append(f"- **Taxonomy tag:** {tag}")
    if pred:
        lines.append(f"- **Predicate:** {pred}")
    if cause not in ("unexplained → next tier", "inconclusive: more data", "ambiguous"):
        lines.append(f"- **Implication:** {implication}")
    if candidates:
        lines.append(f"- **Candidates / partial matches:** {', '.join(candidates)}")
    if hazard:
        # FR-H-6 caveats: trigger invisible in-frame; subjective labels; audit mandatory
        lines.append("- ⚠️ The trigger is not visible in the image; the label rests on an "
                     "assumed foreseeable condition. Likelihood/severity are subjective — "
                     "the human audit sample is mandatory for this cluster.")
    if cause in ("unexplained → next tier", "inconclusive: more data"):
        lines.append("- ⚠️ Queued for T3/T4 attribution.")
    elif cause == "ambiguous":
        lines.append("- ⚠️ Multiple causes match; need additional data to disambiguate.")
    return "\n".join(lines)


# ----------------------------------------------------------------- write_report

def write_report(
    clusters_path: str | Path,
    diag_vec_path: str | Path,
    out_dir: str | Path,
    validate_csv: str | Path | None = None,
    cfg: dict | None = None,
) -> tuple[Path, Path]:
    """Generate root_causes.md and root_causes.csv (FR-E-7).

    Args:
        clusters_path:  path to clusters.json (from escalate()).
        diag_vec_path:  path to diag_vector.parquet.
        out_dir:        directory to write outputs.
        validate_csv:   optional path to outputs/validate/confusion.csv.
        cfg:            config dict; uses defaults if None.

    Returns (md_path, csv_path).
    """
    cfg = cfg or {}
    min_prec = float(cfg.get("validate", {}).get("min_precision", 0.8))
    min_agreement = float(cfg.get("hazard", {}).get("min_agreement", 0.6))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters = json.loads(Path(clusters_path).read_text(encoding="utf-8"))
    diag_vec = pd.read_parquet(diag_vec_path)
    validate_ok = validate_csv is not None and Path(validate_csv).exists()
    _load_precision(Path(validate_csv) if validate_csv else None)

    total_errors = len(diag_vec)
    cause_df = _cause_table(diag_vec)

    # ---- markdown report
    md_lines = [
        "# Root-cause report",
        "",
        f"Total errors: **{total_errors}**",
        "",
        "## Cause distribution (per-error accounting, C17)",
        "",
        _df_to_markdown(cause_df),
        "",
    ]

    # Deployment vs architecture split (FR-E-8)
    dep_n = int(diag_vec[diag_vec.cause.apply(_is_deployment)].shape[0])
    arch_n = total_errors - dep_n
    md_lines += [
        "## Deployment vs architecture split (FR-E-8)",
        "",
        f"- Deployment-side causes: **{dep_n}** ({dep_n / max(total_errors, 1) * 100:.1f}%)",
        f"- Architecture-side causes: **{arch_n}** ({arch_n / max(total_errors, 1) * 100:.1f}%)",
        "",
    ]

    # Per-cluster sections
    md_lines += ["## Cluster details", ""]
    for c in clusters:
        md_lines.append(_cluster_section(c, min_prec, validate_ok, min_agreement))
        md_lines.append("")

    # Accounting identity (FR-E-11)
    md_lines += [
        "## Accounting identity (FR-E-11)",
        "",
        f"Every error appears in exactly one row of diag_vector.parquet.",
        f"Row count: {total_errors} (must equal total errors above).",
        "",
    ]

    # Scope of evidence (FR-E-12)
    md_lines += ["## Scope of evidence (FR-E-12)", "", _SCOPE_LIMITS, ""]

    if not validate_ok:
        md_lines += [
            "> [!WARNING]",
            "> `python -m ea validate` has not been run. All causes labelled **candidate**.",
            "> Run validate before marking any cause decisive.",
            "",
        ]

    md_text = "\n".join(md_lines)
    md_path = out_dir / "root_causes.md"
    md_path.write_text(md_text, encoding="utf-8")

    # ---- csv
    csv_path = out_dir / "root_causes.csv"
    cause_df["implication"] = cause_df.cause.map(lambda c: IMPL.get(c, ""))
    cause_df["side"] = cause_df.cause.apply(lambda c: "deployment" if _is_deployment(c) else "architecture")
    cause_df["verdict"] = cause_df.cause.apply(
        lambda c: _decisive(c, min_prec) if validate_ok else "candidate (validate not run)"
    )
    cause_df.to_csv(csv_path, index=False)

    print(f"[report] wrote {md_path} and {csv_path}")
    return md_path, csv_path
