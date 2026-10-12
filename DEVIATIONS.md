# DEVIATIONS.md — every overrun and departure (GC-1, FR-Q-9)

Per `requirements.md` GC-1 and `design.md` §0.1: the line budget is soft; every
deviation from the plan/design is logged here with reason and requirement IDs.

| # | Module / item | Deviation | Reason | FR / source |
|---|---|---|---|---|
| 1 | environment | Used `F:\URCA_PROJECTS\.venv` (Python 3.13.5, torch 2.14+cu126, transformers 5.17.0) instead of a project-local venv; added `statsmodels`, `pyarrow` via the owner's `setup_env.py` mechanism | Owner instruction (session 2026-10-12); both are allowed deps (§3.2) | §3.2 |
| 2 | model.py `VLM.__init__` | Default quantization for real runs on this machine is `"8bit"` (bitsandbytes), bf16 does not fit in the 6 GB Quadro RTX 3000; `llm_int8_skip_modules=["lm_head"]` keeps the margin head exact | Hardware constraint; FR-M-1 allows `quant` arm; noted in docstring | FR-M-1 |
| 3 | metrics.py `fpr_at_recall` | Implemented as *smallest* FPR whose recall ≥ target, scanning `roc_curve(..., drop_intermediate=False)`, instead of the design sketch's `np.interp(rec, t, f)` | The sketch returns FPR=1.0 for a perfect scorer at recall=1.0 because sklearn's ROC keeps the trivial last point (fpr=1,tpr=1) and drops collinear ones; the metric's standard meaning is the minimal FPR at the recall target (ROC operating-point selection; cf. sklearn roc_curve docs, drop_intermediate caveat) | FR-B-2 |
| 4 | metrics.py `coverage_at_recall` | Max accepted fraction over ALL prefixes (recall can dip and recover), not first-prefix-break | Selective-recall coverage is the largest accepted set meeting the target; early break under-reports (ReCoVERR-style selective prediction, 2402.15610) | FR-B-13 |
| 5 | selfcheck | Split into structure checks (default) and forward-pass checks (`--load`); items 3/4/6/10 need weights | [VERIFY] items differ in cost; structure checks must run weight-free per NFR-6 | §9, NFR-3 |
| 6 | config.yaml | Added `model.path` (local snapshot) beside `model.name` | Windows without Developer Mode cannot create the HF hub symlink cache; the owner supplied local weights at F:\URCA_PROJECTS\.hf_cache | GC-6 |
| 7 | run.py `one()` | `gold` is computed via `p["gold"](s)` — for O2.wrongfacts the override changes `facts`, but gold is taken from the ORIGINAL sample | Pitfall 2 (gold under interventions = original gold) | FR-T2-2, pitfall 2 |
| 8 | config.yaml | The N0.null section key must be written `"null":` (quoted) — a bare `null:` key parses as YAML null and silently produced a `None` key in the config dict | YAML 1.1 scalar typing; found by the manifest test (json sort_keys TypeError) | GC-6, §3.3 |
| 9 | run.py | `INTERV` (interventions.py, phase 4) is imported lazily inside `one()` only for non-base arms; `failures` defaults next to the output file instead of cwd | Keeps the phase-3 runner testable before interventions exist; failures must land beside the runs they belong to | FR-R-5 |
| 10 | attrib.py | Used `np.trapezoid` (with fallback to `np.trapz`) | NumPy 2.0+ removed `np.trapz` | FR-T3-5 |
| 11 | metrics.py / escalate.py | Added optional `ci=True` parameter to `margin_shift`; `explain_recover` passes `ci=False` | `explain_recover` only consumes the point estimate Δs; skipping unnecessary bootstrap in inner loops accelerates verification by orders of magnitude | FR-E-3, GC-1 |

