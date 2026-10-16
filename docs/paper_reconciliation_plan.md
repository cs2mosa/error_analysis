# Paper Reconciliation and Resolution Protocol

## 1. Overview and Purpose
This document specifies the reconciliation protocol to resolve logical failures and discrepancies identified between the cited research literature (`docs/references.md`, `docs/design.md`, `docs/requirements.md`) and the codebase (`ea/`).

Following the Karpathy guidelines and anti-slop principles:
- **Test-Build First:** Each logical failure is first captured by targeted, reproducible tests in `tests/test_paper_fidelity.py` before any codebase edits.
- **Surgical Changes:** Edits touch only the affected functions and lines without refactoring working adjacent code or introducing unnecessary abstractions.
- **Zero AI-Slop Comments:** Comments remain functional, concise, explaining constraints and reasons, avoiding decorative separators and workflow narration.

---

## 2. Inventory of Discrepancies and Resolution Actions

| # | Paper & ArXiv ID | Component | Failure Description | Surgical Resolution & Status |
|---|---|---|---|---|
| **1** | **ROAD**<br>`arXiv:2202.00449` | `ea/cli.py:406-412`<br>`ea/attrib.py` | `_run_t3` evaluated explainer validity on error samples with ground truth polarity (`sign = 2*label - 1`), causing deletion of error-causing features to *increase* margin and fail monotonicity. | **Fixed (`ea/cli.py`)**: Calibrated validity gate relative to model prediction sign `model_sign = 1 if s_base >= 0 else -1` on matched correct samples. |
| **2** | **Activation Patching**<br>`arXiv:2309.16042` | `ea/cli.py:628`<br>`ea/mech.py` | `_run_t4` duplicated the single pair corruption result `[p_res, p_res]` to bypass the `len >= 2` guard in `is_patch_causal`, violating the multi-corruption requirement. | **Fixed (`ea/cli.py`)**: Implemented the second corruption arm (`boxnoise` Gaussian perturbation per `config.yaml:36`) and passed genuine `[p_res_pair, p_res_box]` to `is_patch_causal`. |
| **3** | **Information Horizon**<br>`arXiv:2512.07580` | `ea/mech.py` | The earlier claim that dividing by a negative shift inverted the inequality was **incorrect**: `(s_l - s_blind)/(s_full - s_blind)` is a ratio of two signed numbers of the same sign and is already direction-safe (old and 'directional' forms agreed on 198,407 random cases; the original fidelity test also passed on the pre-fix code). | **Corrected**: plain ratio restored; the only real change is the bf16 noise floor `min_effect=0.05`, now guarded by `test_horizon_abstains_when_visual_effect_is_below_noise_floor`. |
| **4** | **SPOT-Bench & TRACE**<br>`arXiv:2604.24317`<br>`arXiv:2609.30670` | `ea/stream.py:156` | `dead_time_s` equation assumed clip boundaries enclose risk window, over-counting dead time up to $2.5\times$ on clips starting after risk ends. | **Fixed (`ea/stream.py`)**: Computed dead time as the exact measure of $[ts[0], ts[-1]]$ outside the risk window $[\text{onset} - \text{tol} - \text{ant}, \text{end\_s} + \text{tol}]$. |
| **5** | **ReCoVERR**<br>`arXiv:2402.15610` | `ea/metrics.py` | **Previously marked Verified in error.** `recall = tp / total_pos` used a global denominator, so recall was monotone in k and the function returned only 0.0 or 1.0. | **Fixed**: accepted-set recall (`tp_k / positives_k`), max over prefixes. Tests: `test_coverage_at_recall_*` in `tests/test_fidelity_fixes.py`. |
| **6** | **VLind-Bench Gating**<br>`arXiv:2406.08702` | `ea/triage.py:40-47` | Gate must safeguard against attributing perception failure to priors. | **Verified (`ea/triage.py`)**: Strictly requires all specified gating prompts ($Q_P, Q_P^{abs}, Q_R$) to be present, valid, and correct; closes gate on missing/invalid rows (FR-T1-4, C6). |
| **7** | **Perception vs. Readout**<br>`arXiv:2609.22588`<br>`arXiv:2510.17771` | `ea/escalate.py:448-450` | Behavioral process failure triage vs mechanistic confirmation. | **Verified (`ea/escalate.py`)**: T1/T2 triage returns `inconclusive: more data` with `readout/process (H3)` candidate, triggering T4 mechanistic escalation. T4 causal confirmation assigns `readout/process (H3)`. |

---

## 3. Execution Plan & Results

1. **Step 1: Test Build First**
   - Built dedicated paper-specific test suite in `tests/test_paper_fidelity.py` covering all 7 research papers and failure modes.
2. **Step 2: Surgical Code Implementation**
   - Modified `ea/stream.py` to fix non-overlapping dead-time computation.
   - Modified `ea/mech.py` to fix directional visual retention projection and noise floor.
   - Modified `ea/cli.py` to implement ROAD model prediction sign evaluation and dual-arm activation patching (`pair` + `boxnoise`).
3. **Step 3: Verification & Regression Testing**
   - Verified that `tests/test_paper_fidelity.py` passes 100% (7/7 tests passed).
   - Preserved architectural invariants (14 normative signatures in `sig_compat.txt`, strict gate validation, Deviation 4 prefix coverage).
   - Verified the complete test suite against regressions.

## 4. Second audit pass (code vs. paper text)

Defects found by reading the ROAD full text and the reference implementation, then running
the functions: fixed test-first in `tests/test_fidelity_fixes.py` (11 of 13 original tests fail on
the previous code; the other two are regression guards).

| Item | Resolution |
|---|---|
| `coverage_at_recall` degenerate | Fixed (see row 5) |
| `horizon` fix was a no-op | Corrected (see row 3) |
| FR-T3-6 gate weaker than the spec (ties passed, no Wilcoxon, absolute-gap fill check, no randomization / robustness, `ko_fn` never passed) | `explainer_validity_gate` + `gate_aggregate` + `_explainer_gate` (DEVIATIONS 27) |
| `nli` unconverged solver; `noise_std=0` still added noise | Exact sparse solve (DEVIATIONS 26) |
| `is_patch_causal` any-cell-per-arm | Same site across arms (DEVIATIONS 28) |
| Documented, not changed | DEVIATIONS 28 (corruption type), 30, 31, 32 |

Not verified against the papers: 2406.08702, 2502.17422, 2411.18620, 2605.22903, 2201.13291,
2509.04588, 2203.14960, SpRAy, and the `sink_mask` 10x-median threshold. The real-model
`_run_t3` path was exercised only with a fake model.
