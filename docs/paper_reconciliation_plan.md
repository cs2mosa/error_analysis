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
| **3** | **Information Horizon**<br>`arXiv:2512.07580` | `ea/mech.py:226-233` | `(s_l - s_blind) / (s_full - s_blind) >= frac` assumed $(s_{\text{full}} - s_{\text{blind}}) > 0$. When visual evidence reduces the margin (negative class), dividing by negative shift inverted the inequality. | **Fixed (`ea/mech.py`)**: Implemented directional projection `((s_l - s_blind) * sign_vis) / abs(delta_vis) >= frac` with a bfloat16-safe noise floor ($0.05$). |
| **4** | **SPOT-Bench & TRACE**<br>`arXiv:2604.24317`<br>`arXiv:2609.30670` | `ea/stream.py:156` | `dead_time_s` equation assumed clip boundaries enclose risk window, over-counting dead time up to $2.5\times$ on clips starting after risk ends. | **Fixed (`ea/stream.py`)**: Computed dead time as the exact measure of $[ts[0], ts[-1]]$ outside the risk window $[\text{onset} - \text{tol} - \text{ant}, \text{end\_s} + \text{tol}]$. |
| **5** | **ReCoVERR**<br>`arXiv:2402.15610` | `ea/metrics.py:252-263` | Selective recall search over prefix ordering. | **Verified (`ea/metrics.py`)**: Evaluates max accepted fraction over all prefixes (Deviation 4, FR-B-13). Returns 1.0 when reachable and 0.0 when unreachable. |
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

