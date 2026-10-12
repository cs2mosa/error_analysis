# tracking-state.md — Project State, Audit Findings & Roadmap

**Date:** 2026-10-12  
**System:** Qwen3-VL-4B Error-Analysis Codebase  
**Environment:** Windows (Python 3.13.5, PyTorch 2.14+cu126, Transformers 5.17.0, `F:\URCA_PROJECTS\.venv`)  
**Repository Root:** [`f:/error_analysis`](file:///f:/error_analysis)

---

## 1. Executive Summary

All core engineering milestones (**M1 through M6**) AND the **Latent-Hazard track (§5.14 / FR-H)** for the Qwen3-VL-4B Error Analysis engine are **100% implemented, verified, and passing**.

- **Test Suite Status:** **198 passed**, 5 GPU smoke tests deselected (203 total collected). Execution time: ~140 seconds total across all suites.
- **CLI Commands:** All 10 subcommands (`selfcheck`, `t0`, `t1`, `t2`, `hazard`, `stream`, `t3`, `t4`, `report`, `validate`) are fully wired, tested, and operational.
- **Integration Test:** Verified end-to-end pipeline execution from synthetic data generation through tiers T0–T2, triage, escalation, and Markdown/CSV reporting (`tests/test_integration.py`).
- **Hazard Integration Test:** Verified end-to-end latent hazard pipeline execution with planted under-reporting and secured twin controls (`tests/test_hazard.py::test_end_to_end_hazard_under_reporting`).
- **Accounting Identity:** Formally verified that every error partitions into exactly one terminal state, summing to 100% of the error corpus.

---

## 2. Milestone & Module Implementation Matrix

| Milestone | Scope & Functional Requirements | Primary Modules | Test Suite | Status |
|---|---|---|---|:---:|
| **M1: Foundation** | Data loading, schemas, frame extraction, prompt templates, polarity conventions, runner core, VLM adapter, hooks, determinism. | [`ea/data.py`](file:///f:/error_analysis/ea/data.py)<br>[`ea/prompts.py`](file:///f:/error_analysis/ea/prompts.py)<br>[`ea/model.py`](file:///f:/error_analysis/ea/model.py)<br>[`ea/run.py`](file:///f:/error_analysis/ea/run.py) | `test_data.py`<br>`test_prompts.py`<br>`test_plans.py`<br>`test_run.py` | ✅ **Complete** |
| **M2: Metrics & Slices** | Behavioral metrics (ECE, Brier, AURC, selective risk-coverage, flips, framing, temporal gains), multi-order slice search with FDR control. | [`ea/metrics.py`](file:///f:/error_analysis/ea/metrics.py)<br>[`ea/slices.py`](file:///f:/error_analysis/ea/slices.py) | `test_metrics.py`<br>`test_slices.py` | ✅ **Complete** |
| **M3: T1–T2 Escalation** | Triage table (`qp`, `qr`, `qd`), intervention arms (O1–O7, bgmask, occlude, null), cluster builder, 3-valued Wilson CIs, signature matching table `SIG`. | [`ea/triage.py`](file:///f:/error_analysis/ea/triage.py)<br>[`ea/interventions.py`](file:///f:/error_analysis/ea/interventions.py)<br>[`ea/escalate.py`](file:///f:/error_analysis/ea/escalate.py) | `test_triage.py`<br>`test_interventions.py`<br>`test_escalate.py` | ✅ **Complete** |
| **M3b: Validation Engine** | False-attribution measurement via synthetic injections (`INJ.*`), confusion matrix generation, decisive vs. candidate precision gating. | [`ea/escalate.py`](file:///f:/error_analysis/ea/escalate.py)<br>[`ea/interventions.py`](file:///f:/error_analysis/ea/interventions.py)<br>[`ea/report.py`](file:///f:/error_analysis/ea/report.py) | `test_escalate.py`<br>`test_report.py` | ✅ **Complete** |
| **M4: Streaming Track** | Causal video evaluation, $k$-of-$n$ alert persistence, early-warning anticipation, error decomposition (ok, anticipated, silent, late, premature, spam), H11 window-vs-stream gap. | [`ea/stream.py`](file:///f:/error_analysis/ea/stream.py) | `test_stream.py` (20 tests) | ✅ **Complete** |
| **M5: T3 Attribution** | Sink-excluded attention maps, gradient saliency, bounding box localization metrics, 3-way reading, ROAD `nli` deletion/insertion, explainer validity gate, SpRAy, planted confounders. | [`ea/attrib.py`](file:///f:/error_analysis/ea/attrib.py) | `test_attrib.py` (8 tests) | ✅ **Complete** |
| **M6: T4 Mechanistic** | Pooled float16 feature caching, cross-validated linear probing with selectivity checks, `probe_verdict`, knockout sweep, logit lens, information horizon, activation patching, dual-evidence rule. | [`ea/mech.py`](file:///f:/error_analysis/ea/mech.py) | `test_mech.py` (5 tests) | ✅ **Complete** |
| **Latent-Hazard Track** | Physical consequence under trigger (§5.14, FR-H-1..7): `hz_*` prompts, `HZ.cue` vs `HZ.cue_twin`, secured twins, `explain_cue`, `hazard_metrics`, `hz_checklist`, reporting cap. | [`ea/prompts.py`](file:///f:/error_analysis/ea/prompts.py)<br>[`ea/interventions.py`](file:///f:/error_analysis/ea/interventions.py)<br>[`ea/run.py`](file:///f:/error_analysis/ea/run.py)<br>[`ea/escalate.py`](file:///f:/error_analysis/ea/escalate.py)<br>[`ea/metrics.py`](file:///f:/error_analysis/ea/metrics.py) | `test_hazard.py` (16 tests) | ✅ **Complete** |
| **Robustness & Mutation** | 8 deliberate-bug mutation tests verifying that subtle flaws in recovery inequalities, controls, or gates immediately fail tests. | [`tests/test_mutation.py`](file:///f:/error_analysis/tests/test_mutation.py) | `test_mutation.py` (8 tests) | ✅ **Complete** |

---

## 3. Design Document Audit & Alignment Findings

A systematic review of [`design.md`](file:///f:/error_analysis/design.md) against the codebase identified the following resolutions and deviations:

### 3.1 Resolution of Initial `[VERIFY]` Annotations
The design document initially drafted reference sketches marked with `[VERIFY]` pending empirical model testing. All have been verified and codified:
1. **Qwen3-VL Temporal Patching ([design.md:L204](file:///f:/error_analysis/design.md#L204)):**
   - *Requirement:* Qwen3-VL merges 2 frames per temporal patch; odd frame counts fail or misalign.
   - *Resolution in Code:* [`ea/data.py::frames`](file:///f:/error_analysis/ea/data.py#L112-L135) automatically pads odd sequences by repeating the final frame, preserving exact alignment.
2. **DeepStack Layer Injection Offset ([design.md:L341](file:///f:/error_analysis/design.md#L341)):**
   - *Requirement:* Visual features are added to layers 0–2 hidden states outside layer modules.
   - *Resolution in Code:* [`ea/mech.py::patch_scan`](file:///f:/error_analysis/ea/mech.py#L173-L215) enforces starting patch sweeps at layer $\ge 3$.
3. **Eager Attention Hooking ([design.md:L369](file:///f:/error_analysis/design.md#L369)):**
   - *Requirement:* Materialize 4D attention weights for non-invasive extraction.
   - *Resolution in Code:* [`ea/model.py::attn_last`](file:///f:/error_analysis/ea/model.py#L225-L245) captures layer weights via `o[1][0, :, -1, :]` under eager mode.
4. **ROAD Debiased Imputation Algorithm ([design.md:L753](file:///f:/error_analysis/design.md#L753)):**
   - *Requirement:* Noisy linear imputation (ROAD-style) without introducing external heavy dependencies.
   - *Resolution in Code:* Implemented directly in [`ea/attrib.py::nli`](file:///f:/error_analysis/ea/attrib.py#L125-L157) using boundary linear interpolation + Gaussian noise.
5. **Question-Token Visual Migration ([design.md:L784](file:///f:/error_analysis/design.md#L784)):**
   - *Requirement:* Visual information migrates from visual tokens into question tokens before the decision.
   - *Resolution in Code:* [`ea/mech.py::cache_sample`](file:///f:/error_analysis/ea/mech.py#L40-L65) caches question-position states `L{i}_q` alongside box and last-token states.

### 3.2 Methodological Refinements Logged in [DEVIATIONS.md](file:///f:/error_analysis/DEVIATIONS.md)
The following enhancements were logged in [`DEVIATIONS.md`](file:///f:/error_analysis/DEVIATIONS.md):
- **Deviation 3 (`metrics.py`):** `fpr_at_recall` updated from naive interpolation to selecting the minimal operational FPR satisfying $\text{recall} \ge \text{target}$.
- **Deviation 4 (`metrics.py`):** `coverage_at_recall` updated to evaluate max accepted fraction across all prefixes rather than breaking on first dip.
- **Deviation 8 (`config.yaml`):** Quote the `"null":` key under YAML 1.1 rules to avoid silent parsing as Python `None`.
- **Deviation 10 (`attrib.py`):** Use `getattr(np, "trapezoid", getattr(np, "trapz", None))` for NumPy 2.0+ compatibility.
- **Deviation 11 (`metrics.py` / `escalate.py`):** Added optional `ci=True` parameter to `margin_shift`. Inner diagnostic sweeps pass `ci=False`, cutting verification runtime from >5 minutes to <3 seconds.

### 3.3 Soft Line Budget Analysis (§0.1 & §2)
[`design.md`](file:///f:/error_analysis/design.md) budgeted ~2,000 lines across 14 modules.
- **Actual Size:** ~3,300 lines across `ea/`.
- **Reasoning:** In accordance with §0.1 (GC-1: correctness, independent controls, validity gates, and Wilson CIs strictly outrank brevity), code expansions occurred in [`ea/escalate.py`](file:///f:/error_analysis/ea/escalate.py) (~700 lines) and [`ea/interventions.py`](file:///f:/error_analysis/ea/interventions.py) (~315 lines) to accommodate exhaustive testing, latent hazard controls, and the accounting identity.

---

## 4. Latent-Hazard (Prediction) Track Audit (§5.14 / Change C19)

### 4.1 Specification in `design.md`
The Latent-Hazard track was introduced during the v0.5 methodology audit (Change C19) to address industrial situations where a condition is not yet an active violation, but foreseeably causes an incident under a trigger (e.g. an unbraced open-lid container under wind vibration):
- **Distinction from Streaming:** Single-frame evaluation of latent risk, not time-series video onset forecasting.
- **Prompts Implemented (§5.14, FR-H-3):**
  - `hz_s0`, `hz_s0_abs`, `hz_s1`, `hz_s1_abs`: Perception check of object and state attributes (conjunction in `hz_qp_ok_rate`).
  - `hz_r`: Text-only physical reasoning from oracle conditions and triggers (`hz_qr_ok_rate`).
  - `hz_d`: Uncued decision prompt.
  - `hz_cue`: Cued arm explicitly naming consequence and trigger.
- **Intervention Arm & Control (FR-H-4):**
  - `HZ.cue` paired with control `HZ.cue_twin` on matched secured scenes (e.g. lid clamped).
- **Pitfall #24 Guard:** Never evaluate cued recovery without twins; a cue that causes false alarms on safe twins reflects prompt bias, not true hazard awareness. Evaluated in [`explain_cue`](file:///f:/error_analysis/ea/escalate.py#L388).

### 4.2 Verification Status in Codebase
- **In [`ea/prompts.py`](file:///f:/error_analysis/ea/prompts.py):** All `hz_*` templates registered.
- **In [`ea/interventions.py`](file:///f:/error_analysis/ea/interventions.py):** `HZ.cue` and `HZ.cue_twin` registered with controls.
- **In [`ea/run.py`](file:///f:/error_analysis/ea/run.py):** `plan_hazard` generates balanced paired jobs.
- **In [`ea/escalate.py`](file:///f:/error_analysis/ea/escalate.py):** `explain_cue`, `hz_qp_ok_rate`, `hz_qr_ok_rate`, and `"uncued-hazard under-reporting"` in `SIG`.
- **In [`ea/metrics.py`](file:///f:/error_analysis/ea/metrics.py):** `hazard_metrics` and `hz_checklist` implemented.
- **In [`tests/test_hazard.py`](file:///f:/error_analysis/tests/test_hazard.py):** All 16 unit and end-to-end integration tests are **passing**.

---

## 5. Finishing Roadmap & Completion Criteria

The engineering phase is **complete**. The codebase is ready for final sign-off.

```mermaid
graph LR
    A[Codebase Complete: 198/198 Tests Passed] --> B[Final Documentation Sync]
    A --> C[Ready for Real Benchmark Datasets]
    
    B --> B1[Update design.md: mark verified items & record actual module budgets]
    B --> B2[Deliver Final Phase Report to user]
    C --> C1[When CS10K / PaSBench data is supplied, run T0 on GPU]
```

1. **Step 1 (Immediate):** Update [`design.md`](file:///f:/error_analysis/design.md) to replace historical planning sketches and `[VERIFY]` notes with the final reference architecture, reflecting all documented deviations.
2. **Step 2 (Immediate):** Deliver the **Final Phase Report** summarizing the completed architecture, verified invariants, test coverage, and diagnostic capabilities.
3. **Step 3 (Operational):** Whenever real benchmark datasets (CS10K, PaSBench, SafeBuild) are connected to `data/<bench>/`, execute `ea t0` through `ea report` on real weights.

---

## 6. Docker Containerization & Remote Server Deployment

The codebase has been fully containerized and verified for deployment to remote server environments:

- **Docker Environment:**
  - [`Dockerfile`](file:///f:/error_analysis/Dockerfile): CUDA 13.0, Ubuntu 24.04, Python 3.13 (deadsnakes), PyTorch 2.14+cu130, headless Matplotlib (`Agg`), RTX 5090 / sm_120 compatibility fix.
  - [`docker-compose.yml`](file:///f:/error_analysis/docker-compose.yml): Unified targets for `selfcheck`, `test`, `test-gpu`, `t0`, `t1`, `t2`, `t3`, `t4`, `stream`, `validate`, `report`, and `shell`.
  - [`docker/entrypoint.sh`](file:///f:/error_analysis/docker/entrypoint.sh): Modular entrypoint script routing CLI verbs.
  - [`.env.example`](file:///f:/error_analysis/.env.example): Configurable host mount directories (`HF_CACHE_DIR`, `DATA_DIR`, `OUTPUTS_DIR`, `HF_TOKEN`).
- **Remote Server Operation Without Pre-downloaded VLM:**
  - `docker build` requires **zero** model weights (build phase only installs dependencies).
  - Test suites (`docker compose run test`) run completely offline via synthetic fixtures and `FakeVLM` without downloading the model.
  - Inference commands (`selfcheck --load`, `t0`, `t1`, etc.) dynamically resolve model weights via [`ea/selfcheck.py::_model_ref`](file:///f:/error_analysis/ea/selfcheck.py#L19-L38):
    1. CLI `--model <path_or_repo>` override
    2. `MODEL_PATH` environment variable (if path exists or valid repo)
    3. `cfg.model.path` local snapshot path (if path exists)
    4. Auto-discovery in `$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*`
    5. Fallback to Hugging Face Hub (`Qwen/Qwen3-VL-4B-Instruct`) for automatic on-the-fly download if online.


