# requirements.md — Qwen3-VL-4B Error-Analysis Codebase

**Source plan:** `qwen3vl_error_analysis_complete.md` (v0.4). Section numbers (§) below refer to that plan.
**Companion files:** `design.md` (how to implement), `references.md` (paper links).
**Audience:** an AI coding agent (and the human reviewing it).

> **Revision v0.5 — literature-backed audit of the root-cause logic.** After checking the methodology papers in `references.md` (abstract level; see its *Evidence map for v0.5*), changes **C1–C18** were made; they are tagged in the text. The headline change: a cause is no longer assigned by "first test that passes" but by a **signature of several diagnostics with explicit controls**, ambiguity is reported instead of resolved silently, and **no cause may be called decisive without a measured precision** (`validate`, FR-V). **C19** adds the latent-hazard (*prediction*) track (§4.14) following the project owner's definition; **C20** adds the accounting identity and the scope-of-evidence statement (§1.4, FR-E-11/12). **C21** makes the size budget soft (GC-1) and adds the implementation-correctness protocol (§6.3, FR-Q). Items marked **[DECISION NEEDED]** require the project owner.

---

## 0. How to read this document

- **MUST / SHOULD / MAY** follow RFC 2119.
- Every requirement has an ID. Tests and code comments should cite IDs (e.g. `# FR-B-4`).
- **Priority** controls build order:
  - **P0** — needed for T0, T1, T2 (no model internals). Build first.
  - **P1** — T3 attribution and the streaming track.
  - **P2** — T4 mechanistic analysis.
  - **P3** — optional. Do not build unless P0–P2 are done and tested.
- Each requirement has an **acceptance criterion (AC)** that can be checked by a test or by inspecting an output file.
- Plan items marked **[PROPOSED]** become *config defaults* (§7), not hard-coded constants. Items marked **[VERIFY]** become *runtime assertions or unit tests* that fail loudly if the assumption is wrong (§9).

---

## 1. Purpose, scope, non-goals

### 1.1 Purpose
Build the smallest codebase that is still **correct and verifiable** and that can (a) benchmark baseline Qwen3-VL-4B on safety-violation data, (b) discover *why* it fails, using the cheapest-first ladder T0→T4 plus a streaming track (plan §7), and (c) emit a root-cause map (plan §9 deliverable 7).

### 1.2 In scope
Baseline pretrained Qwen3-VL-4B only. The resulting root-cause map characterises the model **as a standalone classifier on the benchmark domain**; an upstream frame router or a different camera domain changes the error population (FR-D-7). Behavioral metrics, slice discovery, triage, oracle interventions, deployment-knob sweeps, attribution, probes, attention knockout, logit lens, activation patching, streaming metrics, escalation logic, report generation.

### 1.3 Non-goals (MUST NOT be built)
- Fine-tuning, training loops, LoRA, comparison with fine-tuned checkpoints, implementing fixes.
- A general-purpose experiment framework, plugin system, web UI, database, job scheduler, or experiment-tracking service.
- Support for models other than Qwen3-VL (a thin adapter boundary is enough; no multi-model abstraction).
- Re-implementations of third-party papers' full pipelines (HERMES hierarchical KV cache, Domino, etc.). See §3 for the "do not build" list.

---

### 1.4 What the analysis can and cannot reveal (C20)

Nothing here can *guarantee* that every cause of every error is found: the analysis is black-box inference on one checkpoint plus internal-state probes, and the diagnostics are only as good as `validate` shows. What the requirements **do** guarantee is bookkeeping and honesty: no error is silently dropped (FR-E-11), no cause is called decisive without a measured precision (FR-V-3), and ambiguity is reported instead of resolved. Causes that this design **cannot** establish, and that the report MUST list as out of scope (FR-E-12):

- training-data coverage or label-distribution causes (no access to the training data; only slice-level performance is observable);
- whether a fix (fine-tuning, a different encoder) would work — only weight-free interventions are tested;
- **mixtures** of causes: co-occurring causes are reported as `ambiguous`, not separated;
- benchmark label noise beyond the audit sample (`audit.k_per_cluster` per cluster);
- errors in no significant slice with too few samples (`underpowered`);
- interactions between deployment knobs (sweeps are one knob at a time);
- free-text generation behaviour, except the lenient `hz_report` checks (FR-H-7);
- causes outside the VLM in a deployed pipeline (frame router, tracker, production frame sampling, camera) unless modelled in the data (FR-D-7);
- anything derived from T3/T4 explainers or probes that failed their validity gates (tagged `UNVALIDATED`).

## 2. Definitions and global conventions

These definitions are normative. Every module MUST use them identically.

| Term | Definition |
|---|---|
| **Margin `s`** | `s = logsumexp(logit[pos_ids]) − logsumexp(logit[neg_ids])` at the first answer-token position, one forward pass, no sampling (plan §6.1). `pos`/`neg` token sets are defined per prompt. |
| **Native decision** | `pred = (s > 0)`. Used for error definition, consistency metrics, recovery rates. |
| **Calibrated decision** | `s > τ`, where τ is frozen at T0 (e.g. threshold at target FPR). Used only for operating-point metrics (recall@FPR etc.) and streaming alerts. |
| **Gold** | Per-prompt ground truth `gold(sample) ∈ {0,1}`. For violation prompts, `gold = sample.label`. Prompts whose "positive" token means *safe* (e.g. "Is the worker wearing a helmet?" → violation is "No") set `pos_ids` to the "No" token so that **`s > 0` always means "model asserts the prompt's positive class"** and `gold` is defined accordingly. |
| **Error** | A (sample, prompt) row with `pred != gold` under the base arm. A *sample-level error* uses the primary prompt set (mean margin over paraphrases, or majority vote — configurable, default majority). |
| **Failure type** | One or more of the 10 labels in plan §5 (taxonomy). |
| **Cluster** | A set of error samples sharing a slice predicate (from slice search) or a taxonomy label. The unit that is escalated through tiers. |
| **Matched correct sample** | A correctly answered sample with the same `cls` and `obj_size_bin` as an error sample, drawn with a fixed seed (plan §6.7). |
| **Arm** | A named input transformation (intervention, control, or knob value). `arm="base"` is the untouched input. |
| **Knob** | A deployment parameter swept one at a time (plan §7 T2 table). |
| **Recovery rate** | Among a cluster's **confident** error samples (see below), the fraction whose native decision becomes correct under an arm; always reported with its Wilson CI **and** the mean margin shift Δs (toward gold). |
| **Control recovery / breakage** | Recovery under the matched control arm; and, symmetrically, the fraction of *matched correct* samples that become wrong under the same arm. |
| **Explained** | A **three-valued** outcome (`True/False/None`). `True` iff, on *confident* errors, (T0–T2 recovery tests) the Wilson lower bound of recovery ≥ `exit.recovery` **and** the Wilson upper bound of the worst matched control ≤ `exit.control` and `n ≥ exit.min_n`; `False` iff the upper bound of recovery < `exit.recovery` or the lower bound of a control > `exit.control`; otherwise `None` (inconclusive, can never support a cause). Necessity/consistency use their own predicates (FR-E-4). (C5) |
| **Coordinates** | All boxes are stored and passed as **normalized `[x0,y0,x1,y1] ∈ [0,1]`** relative to the original media, so they survive the processor's resize. |
| **Token grid** | Visual tokens after the 2×2 merger form an `(h/2, w/2)` row-major grid per image (per frame-group for video). All spatial maps (attention, saliency, probes, boxes) live on this grid. |
| **Unit of resampling** | `sample_id` (or video id), never individual rows — multiple prompts/arms per sample are not independent. |
| **Confident error** | An error with `\|s_base\| ≥ exit.conf_margin` and a valid answer format. Borderline errors flip under *any* perturbation (regression to the mean) and are excluded from recovery/necessity tests (reported separately). Own methodological guard; no source in `references.md` prescribes it. (C5) |
| **Null arm `N0.null`** | An imperceptible perturbation (JPEG q=`null.jpeg_q`, `null.shift_px` translation). The noise-floor control for knobs and O6. A deterministic base re-run is **not** a control (it recovers exactly 0). (C4) |
| **Answer mass** | `mass = Σ softmax(logits)[pos_ids ∪ neg_ids]` at the answer position. Rows with `mass < answer.min_mass` are `invalid_format`: excluded from recovery/consistency tests and counted. (C16) |
| **Diagnostic / signature** | A diagnostic is one test returning `True/False/None` (e.g. `crop`, `bgmask`, `hires`, `necessity`). A **signature** is the conjunction of required diagnostic outcomes that defines a cause (design.md §5.8). A cause is assigned only if exactly one signature matches. (C1) |

---

## 3. Global constraints

### 3.1 Size, simplicity and correctness (the size budget is soft; correctness is hard)
- **GC-1 (P0)** **Planning budget (soft):** ≈ **2,000 source lines** (excluding tests, blank lines, comments) in ≈ **13 modules** (list in `design.md` §2). **Tests have no size cap.** A module or the total MAY exceed the budget whenever correctness or verifiability needs it. Every overrun is logged in `DEVIATIONS.md` (module, lines over, reason, FR ids); an overrun beyond **+50 % of a module**, a new module, a new dependency or a §1.3 non-goal requires a stop-and-report to the human before continuing. Size MUST NOT be reduced by dropping a test, control arm, validity gate or input check, nor by merging distinct definitions. AC: `DEVIATIONS.md` exists and lists every overrun; no check in FR-Q is missing.
- **GC-2 (P0)** Exactly **one module** (`model.py`) may import model-specific classes from `transformers` or touch model internals (`.visual`, `.language_model`, layer lists, attention interface). All other modules interact through its small API. AC: `grep -l "transformers" ea/*.py` lists only `model.py`.
- **GC-3 (P0)** All experiment output is one **append-only long table** (`runs.jsonl`) plus small derived CSVs. No database, no ORM, no custom file formats.
- **GC-4 (P0)** Every analysis is a **pure function of a pandas DataFrame** (inputs → DataFrame/dict). No analysis function may call the model.
- **GC-5 (P0)** Interventions are **pure functions** `(sample, **params) → (images/frames, optional text override)` registered in one dict. Adding an intervention = adding one function + one registry line.
- **GC-6 (P0)** One config file (`config.yaml`) holds all [PROPOSED] values and sweep grids. No hidden constants in code.
- **GC-7 (P0)** **Correctness over brevity.** Every metric, diagnostic and analysis tool has a verification card (§6.3, FR-Q) that passes before it counts as implemented. AC: FR-Q tests present and green.

### 3.2 Allowed dependencies
`torch`, `transformers` (version pinned in `requirements.txt` after the §9 checks pass), `accelerate`, `pillow`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `statsmodels`, `pyyaml`, `matplotlib`, `pytest`. Optional: `bitsandbytes` (quantization arms), `opencv-python` or `av` (video decoding), `qwen-vl-utils` (video preprocessing helper), `ffmpeg` binary (bitrate arm only).

### 3.3 Dependencies that MUST NOT be added
`quantus`, `domino`, `captum`, `transformer_lens`, `nnsight`, `pytorch-lightning`, `hydra`, `wandb`, `mlflow`, `sqlalchemy`. Rationale: the needed metrics are < 60 lines each; TransformerLens/nnsight do not natively support Qwen3-VL and add more code than plain PyTorch hooks.

### 3.4 "Do not build" list (documented descopes)
| Item from plan | Decision | Reason |
|---|---|---|
| Hierarchical KV-cache policy (HERMES) | Not built | Re-implementing it dwarfs the rest. Sliding-window **recompute** (frame window N) is built and is the cache-free analogue. Plan §7 knob "KV-cache policy" is reported as *window-recompute only*, with this limitation stated in the report. |
| Domino / LADDER / LaVMD / SliceLens | Not built (P3 alternative: k-means on embeddings) | Metadata slicing is primary (plan §6.3). |
| Full explainer zoo (rollout, TAM, GLIMPSE) | Not built | Two explainers only (design.md §7). |
| Intermediate-representation injection, head ranking (SparseMM) | P3 | Not required for the evidence rule. |
| Thinking-variant analysis (H10) | P3 | Only if the Thinking checkpoint is chosen (plan §11 Q3). |

---

## 4. Functional requirements

### 4.1 Data layer (FR-D)

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-D-1 | P0 | Define one `Sample` schema (below) and one on-disk interface: `data/<bench>/samples.jsonl` + media files. **No other module reads raw benchmark files.** | Downstream code runs on a hand-written `samples.jsonl` with no benchmark installed. |
| FR-D-2 | P0 | Provide one adapter function per benchmark (`ConstructionSite10k`, `iSafetyBench`, `FieldWorkArena`, `PaSBench-Video`, `OVO-Bench` RT subset) that converts the benchmark's native format to `samples.jsonl`. Adapters MAY be stubs raising `NotImplementedError("needs access/format")` until access exists (plan §12 step 3). | Each adapter has a docstring listing the [VERIFY] items from plan §4. |
| FR-D-3 | P0 | Every field except `id`, `media`, `label` is optional. Missing metadata MUST NOT crash any module; slice search skips missing columns, metrics report `n` and `NaN`. | Test with a samples file where all metadata is `null`. |
| FR-D-4 | P0 | Provide a **synthetic dataset generator** (`make_toy(n)`) producing images with colored shapes standing in for "helmet present/absent", with boxes, facts, pair ids, size bins. Used by every test. | `make_toy(200)` returns valid `samples.jsonl` + PNGs in < 10 s. |
| FR-D-5 | P0 | A `bin_meta(df)` helper discretizes continuous metadata (object size, distance, blur, brightness, bitrate) into labeled bins (quantiles, configurable). | Output columns are categorical; no NaN-to-category leaks. |
| FR-D-6 | P0 | Video loading: `frames(sample, fps, n, order, t_end)` returns PIL frames sampled causally (all timestamps ≤ `t_end`), with `order ∈ {fwd, shuffle, reverse}` and optional `single=True` (one frame). | Unit test: no returned timestamp exceeds `t_end`. |
| FR-D-7 | P1 | **Domain/population check (C18).** `meta.router_pass` (0/1, optional) marks samples an upstream frame router would forward; all T0 metrics MUST be stratifiable by it. If labelled in-domain footage exists, run T0 on it and report the benchmark→domain gap in per-class recall/FPR. Root-cause claims are labelled `benchmark-domain` unless verified in-domain (large pre-trained VLMs need additional training before applying to real construction sites: 2508.11011). | Report prints the domain label; `router_pass=null` runs without error. |

**`Sample` schema** (all fields JSON-serializable):
```
id, bench, modality ("image"|"video"), media (path), label (1=violation, 0=safe), cls (rule id),
rule_text, item (PPE item name or null),
facts: {item: "present"|"absent"|"not_visible"},            # oracle facts for O2 and Q_P/Q_R
boxes: [{role: "violator"|"person"|"zone", xyxy: [..normalized..]}],
pair_id (null | str),                                        # matched safe/violation pair
meta: {scene, viewpoint, vtype ("absence"|"presence"|"proximity"), n_workers, feature_share,
       obj_size, dist, occlusion, blur, brightness, bitrate, temporal_necessity},
video: {fps, duration_s, onset_s, end_s}                     # null for images
```

### 4.2 Model adapter (FR-M) — the only model-coupled module

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-M-1 | P0 | `VLM(name, quant=None, min_pixels=None, max_pixels=None)` loads Qwen3-VL-4B (bf16 default; `quant ∈ {None,"8bit","4bit"}`), eval mode, greedy, batch size 1. | Loads from config; quantized variants load with `bitsandbytes`. |
| FR-M-2 | P0 | `vlm.margin(images|frames, text, pos, neg) → float` implements the margin of §2 using one forward pass with `logits_to_keep=1` (or equivalent), and also returns `mass` (answer mass, §2). `pos`/`neg` are word lists whose single-token ids are resolved from the tokenizer; multi-token words are dropped with a warning; an empty set raises. | Test: `margin == logit[Yes]−logit[No]` computed manually, tol 1e-3; `mass ∈ (0,1]`. |
| FR-M-3 | P0 | `vlm.grid(inputs)` returns visual-token positions and the merged grid `(t, h/2, w/2)` for images/video. | Test: `len(positions) == t*(h/2)*(w/2)`. |
| FR-M-4 | P0 | Expose knobs: `set_pixels(min,max)`; quantization at load; per-call latency (`ms`) and peak memory in the return metadata. | Latency recorded for every run row. |
| FR-M-5 | P0 | Determinism: re-running the same input twice gives \|Δs\| < 0.05 (bf16 tolerance). | Test (GPU). |
| FR-M-6 | P2 | `Tap(layers, patch)` context manager: captures decoder-layer outputs and optionally replaces them (for patching). | Test: capturing then patching with the identical tensor leaves `s` unchanged (tol 1e-3). |
| FR-M-7 | P2 | `Tap` also captures visual-encoder sites: ViT output (input to the merger), the three DeepStack tap inputs, and merger output, each reshaped to one vector per merged visual token. | Test: `n_pre_merger == 4 × n_visual_tokens`. |
| FR-M-8 | P2 | `knockout(block=[(frm_group, to_group)], layers)` context manager blocks attention from `to` queries to `frm` keys in selected layers (implemented with per-layer attention pre-hooks on an **eager-attention** model instance). Groups: `vis`, `q` (question text), `last`. | Tests: (a) empty knockout ⇒ `s` unchanged (tol 1e-3); (b) blocking `vis→last` and `vis→q` in all layers ⇒ `s` equals the blind-image margin (tol 0.1). |
| FR-M-9 | P1 | `attn_last(vlm, inputs)` returns last-token attention rows `[layers, heads, T]` (slice to visual tokens for `[layers, heads, Nv]`), using an eager-attention instance and forward hooks that keep **only the last row** (memory-light). | Rows sum to 1 over all keys; matches `output_attentions=True` on one sample (tol 1e-2). |
| FR-M-10 | P1 | `vlm.grad_x_input(inputs, pos, neg)` returns a per-visual-token saliency by taking the gradient of `s` w.r.t. the visual features (merged + DeepStack) and multiplying by them. | Test: saliency shape `[Nv]`; non-negative after `abs`. |
| FR-M-11 | P1 | `sink_mask(hidden_vis)` flags attention-sink visual tokens (massive-activation test, threshold from config). | Test: planted token with 100× magnitude is flagged. |

### 4.3 Prompts (FR-P)

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-P-1 | P0 | Prompts live in one versioned dict. Each prompt: `id, group, text template, pos, neg, gold(sample), needs_image`. `PROMPT_VERSION` string is written to every run row. | Changing any template text changes the version hash (hash computed, not hand-bumped). |
| FR-P-2 | P0 | Groups: **D** (≥3 paraphrases of the violation question), **POL** (positive "wearing X?", negated "NOT wearing X?", "anything missing?"), **FRM** (neutral / leading-positive / leading-negative), **QP** (perception sub-question in **both polarities**: `qp` "visible?" and `qp_abs` "missing?" — C7), **QR** (rule + oracle facts in text, no image). | All groups present; each prompt answerable with a single Yes/No token. |
| FR-P-3 | P0 | Prompts needing `item` or `facts` are skipped (not errored) for samples lacking them; skip counts are reported. | Run on toy data with `item=None` completes. |
| FR-P-4 | P3 | Thinking variant: budgeted reasoning then forced answer prefix for margin readout. | Only if the Thinking checkpoint is chosen. |

### 4.4 Runner and results store (FR-R)

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-R-1 | P0 | `run(tier)` iterates `model_config groups → samples → jobs (prompt × arm × params)`, loads each model configuration **once**, and appends one JSON line per job to `runs.jsonl`. | Number of model loads equals number of distinct model configs. |
| FR-R-2 | P0 | **Resumable & idempotent:** each job has a content hash key (sample, prompt version+id, arm, params, model cfg); existing keys are skipped. | Killing and restarting produces no duplicate rows. |
| FR-R-3 | P0 | Row schema: `key, sample_id, prompt_id, group, arm, knob, value, model_cfg, s, pred, gold, correct, n_vis_tokens, latency_ms, t (stream time or null), extra (json; contains `mass`)`. | Schema test on toy run. |
| FR-R-4 | P0 | Run manifest saved once per tier: git hash, package versions, config hash, prompt version, seed, model name/revision, GPU. | `manifest.json` exists and is referenced by the report. |
| FR-R-5 | P0 | Failures of a single job (e.g. OOM, unreadable media) are logged to `failures.jsonl` and skipped; the run continues. | Injected failure does not stop the run. |
| FR-R-6 | P0 | Tier commands: `t0`, `t1`, `t2`, `t3`, `t4`, `stream`, `report`. Each reads `runs.jsonl` and config, and writes CSVs to `outputs/<tier>/`. | `--help` lists them. |
| FR-R-7 | P0 | **Validity flag (C16).** Rows with `mass < answer.min_mass` get `valid=False`; analysis functions MUST ignore them and the tier summary MUST count them per prompt group. `selfcheck` fails if mass < `answer.min_mass` on a calibration sample of any prompt group. | Test: an injected low-mass row is excluded and counted. |

### 4.5 Behavioral metrics (FR-B) — plan §6.2

All metrics take a DataFrame, return numbers/DataFrames, accept `by=[cols]` stratification, and report `n`. CIs use **cluster bootstrap over `sample_id`** (FR-B-0).

| ID | Pri | Metric | Notes / AC |
|---|---|---|---|
| FR-B-0 | P0 | `boot_ci(fn, df, n=1000)` resampling sample ids | Test: CI width shrinks ~√n on synthetic data. |
| FR-B-1 | P0 | Per-class recall and FPR on safe samples, stratified by class/size bin/scene | |
| FR-B-2 | P0 | Recall @ fixed FPR; FPR @ fixed recall (from ROC of `s`); frozen τ for the calibrated decision | Test on a synthetic score with known ROC. |
| FR-B-3 | P0 | Cost-weighted error with configurable `C_miss`, `C_fa`; also min expected cost over thresholds | Costs from config (plan §11 Q6 unresolved → placeholder default). |
| FR-B-4 | P0 | Worst-slice recall / slice gap (uses FR-S output) | |
| FR-B-5 | P0 | Pair accuracy (both items correct) **and** threshold-free pair ranking accuracy (`s_viol > s_safe`) | Test: perfect/inverted/random scorers → 1 / 0 / ≈0.5 pair ranking; pair accuracy ≤ min(item accuracies). |
| FR-B-6 | P0 | Polarity-flip consistency (all POL prompts agree), per-prompt accuracy, Yes-rate (affirmation bias) | Test: an "always Yes" synthetic model gets consistency 0 on opposite-polarity prompts and Yes-rate 1. |
| FR-B-7 | P0 | Paraphrase flip rate, **always reported next to the same prompts on the blank frame** (arm `O3.blank`) | Report refuses to print flip rate without the blank counterpart. |
| FR-B-8 | P0 | Visual Necessity = mean KL(Bern(σ(s_real)) ‖ Bern(σ(s_other))) for `other` ∈ {blank, noise, scene-swap, **evidence-box occlusion `O3.occlude`**} (C3). Whole-image blanking alone cannot tell "ignores the evidence" from "ignores the image"; localized occlusion is a distinct test (2605.22903). | Test: identical margins ⇒ 0. |
| FR-B-9 | P0 | Prompt-framing sensitivity = mean Δs (leading-pos − neutral, leading-neg − neutral) | |
| FR-B-10 | P0 | Perception–Decision consistency and right-for-wrong-reason rate (QP vs D) | Uses FR-T1. |
| FR-B-11 | P1 | Counterfactual consistency on matched pairs: margin must reorder (`s_viol > s_safe`) when evidence is edited in/out; the cited-evidence variant (ask which item) is P3 | |
| FR-B-12 | P0 | Calibration: ECE (15 bins), Brier, reliability-diagram data, on `σ(s)` vs gold; per class and slice | Test: perfectly calibrated synthetic ⇒ ECE < 0.02. |
| FR-B-13 | P0 | Selective prediction: risk–coverage curve, AURC, Coverage@Accuracy and Coverage@Recall, confidence = `|s|` | Test: perfectly ranked confidence ⇒ AURC equals the oracle minimum. |
| FR-B-14 | P0 | Calibration/AURC recomputed **per knob value** | Output has a `knob,value` index. |
| FR-B-15 | P0 | Oracle recovery rate and breakage per (cluster, arm) | Core of the escalation engine (FR-E). |
| FR-B-16 | P0 | Quality-sweep curves: accuracy/recall vs severity, per corruption type | |
| FR-B-17 | P0 | Temporal controls: multi-frame gain, order gain, reverse sensitivity (accuracy of full − single/shuffled/reversed) | |
| FR-B-18 | P0 | **Margin-shift reporting (C5).** For every (cluster, arm) report the mean/CI of Δs = (s_arm − s_base) signed toward gold, next to recovery, and the same for `N0.null`. A model's internal support for the answer can weaken without the prediction flipping (2605.22903), so sign flips alone under-report both effects and noise. | Test: `N0.null` Δs ≈ 0 within bf16 tolerance (\|Δs\| < 0.05) on the fake model. |

### 4.6 Slice discovery (FR-S) — plan §6.3

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-S-1 | P0 | Metadata slice search: enumerate conjunctions of up to `max_order` (default 2) categorical metadata columns; for each slice with support ≥ `min_support`, report error rate, support, Wilson CI, gap to base rate, one-sided Fisher p, and **Benjamini–Hochberg** q-value over all tested slices. | Planted-slice test: a toy dataset where one slice has 3× error rate returns that slice as top-1 with q < 0.05; a dataset with no planted slice returns none. |
| FR-S-2 | P0 | Run separately for **misses** (on positives) and **false alarms** (on negatives). | Two output files. |
| FR-S-3 | P0 | Return slices as predicates usable by the escalation engine (`dict col→value`). | Round trip: predicate → sample ids. |
| FR-S-4 | P3 | Embedding slices: k-means on image embeddings, per-cluster error rate + CI; every slice validated against matched contrasts. | |

### 4.7 T1 triage (FR-T1) — plan §7 T1

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-T1-1 | P0 | For every error sample and its matched correct sample run four prompts on the same media: **Q_P** (perception, both polarities: `qp`, `qp_abs`), **Q_R** (rule + oracle facts, text only), **Q_D** (decision). | Rows exist for all four. |
| FR-T1-2 | P0 | `triage(qp_ok, qr_ok, qd_ok)` returns the plan's reading: perception failure / process failure / rule-reasoning failure / right-for-wrong-reason / ok, with `qp_ok = qp ∧ qp_abs`, and **`untriaged (no item)`** when Q_P is undefined (proximity/zone violations). The perceptual-vs-process split follows 2609.22588. A reading is a *provisional label* only; it becomes a cause through FR-E-5. | Exhaustive 8-case unit test + the `None` case. |
| FR-T1-3 | P0 | Emit per-cluster counts of each reading, with matched-correct comparison, and the T2 tests to run next (from the "Next" column). | `t1_triage.csv`. |
| FR-T1-4 | P0 | **Perception/knowledge gate (C6).** A "prior-driven" (necessity) diagnosis is only issued for samples where Q_P (both polarities) and Q_R are correct. Language-prior tests must first establish that the basic abilities pass, otherwise a perception failure is mistaken for a prior (2406.08702). Samples failing the gate are counted as `gated-out`. | Test: necessity returns `None` when the gate fails. |

### 4.8 T2 interventions and knobs (FR-T2) — plan §7 T2

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-T2-1 | P0 | Interventions O1–O7 implemented as registered pure functions, **each with its named matched control arm** (plan table): O1 crop ladder (full → GT box+margin → tight → *attention-derived crop P1*) with controls *random same-size crop* and *full frame resized to the same token count*; O2 oracle facts with *wrong-facts* control; O3 blind / noise / scene-swap; O4 polarity + framing arms; O5 per-person crop and box-overlay with *overlay-on-random-person* control; O6 quality sweep (degrade corrects, sharpen errors); O7 temporal ablations; **plus (C3, C4)** `O1.bgmask` (full frame at native resolution, everything outside the box+margin grayed ⇒ same token count; control: same-size random box kept), `O3.occlude` (evidence box grayed; control: random same-size box) and `N0.null` (the noise-floor control for knobs and O6). | Registry lookup returns `(fn, control_name)` for each. |
| FR-T2-2 | P0 | Every intervention runs on error samples **and** matched correct samples (symmetry, plan §7 T2). | Both appear in `runs.jsonl` for each arm. |
| FR-T2-3 | P0 | Knob sweeps, one knob at a time, from config grids: resolution (`max_pixels`), visual-token budget (random vs attention-based pruning), fps, frame window N, quantization, camera conditions (motion/defocus blur, low-light noise, haze, JPEG quality, H.264/H.265 bitrate, resampling method). The token-pruning knob SHOULD be implemented as **dropping visual tokens at the input of the LLM** (random vs lowest-attention-first), not by re-implementing a pruning paper. **The control of every knob is `N0.null`, never a base re-run.** | Each knob value is one `(knob, value)` pair in `runs.jsonl`. |
| FR-T2-4 | P0 | For every knob run, re-measure accuracy, mean margin, ECE, AURC and latency. | `t2_knobs.csv`. |
| FR-T2-5 | P0 | Report the crop caveat: cropped inputs change token count; always compare with the same-token control. | Report text check. |
| FR-T2-6 | P0 | **O2 is an exclusion test (C2).** With oracle facts in the text any LLM answers correctly, so `qr_ok=True` MUST NOT be used to name "perceptual failure" or any cause; only `qr_ok=False` (⇒ rule-reasoning failure) is informative. Q_R (FR-T1-1) and O2 are the **same prompt** and one diagnostic, `qr_ok`. | Test: a fake model with a perception failure gets `qr_ok=True` and no cause from it. |
| FR-T2-7 | P0 | **Resolution vs clutter (C3).** `O1.gtcrop` recovery alone MUST NOT yield "pixel/token-budget limit": the signature also requires the high-`max_pixels` knob (`hires`) to recover and `O1.bgmask` not to. Crop-only recovery ⇒ `unexplained → next tier`. | Table-driven test on the signature table. |

### 4.9 T3 attribution (FR-T3) — plan §6.4 — P1

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-T3-1 | P1 | Two explainers on the token grid: `attn_last` (layer-band — default **upper half** of the LLM layers (C9), head-averaged, **sink-excluded**, renormalized over non-sink visual tokens) and `grad_x_input`. | Both return `[h/2, w/2]` maps. |
| FR-T3-2 | P1 | Localization metrics against the GT violator box: mass accuracy, rank accuracy, pointing game, and **enrichment = mass_in_box / box_area_fraction** (chance = 1). | Test: a map concentrated in the box ⇒ enrichment ≈ 1/area_fraction; uniform map ⇒ ≈ 1. |
| FR-T3-3 | P1 | Background/scene attribution mass per scene type. | |
| FR-T3-4 | P1 | Error vs matched-correct localization comparison with the **3-way reading** (plan §6.4a) returning a `loc` diagnostic: `loc_equal_high` / `loc_lower_in_errors` / `loc_low_both` (medians with CIs; thresholds from config). **`loc_equal_high` is *not* H3 by itself (C8):** MLLMs localize evidence even when wrong because of small-detail limits (2502.17422), and "seeing but not believing" is the process-failure counterpart (2510.17771); H3 additionally needs `crop=False ∧ hires=False` and robust causal evidence (FR-T4-8/10). `loc_low_both` ⇒ shortcut (H5) only with `necessity=True`. | Synthetic test for all three branches + the signature test. |
| FR-T3-5 | P1 | Deletion/insertion AUC (and comprehensiveness/sufficiency at top-k) by masking token-grid cells in image space and re-scoring. The **primary fill is debiased** (noisy linear imputation, ROAD: removal *shape* leaks information and rankings depend on the fill, 2202.00449); gray and blur are stability checks (C10). | Test: with a synthetic model whose margin depends only on the box, deletion-AUC of a perfect map < random map. |
| FR-T3-6 | P1 | **Explainer-validity gate (§6.4b):** (a) beats random-order baseline and raw-attention baseline on deletion/insertion; (b) ranking stable across three pixel-replacement strategies (`nli`, `gray`, `blur`); (c) randomization sanity: map similarity to the original drops when LLM weights are re-initialized; (d) robustness: Spearman similarity under small input noise; (e) **[P2] internal agreement:** `knockout` of the top-k attributed tokens reduces \|s\| more than random-k (output-faithfulness curves can be maximised by routes unrelated to the computation actually used: 2509.04588, shown on CNNs). Deletion/insertion uses only the *ranking* of scores (2201.13291), so a pass says nothing about score magnitudes. Downstream T3 tables are written only if the gate passes; otherwise the output is flagged `UNVALIDATED`. | Gate result in `t3_gate.json`. |
| FR-T3-7 | P1 | SpRAy-style clustering of error attribution maps (k-means on downsampled maps). Output is **exploratory** unless FR-T3-8 passes and group labels exist (SpRAy struggles with complex features and severe imbalance: 2604.04518) (C11). | Cluster ids written per error. |
| FR-T3-8 | P1 | Planted-confounder check: on toy data with a scene shortcut, the explainer's background mass flags it. | Passing test. |

### 4.10 T4 mechanistic analysis (FR-T4) — plan §6.5 — P2

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-T4-1 | P2 | Cache pooled site features for sampled (stratified) samples to disk: pooled over the GT box on the token grid for sites ViT-out, DeepStack taps (3), merger-out, LLM layer `l` visual positions (all layers, configurable stride), and LLM layer `l` question positions (`q`, C12) and last position. Targets: `present/absent/not_visible`, person-in-zone, proximity. | `cache/<site>.npy` + `labels.csv`; peak GPU memory not growing across samples. |
| FR-T4-2 | P2 | Linear probes (standardized logistic regression, 5-fold CV, grouped by `pair_id` if present) per site/layer: balanced accuracy, **selectivity** vs shuffled labels, and accuracy restricted to error-cluster vs matched-correct items. **A site counts as lost for a cluster only if the matched corrects remain decodable there** (C12): visual information vanishes at an intermediate layer for every input (information horizon, 2512.07580) and migrates into question tokens and the last position (2411.18620), so absolute loss at box tokens is generic. Verdicts with a CI straddling `probe.decodable_acc` are `underpowered`. | `t4_probes.csv`. |
| FR-T4-3 | P2 | **Vision-only encoder ceiling** (DINOv2) probe on the same boxes. | Column `site="dinov2"`. |
| FR-T4-4 | P2 | `probe_verdict(site→decodable)` implements the plan §7 T4 decision rule (not decodable at ViT ⇒ encoder limit; ViT yes / merger no ⇒ H1; merger yes / lost at L ⇒ H2 with layer L; decodable throughout ⇒ H3); "decodable" at layer L means decodable at box ∪ question ∪ last-position sites, and the verdict is only issued if matched corrects stay decodable at the lost site (C12). | Table-driven unit test, all branches. |
| FR-T4-5 | P2 | Attention knockout sweep over layer windows for `vis→q`, `vis→last`, `q→last` (and temporal/spatial for video P3): margin-drop effect size. | `t4_knockout.csv`. |
| FR-T4-6 | P2 | Logit lens: margin read out from every layer's last-token state through the final norm and LM head. | Test: last layer equals the actual margin (tol 1e-2). |
| FR-T4-7 | P2 | Information-horizon layer per sample = smallest `l` such that blocking visual→text flow from layer `l` onward changes `s` by < 5% of the full visual effect; samples with negligible visual effect are excluded and counted. Reported **per cluster as errors vs matched corrects with CI**; H2 requires a significant difference (C12). | Synthetic-model test. |
| FR-T4-8 | P2 | Activation patching, matched-counterfactual corruption (same-scene safe/violation pair at **identical token length**), denoising and noising directions, normalized logit-difference recovery `(s_patched − s_corrupt)/(s_clean − s_corrupt)`, layer windows × position groups (`vis_in_box`, `vis_all`, `q`, `last`). Pairs whose token lengths differ are resized to match or skipped with a count. **Robustness (C13):** a patching result counts as *causal* only if it holds under ≥ 2 corruption methods (`patch.corruptions`: matched counterfactual; gray/noise replacement of the box tokens) and both metrics (normalized logit-difference recovery; raw Δs), because conclusions depend on corruption method and metric (2309.16042). | Heatmap data in `t4_patch.csv`; sanity: patching all layers at all positions ⇒ recovery ≈ 1. |
| FR-T4-9 | P2 | Visual-token similarity across depth (mean pairwise cosine per layer). | |
| FR-T4-10 | P2 | **Evidence rule enforced in code:** an internal root-cause label (H1/H2/H3) is only emitted when ≥ 2 independent methods agree, at least one decodability method (probe) and one causal method (knockout or patching). "Causal" means robust under FR-T4-8. Otherwise `label="unresolved (single method)"`. | Unit test with one-method evidence ⇒ unresolved. |

### 4.11 Streaming track (FR-ST) — plan §6.6, §7 — P1

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-ST-1 | P1 | **Causal online evaluation:** at stream time `t` the model receives only frames with timestamp ≤ `t` (window of last N frames at fps f). The runner writes one `(video, t, s)` row per step. | Causality test (FR-D-6). |
| FR-ST-2 | P1 | **Trigger policies are applied offline** to the stored margin series (threshold τ, k-of-n persistence, hysteresis) — changing the policy MUST NOT require new model runs. | Re-analysis with a new policy touches only `stream.py`. |
| FR-ST-3 | P1 | Metrics: onset error, warning latency, asymmetric early/late score (miss costs more than early alarm; constants from config), spam rate in dead time, silent-miss rate, premature-alarm rate. `stream.anticipation_s` is an **optional** temporal early-warning window (0 = off): alerts inside it are scored as hits (`anticipated`), not premature. The project's *prediction* (latent hazard) is a single-frame task specified in §4.14; for hazard videos `onset` is the **risk onset** (first visible sign of the hazard condition), as annotated in PaSBench-Video (2606.02443). TRACE's false-alarm notion and SPOT-Bench's dead-time spam (2609.30670; 2604.24317) apply to the timing metrics. | Tests on hand-built series with known answers. |
| FR-ST-4 | P1 | Timing-error decomposition per video: `ok / anticipated (only if anticipation_s > 0) / silent / late / premature / spam`, and for `late`: **perception-late** (window-level margin at/after onset is below τ) vs **policy-late** (window-level margin exceeded τ earlier than the alert). | Tests for each class. |
| FR-ST-5 | P1 | Window-level vs stream-level metric gap (H11 test): window-level = classify windows overlapping the violation interval; stream-level = alert timing metrics. | Both reported side by side. |
| FR-ST-6 | P1 | Persistence/recovery: frames until correct after a degraded frame (uses T2 quality arms on video). | |
| FR-ST-7 | P1 | Latency profile per knob configuration (p50/p95 forward time, equals time-to-first-token for a prefill-only readout; per-frame cost). | In `stream_latency.csv`. |
| FR-ST-8 | P1 | Protocol disclosure block (evidence boundary, history mechanism = window-recompute, trigger rule) printed at the top of every streaming result file (plan §6.6 / TRACE). | Header check. |
| FR-ST-9 | P3 | Recency analysis of attention over cached video tokens vs time distance. | |

### 4.12 Escalation engine and report (FR-E)

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-E-1 | P0 | Cluster builder: error set × slice predicates (+ taxonomy labels) ⇒ clusters with ids, sizes, matched correct ids. | `clusters.json`. |
| FR-E-2 | P0 | Taxonomy tagger: assign plan §5 failure types **automatically where rules exist** (e.g. 2a from `vtype=="absence"`, 6 from paraphrase flips, 7 from per-person arm recovery, 8 from quality arm recovery, 10 from T1) and leave the rest `"needs-manual"`. | Tags column in `errors.parquet`. |
| FR-E-3 | P0 | **Exit rule (plan §7.0, revised C1/C4/C5):** every diagnostic is **three-valued** and computed on **confident errors** with Wilson CIs (definition of *Explained*, §2); controls are the arm-specific matched controls, and `N0.null` for knobs and O6 (never a base re-run); with several controls the **worst** one is used. Clusters with `n < exit.min_n` are `underpowered`, never `explained`. A cause is assigned only through FR-E-5. | Table-driven unit tests with synthetic recoveries incl. CI-straddle ⇒ `None`. |
| FR-E-4 | P0 | Different diagnostics use different rules, implemented as small predicates: **recovery** (O1.gtcrop, O1.bgmask, O5, O6, knobs/`hires`), **exclusion** (O2/Q_R: only `qr_ok=False` is informative, FR-T2-6), **necessity** (O3 blank **and** occlude, gated by FR-T1-4), **consistency** (O4: polarity/framing flips must exceed paraphrase noise **and** the matched corrects' flip rate). | Unit tests per predicate. |
| FR-E-5 | P0 | **Signature table (C1):** cause = conjunction of required diagnostic outcomes, as a literal dict (design.md §5.8). `decide` returns a **unique** cause, `ambiguous` (several fit; candidates listed), `inconclusive: more data` (candidates blocked only by `None`), or `unexplained → next tier`. Tiers are not "first pass wins": all T1/T2 diagnostics are computed per cluster, then `decide` runs; T3/T4 are added only for non-unique results. The table (with implications) is printed in the report. | Unit tests: unique / ambiguous / inconclusive / unexplained; `crop` alone ⇒ unexplained. |
| FR-E-6 | P0 | Label/benchmark-issue bucket: sample *k* errors (config) per cluster for manual audit before any root-cause claim; the report shows audit status per cluster (`pending/done`) and refuses `final` status for unaudited clusters. | `audit_sample.csv` generated. |
| FR-E-7 | P0 | Final report `root_causes.md` + `root_causes.csv`: per cluster — size, failure types, provisional label (T1), tier at which explained, cause, evidence list (tier, method, statistic, n), number of agreeing methods, implication (from §8 table), caveats (§10 of the plan). Counts and percentages per cause. | Generated by `report` command on toy data. |
| FR-E-8 | P0 | The report also prints the deployment-vs-architecture split (plan §1): share of errors explained by each family, plus the share of the **unexplained** residue. | |
| FR-E-9 | P0 | **Per-error accounting (C17):** clusters overlap, so each error takes the cause of the most specific (smallest) cluster containing it with a unique cause; ties or none ⇒ `ambiguous` / the `ALL_ERRORS` cause. `diag_vector.parquet` has one row per error; FR-E-8 percentages use it. | Test: an error in two clusters is counted once. |
| FR-E-10 | P0 | **`ALL_ERRORS` cluster (C17):** always built, so diffuse errors outside any significant slice are diagnosed. Slice clusters also carry a human *coherence* verdict in the audit sample (incoherent slices are common: 2203.14960). | `clusters.json` contains it. |
| FR-E-11 | P0 | **Accounting identity (C20):** every error sample ends in exactly one final state — `unique cause`, `ambiguous`, `inconclusive`, `unexplained`, `untriaged (no item)`, `gated-out`, `invalid_format`, `underpowered`, `audit: label issue`. The report prints the table and the counts MUST sum to the total number of errors. | Unit test: sum of states == number of errors on the fake model and on a shuffled variant. |
| FR-E-12 | P0 | **Scope of evidence (C20):** `root_causes.md` ends with the list in §1.4 (what this analysis cannot establish), the `validate` precision of each printed cause, and which diagnostics were `UNVALIDATED` or `underpowered`. | Report lint. |

### 4.13 Validation of the diagnostics (FR-V) — C15

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-V-1 | P0 (runs at M3b) | `python -m ea validate`: from samples that are correct and confident under `base`, apply **known-cause injections** (`INJ.small`, `INJ.clutter`, `INJ.degrade`, `INJ.lead`) and a **label-noise null** (`INJ.label`: gold flipped for `validate.label_noise` of the pool, input unchanged); run the same diagnostics and `decide` on each injected cluster (≥ `validate.per_cause_n` errors each). | `outputs/validate/confusion.csv`: injected cause × assigned cause. |
| FR-V-2 | P0 | Per-cause **precision and recall with Wilson CIs** from the confusion matrix. The `INJ.label` row MUST end `unexplained`/audit; any confident cause there counts as a false attribution. | Test on the fake model: label-noise never gets a confident cause. |
| FR-V-3 | P0 | **Decisive-gating:** the report prints a cause as `decisive` only if its validated precision ≥ `validate.min_precision`; otherwise `candidate (validated precision p [lo, hi])`. Causes without an injection analogue (H3, H1-merger, H2) are always `candidate` unless FR-T4-10 passes. The report states that injections are synthetic (optimistic bound) and single-cause (real errors are mixtures). | Report lint. |

### 4.14 Latent-hazard ("prediction") track (FR-H) — C19

*Definition (project owner):* the VLM must report a scene condition that could foreseeably cause an incident under a trigger, e.g. an open-lid container with no support that can fall in wind. Single-frame property; design.md §5.14.

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-H-1 | P0 | Sample type `vtype="latent_hazard"` with `facts.conditions` (list of `{obj, state, support}`), `trigger`, `consequence`, `hazard_gold ∈ {0,1}`, optional ordinal `likelihood`/`severity`. Adapters that cannot supply them skip the track and say so. | Loader test; missing fields ⇒ track skipped with a message. |
| FR-H-2 | P0 | **Secured twins:** every hazard sample has ≥ 1 matched scene with the hazard removed (lid closed/supported), `pair_id` linking them. Recall without false-positive measurement is not reported (recall and false-positive rate are tightly coupled and models fire indiscriminately on look-alike scenes: 2606.02443). | Report refuses hazard recall without twins. |
| FR-H-3 | P0 | Prompts `hz_d` (uncued decision), `hz_cue` (trigger and consequence named), `hz_s_<k>` and `hz_s_<k>_abs` (per-condition perception, both polarities), `hz_r` (text-only reasoning from oracle conditions); all single-token margins via FR-M-2; triage maps Q_P/Q_R/Q_D onto them (design.md §5.14). | Prompt render tests. |
| FR-H-4 | P0 | Arm `HZ.cue` with control `HZ.cue_twin` (same cued prompt on secured twins). Diagnostic `cue` = `True` only if cued recovery passes FR-E-3 **and** twin false-alarm rate ≤ `exit.control`. Signature `uncued-hazard under-reporting` = `cue ∧ qp_ok ∧ qr_ok`; `qr_ok` for hazard clusters comes from `hz_r` and is an exclusion test only (FR-T2-6). | Table-driven test; a cue that raises "Yes" on twins ⇒ `cue=False`. |
| FR-H-5 | P0 | Metrics: recall at calibrated τ **with** false-alarm rate on twins; `pair_acc`/`pair_rank_acc` hazard-vs-twin as headline; slices by `trigger`, `consequence`, object class. | Unit tests on synthetic pairs. |
| FR-H-6 | P0 | Label quality: ≥ 2 annotators or an agreement estimate per `hazard_gold`; hazard clusters with agreement below `hazard.min_agreement` (0.6, own proposal) are capped at `candidate`; the audit sample (FR-E-6) is mandatory for them. The report states that the trigger is not visible in the image and that likelihood/severity are subjective. | Report lint. |
| FR-H-7 | P1 | `hz_report`: under the deployed commenting prompt generate ≤ 60 tokens and score with a keyword/synonym checklist (object, condition, consequence from `facts`); lenient by design and always followed by the human audit sample. Video hazards: `onset` = risk onset (first visible sign of the condition), not incident time. | Checklist unit test; stream tests use `risk_onset`. |

---

## 5. Non-functional requirements

| ID | Requirement | AC |
|---|---|---|
| NFR-1 | **Reproducibility.** Seeds fixed (default 0) for sampling, bootstrap, matched-control draws, noise arms. Greedy decoding only. | Two full toy runs ⇒ byte-identical CSVs (GPU-free fake model). |
| NFR-2 | **Resource safety.** One sample at a time on GPU; cached tensors are pooled before storage; per-sample `torch.cuda.empty_cache()` is NOT used (fragmentation noise) — instead delete references. T4 caches stored on disk as float16. | Peak memory logged per tier. |
| NFR-3 | **Fail loudly on wrong assumptions.** Every [VERIFY] item of §9 is a startup assertion or unit test with an actionable message. | Run `python -m ea selfcheck`. |
| NFR-4 | **Readable by a newcomer.** Each module starts with a ≤ 10-line docstring: purpose, inputs, outputs, plan sections implemented. Functions SHOULD be ≤ 40 lines; longer is allowed (with a comment saying why) when splitting would obscure a numerical procedure. | Review checklist. |
| NFR-5 | **No silent skips.** Anything skipped (missing metadata, missing facts, token-length mismatch in patching) is counted and printed in the tier summary. | Summary table. |
| NFR-6 | **Speed.** The toy end-to-end test (fake model) runs in < 60 s on CPU. Real-model tiers are resumable. | CI timing. |
| NFR-7 | **Statistics hygiene.** Every reported rate has `n`; every comparison across slices uses CIs or FDR control; no p-values without multiplicity handling. | Report lint. |

---

## 6. Testing requirements

### 6.1 Test layers
1. **Unit tests (CPU, no model):** every pure function in `metrics.py`, `slices.py`, `escalate.py`, `triage.py`, `stream.py` against synthetic DataFrames with *known* answers (see ACs above).
2. **Fake-model integration test (CPU):** a `FakeVLM` whose margin is a known function of the image (e.g. depends on whether the "helmet" patch is visible at a given scale) is injected into the runner. The full pipeline — T0 → T1 → T2 → escalation → report — runs on `make_toy`, and the test asserts:
   - a cluster of "small object" errors is labeled **pixel/token-budget limit** because the crop arm recovers it while the random-crop control does not;
   - a cluster of "scene shortcut" errors is labeled **prior-driven** because the blank-frame margin equals the real-frame margin;
   - a cluster of "absence" errors is labeled **polarity/absence** via polarity-flip inconsistency;
   - an unexplained cluster stays **unexplained** and is queued for T3/T4;
   - a "clutter" cluster is labeled **clutter/binding/context** (crop and `O1.bgmask` recover, `hires` does not) and the small-object cluster is **not** recovered by `bgmask` (C3);
   - a label-noise cluster (flipped gold) stays **unexplained** (C15).
   *Scope:* this test shows the **implementation** matches the rules (the fake model's mechanisms were built to match them). Whether the rules name real causes is measured by `validate` (FR-V) on the real model.
3. **Model smoke tests (GPU, marker `gpu`):** the FR-M-2/3/5/6/7/8/9/10 checks on 2 real images with the real checkpoint (can use the smaller Qwen3-VL variant for CI if the 4B is too large, as long as the same code path is exercised).
4. **Invariant tests:** (a) `s` under `base` arm equals `s` from a manual forward pass; (b) hooks that do nothing do not change `s`; (c) resume produces no duplicates; (d) matched controls never include the error sample itself; (e) bootstrap resamples whole `sample_id`s.

### 6.2 Acceptance gates per milestone
| Milestone | Gate |
|---|---|
| M1 data + model + runner | `make_toy` → T0 run on the real model produces `runs.jsonl`; FR-M-2/3/5 pass. |
| M2 metrics + slices | All FR-B/FR-S unit tests pass; T0 report on toy data. |
| M3 T1 + T2 + escalation + report | Fake-model integration test passes (the planted-cause cluster labels above). |
| M3b `validate` | Confusion matrix produced on the **real** model; `INJ.label` row has no confident cause; per-cause precision recorded; `decisive` gating active in the report. |
| M4 streaming | FR-ST unit tests + a 1-video causal run. |
| M5 T3 | Explainer gate runs; planted-confounder test passes. |
| M6 T4 | Tap/knockout/lens/patching invariants pass; evidence-rule test passes. |

---

### 6.3 Implementation-correctness protocol (FR-Q) — C21

Applies to **every** metric, diagnostic, intervention, attribution, probe, patching and streaming tool. Per-tool test content is in `design.md` §7.6; thresholds are in `config.yaml: tests`.

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-Q-1 | P0 | **Verification card:** each tool has (a) a **known-answer** test with a hand-computed expected value, (b) at least one **property/invariant** test, and (c) a docstring stating *Measures / Does not measure / Other explanations*. | Review checklist; tests name the FR they cover. |
| FR-Q-2 | P0 | **Independent cross-check** where a standard implementation exists: Wilson CI vs `statsmodels`, BH-FDR vs `statsmodels`, ROC/AUROC vs `sklearn`, Fisher/hypergeometric vs `scipy`, binned calibration vs a naive loop, bootstrap coverage by simulation; every vectorised routine vs a naive loop on random inputs. Tolerances are documented in the test. | Tests compare outputs within tolerance. |
| FR-Q-3 | P0 | **Null and planted-effect tests** for every diagnostic that returns `True/False/None` (recovery, necessity, consistency, 3-way, probes, slice search): on a fake model with **no** planted effect, the false-positive rate over `tests.null_runs` seeded runs ≤ `tests.null_fpr_max`; with a planted effect, the true-positive rate ≥ `tests.planted_tpr_min` at `exit.min_n`. | Tests print the measured rates. |
| FR-Q-4 | P0 | **Anchor tests for model-touching tools:** empty knockout ⇒ `s` equals base; `patch(clean→clean)` ⇒ no change; patching everything ⇒ counterfactual output (recovery 1) and nothing ⇒ 0; logit lens at the final layer equals the model logits; attention rows sum to 1 and blocked positions get exactly 0; hooks leave no state (a second base run is identical); batch vs single forward agree; `n_vis == t·h/2·w/2` and the box→cells→box round trip hold on the real processor; `grad×input` matches a finite-difference directional derivative within `tests.finite_diff_rtol`. | `selfcheck` + GPU smoke tests. |
| FR-Q-5 | P0 | **Intervention property tests:** `O1.bgmask` keeps inside pixels bit-identical and sets the outside to a constant; `O3.occlude` is its complement; the same-token control matches the crop's visual-token count (±1); random controls match size and have IoU = 0 with the evidence box; `N0.null` differs from the base image yet PSNR ≥ `tests.psnr_null_min_db`; `degrade` PSNR is monotone in severity. | Unit tests. |
| FR-Q-6 | P0 | **Mutation checks:** for `explain_recover`, `explain_necessity`, `explain_consistency`, `gate_ok`, `conf_ids`, `decide`, `classify`, a parametrized test injects one deliberate bug (flip an inequality, drop the control, drop the gate, use a base re-run as control) and asserts that at least one other test fails. | `tests/test_mutation.py`. |
| FR-Q-7 | P0 | **Leak and causality checks:** streaming results at time *t* are unchanged when future frames are perturbed; probes use group CV by `pair_id` (assert disjoint) and label-shuffle accuracy ≈ chance; matched controls balance their matching variables (\|SMD\| ≤ `tests.smd_max`) and exclude the error itself; slice columns contain nothing derived from outcomes. | Unit tests. |
| FR-Q-8 | P0 | **Signature-table audit:** a test enumerates the diagnostic outcome domain, asserts `decide` never raises and is deterministic, and writes the compatibility matrix (which causes can co-match) to `tests/snapshots/sig_compat.txt`; editing the signature table requires updating the snapshot. | Test + snapshot. |
| FR-Q-9 | P0 | **Deviation log:** `DEVIATIONS.md` records every size overrun, every departure from a plan definition and every failed [VERIFY] item with the actual value. | File present; `selfcheck` failures appear in it. |
| FR-Q-10 | P0 | **Finished means verified:** an implementation step (design.md §8) is complete only when the verification cards of the tools it introduces pass. | CI runs FR-Q tests per step. |

---

## 7. Configuration defaults (all from plan [PROPOSED] items)

Single `config.yaml`. These are defaults, to be tuned on a pilot (plan §7.0).

| Key | Default | Plan source |
|---|---|---|
| `exit.recovery` / `exit.control` / `exit.min_n` | 0.5 / 0.1 / 10 | §7.0 (min_n added) |
| `prompts.paraphrases` | ≥ 3 | §6.2, §7 T0 |
| `seeds` | `[0]` (≥ 3 for variance studies) | §6.7 |
| `cost.miss` : `cost.false_alarm` | 10 : 1 (placeholder) | §11 Q6 |
| `ece.bins` | 15 | — |
| `slices.max_order` / `min_support` / `fdr` | 2 / 30 / 0.05 | §6.3 controls |
| `bootstrap.n` | 1000 | §6.7 |
| `crop.margin` | 0.25 | §7 O1 |
| `knobs.max_pixels` | low / native / high (set after reading the checkpoint config) | §7 |
| `knobs.token_budget` | 100 / 50 / 25 / 10 % (random vs attention-based) | §7 |
| `knobs.fps` | 1 / 2 / 5 / 8 | §7 |
| `knobs.window` | 4 / 8 / 16 / 32 | §7 |
| `knobs.quant` | bf16 / 8bit / 4bit | §7 |
| `knobs.quality` | blur, low-light, noise, haze, jpeg, bitrate, resample × severity 1–5 | §7 |
| `sink.ratio` | 10 (token L∞ activation > 10× median ⇒ sink) | §6.4 |
| `loc.shortcut_enrichment` / `loc.equal_delta` | 1.5 / 0.15 | §6.4 3-way |
| `probe.decodable_acc` / `probe.selectivity` | 0.70 / 0.15 | §7 T4 |
| `horizon.frac` | 0.95 | §6.5 |
| `stream.tolerance_s` / `trigger` | 2.0 / k-of-n = 2 of 3 | §6.6 |
| `audit.k_per_cluster` | 20 | §6.7 |
| `exit.conf_margin` | 1.0 (own proposal) | C5 |
| `hazard.min_agreement` | 0.6 (own proposal) | FR-H-6 |
| `tests.null_runs` / `null_fpr_max` / `planted_tpr_min` | 100 / 0.1 / 0.9 (own proposals) | FR-Q-3 |
| `tests.finite_diff_rtol` / `smd_max` / `psnr_null_min_db` | 0.05 / 0.1 / 40 (own proposals) | FR-Q-4/5/7 |
| `answer.min_mass` | 0.5 (own proposal) | C16 |
| `null.jpeg_q` / `null.shift_px` | 95 / 1 (own proposal) | C4 |
| `attn.band` | upper_half | 2510.17771 (C9) |
| `patch.corruptions` | [pair, boxnoise] | 2309.16042 (C13) |
| `validate.min_precision` / `per_cause_n` / `label_noise` | 0.8 / 30 / 0.1 (own proposal) | C15 |
| `stream.anticipation_s` | 0.0 (optional early-warning window; not the latent-hazard task) | C14 |

---

## 8. Traceability: hypotheses → experiments → requirements → outputs

| Hyp. | Experiment(s) | Requirements | Output |
|---|---|---|---|
| H1 merger | O1 crop ladder + **O1.bgmask** + token-budget/resolution knobs (`hires`) → T4 probes ViT vs merger (differential) | FR-T2-1/3, FR-T4-2/4 | `t2_recovery.csv`, `t4_probes.csv` |
| H2 depth fading | knockout, information horizon, logit lens, token similarity | FR-T4-5/6/7/9 | `t4_*.csv` |
| H3 readout | T1 Q_P (both polarities)/Q_R/Q_D; **T2 crop and `hires` must NOT recover**; T3 3-way localization; T4 robust causal tests | FR-T1, FR-T3-4, FR-T4-8 | `t1_triage.csv`, `t3_loc.csv` |
| H4 deployment | knob sweeps; calibration under knobs | FR-T2-3/4, FR-B-14 | `t2_knobs.csv` |
| H5 scene shortcut | blank/swap; background mass; SpRAy clusters | FR-B-8, FR-T3-3/7 | `t3_loc.csv` |
| H6 absence | polarity-flip; absence-vs-presence slices at matched size | FR-B-6, FR-S-1 | `b_polarity.csv`, `slices_*.csv` |
| H7 prompt/prior | blank frame **+ evidence-box occlusion (after the FR-T1-4 gate)**, framing arms, paraphrase flips | FR-B-7/8/9 | `b_prompt.csv` |
| H8 binding | O5 per-person crops/overlay; **O1.bgmask**; worker-count slices | FR-T2-1 (O5), FR-S-1 | `t2_recovery.csv` |
| H9 input quality | O6 quality sweep; bitrate/resample arms | FR-T2-1 (O6), FR-B-16 | `t2_quality.csv` |
| H10 reasoning loss | (P3) CoT-length curve, image swap | FR-P-4 | — |
| H11 trigger policy | streaming track, window vs stream gap | FR-ST-* | `stream_*.csv` |

---

## 9. Assumptions the code must verify at startup ([VERIFY] items)

`python -m ea selfcheck` MUST assert or print, and fail with an actionable message if wrong:

1. Checkpoint identity: Instruct vs Thinking (decides whether H10 applies); LLM layer count, hidden size, heads/KV-heads, ViT depth, `deepstack_visual_indexes`, default pixel limits, patch size and merge size (patch 16 and 2×2 merge ⇒ one token per 32×32 px is expected; if not, stop).
2. Module paths used by `model.py` exist (`model.model.visual`, `.blocks`, `.merger`, `.deepstack_merger_list`, `model.model.language_model.layers`, final norm, `lm_head`). A mismatch prints the actual module tree.
3. The visual-token order equals positions of `image_token_id`/`video_token_id`, row-major per image / time-major per video.
4. Pre-merger count is 4× merged count and consecutive groups of four pre-merger vectors belong to one merged token (the merger flattens consecutive groups of four).
5. DeepStack injection occurs **outside** the decoder-layer modules (so layer-output hooks for the first three layers do not include it). Patching windows MUST therefore start at layer ≥ 3, or report the limitation.
6. With `attn_implementation="eager"`, the text self-attention modules (a) receive a materialized 4-D `attention_mask` kwarg (needed by knockout) and (b) return the attention weights as their second output (needed by `attn_last`). If either is false in the pinned `transformers`, stop and report; do not build a workaround silently.
7. Video processor accepts frame lists plus fps/timestamps and preserves timestamp text across knobs.
8. Prompt answer tokens ("Yes", "No") are single tokens in the tokenizer at the answer position.
9. Benchmark formats/licenses/splits (plan §4 [VERIFY]) — recorded in each adapter docstring.
10. Answer mass: `mass ≥ answer.min_mass` on a calibration sample of every prompt group, and the prompt ends with the assistant generation prefix (C16).
11. The `nli` (noisy linear imputation) port reproduces the removal behaviour described by ROAD on a toy image **[VERIFY against the authors' reference implementation, 2202.00449]**; no new dependency is added.
12. The upper-half attention band and the question-token routing of visual information hold for Qwen3-VL (the cited evidence is for LLaVA-family and other models: 2510.17771, 2411.18620) — check with `attn_last`/`knockout` on 2 real images.

---

## 10. Deliverables (maps to plan §9)

| Plan deliverable | Produced by |
|---|---|
| 1. Versioned eval harness | `data.py`, `model.py`, `prompts.py`, `run.py`, `manifest.json` |
| 2. Metric suite with CIs and slice search | `metrics.py`, `slices.py` |
| 3. Annotated error set with taxonomy and labels | `errors.parquet`, `clusters.json`, `audit_sample.csv` |
| 4. Triage/oracle/knob results + latency profile | `outputs/t1`, `outputs/t2`, `stream_latency.csv` |
| 5. Attribution + explainer-validity report | `outputs/t3`, `t3_gate.json` |
| 6. Probe, information-flow, patching results | `outputs/t4` |
| 7. Final root-cause map | `root_causes.md`, `root_causes.csv` |
| (new) Known-cause calibration of the diagnostics | `validate` → `outputs/validate/confusion.csv` |

---

## 11. Definition of done

- All P0 requirements implemented and passing tests; P1/P2 implemented or explicitly listed as *deferred* in the report with the reason.
- Source size is within the planning budget **or** every overrun is logged in `DEVIATIONS.md` (GC-1); `grep` check for GC-2 passes; no forbidden dependency.
- All FR-Q verification cards pass (known-answer, cross-check, null/planted, anchor, mutation, leak/causality, signature audit).
- The fake-model integration test passes (the planted-cause cluster labels).
- `python -m ea validate` has been run on the real checkpoint; `root_causes.md` labels a cause `decisive` only where validated precision ≥ `validate.min_precision`, and prints `candidate` otherwise.
- `python -m ea selfcheck` passes on the target machine with the real checkpoint.
- `root_causes.md` is generated end-to-end from `runs.jsonl` without manual edits.
