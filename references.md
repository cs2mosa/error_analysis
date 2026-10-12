# references.md — Papers cited in the error-analysis plan

**How these links were made.** Every arXiv link below is built from the ID written in your plan (`https://arxiv.org/abs/<ID>`). I did **not** open most of them. Entries marked *opened (abstract)* in the Note column were opened in the v0.5 audit session via their arXiv abstract page: the title matched the plan and I read the abstract only (not the full text). The two entries marked *opened this session* in the "without an arXiv ID" section were checked earlier. If a link does not resolve or shows a different title, the ID in the plan probably has a typo: search the title instead. Several IDs in the plan carry months later than mid-2026; treat each one as unconfirmed until you open it.

The plan itself notes that all literature was read at abstract level only, so nothing here is a verified citation.


## 13.1 Internal analysis of VLMs — methods we adopt (T3/T4)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Probing Visual Concepts in Lightweight VLMs for Automated Driving | 2603.06054 | https://arxiv.org/abs/2603.06054 |  |
| Cross-modal Information Flow in MLLMs | 2411.18620 | https://arxiv.org/abs/2411.18620 | opened (abstract); title matches |
| Towards Interpreting Visual Information Processing in VLMs | 2410.07149 | https://arxiv.org/abs/2410.07149 |  |
| Map the Flow: Hidden Pathways of Information in VideoLLMs | 2510.13251 | https://arxiv.org/abs/2510.13251 |  |
| An Empirical Study on How Video-LLMs Answer Video Questions | 2508.15360 | https://arxiv.org/abs/2508.15360 |  |
| Linear Mechanisms for Spatiotemporal Reasoning in VLMs | 2601.12626 | https://arxiv.org/abs/2601.12626 |  |
| Causal Tracing of Object Representations in LVLMs (FCCT) | 2511.05923 | https://arxiv.org/abs/2511.05923 |  |
| SparseMM | 2506.05344 | https://arxiv.org/abs/2506.05344 |  |

## 13.2 Where visual information lives across depth (H2)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| When Token Pruning is Worse than Random (Information Horizon) | 2512.07580 | https://arxiv.org/abs/2512.07580 | opened (abstract); title matches |
| The Hidden Evolution of Disguised Visual Context inside the VLM | 2606.20077 | https://arxiv.org/abs/2606.20077 |  |
| Layer-wise image-token redundancy study (matrix entropy, intrinsic dimension, curvature) | 2604.09425 | https://arxiv.org/abs/2604.09425 | VERIFY flagged in plan |
| Seeing without Looking: Do VL Benchmarks Really Test Vision? | 2605.22903 | https://arxiv.org/abs/2605.22903 | opened (abstract); title matches |

## 13.3 Perception vs readout (H3)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| MLLMs Know Where to Look | 2502.17422 | https://arxiv.org/abs/2502.17422 | opened (abstract); title matches |
| Seeing is not Enough: VLMs Perceive Evidence but Fail to Act | 2609.22588 | https://arxiv.org/abs/2609.22588 | opened (abstract); title matches |
| SQuINTing at VQA Models | 2001.06927 | https://arxiv.org/abs/2001.06927 |  |
| Cognitive Paradigms for Evaluating VLMs on Visual Reasoning | 2501.13620 | https://arxiv.org/abs/2501.13620 |  |

## 13.4 Absence, priors, sycophancy, reasoning (H6, H7, H10)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Vision-Language Models Do Not Understand Negation (NegBench) | 2501.09425 | https://arxiv.org/abs/2501.09425 | opened (abstract); title matches; plan note: VERIFY flagged in plan |
| MissingBench-Verified | 2607.18673 | https://arxiv.org/abs/2607.18673 |  |
| To See or To Please: Visual Sycophancy and Split Beliefs in VLMs | 2603.18373 | https://arxiv.org/abs/2603.18373 |  |
| V-FAT: Visual Fidelity Against Text-bias | 2601.04897 | https://arxiv.org/abs/2601.04897 |  |
| VLind-Bench | 2406.08702 | https://arxiv.org/abs/2406.08702 | opened (abstract); title matches |
| PriVE-Bench | 2607.16311 | https://arxiv.org/abs/2607.16311 |  |
| More Thought, Less Accuracy? On the Dual Nature of Reasoning in VLMs (VAPO) | 2509.25848 | https://arxiv.org/abs/2509.25848 |  |
| Are VLMs Seeing or Just Saying? (VisualSwap / VS-Bench) | 2605.15864 | https://arxiv.org/abs/2605.15864 | VERIFY flagged in plan |
| When to Think and When to Look: Uncertainty-Guided Lookback | 2511.15613 | https://arxiv.org/abs/2511.15613 |  |

## 13.5 Multi-object binding and encoder blind spots (H8, H1)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Understanding the Limits of VLMs Through the Lens of the Binding Problem | 2411.00238 | https://arxiv.org/abs/2411.00238 |  |
| The Binding Problem in VLMs (position IDs) | 2506.15871 | https://arxiv.org/abs/2506.15871 |  |
| Visual Structures Helps Visual Reasoning | 2506.22146 | https://arxiv.org/abs/2506.22146 |  |
| Eyes Wide Shut? (CLIP-blind pairs, MMVP) | 2401.06209 | https://arxiv.org/abs/2401.06209 |  |

## 13.6 Deployment knobs: compression, quantization, sampling, input quality (H4, H9)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Feather the Throttle | 2412.13180 | https://arxiv.org/abs/2412.13180 |  |
| CoverPruner ("Who Speaks for the Pruned?") | 2609.03158 | https://arxiv.org/abs/2609.03158 |  |
| TORINO | 2607.04593 | https://arxiv.org/abs/2607.04593 |  |
| CaRe (Calibrate Before Reason) | 2607.27700 | https://arxiv.org/abs/2607.27700 |  |
| VisionSelector | 2510.16598 | https://arxiv.org/abs/2510.16598 |  |
| AgilePruner | 2603.01236 | https://arxiv.org/abs/2603.01236 |  |
| Does Visual Token Pruning Improve Calibration? | 2604.12035 | https://arxiv.org/abs/2604.12035 |  |
| VLMQ | 2508.03351 | https://arxiv.org/abs/2508.03351 | VERIFY flagged in plan |
| Fine-Grained PTQ for LVLMs (QIG) | 2603.17809 | https://arxiv.org/abs/2603.17809 |  |
| Moment-Video | 2606.02522 | https://arxiv.org/abs/2606.02522 |  |
| Shot-Aware Frame Sampling (InfoShot) | 2603.17374 | https://arxiv.org/abs/2603.17374 |  |
| DIVE with GRT | 2509.14199 | https://arxiv.org/abs/2509.14199 |  |
| VLM-RobustBench | 2603.06148 | https://arxiv.org/abs/2603.06148 |  |
| DIQ-H | 2512.03992 | https://arxiv.org/abs/2512.03992 |  |
| Robustness of VLMs to Common Corruptions | 2504.13690 | https://arxiv.org/abs/2504.13690 |  |

## 13.7 Streaming and real-time behavior (H11, H4)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| A Simple Baseline for Streaming Video Understanding (SimpleStream) | 2604.02317 | https://arxiv.org/abs/2604.02317 |  |
| HERMES: KV Cache as Hierarchical Memory | 2601.14724 | https://arxiv.org/abs/2601.14724 |  |
| Going Down Memory Lane | 2602.18434 | https://arxiv.org/abs/2602.18434 |  |
| OVO-Bench | 2501.05510 | https://arxiv.org/abs/2501.05510 |  |
| Don't Pause! Every prediction matters (SPOT-Bench) | 2604.24317 | https://arxiv.org/abs/2604.24317 | opened (abstract); title matches |
| Omni-DuplexEval | 2605.17360 | https://arxiv.org/abs/2605.17360 |  |
| TRACE | 2609.30670 | https://arxiv.org/abs/2609.30670 | opened (abstract); title matches |
| ProactiveLLM | 2606.00523 | https://arxiv.org/abs/2606.00523 |  |
| StreamPro | 2605.16381 | https://arxiv.org/abs/2605.16381 |  |

## 13.8 Temporal shortcuts (video errors)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| What Does a Temporal Benchmark Score Measure? | 2607.12304 | https://arxiv.org/abs/2607.12304 |  |
| TempCompass | 2403.00476 | https://arxiv.org/abs/2403.00476 |  |
| VidHal | 2411.16771 | https://arxiv.org/abs/2411.16771 |  |

## 13.9 Attention-sink confounds and encoder retention (T3/T4)

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Intriguing Properties of Large Language and Vision Models | 2410.04751 | https://arxiv.org/abs/2410.04751 |  |
| Visual Attention Sink ("See What You Are Told") | 2503.03321 | https://arxiv.org/abs/2503.03321 |  |
| Vocabulary Hijacking / Inert Tokens | 2605.10622 | https://arxiv.org/abs/2605.10622 |  |

## 13.10 XAI and error-exposing metrics

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Quantus toolkit (JMLR MLOSS) | 2202.06861 | https://arxiv.org/abs/2202.06861 |  |
| Metrics for saliency-map evaluation (DAUC/IAUC critique) | 2201.13291 | https://arxiv.org/abs/2201.13291 | opened (abstract); title matches |
| Beyond Output Faithfulness | 2509.04588 | https://arxiv.org/abs/2509.04588 | opened (abstract); title matches |
| GLIMPSE | 2506.18985 | https://arxiv.org/abs/2506.18985 |  |
| Saliency-R1 | 2604.04500 | https://arxiv.org/abs/2604.04500 |  |
| Explanation evaluation for VLN (COMP / SUFF / decision-flip) | 2310.06654 | https://arxiv.org/abs/2310.06654 |  |
| Benchmarking Faithfulness of NL Explanations in VL tasks | 2304.08174 | https://arxiv.org/abs/2304.08174 |  |
| Surrogate faithfulness (SURF) | 2504.10833 | https://arxiv.org/abs/2504.10833 |  |
| Ground Truth Evaluation of Neural Network Explanations with CLEVR-XAI | 2003.07258 | https://arxiv.org/abs/2003.07258 |  |
| Finding and Removing Clever Hans | 1912.11425 | https://arxiv.org/abs/1912.11425 |  |
| The Clever Hans Effect in Anomaly Detection | 2006.10609 | https://arxiv.org/abs/2006.10609 |  |
| Reproducibility study on spurious correlations / Clever Hans | 2604.04518 | https://arxiv.org/abs/2604.04518 | opened (abstract); title matches |
| Domino | 2203.14960 | https://arxiv.org/abs/2203.14960 | opened (abstract); title matches |
| VISLIX | 2505.03132 | https://arxiv.org/abs/2505.03132 |  |
| LADDER | 2408.07832 | https://arxiv.org/abs/2408.07832 |  |
| LaVMD (Language-assisted Vision Model Debugger) | 2312.05588 | https://arxiv.org/abs/2312.05588 |  |
| SliceLens / FeSD | 2512.24592 | https://arxiv.org/abs/2512.24592 |  |
| Detecting Systematic Weaknesses along Predefined Dimensions | 2502.12360 | https://arxiv.org/abs/2502.12360 |  |
| VLM2-Bench | 2502.12084 | https://arxiv.org/abs/2502.12084 |  |
| Vinoground | 2410.02763 | https://arxiv.org/abs/2410.02763 |  |
| Test-Time Matching (Winoground group score analysis) | 2510.07632 | https://arxiv.org/abs/2510.07632 |  |
| EDCT-Bench | 2609.17953 | https://arxiv.org/abs/2609.17953 |  |
| Paraphrase consistency with image-reliance check | 2603.20985 | https://arxiv.org/abs/2603.20985 |  |
| Language-Guided Invariance Probing (LGIP) | 2511.13494 | https://arxiv.org/abs/2511.13494 |  |
| What Matters for Calibrating VLMs | 2402.07417 | https://arxiv.org/abs/2402.07417 |  |
| Instinct vs Reflection (token vs verbalized confidence) | 2604.17274 | https://arxiv.org/abs/2604.17274 |  |
| Selective "Selective Prediction" (ReCoVERR) | 2402.15610 | https://arxiv.org/abs/2402.15610 | opened (abstract); title matches |
| Conformal abstention policies for LLMs/VLMs | 2502.06884 | https://arxiv.org/abs/2502.06884 |  |
| Calibration/selective prediction in multimodal clinical classification | 2603.02719 | https://arxiv.org/abs/2603.02719 |  |
| Towards Best Practices of Activation Patching in Language Models | 2309.16042 | https://arxiv.org/abs/2309.16042 | opened (abstract); title matches |

## 13.11 Domain: construction / industrial safety with VLMs

| Paper (as named in plan) | arXiv ID | Link | Note |
|---|---|---|---|
| Are Large Pre-trained VLMs Effective Construction Safety Inspectors? (ConstructionSite 10k) | 2508.11011 | https://arxiv.org/abs/2508.11011 | opened (abstract); title matches |
| Automated Hazard Detection in Construction Sites Using LLMs and VLMs | 2511.15720 | https://arxiv.org/abs/2511.15720 |  |
| SafeBuild-Bench | 2608.00068 | https://arxiv.org/abs/2608.00068 | opened (abstract); title matches; plan note: VERIFY flagged in plan |
| iSafetyBench | 2508.00399 | https://arxiv.org/abs/2508.00399 |  |
| FieldWorkArena | 2505.19662 | https://arxiv.org/abs/2505.19662 |  |
| PaSBench-Video | 2606.02443 | https://arxiv.org/abs/2606.02443 | opened (abstract); title matches |
| Qwen3-VL Technical Report | 2511.21631 | https://arxiv.org/abs/2511.21631 |  |


## Entries the plan lists without an arXiv ID

| Paper | Where in plan | Status | Link / how to find it |
|---|---|---|---|
| Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs (ICLR 2026) | 13.3, 13.0 | **ID resolved; opened this session** | https://arxiv.org/abs/2510.17771 |
| StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding | 13.7 | **ID resolved; opened this session** | https://arxiv.org/abs/2411.03628 |
| TVBench | 13.8 | ID recalled from memory, **not checked** | https://arxiv.org/abs/2410.07752 (search "TVBench: Redesigning Video-Language Evaluation" if it does not match) |
| SliceFinder (Chung et al. 2019) | 13.10 | ID recalled from memory, **not checked** | https://arxiv.org/abs/1807.06068 (search "Slice Finder: Automated Data Slicing for Model Validation") |
| SliceLine (Sagadeeva & Boehm, SIGMOD 2021) | 13.10 | No arXiv ID known | Search "SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging" |
| Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations? (Sun et al., MICCAI 2023) | 13.10 | ID not located | Search the title |
| Selectively Answering Visual Questions | 13.10 | ID not captured in plan | Search the title |
| StreamReady / ProReady-QA (CVPR 2026) | 13.7 | ID not captured in plan | Search the title |
| ImpQuant (ICML 2026 poster) | 13.6 | ID not captured in plan | Search the title |
| ProViP (head-aware token pruning) | 13.1 | Plan says primary source not located | Search the title; treat as unconfirmed |
| ROAD (debiased deletion/insertion) | 6.4(b) | **ID resolved; abstract read in the v0.5 audit** (ICML 2022; title matches) | https://arxiv.org/abs/2202.00449 |
| MPRT / randomization sanity checks | 6.4(b) | Underlying paper not listed in the plan | Search "Sanity Checks for Saliency Maps" (Adebayo et al., 2018) |

## Model and tooling pages (not in the plan, useful for implementation)

| Resource | Link | Status |
|---|---|---|
| Qwen3-VL Technical Report | https://arxiv.org/abs/2511.21631 | ID from plan, not opened |
| Qwen3-VL-4B-Instruct model card | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Not opened; confirm the exact repo name and whether you use Instruct or Thinking |
| Hugging Face Transformers Qwen3-VL docs | https://huggingface.co/docs/transformers/model_doc/qwen3_vl | Not opened; read this and the model's `modeling_qwen3_vl.py` for the [VERIFY] items in `requirements.md` §9 |

A search this session turned up a public Qwen3-VL config listing `patch_size: 16` and a `deepstack_visual_indexes` field, and module names `model.visual`, `model.visual.merger`, and `model.language_model.layers`, consistent with `design.md`. That was a third-party config file, so `selfcheck` should still confirm these against the real checkpoint.


## Evidence map for v0.5 (what each source does and does not support)

Reading level: **abstract only**, via the arXiv abstract page (ROAD: abstract on arXiv/ICML pages). Claims below are limited to what those abstracts state. None of these papers evaluates this pipeline or Qwen3-VL-4B on safety data, so each is *motivation for a design choice*, not validation of it. "Own design" marks choices with no source in this file.

| Change | Source(s) | What the abstract supports | What it does **not** show |
|---|---|---|---|
| C1 signature table / explicit ambiguity | 2406.08702, 2502.17422, 2609.22588 | Several distinct mechanisms (language prior, small-detail perception, process failure) produce wrong answers and need separate tests | That any specific signature is correct for Qwen3-VL — hence C15 |
| C2 O2 as exclusion test | 2609.22588 | Defines perceptual failure (evidence not recognized) vs process failure (recognized evidence fails to constrain the decision); text-with-facts cannot separate these | — (own reasoning that oracle facts make any LLM correct) |
| C3 `O1.bgmask`, `O3.occlude` | 2605.22903; 2502.17422 | Localized occlusion is a distinct analysis from global degradation; sensitivity to the size of the visual subject is causal (intervention study) | That a gray-out arm cleanly separates clutter from resolution (own design) |
| C4 `N0.null` | own design | — | — |
| C5 CI-based rule, confident errors, Δs | 2605.22903 (Δs); own design (CI, confident-error guard) | Internal support for the answer can weaken while the prediction is unchanged | The regression-to-the-mean guard has no source here; `conf_margin = 1.0` is a guess |
| C6 perception/knowledge gate | 2406.08702 | A language-prior benchmark ensures basic tests (commonsense knowledge, visual perception, commonsense bias) pass before measuring priors | That this gate suits yes/no safety prompts |
| C7 `qp_abs`, `untriaged` | 2501.09425; 2609.22588 | Negation is a documented VLM weakness (NegBench, mainly retrieval/MCQ with CLIP-style models); perception vs process split | That Qwen3-VL's yes/no polarity bias is large here — measured by T1 |
| C8 equal enrichment ≠ H3 | 2502.17422; 2510.17771 | MLLMs "know where to look" even when wrong; VLMs often attend to the evidence yet answer wrongly | Which of the two explains a given safety cluster — hence the crop/hires conjunction |
| C9 upper-half band; `q` site | 2510.17771; 2411.18620 | Shallow layers attend mostly to text, deeper layers to evidence; visual information is moved into question tokens before the last position (LLaVA-family) | That the same holds for Qwen3-VL — `selfcheck` item 12 |
| C10 debiased fill, rank-only caveat, internal agreement | 2202.00449; 2201.13291; 2509.04588 | Removal shape leaks information and rankings depend on the fill; DAUC/IAUC use only ranking and show out-of-distribution images; insertion/deletion can be maximised via other pathways (CNNs) | That `nli` as ported here behaves like the authors' version — `selfcheck` item 11 |
| C11 SpRAy exploratory | 2604.04518 | SpRAy struggles with complex features and severe imbalance; minority scarcity makes model selection unreliable | Behaviour on VLM attribution maps |
| C12 differential probes/horizon | 2512.07580; 2411.18620 | An information horizon exists, is task/model-dependent, and token information vanishes beyond it; information migrates to question/last positions | That the horizon for safety clusters differs from generic VQA — that is the measurement |
| C13 patching robustness | 2309.16042 | Choice of metric and corruption method can change patching conclusions; best-practice recommendations | The specific "≥ 2 corruptions, 2 metrics" rule is our operationalisation (abstract-level read) |
| C14 anticipation window (now optional) | 2609.30670; 2604.24317; 2606.02443 | TRACE separates delay, false alarms (response while no target window is valid) and missed windows; SPOT-Bench reports offline models spamming in dead time; PaSBench-Video annotates frame-level risk onset and accident boundaries | Neither defines the project's *prediction* (latent hazard) — that is the owner's definition (C19) |
| C15 `validate` | own design; 2406.08702 (gating idea) | — | Injections are synthetic, so the confusion matrix is an optimistic bound |
| C16 answer mass | own design | — | — |
| C17 `ALL_ERRORS`, coherence audit | 2203.14960 | Slice discovery methods often return incoherent slices; slices should be underperforming *and* coherent | Anything about metadata slicing specifically |
| C18 domain / router population | 2508.11011 | Large pre-trained VLMs show zero-/few-shot ability on construction safety but need additional training for actual sites | That a warehouse domain behaves like construction |
| C19 latent-hazard track | owner's definition; 2606.02443; 2608.00068 | PaSBench-Video: includes no-risk videos for false-positive measurement; recall tightly coupled with false-positive rate (Pearson 0.64); models rely on scene-level activity cues rather than reasoning about emerging harm. SafeBuild-Bench: hazard identification (multiple choice) and free-form hazard description tasks; best overall score near 60 | That any benchmark here contains *latent* hazards of the open-lid type, or has secured twins; the cued/uncued arm and twin requirement are our own design |
| C20 accounting identity; scope of evidence | own design | — | That the listed unobservable cause classes are exhaustive |
| C21 soft budget; correctness protocol | own design | — | That the listed tests are sufficient; they verify the code against its definitions, not the diagnostics against real causes (that is `validate`) |

**Opened but not used for a change:** 2402.15610 (selective prediction / ReCoVERR) — relevant to the AURC/coverage metrics already in FR-B-13; no change needed.

**Still unverified (abstract not opened):** every other arXiv ID in sections 13.1–13.11 above, including all IDs whose month is later than mid-2026 that are not marked *opened (abstract)*.
