# design.md — Implementation Guide for the Qwen3-VL-4B Error-Analysis Codebase

**Implements:** `requirements.md` (IDs cited as FR-xxx) for the plan `qwen3vl_error_analysis_complete.md` (§ = plan section).
**Goal:** a **correct, verifiable** codebase that is as small as correctness allows. Correctness of every metric, test and analysis tool outranks line count; the line figures in §2 (≈ 2,000 in total) are **planning budgets, not limits** (§0.1). Simplicity is pursued by choosing simple *designs*, never by dropping a check.

> **Revision v0.5 — literature-backed audit of the cause-assignment logic.** Changes C1–C18 were made after reading the methodology papers listed in `references.md` (abstract level; see its section *Evidence map for v0.5* for exactly what each source does and does not support). Net effect: cause assignment is a **signature table with explicit ambiguity** instead of first-match-wins; every diagnostic has an independent control; a **`validate`** step measures the false-attribution rate on the real model; no cause may be called *decisive* without a measured precision.
>
> | ID | Change | Basis |
> |---|---|---|
> | C1 | `decide` → signature table over three-valued diagnostics (unique / ambiguous / inconclusive / unexplained) | own design; motivated by 2406.08702, 2502.17422, 2609.22588 |
> | C2 | O2 (oracle facts) demoted to an **exclusion** test (rule reasoning OK) — it never names "perception" | own design; 2609.22588 (perception vs process failure) |
> | C3 | New arms `O1.bgmask` (clutter vs resolution) and `O3.occlude` (evidence-region necessity) | 2605.22903 (localized occlusion), 2502.17422 (size is causal) |
> | C4 | `N0.null` perturbation replaces "base re-run" as the noise-floor control | own design (a deterministic re-run recovers exactly 0) |
> | C5 | Exit rule is CI-based/three-valued, on *confident* errors; Δs reported next to recovery | own design; 2605.22903 (internal support can weaken without a flip) |
> | C6 | "Prior-driven" only after a perception/knowledge gate | 2406.08702 |
> | C7 | Polarity-balanced Q_P (`qp` + `qp_abs`); `untriaged` for samples without `item` | 2501.09425; 2609.22588 |
> | C8 | Equal enrichment no longer ⇒ H3 by itself; H3 needs crop **and** hires to fail | 2502.17422, 2510.17771 |
> | C9 | `attn_last` band = upper half of layers; `q`-position site added | 2510.17771, 2411.18620 |
> | C10 | Debiased removal fill (ROAD-style) for deletion/insertion; rank-only caveat; internal-agreement check | 2202.00449, 2201.13291, 2509.04588 |
> | C11 | SpRAy output flagged exploratory | 2604.04518 |
> | C12 | Probes/horizon are **differential** vs matched corrects | 2512.07580, 2411.18620 |
> | C13 | Patching "causal" requires ≥ 2 corruption methods and 2 metrics | 2309.16042 |
> | C14 | `stream.anticipation_s` kept only as an *optional* temporal early-warning window; "prediction" is **not** onset forecasting (see C19) | 2609.30670, 2604.24317 (semantics of early alerts) |
> | C15 | `validate`: known-cause injections on the real model + decisive-gating by precision | own design; gating idea from 2406.08702 |
> | C16 | `answer_mass` validity column | own design |
> | C17 | `ALL_ERRORS` cluster; most-specific-cluster accounting; slice coherence audit | 2203.14960; own design |
> | C18 | Domain / router-population caveat (`meta.router_pass`) | 2508.11011; own design |
> | C19 | **Latent-hazard (prediction) track** §5.14: conditions → physical consequence under a trigger; uncued vs cued arm; secured twins | project owner's definition; 2606.02443, 2608.00068 (formats / false-alarm coupling); own design |
> | C20 | **Accounting identity** (every error ends in exactly one named state) and a **scope-of-evidence** section listing causes this black-box analysis cannot reveal | own design |
> | C21 | Line budget is **soft** (logged deviations); **implementation-correctness protocol** §7.6 (known-answer, cross-check, null/planted, anchor and mutation tests for every tool) | own design |

> **Code in this file is reference sketches, not final code.** Names and shapes are normative; exact HuggingFace API details are marked **[VERIFY]** and are covered by `selfcheck` (requirements §9). If a sketch disagrees with the pinned `transformers` version, fix the sketch, keep the interface.

---

## 0. The seven decisions that keep the code small

| # | Decision | What it removes |
|---|---|---|
| 1 | **One scalar per question: the margin `s`** (§6.1). Every metric, attribution, probe target, and patching score is computed from `s`. | Separate code paths for "answers", "probabilities", "confidences". |
| 2 | **One append-only long table** `runs.jsonl`. Every tier only *adds rows*; every analysis is a pandas function over it. | Per-experiment result classes, loaders, databases. |
| 3 | **Interventions are pure functions in a dict** (`INTERV`). A new arm = one function + one registry line. | Class hierarchies, config-driven pipelines. |
| 4 | **Exactly one file touches model internals** (`model.py`). It exposes ~10 functions. | Version-sensitive code spread over the repo. |
| 5 | **Everything is normalized to the visual-token grid** `(t, h/2, w/2)`: boxes, attention, saliency, probes. | Per-method coordinate bookkeeping. |
| 6 | **Streaming = store the margin time series once; evaluate trigger policies offline.** | Re-running the model per trigger rule. |
| 7 | **Tier = a CLI subcommand that builds a job list, runs it, then calls analysis functions.** No framework. | Orchestrators, DAGs, config systems. |

---

### 0.1 Priorities and the flexible budget (C21)

Priority order when goals conflict: **(1) every measurement and diagnostic is implemented correctly and verified, (2) conclusions are honest (controls, CIs, validity gates), (3) the code is simple and readable, (4) the line count.** Therefore:

- The numbers in §2 are estimates. A module (or the total) **may exceed them whenever correctness needs it**. Do not shorten code by removing an input check, a control arm, a validity gate, a null/planted test, or by merging two distinct definitions into one function.
- Every overrun is logged in `DEVIATIONS.md` (module, lines over budget, reason, requirement IDs). Up to **+50 % per module** needs no approval. Beyond that, or a new module, or a new dependency, or a non-goal (requirements §1.3), **stop and report to the human**; continuing after a report is the normal outcome, not a failure.
- The seven decisions below stay the *default design* because they make code easy to verify (one scalar, one table, pure functions); change one only if it makes a tool incorrect, and log it.
- If a plan definition cannot be implemented correctly as written, that is a reported deviation (pitfall 15), not something to hide for brevity.

---

## 1. Data flow

```
data/<bench>/samples.jsonl ──► data.py ─────────────┐
config.yaml ─────────────────► prompts.py ──────────┤
                                                    ▼
                         run.py  (jobs = sample × prompt × arm × params × model_cfg)
                           │ uses interventions.py (INTERV) and model.py (VLM)
                           ▼
                      runs.jsonl  (append-only, resumable, content-hash keys)
                           │
   ┌──────────┬────────────┼─────────────┬──────────────┐
   ▼          ▼            ▼             ▼              ▼
metrics.py  slices.py   triage.py     stream.py   attrib.py / mech.py (read model again)
   └──────────┴────────────┼─────────────┴──────────────┘
                           ▼
                    escalate.py  (clusters → recovery/necessity/consistency → cause)
                           ▼
                    report.py → root_causes.md / .csv
```

---

## 2. Repository layout and line budget (planning estimates — see §0.1)

```
ea/                      # python package, run as: python -m ea <cmd>
  config.yaml            # all [PROPOSED] values & sweep grids          (~60 lines yaml)
  data.py        ~150    # Sample, load, frames(), box_mask(), bin_meta(), make_toy()
  prompts.py      ~95    # PROMPTS dict, render(), PROMPT_VERSION
  model.py       ~190    # VLM, Tap, knockout, attn_last, grad_x_input, sinks
  interventions.py ~200  # INTERV registry: O1–O7, bgmask/occlude/null arms, INJ.* injections
  run.py         ~120    # job plans per tier, resumable runner
  metrics.py     ~215    # all behavioral metrics (pure functions)
  slices.py       ~60    # slice search
  triage.py       ~40    # Q_P/Q_R/Q_D table
  escalate.py    ~180    # clusters, three-valued exit rule, signature table, validate()
  attrib.py      ~160    # T3: maps, localization, deletion/insertion, gate
  mech.py        ~240    # T4: sites cache, probes, knockout, lens, horizon, patching
  stream.py      ~130    # online steps, trigger, decomposition, metrics
  report.py       ~80    # root_causes.md/.csv
  cli.py          ~40    # argparse subcommands
tests/                   # ≤ 700 lines, incl. FakeVLM and toy generator usage
requirements.txt
```
**Rule:** if a module exceeds its budget by >30%, stop and simplify instead of adding code.

---

## 3. Schemas and config

### 3.1 `Sample` (data.py)
See `requirements.md` §4.1. Implement as a plain `@dataclass` with `from_json`. Flatten `meta` into columns for slicing via `to_df(samples)`.

### 3.2 `runs.jsonl` row (FR-R-3)
```json
{"key":"…md5…","sample_id":"cs_0012","prompt_id":"d0","group":"D","arm":"base",
 "knob":null,"value":null,"model_cfg":{"quant":null,"max_pixels":null},
 "s":-1.37,"pred":0,"gold":1,"correct":0,"n_vis_tokens":412,"latency_ms":88.1,
 "t":null,"extra":{"mass":0.97}}
```
Key = `md5(json.dumps([sample_id, PROMPT_VERSION, prompt_id, arm, params, model_cfg], sort_keys=True))`.

### 3.3 `config.yaml` (excerpt)
```yaml
model: {name: Qwen/Qwen3-VL-4B-Instruct}
seed: 0
exit: {recovery: 0.5, control: 0.1, min_n: 10, conf_margin: 1.0}   # decisions use Wilson CIs (three-valued); errors with |s_base| < conf_margin are borderline and excluded from recovery tests [PROPOSED]
necessity: {ratio: 0.25}
consistency: {min_flip: 0.2, ratio: 2.0}
cost: {miss: 10, false_alarm: 1}
slices: {max_order: 2, min_support: 30, fdr: 0.05}
crop: {margin: 0.25}
knobs:
  max_pixels: [low, native, high]       # fill numbers after reading the checkpoint config
  token_budget: [1.0, 0.5, 0.25, 0.1]
  fps: [1, 2, 5, 8]
  window: [4, 8, 16, 32]
  quant: [null, 8bit, 4bit]
  quality: {kinds: [blur, lowlight, noise, haze, jpeg, resample], severities: [1,2,3,4,5]}
sink: {ratio: 10}
loc: {shortcut_enrichment: 1.5, equal_delta: 0.15}
probe: {decodable_acc: 0.70, selectivity: 0.15}
horizon: {frac: 0.95}
stream: {tolerance_s: 2.0, k: 2, n: 3, c_miss: 1.0, c_early: 0.3, decay_s: 5.0, anticipation_s: 0.0}   # optional temporal early-warning window (0 = off). "Prediction" in this project = latent hazard (§5.14), not onset forecasting
audit: {k_per_cluster: 20}
answer: {min_mass: 0.5}                  # P(answer tokens) over the full vocab; below ⇒ row flagged invalid_format
null: {jpeg_q: 95, shift_px: 1}          # N0.null: the noise-floor control arm
attn: {band: upper_half}                 # layer band for attn_last (shallow layers attend mostly to text)
patch: {corruptions: [pair, boxnoise]}   # ≥ 2 corruption methods required for a "causal" claim
validate: {min_precision: 0.8, per_cause_n: 30, label_noise: 0.1}
hazard: {min_agreement: 0.6}               # latent_hazard labels below this annotator agreement are capped at 'candidate' (own proposal)
tests: {null_runs: 100, null_fpr_max: 0.1, planted_tpr_min: 0.9, finite_diff_rtol: 0.05, smd_max: 0.1, psnr_null_min_db: 40}   # §7.6 thresholds (own proposals)
```

---

## 4. Core conventions (implement once, in `prompts.py` and `model.py`)

**Prompt record**
```python
P = dict(id, group, text, pos, neg, gold, img)   # pos/neg: lists of answer words
```
- `s = logsumexp(logit[pos]) − logsumexp(logit[neg])`; `pred = s > 0`; `correct = pred == gold(sample)`.
- For prompts where the violation answer is "No" (e.g. *"Is the worker wearing a helmet?"*), set `pos=["No"], neg=["Yes"]`. Then **`s > 0` always means "asserts the positive class"** and `gold = sample.label`. No special cases downstream.
- Two-token softmax ⇒ `p_pos = sigmoid(s)`. This is the **only** probability used (ECE, Brier, KL, AURC confidence `|s|`).
- Answer tokens: take the first token of each word variant (`"Yes"`, `"yes"`, `" Yes"`), keep only variants that tokenize to one token. Check at start-up that the prompt ends with the assistant generation prefix so the first generated token is the answer.

**Decisions**
- Error set, recovery, consistency ⇒ **native** (`s > 0`).
- Recall@FPR, streaming alerts ⇒ **calibrated** (`s > τ`, τ frozen from T0 at the target FPR; saved in `outputs/t0/tau.json`).

**Answer-mass validity (C16).** `s` is a two-token margin; it is only meaningful if the model would actually answer with one of those tokens. Every row stores `mass`; rows with `mass < answer.min_mass` get `valid=False`, are excluded from recovery/consistency tests, and are counted in the tier summary. `selfcheck` also requires mass ≥ `answer.min_mass` on a calibration sample of every prompt group.

**Always compare against the original gold.** Under any arm (crop, wrong-facts, blank…), `gold` in the row is the *original sample's* gold. Recovery = fraction correct under the arm. The wrong-facts control should therefore show ≈0 recovery.

---

## 5. Module guides

### 5.1 `data.py`

```python
@dataclass
class Sample: id; bench; modality; media; label; cls; rule_text=None; item=None
              facts: dict = field(default_factory=dict); boxes: list = ...; pair_id=None
              meta: dict = ...; video: dict | None = None

def load(bench) -> list[Sample]          # reads data/<bench>/samples.jsonl
def to_df(samples) -> pd.DataFrame       # one row per sample, meta flattened
def bin_meta(df, cols, q=3) -> df        # pd.qcut(..., duplicates="drop") → categorical; NaN stays NaN

def box_mask(box, grid):                 # box normalized [x0,y0,x1,y1]; grid=(h,w) or (t,h,w)
    t, (h, w) = (grid[0], grid[1:]) if len(grid) == 3 else (1, grid)
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    m[int(np.floor(y0*h)):int(np.ceil(y1*h)), int(np.floor(x0*w)):int(np.ceil(x1*w))] = True
    return np.tile(m.ravel(), t)         # flat mask over visual tokens (row-major, time-major)

def frames(path, fps, n, t_end, order="fwd", single=False, seed=0):
    # causal by construction: timestamps = t_end - arange(n)[::-1]/fps, clipped at 0
    # cv2.VideoCapture; set(CAP_PROP_POS_MSEC, t*1000); read(); BGR→RGB
    # single → last frame only; shuffle → rng.permutation; reverse → [::-1]
    # Qwen3-VL merges 2 frames per temporal patch: if len(out) is odd, repeat the last frame  [VERIFY]
```

**`make_toy(n)` (FR-D-4)** — the backbone of all tests. Draw 224–448 px images:
- background color = `scene` (3 scenes);
- a **red square** of side `obj_size ∈ {small, medium, large}` = evidence of a violation (label 1); label 0 = no red square, or a **green square** (compliant);
- a **blue border** marks `vtype="absence"` scenes;
- `boxes=[{role: violator, xyxy}]`, `facts={item: "absent"/"present"}`, `pair_id` links a violation image to the same scene with the red square replaced by green;
- `meta` filled consistently (size bin, n_workers random, brightness).
A `FakeVLM` (§7) reads these pixels, so interventions (crop, blank, blur, resize) have *real, predictable* effects without any GPU.

**Adapters** (FR-D-2): one function per benchmark, each ≤ 30 lines, each ending in `write_jsonl(...)`. Until access exists, they raise `NotImplementedError` with the [VERIFY] list from plan §4. Do not write adapters before M3 — the toy data drives everything else.

---

### 5.2 `prompts.py`

```python
Y, N = ["Yes", "yes"], ["No", "no"]
viol    = lambda s: s.label
present = lambda s: int(s.facts.get(s.item) == "present")
D = "Safety rule: {rule_text}\n"

PROMPTS = {p["id"]: p for p in [
  dict(id="d0", group="D",   text=D+"Does this image show a violation of the rule? Answer Yes or No.", pos=Y, neg=N, gold=viol, img=True),
  dict(id="d1", group="D",   text="Is the following rule being violated in this image: {rule_text}? Answer Yes or No.", ...),
  dict(id="d2", group="D",   text="As a site safety inspector, would you flag this scene for: {rule_text}? Answer Yes or No.", ...),
  dict(id="pol_pos",  group="POL", text="Is the worker wearing {item}? Answer Yes or No.",            pos=N, neg=Y, gold=viol, img=True),  # "No" = violation
  dict(id="pol_neg",  group="POL", text="Is the worker NOT wearing {item}? Answer Yes or No.",        pos=Y, neg=N, gold=viol, img=True),
  dict(id="pol_miss", group="POL", text="Is any required protective equipment missing? Answer Yes or No.", pos=Y, neg=N, gold=viol, img=True),
  dict(id="frm_neu",  group="FRM", text=D+"Is the rule violated? Answer Yes or No.", ...),
  dict(id="frm_pos",  group="FRM", text="This scene very likely contains a violation. "+D+"Is the rule violated? Answer Yes or No.", ...),
  dict(id="frm_neg",  group="FRM", text="This site has an excellent safety record. "+D+"Is the rule violated? Answer Yes or No.", ...),
  dict(id="qp", group="QP", text="Is {item} clearly visible on the worker? Answer Yes or No.", pos=Y, neg=N, gold=present, img=True),
  dict(id="qp_abs", group="QP", text="Is the worker missing {item}? Answer Yes or No.", pos=Y, neg=N, gold=lambda s: 1 - present(s), img=True),   # polarity-balanced partner of qp
  dict(id="qr", group="QR", text="Rule: {rule_text}\nFacts: {facts_text}\nDoes this situation violate the rule? Answer Yes or No.", pos=Y, neg=N, gold=viol, img=False),
]}

def render(p, s):          # returns None if a required field is missing (FR-P-3)
    f = dict(rule_text=s.rule_text, item=s.item, facts_text="; ".join(f"{k}: {v}" for k, v in s.facts.items()) or None)
    need = re.findall(r"{(\w+)}", p["text"])
    return None if any(f.get(k) is None for k in need) else p["text"].format(**f)

PROMPT_VERSION = md5(json.dumps({k:(v["text"],v["pos"],v["neg"]) for k,v in PROMPTS.items()}, sort_keys=True))[:8]
```
Notes: **`qp_ok` = `qp` AND `qp_abs` both correct** — a single-polarity Q_P under affirmation bias would label an H6 error as "perception failure" (negation is a documented VLM weakness: 2501.09425). `qp` has `gold=present` (Q_P asks about perception, not the violation); its `pos=Y` so `s>0` means "model says visible". The `facts: not_visible` value is treated as *not present* for Q_P.

---

### 5.3 `model.py` (the only model-coupled file)

#### Loading and scoring
```python
class VLM:
    def __init__(self, name, quant=None, min_pixels=None, max_pixels=None, attn="sdpa"):
        qc = {"8bit": BitsAndBytesConfig(load_in_8bit=True),
              "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)}.get(quant)
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cuda",
            quantization_config=qc, attn_implementation=attn).eval()        # use `dtype=` if the pinned version renamed it
        self.p = AutoProcessor.from_pretrained(name)
        if max_pixels or min_pixels:                                         # [VERIFY] key names for Qwen3-VL's image processor
            self.p.image_processor.size = {"shortest_edge": min_pixels or 0, "longest_edge": max_pixels}

    def prep(self, text, images=None, frames=None):
        kind = [{"type": "image"}] * len(images or []) + ([{"type": "video"}] if frames else [])
        msgs = [{"role": "user", "content": kind + [{"type": "text", "text": text}]}]
        prompt = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.p(text=[prompt], images=images or None, videos=[frames] if frames else None,
                      return_tensors="pt").to(self.m.device)               # video fps/timestamps kwargs [VERIFY]

    @lru_cache(None)
    def ids(self, words):                                                    # words: tuple
        out = {t[0] for w in words if len(t := self.p.tokenizer.encode(w, add_special_tokens=False)) == 1}
        assert out, f"no single-token variant for {words}"; return sorted(out)

    def _s(self, lg, pos, neg):
        return (torch.logsumexp(lg[self.ids(tuple(pos))], 0) - torch.logsumexp(lg[self.ids(tuple(neg))], 0))

    @torch.no_grad()
    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        inp = self.prep(text, images, frames); t0 = time.perf_counter()
        lg = self.m(**inp, logits_to_keep=1).logits[0, -1].float(); torch.cuda.synchronize()
        return dict(s=self._s(lg, pos, neg).item(), n_vis=int(self.groups(inp)["vis"].numel()),
                    ms=(time.perf_counter() - t0) * 1e3,
                    mass=float(torch.softmax(lg, 0)[self.ids(tuple(pos)) + self.ids(tuple(neg))].sum()))   # P(answer tokens) over the full vocab
```

`score_inp(inp, pos, neg)` is the same forward as `score` but takes already-prepared inputs and returns only the float margin (used by T3/T4 so inputs are prepared once and hooks wrap the call). Implement `score` as `prep` + `score_inp` + timing.

#### Token groups and grid (FR-M-3)
```python
    def groups(self, inp):
        ids = inp["input_ids"][0]
        key, vid = ("image_grid_thw", self.m.config.image_token_id) if "image_grid_thw" in inp \
              else ("video_grid_thw", self.m.config.video_token_id) if "video_grid_thw" in inp else (None, None)
        last = ids.numel() - 1
        if key is None: return dict(vis=ids[:0], q=torch.arange(0, last, device=ids.device), last=torch.tensor([last], device=ids.device), grid=None)
        vis = (ids == vid).nonzero().squeeze(1); t, h, w = inp[key][0].tolist()
        return dict(vis=vis, q=torch.arange(vis.max() + 1, last, device=ids.device),
                    last=torch.tensor([last], device=ids.device), grid=(t, h // 2, w // 2))
```
Visual features, in order, are scattered into `inputs_embeds` at exactly the positions of the image/video token id, so **index `i` of any per-visual-token tensor ↔ `vis[i]` ↔ grid cell `i` (row-major, time-major)**. This single fact makes every spatial analysis certain.

#### Hooks: `Tap`, `knockout`, `attn_last` (use `attn="eager"` for T3/T4)

**Why eager:** with `attn_implementation="eager"` the attention module receives a materialized 4-D mask and returns the weights; both can be edited/read with plain `register_forward_(pre_)hook`. This avoids relying on private attention-registry APIs. Use the `sdpa` model for T0–T2 and an `eager` instance for T3/T4 (`VLM(..., attn="eager")`). Single-sample runs keep the memory acceptable; if a very large image runs out of memory, lower `max_pixels` for T3/T4 only and state it in the report.

```python
class Tap:
    """Context manager. llm: capture decoder-layer outputs. vit: capture visual sites.
       patch: {layer: fn(h)->h} replaces layer output (activation patching)."""
    def __init__(s, vlm, vit=False, patch=None): s.v, s.vit, s.patch = vlm, vit, patch or {}; s.cap, s.site, s.hk = {}, {}, []
    def __enter__(s):
        m = s.v.m.model
        for i, L in enumerate(m.language_model.layers): s.hk.append(L.register_forward_hook(s._layer(i)))
        if s.vit:
            s.hk.append(m.visual.merger.register_forward_pre_hook(lambda mod, a: s.site.__setitem__("vit", a[0].detach())))
            s.hk.append(m.visual.merger.register_forward_hook(lambda mod, a, o: s.site.__setitem__("merger", o.detach())))
            for k, dm in enumerate(m.visual.deepstack_merger_list):
                s.hk.append(dm.register_forward_pre_hook(lambda mod, a, k=k: s.site.__setitem__(f"ds{k}", a[0].detach())))
        return s
    def _layer(s, i):
        def f(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            s.cap[i] = h.detach()
            if i in s.patch:
                h2 = s.patch[i](h); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return f
    def __exit__(s, *a): [h.remove() for h in s.hk]

def per_token(site, x, n_vis):             # pre-merger sites: 4 consecutive vectors = 1 merged token
    return x.view(n_vis, -1, x.shape[-1]).mean(1) if site != "merger" else x
```
Why "4 consecutive": the merger flattens consecutive groups of four patch vectors (`view(-1, 4*D)`), so that is how the ViT sequence is ordered. `selfcheck` asserts `x.shape[0] == 4 * n_vis`.

**DeepStack caution:** DeepStack adds visual features to the hidden states of the first three LLM layers *inside the language-model forward loop, outside the layer modules*. Layer-output hooks on layers 0–2 therefore see states **without** the injection added after them. Start patching windows at layer ≥ 3, or state the limitation. **[VERIFY]**

```python
class KO:                                   # global state read by the pre-hook
    layers, blocks = set(), []              # blocks: list of (frm_idx, to_idx) tensors; to-queries may not attend frm-keys

@contextmanager
def knockout(vlm, inp, spec, layers):       # spec: [("vis","last"), ("vis","q"), ("q","last")]
    g = vlm.groups(inp); KO.layers = set(layers); KO.blocks = [(g[a], g[b]) for a, b in spec]
    hk = []
    for i, L in enumerate(vlm.m.model.language_model.layers):
        def pre(mod, args, kwargs, i=i):
            if i not in KO.layers: return
            m = kwargs["attention_mask"]; assert m is not None, "eager attention must pass a 4-D mask"
            m = m.clone(); neg = torch.finfo(m.dtype).min if m.dtype.is_floating_point else False
            for frm, to in KO.blocks: m[:, :, to[:, None], frm[None, :]] = neg
            kwargs["attention_mask"] = m; return args, kwargs
        hk.append(L.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try: yield
    finally: [h.remove() for h in hk]; KO.layers, KO.blocks = set(), []

def attn_last(vlm, inp):                    # → [layers, heads, T_all]: last-token attention row per layer
    out, hk = {}, []
    for i, L in enumerate(vlm.m.model.language_model.layers):
        hk.append(L.self_attn.register_forward_hook(
            lambda mod, a, kw, o, i=i: out.__setitem__(i, o[1][0, :, -1, :].float().cpu()), with_kwargs=True))
    with torch.no_grad(): vlm.m(**inp, logits_to_keep=1)
    [h.remove() for h in hk]
    return torch.stack([out[i] for i in sorted(out)])          # o[1]: attention weights, returned by eager attention [VERIFY]
```
Only the last row is kept (memory-light). Slice with `vis` indices to get `[layers, heads, Nv]`.

#### Sinks (FR-M-11) and gradient saliency (FR-M-10)
```python
def sink_mask(h_vis, ratio):                # h_vis: [Nv, D] hidden states of visual tokens at a middle layer
    m = h_vis.abs().amax(-1); return m > ratio * m.median()

def grad_x_input(vlm, inp, pos, neg):
    feats = []
    def hook(mod, a, out):                  # out = (merged [Nv,D], [deepstack0..2 each [Nv,D]])   [VERIFY]
        merged, ds = out
        merged = merged.detach().requires_grad_(True); ds = [d.detach().requires_grad_(True) for d in ds]
        feats.extend([merged, *ds]); return merged, ds
    hk = vlm.m.model.visual.register_forward_hook(hook)
    with torch.enable_grad():
        s = vlm._s(vlm.m(**inp, logits_to_keep=1).logits[0, -1].float(), pos, neg); s.backward()
    hk.remove()
    return sum((f.grad * f).abs().sum(-1) for f in feats).detach().float().cpu()      # [Nv]
```
Gradients flow to *features* (inputs of the LLM side), not pixels — which sidesteps patch-ordering questions entirely.

---

### 5.4 `interventions.py`

```python
INTERV = {}
def reg(name, control=None, kind="recover"):
    def d(f): INTERV[name] = dict(fn=f, control=control, kind=kind); return f
    return d
# every fn: (sample, **params) -> (images | None, sample_override | None)
```

| Arm | Implementation (≈ lines) | Control arm |
|---|---|---|
| `O1.gtcrop` | crop violator box expanded by `crop.margin` (6) | `O1.randcrop` (random same-size crop, seeded by sample id) and `O1.sametok` (full frame resized to the **same pixel area** as the crop) |
| `O1.tightcrop` | margin 0 (1) | same |
| `O1.attncrop` (P1) | crop around argmax of `attn_last` map (needs T3 eager pass) | same |
| `O2.facts` | `images=None`; job uses prompt `qr`; sample unchanged (3) | `O2.wrongfacts` (flip each fact present↔absent in a sample copy) |
| `O3.blank` / `O3.noise` / `O3.swap` | gray image; Gaussian noise image; image of a random other-scene sample of the other label (3 each) | — (`kind="necessity"`) |
| `O1.bgmask` (C3) | keep the **full frame at native resolution**; gray out everything outside violator box + `crop.margin` (4). Token count is unchanged ⇒ separates *clutter/context* (recovers) from *resolution* (does not) | `O1.randmask` (same-size box at a random location kept, rest grayed) |
| `O3.occlude` (C3) | gray the GT evidence box only, native resolution (3). Tests whether the answer depends on the **evidence region**, not just on "an image being present" | `O3.randoccl` (random same-size box); compare Δs (`kind="necessity"`) |
| `N0.null` (C4) | imperceptible perturbation: JPEG quality `null.jpeg_q` + `null.shift_px` translation (3). **The noise-floor control for every knob and for O6.** A deterministic base re-run is *not* a control: it recovers exactly 0 by construction | — |
| `INJ.small/clutter/degrade/lead/label` (C15) | known-cause injections used only by `validate` (§5.8b) (≈ 40 total) | — |
| `O4.*` | **not interventions** — they are *prompts* (POL, FRM groups); the run plan simply includes those prompt ids | paraphrase variance (D group) |
| `O5.personcrop` / `O5.overlay` | crop / draw rectangle on each `person` box (6) | `O5.randperson` / `O5.randoverlay` (random other worker or random box of same size) |
| `O6.degrade` | `degrade(img, kind, sev)` (25) | severity sweep itself |
| `O6.sharpen` | `UnsharpMask` + `autocontrast` on errors (3) | — |
| `O7.*` | **handled in `frames()` params** (`single`, `order`, `n`, `fps`) — no intervention function | `O7.single` keeps timestamps |

`degrade` as one dict of lambdas:
```python
def degrade(img, kind, sev):
    a = np.asarray(img, np.float32) / 255
    if kind == "blur":      return img.filter(ImageFilter.GaussianBlur(1.5 * sev))
    if kind == "lowlight":  return arr2img(np.clip((a ** (1 + .4 * sev)) * (1 - .1 * sev) + rng.normal(0, .01 * sev, a.shape), 0, 1))
    if kind == "noise":     return arr2img(np.clip(a + rng.normal(0, .03 * sev, a.shape), 0, 1))
    if kind == "haze":      return arr2img(a * (1 - .15 * sev) + .15 * sev)
    if kind == "jpeg":      b = io.BytesIO(); img.save(b, "JPEG", quality=max(5, 100 - 18 * sev)); return Image.open(b).convert("RGB")
    if kind == "resample":  w, h = img.size; f = 1 / (1 + sev); return img.resize((int(w*f), int(h*f)), Image.NEAREST).resize((w, h), Image.NEAREST)
    if kind == "motion":    return img.filter(ImageFilter.Kernel((9, 9), [1/9 if r == 4 else 0 for r in range(9) for _ in range(9)]))
```
(`bitrate` for video: `ffmpeg -crf` re-encode via `subprocess`, one 8-line helper; P2.)

**Token-budget knob (random pruning, FR-T2-3).** Do **not** remove tokens (MRoPE positions depend on the grid). Instead zero the visual features of a random subset at the visual-encoder output, which removes the information while keeping positions intact:
```python
def mask_tokens(vlm, keep_frac, seed):                          # context manager in model.py
    def hook(mod, a, out):
        merged, ds = out; g = torch.Generator().manual_seed(seed)
        drop = torch.rand(merged.shape[0], generator=g) > keep_frac; drop = drop.to(merged.device)
        merged = merged.masked_fill(drop[:, None], 0); ds = [d.masked_fill(drop[:, None], 0) for d in ds]
        return merged, ds
    ...
```
State this approximation in the report ("input-level masking, positions kept"). Attention-based dropping (drop lowest-`attn_last` tokens) is P2 and uses the eager instance.

---

### 5.5 `run.py` — job plans and the runner

```python
Job = namedtuple("Job", "sid prompt arm params mcfg")

def plan_t0(cfg, S):    return [Job(s.id, p, "base", {}, BASE) for s in S for p in ("d0", "d1", "d2")]
def plan_t1(cfg, S, E): # E = error ∪ matched-correct ids
    j  = [Job(s.id, p, "base", {}, BASE) for s in S if s.id in E for p in ("qp","qp_abs","qr","pol_pos","pol_neg","pol_miss","frm_neu","frm_pos","frm_neg")]
    j += [Job(s.id, p, "O3.blank", {}, BASE) for s in S for p in ("d0","d1","d2")]       # flip-rate baseline (FR-B-7), all samples
    return j
def plan_t2(cfg, S, E):  # arms from INTERV (+ each arm's control), N0.null on errors AND matched corrects, quality sweep, knob grids; each knob value = one Job.mcfg or params
    ...
PLANS = dict(t0=plan_t0, t1=plan_t1, t2=plan_t2, stream=plan_stream)

def run(jobs, S, out="runs.jsonl"):
    done = {json.loads(l)["key"] for l in open(out)} if exists(out) else set()
    for mc, js in groupby(sorted(jobs, key=lambda j: json.dumps(j.mcfg, sort_keys=True)), key=lambda j: json.dumps(j.mcfg, sort_keys=True)):
        vlm = VLM(**{**cfg.model, **json.loads(mc)})
        for j in js:
            k = key(j)
            if k in done: continue
            try: append(out, one(vlm, S[j.sid], j, k))
            except Exception as e: append("failures.jsonl", dict(key=k, err=repr(e)))

def one(vlm, s, j, k):
    p = PROMPTS[j.prompt]; a = INTERV.get(j.arm)
    imgs, s2 = a["fn"](s, **j.params) if a else (load_media(s, **j.params), None)
    text = render(p, s2 or s)
    if text is None: return None                       # counted as skip
    media = {} if not p["img"] else dict(frames=imgs) if s.modality == "video" else dict(images=imgs)
    r = vlm.score(text, **media, pos=p["pos"], neg=p["neg"])
    g = p["gold"](s)                                   # ORIGINAL sample's gold
    return dict(key=k, sample_id=s.id, prompt_id=j.prompt, group=p["group"], arm=j.arm, **knob_fields(j),
                s=r["s"], pred=int(r["s"] > 0), gold=g, correct=int((r["s"] > 0) == g),
                n_vis_tokens=r["n_vis"], latency_ms=r["ms"], t=j.params.get("t_end"), extra={"mass": r["mass"]})
```
**Model reloads:** jobs are grouped by `mcfg`, so quantization/pixel-limit sweeps load each configuration once. **Knobs implemented as params** (`fps`, `window`, quality kind/severity, token budget) do not reload anything.

**Error set (end of T0), `errors.parquet`:** compute per-sample majority vote over `d0..d2` (native decision), mark errors, join metadata, attach taxonomy tags (FR-E-2), pick matched corrects:
```python
def match(err, cor, keys=("cls","obj_size_bin"), k=1, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _, e in err.iterrows():
        pool = cor[(cor[list(keys)] == e[list(keys)]).all(1) & (cor.sample_id != e.sample_id)]
        out += list(pool.sample(min(k, len(pool)), random_state=int(rng.integers(1e9))).sample_id)
    return out
```
Matching falls back to `cls` only when the exact bin has no corrects (counted and reported).

---

### 5.6 `metrics.py` — formulas

All functions take arrays or DataFrames with columns `sample_id, s, pred, gold, arm, prompt_id, …` and return floats/DataFrames.

```python
def p(s): return 1 / (1 + np.exp(-s))

def boot_ci(fn, df, n=1000, seed=0, a=0.05):               # cluster bootstrap over sample_id (FR-B-0)
    ids = df.sample_id.unique(); rng = np.random.default_rng(seed); g = dict(tuple(df.groupby("sample_id")))
    v = [fn(pd.concat([g[i] for i in rng.choice(ids, len(ids))])) for _ in range(n)]
    return np.nanpercentile(v, [100*a/2, 100*(1-a/2)])

def recall_at_fpr(s, y, fpr):   f, t, _ = roc_curve(y, s); return float(np.interp(fpr, f, t))
def fpr_at_recall(s, y, rec):   f, t, _ = roc_curve(y, s); return float(np.interp(rec, t, f))
def expected_cost(pred, y, cm, cf): return (cm * ((y == 1) & (pred == 0)).mean() + cf * ((y == 0) & (pred == 1)).mean())
def min_cost_threshold(s, y, cm, cf): ts = np.unique(s); return min((expected_cost((s > t).astype(int), y, cm, cf), t) for t in ts)

def ece(prob, y, bins=15):
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(prob[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any())
def brier(prob, y): return float(np.mean((prob - y) ** 2))

def risk_coverage(conf, err):                               # conf = |s|; err = 1 if native decision wrong
    o = np.argsort(-conf); k = np.arange(1, len(o) + 1); return k / len(o), np.cumsum(err[o]) / k
def aurc(conf, err): return float(risk_coverage(conf, err)[1].mean())
def coverage_at_risk(conf, err, r): c, rk = risk_coverage(conf, err); return float(c[rk <= r].max()) if (rk <= r).any() else 0.0
# Coverage@Recall: same sort, but track recall of positives among accepted items.

def kl_bern(a, b, e=1e-6): a, b = np.clip(a, e, 1-e), np.clip(b, e, 1-e); return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))
def visual_necessity(real, other):                          # aligned Series of s over sample_ids
    return float(np.mean(kl_bern(p(real), p(other))))

def flip_rate(df):  return float((df.groupby("sample_id").pred.nunique() > 1).mean())          # over paraphrases
def pair_acc(df):   # df has pair_id, gold, correct (one prompt)
    g = df.dropna(subset=["pair_id"]).groupby("pair_id"); both = g.correct.min(); return float(both.mean())
def pair_rank_acc(df):                                      # threshold-free
    w = df.pivot(index="pair_id", columns="gold", values="s"); return float((w[1] > w[0]).mean())
def polarity_consistency(df):                               # df: prompts pol_pos/pol_neg/pol_miss, base arm
    return float((df.pivot(index="sample_id", columns="prompt_id", values="pred").nunique(axis=1) == 1).mean())
def yes_rate(df, yes_is_pos): ...                           # P(model says "Yes") per prompt → affirmation bias

def framing_shift(w):       # w: pivot sample×prompt of s
    return dict(pos=(w.frm_pos - w.frm_neu).mean(), neg=(w.frm_neg - w.frm_neu).mean())

def temporal_gains(acc): return dict(multi_frame=acc["full"] - acc["single"], order=acc["full"] - acc["shuffle"], reverse=acc["full"] - acc["reverse"])
```
Rules: (i) every function returns `n` alongside values when stratified; (ii) the report prints flip-rate only together with the blank-frame flip-rate (FR-B-7), enforced by a helper `flip_report(df)` that fetches both arms; (iii) `Coverage@Recall` is the coverage achievable while keeping recall ≥ target on accepted positives.

---

### 5.7 `slices.py` (FR-S)

```python
def slice_search(df, cols, err="err", max_order=2, min_support=30, alpha=0.05):
    N, E = len(df), int(df[err].sum()); base = E / N; rows = []
    for k in range(1, max_order + 1):
        for cs in combinations(cols, k):
            for key, g in df.groupby(list(cs), observed=True):          # NaN keys dropped: bin first!
                n, e = len(g), int(g[err].sum())
                if n < min_support: continue
                lo, hi = proportion_confint(e, n, method="wilson")
                pv = fisher_exact([[e, n - e], [E - e, N - n - (E - e)]], alternative="greater")[1]
                rows.append(dict(slice=dict(zip(cs, key if isinstance(key, tuple) else (key,))), n=n, err=e/n, lo=lo, hi=hi, gap=e/n - base, p=pv))
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["q"] = multipletests(out.p, method="fdr_bh")[1]
    return out[out.q < alpha].sort_values("lo", ascending=False)       # rank by the CI lower bound
```
- Call twice: **misses** (rows with `gold=1`, `err = miss`) and **false alarms** (`gold=0`, `err = false alarm`), one row per sample (majority-vote native decision).
- Optional redundancy prune: drop a slice if a parent slice has an error rate within 0.05.
- `columns`: `cls, scene, viewpoint, vtype, n_workers_bin, feature_share, obj_size_bin, dist_bin, occlusion, blur_bin, brightness_bin, bitrate_bin`. Skip any column with > 50% missing.
- **P3 embedding slices:** `KMeans(k).fit(embeddings)`, error rate + Wilson CI per cluster. Nothing else.

---

### 5.8 `triage.py` and `escalate.py`

**Triage (FR-T1-2)** — a literal table:
```python
def triage(qp_ok, qr_ok, qd_ok):
    if qp_ok is None: return "untriaged (no item)"      # proximity/zone violations have no `item`: Q_P undefined; counted, never silently dropped
    if qd_ok:  return "ok" if qp_ok else "right_for_wrong_reason"
    if not qp_ok: return "perception_failure"          # Q_R irrelevant
    return "process_failure" if qr_ok else "rule_reasoning_failure"
NEXT = dict(perception_failure="O1,O6 → T4", process_failure="O1.bgmask, O1.gtcrop and hires must NOT recover → then T3/T4",
            rule_reasoning_failure="prompt/rule representation", right_for_wrong_reason="audit")
```
Triage uses the perceptual-failure / process-failure split of 2609.22588 ("recognized evidence fails to constrain the decision"). A `process_failure` reading is only a **provisional label**; it becomes the cause "readout/process (H3)" only through the signature table below (crop and hires must also fail).

**Clusters.** `cluster = (name, predicate dict from slice search or taxonomy tag, error ids, matched correct ids)`. Always add one **`ALL_ERRORS`** cluster (C17): diffuse errors that fall in no significant slice are otherwise never diagnosed. Slices are only as good as their coherence (slice discovery methods often return incoherent slices: 2203.14960), so the audit sample (FR-E-6) also records a human *coherence* verdict per cluster.

**Exit rule — three-valued, CI-based, on confident errors only (C1, C4, C5).** Every diagnostic returns `True / False / None`. `None` (n < `min_n`, or the Wilson CI straddles a threshold) can never support a cause. Recovery/necessity tests use only **confident** errors (`|s_base| ≥ exit.conf_margin`, valid answer format): a near-zero margin flips under *any* perturbation (regression to the mean), so borderline errors are reported separately. *(This guard is our own methodological addition; no source in `references.md` prescribes it.)*
```python
def wil(k, n): return proportion_confint(k, n, method="wilson") if n else (0.0, 1.0)
def conf_ids(runs, ids, X):
    b = runs[(runs.arm == "base") & runs.sample_id.isin(ids) & runs.valid].groupby("sample_id").s.mean()
    return list(b.index[b.abs() >= X.conf_margin])

def recovery(runs, ids, arm):   r = runs[(runs.arm == arm) & runs.sample_id.isin(ids) & runs.valid]; v = r.groupby("sample_id").correct.mean().ge(.5); return int(v.sum()), len(v)   # (k, n)
def breakage(runs, ids, arm):   k, n = recovery(runs, ids, arm); return n - k, n                                   # applied to matched corrects

def explain_recover(runs, c, arm, ctls, X):      # O1.*, O5, O6, hires/knobs; ctls = list of control arms ("N0.null" for knobs/O6)
    ids = conf_ids(runs, c.err_ids, X); k, n = recovery(runs, ids, arm); lo, hi = wil(k, n)
    cs = [recovery(runs, ids, a) for a in ctls]; clo = max(wil(*x)[0] for x in cs); chi = max(wil(*x)[1] for x in cs)   # worst control
    if n < X.min_n or min(x[1] for x in cs) < X.min_n: v = None
    elif lo >= X.recovery and chi <= X.control: v = True
    elif hi < X.recovery or clo > X.control: v = False
    else: v = None
    return dict(test=arm, rec=k/max(n, 1), ci=(lo, hi), ctl=max(x[0]/max(x[1], 1) for x in cs), brk=breakage(runs, c.cor_ids, arm),
                dshift=mean_dshift(runs, ids, arm), n=n, explained=v)            # dshift: mean Δs toward gold (FR-B-18)

def gate_ok(runs, i):                       # VLind-Bench-style gate (C6): perception AND rule knowledge must pass before a *prior* is blamed
    r = runs[(runs.sample_id == i) & (runs.arm == "base") & runs.prompt_id.isin(["qp", "qp_abs", "qr"]) & runs.valid]
    return len(r) == 3 and bool(r.correct.all())

def explain_necessity(runs, c, X):          # O3: insensitive to the image AND to the evidence region ⇒ prior-driven
    ids = [i for i in conf_ids(runs, c.err_ids, X) if gate_ok(runs, i)]
    if len(ids) < X.min_n: return dict(test="O3", n=len(ids), explained=None)
    ci = {a: boot_ci(lambda d, a=a: vn(d, ids, a) / max(vn(d, c.cor_ids, a), 1e-9), runs) for a in ("O3.blank", "O3.occlude")}   # VN_err / VN_cor, (lo, hi)
    v = True if all(hi <= NEC.ratio for lo, hi in ci.values()) else False if any(lo > NEC.ratio for lo, hi in ci.values()) else None
    return dict(test="O3.necessity", ci=ci, n=len(ids), explained=v)

def explain_consistency(runs, c, X):        # O4: polarity/framing flips must exceed paraphrase noise AND the matched corrects' flips
    f_e, f_c, f_par = flip_rate(pol(runs, c.err_ids)), flip_rate(pol(runs, c.cor_ids)), flip_rate(par(runs, c.err_ids))
    lo, hi = boot_ci(lambda d: flip_rate(pol(d, c.err_ids)) - flip_rate(pol(d, c.cor_ids)), runs)      # error-minus-correct flip gap
    ok = f_e >= CONS.min_flip and f_e >= CONS.ratio * f_par
    v = True if ok and lo > 0 else False if (not ok and f_e < CONS.min_flip) or hi <= 0 else None
    return dict(test="O4.flip", f_err=f_e, f_cor=f_c, f_par=f_par, gap_ci=(lo, hi), explained=v)
```
- **O2 is an exclusion test, not a cause (C2).** With oracle facts in the text, any LLM answers correctly, so `qr_ok=True` carries no information about perception or readout. Only `qr_ok=False` is informative (⇒ rule-reasoning failure). `explain_recover("O2.facts", ctls=["O2.wrongfacts"])` is computed for that purpose only.
- **Knob rule:** `hires` = `explain_recover` on `max_pixels = <highest>` with `ctls=["N0.null"]`; `knob` = the same for any other knob (quantization, token budget, fps, window). A base re-run is never a control. Report breakage of matched corrects at that setting; if breakage > 0.2 add the caveat *"intervention changes the input distribution"*.

**Diagnostics → signature table (C1).** Each cause is a *conjunction of required diagnostic outcomes*; omitted diagnostics are "don't care". The cluster gets a cause only if **exactly one** signature matches.

| diag | computed by |
|---|---|
| `crop` | `explain_recover("O1.gtcrop", ["O1.randcrop", "O1.sametok"])` |
| `bgmask` | `explain_recover("O1.bgmask", ["O1.randmask"])` |
| `hires` / `knob` / `sharpen` | `explain_recover(..., ["N0.null"])` |
| `qr_ok` | the text-only rule + oracle-facts arm. **Q_R and O2 are the same prompt, so this is one diagnostic, not two:** `explain_recover("O2.facts", ["O2.wrongfacts"])` on confident errors (exclusion only: `True` says nothing about perception); latent hazards: `hz_r` |
| `cue` | `explain_recover("HZ.cue", ["HZ.cue_twin"])` — latent hazards only (§5.14) |
| `qp_ok` | share of confident errors with `qp ∧ qp_abs` correct, Wilson CI vs 0.5 |
| `necessity`, `consistency` | `explain_necessity`, `explain_consistency` |
| `loc` ∈ {`loc_equal_high`, `loc_low_both`, `loc_lower_in_errors`} | `three_way` (T3) |
| `t4` ∈ {`H1`,`H2`,`H3`,`enc`}, `causal` | `probe_verdict` first token; robust knockout/patching (FR-T4-8/10) |
| `stream` | `"policy"` if `classify` ⇒ `policy_late` dominates |

```python
SIG = {   # cause -> required outcomes (compared with ==; None never matches)
 "pixel/token-budget limit (H1/H4)":  dict(crop=True, hires=True, bgmask=False),          # more pixels help, removing clutter at equal tokens does not
 "clutter/binding/context (H8/H5)":   dict(crop=True, bgmask=True, hires=False),
 "prior-driven (H7)":                 dict(crop=False, bgmask=False, hires=False, qp_ok=True, qr_ok=True, necessity=True),
 "polarity/framing bias (H6/H7)":     dict(consistency=True, qp_ok=True),
 "rule-reasoning failure":            dict(qr_ok=False),                                   # latent hazards: `qr_ok` comes from hz_r (physical/hazard reasoning from oracle conditions)
 "uncued-hazard under-reporting":     dict(cue=True, qp_ok=True, qr_ok=True),              # infers the consequence when cued, does not volunteer it (latent_hazard only)
 "input quality (H9)":                dict(sharpen=True),
 "deployment knob (H4)":              dict(knob=True),
 "scene shortcut (H5)":               dict(loc="loc_low_both", necessity=True),
 "readout/process (H3)":              dict(qp_ok=True, qr_ok=True, crop=False, hires=False, loc="loc_equal_high", t4="H3", causal=True),
 "merger bottleneck (H1)":            dict(t4="H1", causal=True),
 "depth fading (H2)":                 dict(t4="H2", causal=True),
 "encoder/perception limit":          dict(qp_ok=False, crop=False, hires=False, t4="enc"),
 "trigger-policy failure (H11)":      dict(stream="policy"),
}
def decide(ev):                                    # ev: {diag: True|False|None|str}
    hit = [k for k, req in SIG.items() if all(ev.get(d) == v for d, v in req.items())]
    if len(hit) == 1: return hit[0], []
    if len(hit) > 1:  return "ambiguous", hit                                       # several causes fit: never pick one silently
    part = [k for k, req in SIG.items() if all(ev.get(d) in (v, None) for d, v in req.items())]
    return ("inconclusive: more data" if part else "unexplained → next tier"), part  # None-blocked candidates vs nothing fits
```
- *Example:* `crop=True, bgmask=False, hires=False` matches **no** signature (resolution needs `hires`, clutter needs `bgmask`) ⇒ `unexplained → next tier`: the engine escalates to T4 probes instead of mislabelling it "pixel limit".
- Tiers are no longer walked "first explained wins": **all** cheap diagnostics (T1/T2) are computed for every cluster, then `decide` runs once; T3/T4 are added only for `unexplained`/`inconclusive`/`ambiguous` clusters.
- **Evidence rule for internal causes (FR-T4-10):** `t4`/`causal` are only set when the evidence rule passes; otherwise the diag is `None` and the cluster reads `unresolved (single method)`.
- **Per-error accounting (C17).** Clusters overlap, so percentages in the report are per **error**: each error takes the cause of the *most specific* (smallest) cluster containing it that has a unique cause; ties or none ⇒ `ambiguous` / `ALL_ERRORS` cause. `diag_vector.parquet` stores one row per error.

```python
IMPL = {   # cause -> implication (printed in the report)
 "pixel/token-budget limit (H1/H4)":  "higher resolution/tiling, crop-then-ask, less merger compression",
 "clutter/binding/context (H8/H5)":   "person-level cropping/tracking before the VLM; context-matched hard negatives",
 "prior-driven (H7)":                 "counterfactual data; neutral prompt template",
 "polarity/framing bias (H6/H7)":     "balanced polarity prompts/data; fixed template",
 "rule-reasoning failure":            "prompt/rule representation; for latent hazards: add hazard/physics knowledge to the prompt",
 "uncued-hazard under-reporting":     "hazard-checklist prompting (name triggers/consequences explicitly); calibrated threshold on the uncued margin",
 "input quality (H9)":                "image-quality gates; degradation augmentation",
 "deployment knob (H4)":              "adjust serving configuration",
 "scene shortcut (H5)":               "rebalance; hard negatives; scene-matched contrast sets",
 "readout/process (H3)":              "auxiliary grounding loss; perceive-then-decide prompting; training-free evidence highlighting (cf. 2510.17771)",
 "merger bottleneck (H1)":            "unfreeze/redesign merger; detail-preserving tokens",
 "depth fading (H2)":                 "re-inject features (DeepStack-style/IRI); layer-targeted tuning",
 "encoder/perception limit":          "better encoder/resolution; crop-then-ask upstream",
 "trigger-policy failure (H11)":      "trigger head / calibrated thresholds",
}
```

---

### 5.8b `validate` — measure the false-attribution rate (C15; `escalate.py` + `interventions.py`)

**Why.** `tests/fake.py` plants mechanisms that were built to match the decision rules, so passing the integration test shows the code implements the rules — not that the rules name the true cause of a *real* model's errors, where small objects are also low-contrast and cluttered. `validate` measures this on the real checkpoint.

1. `pool` = samples correct under `base` with `|s| ≥ exit.conf_margin`.
2. Apply **one known-cause injection per sample**; keep the samples that now fail (the injected errors):

| INJ | Construction (≈ lines) | Expected cause |
|---|---|---|
| `INJ.small` | put the image in the centre of a k× larger reflect-padded canvas and rescale the box (≈ 8); with `max_pixels` fixed the object occupies fewer tokens | pixel/token-budget limit (H1/H4) |
| `INJ.clutter` | paste m person/object crops from other samples around the violator at native scale, update boxes (≈ 15) | clutter/binding/context (H8/H5) |
| `INJ.degrade` | `degrade(kind, sev)` on the full image (reuses O6) | input quality (H9) |
| `INJ.lead` | prefix `d0` with a misleading framing (`frm_pos` on safe, `frm_neg` on violations) | polarity/framing bias (H6/H7). *Shares the O4 prompts, so this row validates the signature logic, not O4's sensitivity.* |
| `INJ.label` | flip the gold of a random `validate.label_noise` fraction of `pool` (**input unchanged**) | **none** — must end `unexplained`/audit; any confident cause here is a false attribution |

3. Run the T1/T2 diagnostics and `decide` on each injected cluster (≥ `validate.per_cause_n` errors per injection, else mark `underpowered`).
4. Write `outputs/validate/confusion.csv`: rows = injected cause (incl. `INJ.label`), columns = assigned cause (incl. `ambiguous`, `inconclusive`, `unexplained`); per-cause **precision and recall with Wilson CIs**.
5. `report.py` may label a cause **decisive** only if its measured precision ≥ `validate.min_precision`; otherwise it prints `candidate (validated precision p [lo, hi])`. Causes with no injection analogue (H3, H1-merger, H2) are always `candidate` unless the evidence rule (FR-T4-10) passes.

Limits to state in the report: injections are synthetic, so the confusion matrix is an *optimistic* bound; injected causes are single, real errors are mixtures.

---

### 5.9 `attrib.py` — T3 (P1)

**Maps live on the token grid.** Both explainers return `[h/2, w/2]`. `band` comes from `attn.band` (default: **upper half** of the LLM layers — shallow layers attend mostly to text and deeper layers to localized evidence: 2510.17771). Because visual information is first copied into the *question* tokens and only then into the last position (2411.18620, LLaVA-family; **[VERIFY]** for Qwen3-VL), last-token attention alone under-reports routing; the `knockout` experiments (§5.10) include `vis→q` for this reason.
```python
def map_attn(vlm, inp, band, ratio):                      # raw attention, sink-excluded
    g = vlm.groups(inp); A = attn_last(vlm, inp)[:, :, g["vis"].cpu()]          # [L, H, Nv]
    with Tap(vlm) as t: vlm.m(**inp, logits_to_keep=1)
    sink = sink_mask(t.cap[len(t.cap)//2][0, g["vis"]], ratio).cpu()
    a = A[band[0]:band[1]].mean((0, 1)); a[sink] = 0; return (a / a.sum()).reshape(g["grid"][1:])
def map_grad(vlm, inp, pos, neg):
    g = vlm.groups(inp); return grad_x_input(vlm, inp, pos, neg).reshape(g["grid"][1:])
```
**Localization (FR-T3-2)**
```python
def loc(amap, box):
    m = box_mask(box, amap.shape).reshape(amap.shape); a = np.clip(amap, 0, None); a = a / a.sum()
    k = int(m.sum()); top = np.argsort(-a.ravel())[:k]
    return dict(mass=a[m].sum(), enrich=a[m].sum() / m.mean(), rank=m.ravel()[top].mean(), point=float(m.ravel()[a.argmax()]), bg=1 - a[m].sum())
```
**3-way reading (FR-T3-4, C8)** — compare `enrich` between a cluster's errors and matched corrects (medians with `boot_ci`; `n < min_n` ⇒ `None`). **Equal, high enrichment does *not* identify H3 by itself:** MLLMs localize the evidence even when they answer wrongly because of small-detail resolution limits (2502.17422), and "seeing but not believing" (2510.17771) is the process-failure counterpart. The signature table therefore requires `crop=False ∧ hires=False` before it says H3.
```python
def three_way(enr_err, enr_cor, X):                       # returns the `loc` diagnostic consumed by SIG
    if max(np.median(enr_err), np.median(enr_cor)) < X.shortcut_enrichment: return "loc_low_both"                  # scene shortcut (H5)
    if np.median(enr_err) >= np.median(enr_cor) - X.equal_delta * np.median(enr_cor): return "loc_equal_high"       # localized-but-unused: H3 *or* resolution-limited H1
    return "loc_lower_in_errors"                                                                                   # perception/routing failure
```
**Deletion/insertion (FR-T3-5)** — mask cells in *image space* (robust to patch ordering). Gray boxes are not a neutral removal: the *shape* of removed pixels leaks information and rankings change with the fill (2202.00449), and the images are out-of-distribution (2201.13291). The primary fill is therefore the debiased `nli`; `gray`/`blur` are stability checks:
```python
def paint(img, cells, grid, how="nli"):                    # cells masked in *image space*; fill = how
    a = np.asarray(img, np.float32); H, W = a.shape[:2]; h, w = grid; m = np.zeros((H, W), bool)
    for c in cells: r, k = divmod(int(c), w); m[int(r*H/h):int((r+1)*H/h), int(k*W/w):int((k+1)*W/w)] = True
    if how == "gray": a[m] = 128
    elif how == "blur": a[m] = np.asarray(img.filter(ImageFilter.GaussianBlur(8)), np.float32)[m]
    else: a = nli(a, m)          # ROAD-style noisy linear imputation of the masked pixels from their neighbours (≈ 25 lines; implement the algorithm, add no dependency) [VERIFY against the authors' reference implementation]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))
def del_ins(score_fn, img, amap, mode, steps=10, sign=1):
    order = np.argsort(-amap.ravel()); N = len(order); curve = []
    for f in np.linspace(0, 1, steps + 1):
        k = int(f * N); cells = order[:k] if mode == "del" else order[k:]
        curve.append(sign * score_fn(paint(img, cells, amap.shape)))                # sign = 2*gold-1: margin toward gold
    return float(np.trapz(curve, dx=1/steps))
```
**Explainer-validity gate (FR-T3-6)** — run on ≤ 100 samples before any T3 table:
1. `AUC_del(expl) < AUC_del(random order)` and `< AUC_del(raw attention)` (paired Wilcoxon, `p<0.05`); symmetric for insertion.
2. Rankings of explainers stable across `how ∈ {nli, gray, blur}` (a ranking that depends on the fill is not trusted; report the fill used for every number). DAUC/IAUC use only the *ranking* of scores (2201.13291), so a pass says nothing about score magnitudes.
3. **Randomization sanity:** re-initialize the top half of LLM layers (save `state_dict` of those layers, `nn.init.normal_(w, 0, .02)`, restore afterwards); Spearman(map, map_randomized) must drop below 0.5 on average.
4. **Robustness:** Spearman(map, map under Gaussian pixel noise σ=0.02) ≥ 0.6.
5. **[P2] Internal agreement (C10):** `knockout` of the top-k attributed visual tokens (custom `vis` subset) must reduce |s| more than random-k tokens. Output-faithfulness curves can be maximised by routes that do not reflect the computation actually used (2509.04588, shown on CNNs — **[VERIFY]** for VLMs).
Write `t3_gate.json`; if any check fails, tag all T3 outputs `UNVALIDATED`.
**SpRAy:** downsample maps to 8×8, `KMeans(k=4..8)`; report cluster sizes and mean `enrich`/`bg` per cluster. **Exploratory only** unless the planted-confounder test passes: a reproducibility study reports SpRAy struggling with complex features and severe subgroup imbalance (2604.04518).
**Planted-confounder test (FR-T3-8):** toy dataset where the scene color predicts the label; the explainer should put mass on the background region — assert `bg` high for errors in that scene.

---

### 5.10 `mech.py` — T4 (P2)

**Cache pooled features once, analyze many times.** For each sampled (stratified by cluster, matched pairs included) sample, one forward with `Tap(vlm, vit=True)`; pool immediately and store float16:
```python
def cache_sample(vlm, s, prompt, box, out):
    inp = vlm.prep(...); g = vlm.groups(inp); n = g["vis"].numel(); M = torch.from_numpy(box_mask(box, g["grid"])).to(g["vis"].device)
    with Tap(vlm, vit=True) as t: vlm.m(**inp, logits_to_keep=1)
    f = {k: per_token(k, v, n)[M].mean(0) for k, v in t.site.items()}                       # vit, ds0..2, merger
    f |= {f"L{i}_vis": h[0, g["vis"]][M].mean(0) for i, h in t.cap.items()}                 # LLM layers, visual positions in box
    f |= {f"L{i}_last": h[0, -1] for i, h in t.cap.items()}
    f |= {f"L{i}_q": h[0, g["q"]].mean(0) for i, h in t.cap.items()}                         # question positions (C12): visual info migrates here (2411.18620)
    np.savez(out / f"{s.id}.npz", **{k: v.float().cpu().numpy().astype(np.float16) for k, v in f.items()})
```
Memory: ~36 layers × 2560 × 2 B × 3 positions ≈ 0.55 MB/sample — trivial once pooled. Do **not** keep unpooled hidden states.

**DINOv2 ceiling:** `AutoModel.from_pretrained("facebook/dinov2-base")`; image resized to a multiple of 14; mean of patch tokens inside `box_mask(box, (gh, gw))` — same helper.

**Probes (FR-T4-2)**
```python
def probe(X, y, groups=None, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv  = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    ctl  = cross_val_predict(clf, X, np.random.default_rng(seed).permutation(y), cv=cv, groups=groups)
    return pred, balanced_accuracy_score(y, pred), balanced_accuracy_score(y, ctl)    # selectivity = acc − ctl_acc
```
Train on **all** labeled samples (CV predictions), then report accuracy on (a) the error-cluster items and (b) matched correct items (`pred[idx_err] == y[idx_err]`). `decodable_err(site)` = Wilson **lower** bound of balanced accuracy on error items ≥ `probe.decodable_acc` and `acc − ctl_acc ≥ probe.selectivity`; `decodable_cor(site)` is the same on matched corrects. Group CV by `pair_id` so the two members of a pair never straddle train/test.

**Differential rule (C12).** Visual information vanishes at an intermediate layer for *every* input (the "information horizon", task- and model-dependent: 2512.07580) and moves from visual tokens into question tokens and then the last position (2411.18620). So (i) the per-layer flag fed to `probe_verdict` is `decodable_any[l] = box[l] or q[l] or last[l]`, and (ii) a loss counts toward H1/H2 **only if `decodable_cor` still holds at that site/layer**; a loss shared by errors and matched corrects is the generic horizon, and the verdict reads `generic horizon (not error-specific)`. Probe accuracy on a few dozen error items has a CI of about ±0.2 — report it and mark verdicts `underpowered` when the CI straddles `probe.decodable_acc`.

**Verdict (FR-T4-4)**
```python
def probe_verdict(d):           # d: {"dino":bool, "vit":bool, "merger":bool, "layers":{l:bool}}
    if not d["vit"]:   return "encoder/resolution limit" if d["dino"] else "label/too-small: check data"
    if not d["merger"]: return "H1 merger bottleneck"
    lost = [l for l in sorted(d["layers"]) if not d["layers"][l] and not any(d["layers"][m] for m in d["layers"] if m > l)]
    return f"H2 depth fading (lost at layer {lost[0]})" if lost else "H3 decodable throughout → readout"
```

The `t4` diagnostic consumed by `SIG` is the first token of the verdict (`H1`, `H2`, `H3`, `enc` for "encoder/resolution limit").

**Information-flow experiments (all via `knockout`)**
```python
def knock_effect(vlm, inp, pos, neg, spec, window):
    base = vlm.score_inp(inp, pos, neg)
    with knockout(vlm, inp, spec, window): s = vlm.score_inp(inp, pos, neg)
    return base - s                                           # margin drop = effect size
# sweep: for spec in [("vis","last"),("vis","q"),("q","last")]: for w in sliding windows of width 4 over the 36 layers
```
**Logit lens (FR-T4-6)**
```python
def lens(vlm, inp, pos, neg):
    with Tap(vlm) as t: vlm.m(**inp, logits_to_keep=1)
    norm, head = vlm.m.model.language_model.norm, vlm.m.lm_head
    return [vlm._s(head(norm(t.cap[i][0, -1])).float(), pos, neg).item() for i in sorted(t.cap)]
# invariant: lens[-1] == actual margin (tol 1e-2). Answer-emergence layer = first layer after which sign(s_l) never changes.
```
**Information horizon (FR-T4-7)** — reuses knockout:
```python
def horizon(vlm, inp, pos, neg, L, frac=.95):
    s_full  = score(inp);   s_blind = score_with(knockout(spec=[("vis","q"),("vis","last")], layers=range(L)))
    if abs(s_full - s_blind) < eps: return None              # no visual effect → excluded, counted
    for l in range(L + 1):                                    # block visual→text only from layer l onward
        s_l = score_with(knockout(spec=[("vis","q"),("vis","last")], layers=range(l, L)))
        if (s_l - s_blind) / (s_full - s_blind) >= frac: return l
```
Compare the horizon of safety tasks vs a generic VQA reference set (config path) — the plan's "per task type vs generic VQA". The horizon exists for every sample (2512.07580), so H2 is claimed only for a **significant difference** between a cluster's errors and its matched corrects (`boot_ci`).

**Activation patching (FR-T4-8)**
```python
def patch_scan(vlm, clean_inp, corr_inp, pos, neg, L, win=4, groups=("vis_box","vis_all","q","last"), box_mask_idx=None):
    assert clean_inp["input_ids"].shape == corr_inp["input_ids"].shape, "token lengths differ"
    with Tap(vlm) as t: s_clean = vlm.score_inp(clean_inp, pos, neg)
    clean = {i: h.clone() for i, h in t.cap.items()}; s_corr = vlm.score_inp(corr_inp, pos, neg)
    g = vlm.groups(corr_inp); P = dict(vis_box=g["vis"][box_mask_idx], vis_all=g["vis"], q=g["q"], last=g["last"])
    res = {}
    for pg in groups:
        for l in range(3, L - win + 1, win):                   # start ≥ 3: DeepStack caution
            patch = {i: (lambda h, i=i, pg=pg: h.index_copy(1, P[pg], clean[i][:, P[pg]])) for i in range(l, l + win)}
            with Tap(vlm, patch=patch): s_p = vlm.score_inp(corr_inp, pos, neg)
            res[(pg, l)] = (s_p - s_corr) / (s_clean - s_corr)  # 1 = fully restores the clean behaviour
    return res
```
- **Corruption** = the matched-counterfactual image (same scene, evidence removed/added); both images **resized to the identical size** so token counts match. Both directions: *denoising* (patch clean→corrupt, as above) and *noising* (swap roles).
- **Sanity check:** patching all layers at all positions ⇒ recovery ≈ 1.
- **Cost control:** 4 groups × ~8 windows ≈ 32 forwards per pair; run on ≤ 50 pairs per unexplained cluster.
- **Robustness before "causal" (C13).** Activation-patching conclusions depend on the corruption method and metric (2309.16042). A component is called causal only if the effect holds under ≥ 2 corruptions (`patch.corruptions`: matched counterfactual pair; gray/noise replacement of the box tokens) **and** both metrics (normalized logit-difference recovery; raw Δs), in both directions. Otherwise the evidence is `kind="causal"` for *none* and the label stays `unresolved`.

**Evidence rule (FR-T4-10)**
```python
def internal_label(evidence):    # list of dict(method, kind in {"decodability","causal"}, supports=label)
    by = {l: {e["kind"] for e in evidence if e["supports"] == l} for l in {e["supports"] for e in evidence}}
    for l, kinds in by.items():
        if {"decodability", "causal"} <= kinds: return l
    return "unresolved (single method)"
```

---

### 5.11 `stream.py` (P1)

**Step 1 — run the model once per step** (`plan_stream`): for video `v`, for `t` in `arange(1.0, duration, step_s)`:
`frames(v.media, fps, n=window, t_end=t)` → prompt `d0` → one row with `t` set. Everything below is offline.

**Step 2 — trigger policy on the margin series**
```python
def alerts(S, tau, k, n):                  # k-of-n persistence on calibrated decisions
    hit = (np.asarray(S) > tau).astype(int); roll = np.convolve(hit, np.ones(n), "full")[:len(S)]
    on = roll >= k; return on & ~np.r_[False, on[:-1]]          # rising edges = alert events
```
**Step 3 — decomposition (FR-ST-4)**
```python
def classify(ts, S, tau, onset, end, tol, k, n, ant=0.0):
    ev = ts[alerts(S, tau, k, n)]
    if onset is None: return "spam" if len(ev) else "ok"                          # no-risk clip
    early, inwin = ev[ev < onset - tol - ant], ev[(ev >= onset - tol - ant) & (ev <= end)]
    if len(inwin) == 0: return "premature" if len(early) else "silent"
    first = inwin.min()
    if first < onset - tol: return "anticipated"                                 # reachable only when ant > 0 (prediction use-case)
    if first - onset <= tol: return "ok"
    win_pos = ts[(np.asarray(S) > tau) & (ts >= onset - tol)]                      # window-level positives
    return "policy_late" if len(win_pos) and win_pos.min() < first - tol else "perception_late"
```
- **Detection vs prediction (C14, revised).** The project owner defines *prediction* as reporting a scene condition that could foreseeably cause an incident under a trigger (e.g. an open-lid container with no support that can fall in wind) — a property of a **single frame**, handled by §5.14, not by timing. `stream.anticipation_s` stays as an optional early-warning window (alerts in `[onset − tol − anticipation_s, onset − tol)` become `anticipated` instead of `premature`); default 0. For video hazards, `onset` means **risk onset** = first visible sign of the hazard condition (PaSBench-Video annotates frame-level risk onset and accident boundaries: 2606.02443), not the incident.
- **Window-level vs stream-level (H11):** window-level recall/FPR on the stored series (windows overlapping the violation interval = positive) vs stream-level metrics. A large gap ⇒ H11 evidence (`stream.policy` in `CAUSE`).
- **Spam rate** = alert events in dead time (before `onset−tol`, after `end+tol`, or in no-risk clips) per minute of dead time.
- **Asymmetric score** (ARS-style, constants in config): `miss → −c_miss`; `early → −c_early`; `on-time/late → exp(−(t_alert − onset)/decay_s)`.
- **Recovery/persistence:** use a T2 quality arm on a segment; count steps until `correct` returns.
- **Latency:** from the `latency_ms` column per knob config (p50/p95).
- **Header (FR-ST-8):** every streaming CSV starts with `# evidence boundary: frames ≤ t; history: window-recompute (N,fps); trigger: k-of-n`.

---

### 5.12 `report.py`

Assemble from CSVs: `clusters.json` + evidence rows ⇒ `root_causes.csv` (one row per cluster: size, failure types, T1 reading, tier explained, cause, evidence summary, #methods, implication, caveats, audit status) and `root_causes.md` (template with: summary table, counts/percent per cause, deployment-vs-architecture split, unexplained residue, per-cluster evidence, caveats from plan §10, protocol disclosure). Refuse `final` status when `audit_status != done` or `underpowered`. Print every cause with its `validate` precision and CI; label a cause `decisive` only if precision ≥ `validate.min_precision`, otherwise `candidate`. Counts/percentages are per **error** (most-specific-cluster rule, §5.8), not per overlapping cluster. Add the domain caveat (FR-D-7). **Accounting (C20):** the report prints a table in which every error sample has exactly one final state — `unique cause`, `ambiguous (candidates)`, `inconclusive`, `unexplained`, `untriaged (no item)`, `gated-out`, `invalid_format`, `underpowered`, `audit: label issue` — and the counts sum to the total number of errors (this is checked by a unit test). `root_causes.md` ends with a **Scope of evidence** section (FR-E-12).

### 5.13 `cli.py`
```
python -m ea selfcheck
python -m ea t0 | t1 | t2 | t3 | t4 | stream | report [--bench cs10k] [--limit N]
```
`argparse` subparsers, each: build job list → `run()` → call analysis → write `outputs/<tier>/*.csv`. No other options.

---

### 5.14 Latent-hazard (prediction) track (C19)

**Definition (project owner).** *Prediction* = the VLM reports a condition that is not yet a violation or incident but could foreseeably become one under a trigger. Example: an open-lid container that is not supported can fall in wind. This is a property of a single frame; no future frames are needed, so §5.11's timing metrics do not apply (video only: FR-H-7).

**Why it needs its own diagnostics.** A written-rule violation (no helmet) is perceive → compare to rule. A latent hazard is three steps: (1) perceive *conditions* — here two attributes (lid open, unsupported) bound to the same object (H8), (2) apply physical/world knowledge (unsupported open container + wind ⇒ falls), (3) decide, **without being told the trigger**, that it merits a report. The existing T1 triage is reused, with these meanings:

| triage role | prompt | content |
|---|---|---|
| Q_P (perception) | `hz_s_<k>` and `hz_s_<k>_abs` per condition `k` | "Is {obj} {state}?" / "Is {obj} NOT {state}?" (both polarities, as C7); `qp_ok` = all conditions, both polarities, correct (a conjunction: binding errors show up here) |
| Q_R (reasoning) | `hz_r` (text only) | "Facts: {conditions}. Trigger: {trigger}. Could {consequence} occur? Answer Yes or No." — physical/hazard reasoning from oracle conditions (the O2 analogue; exclusion test only, FR-T2-6) |
| Q_D (decision) | `hz_d` (**uncued**) | "Does this scene contain a condition that could foreseeably lead to an incident (for example something falling) under ordinary conditions such as wind, vibration or contact? Answer Yes or No." |
| cued arm | `hz_cue` | "Could {consequence} happen here under {trigger}? Answer Yes or No." — `HZ.cue` |

`hz_cue` leaks the consequence on purpose: it separates **cannot infer** from **does not volunteer**. Its control `HZ.cue_twin` is the same cued prompt on the matched *secured twin* scenes; the diagnostic `cue` is `True` only if cued recovery on the error cluster passes the exit rule **and** the twins' false-alarm rate stays ≤ `exit.control` (otherwise the cue just pushes answers to "Yes": H6/H7).

**Reading the outcomes** (all through the signature table, §5.8):

| `qp_ok` | `qr_ok` (`hz_r`) | `cue` | reading |
|---|---|---|---|
| False | any | any | perception/binding failure → crop/bgmask/hires diagnostics, then T4 |
| True | False | — | hazard/physics reasoning failure (`rule-reasoning failure` signature) |
| True | True | True | **uncued-hazard under-reporting** (new signature): the model can infer it, it does not volunteer it → prompt/threshold fix |
| True | True | False | readout/process (H3) candidate, needs crop/hires fail + T4 evidence |

**Data requirements** (also FR-H-1…6): `vtype="latent_hazard"`; `facts.conditions` = list of `{obj, state, support}`; `trigger`; `consequence`; `hazard_gold ∈ {0,1}`; optional ordinal `likelihood`/`severity`; **secured twins** (same object/scene type, hazard removed: lid closed or supported). Twins are mandatory because recall without false-positive measurement is meaningless: on PaSBench-Video recall is tightly coupled to false-positive rate and models "fire indiscriminately" where routine and hazardous scenes look alike (2606.02443).

**Metrics.** Recall of `hazard_gold=1` at the calibrated τ **and** false-alarm rate on twins; `pair_acc`/`pair_rank_acc` (hazard vs its twin) as the headline; slices by `trigger`, `consequence`, object class. `hz_report` (P1): under the deployed commenting prompt, generate ≤ 60 tokens and check, with a keyword/synonym checklist built from `facts`, that the text names object, condition and consequence (SafeBuild-Bench uses hazard identification and free-form hazard description tasks: 2608.00068). The checklist is lenient and is always followed by the human audit sample.

**Caveats to print.** (a) The trigger (wind) is not visible in the image, so the label rests on an assumed foreseeable condition; (b) hazard labels are subjective (likelihood/severity), so report annotator agreement and treat low-agreement clusters as `candidate` only (FR-H-6); (c) scene-level activity cues can imitate hazard reasoning (2606.02443): run the O3 scene-swap and T3 localization checks on hazard clusters; (d) no source in `references.md` defines this task — the construction is our own.

---

## 6. Hypothesis → recipe (one-page cheat sheet)

| Hyp. | Run | Key metric(s) | Conclude when… |
|---|---|---|---|
| H1 | `t2` O1 crop + bgmask + hires/token-budget; `t4` probes ViT vs merger | recovery (Wilson CI) vs controls and `N0.null`; probe acc by site, differential vs matched corrects | crop **and** hires recover, bgmask does not; then ViT-yes/merger-no |
| H2 | `t4` horizon, lens, knockout, token similarity | horizon layer; layer where decodability is lost (errors vs matched corrects) | merger-yes, lost at layer L **only in errors** + robust causal confirmation |
| H3 | `t1` triage; `t2` O1/hires; `t3` 3-way; `t4` patching | Q_P (both polarities) ok & Q_R ok & Q_D wrong; enrichment equal; **crop and hires do not recover** | process failure + robust patching (≥ 2 corruptions) restores answer |
| H4 | `t2` knobs | recovery under relaxed knob; ECE/AURC per knob | error vanishes at relaxed setting vs `N0.null` control |
| H5 | `t1` blank/swap/occlude; `t3` background mass, SpRAy (exploratory) | Visual Necessity; `bg` mass; enrichment | VN_err ≤ 0.25·VN_cor **and** low enrichment in both (after the perception/knowledge gate) |
| H6 | `t1` polarity prompts; slices on `vtype` | polarity consistency; yes-rate; absence vs presence gap at matched size | flips ≫ paraphrase noise **and** > matched corrects' flips (CI of the gap > 0), with `qp_ok` |
| H7 | `t1` framing + blank + occlude | framing shift; VN (blank **and** evidence-box occlusion); flip rate with blank baseline; **perception/knowledge gate passed** | big framing shift, or margin unchanged on blank and occlude (after the gate) |
| H8 | `t2` O5 + O1.bgmask | recovery vs random-person control; worker-count slices | per-person crop recovers, random-person does not; bgmask recovers while hires does not |
| H9 | `t2` O6 | quality curves; sharpen recovery vs `N0.null` | corrects break under mild degradation & errors recover (CI-based) |
| H10 | (P3) | accuracy vs CoT length | — |
| H11 | `stream` | window vs stream gap; decomposition | policy_late/premature/spam dominate with good window-level metrics |

---

## 7. Testing guide

### 7.1 Principle
Every analysis function is tested against synthetic data **with a planted truth**. The pipeline as a whole is tested with a `FakeVLM` that has known failure mechanisms. These tests (CPU, < 60 s) are the acceptance criteria for P0.

### 7.2 `FakeVLM` (tests/fake.py, ~60 lines)
Same interface as `VLM.score(text, images, frames, pos, neg)`. It reads the toy image:
- `evidence = red-square pixel area after downscaling to a fixed 'token grid' (e.g. 14×14 cells)` → detects the square only if it covers ≥ 1 cell. **Small objects are missed; crops (which enlarge the square relative to the frame) and higher resolution recover them; random crops and same-area resizes do not.**
- **Language prior:** for one rule class (`cls=prior_cls`, recognizable by a distinctive phrase in `rule_text`) the fake adds `+2` to `s` regardless of the image. Safe samples of that class become false alarms, and the margin is unchanged on the blank frame (Visual Necessity ≈ 0), which exercises `explain_necessity`.
- Blue border (absence scenes) + prompt text containing "wearing" (pol_pos) ⇒ `s −= 2` (affirmation bias); "NOT wearing"/"missing" prompts unaffected ⇒ **polarity inconsistency**.
- A fourth failure with **no** mechanism the interventions can touch (random wrong answers for one slice) ⇒ must stay **unexplained**.

### 7.3 Integration test (the four labels)
```python
def test_end_to_end(tmp_path):
    S = make_toy(400); vlm = FakeVLM()
    run_all_tiers(S, vlm, tmp_path)                 # t0 → t1 → t2 → escalate → report
    R = pd.read_csv(tmp_path / "root_causes.csv").set_index("cluster")
    assert R.loc["obj_size=small"].cause     == "pixel/token-budget limit (H1/H4)"   # hires recovers, bgmask does not
    assert R.loc["cls=prior_cls"].cause       == "prior-driven (H7)"
    assert R.loc["vtype=absence"].cause      == "polarity/framing bias (H6/H7)"
    assert R.loc["cls=rand_slice"].status    == "unexplained → next tier"
    assert R.loc["cls=clutter"].cause        == "clutter/binding/context (H8/H5)"      # crop + bgmask recover, hires does not
    assert R.loc["cls=label_noise"].status   == "unexplained → next tier"              # flipped gold: no confident cause
```

**Scope of this test (C15).** `FakeVLM` must expose the `max_pixels` knob (a finer token grid) so `hires` and `bgmask` are exercisable, plus a *clutter* slice (distractor squares that perturb `s`: crop and bgmask recover it, hires does not) and a *label-noise* slice. Passing proves the **implementation** matches the rules. It does **not** show that the rules identify real causes — that is what `validate` (§5.8b) measures on the real model (milestone M3b).

### 7.4 Unit-test checklist (each a few lines)
`ece` (calibrated ⇒ ≈0) · `aurc` (perfect ranking ⇒ minimum) · `recall_at_fpr` (known ROC) · `pair_acc ≤ min item acc` · `pair_rank_acc` for perfect/inverted/random · `polarity_consistency` (always-Yes) · `kl_bern` (identical ⇒ 0) · `boot_ci` (resamples ids; width ∝ 1/√n) · `slice_search` (planted slice found, null dataset ⇒ none, BH applied) · `triage` (all 8 combos) · `explain_recover` (rec/ctl thresholds, `min_n`) · `classify` (each streaming label) · `alerts` (k-of-n) · `frames` (causality) · `box_mask` (area fraction) · `loc` (uniform ⇒ enrichment ≈ 1) · `three_way` (3 branches) · `probe_verdict` (all branches) · `internal_label` (single method ⇒ unresolved) · resume (no duplicate keys) · matched controls exclude the error itself · `explain_recover` three-valued (CI straddle ⇒ `None`; `n < min_n` ⇒ `None`; worst control used) · `conf_ids` drops borderline errors and `valid=False` rows · `N0.null` ≠ base re-run (control recovery must be able to be > 0) · `decide` (unique / ambiguous / inconclusive / unexplained; `crop` only ⇒ unexplained) · `gate_ok` (necessity blocked when Q_P/Q_R fail) · `explain_consistency` uses the matched-correct gap · `classify` returns `anticipated` only when `ant > 0` · `validate` confusion: injected label noise never receives a confident cause · **accounting identity**: the per-state error counts sum to the total error count · `qr_ok` is computed once (Q_R and O2 are the same prompt).

### 7.5 GPU smoke tests (marker `gpu`)
`margin == manual logit difference` · `groups()` count equals `t*(h/2)*(w/2)` · determinism (|Δs| < 0.05) · `Tap` capture+identity patch leaves `s` unchanged · empty `knockout` leaves `s` unchanged · `knockout(vis→q, vis→last, all layers)` ≈ blank-image margin · `lens[-1] ≈ margin` · `attn_last` rows ≤ 1 · `x.shape[0] == 4·n_vis` for the ViT site.

---

### 7.6 Implementation-correctness protocol (C21; requirements FR-Q)

Every metric, diagnostic and tool gets a **verification card**: the tests below that apply to it, written *before or with* the code and passing before the step's checkpoint (§8). A tool that "runs and gives plausible numbers" is not done. Thresholds come from `config.yaml: tests`.

| Tool | Known-answer / anchor test | Property / invariant | Independent cross-check | Null / planted |
|---|---|---|---|---|
| `margin`, `mass` | `s == logit[Yes] − logit[No]` by hand; `mass ∈ (0,1]` | `s(pos,neg) = −s(neg,pos)`; adding a constant to all logits leaves `s` unchanged | — | — |
| Wilson / Bootstrap / BH | Wilson for (k,n) = (0,10), (10,10), (5,10) vs published values | CI ⊂ [0,1]; width shrinks ∝ 1/√n; BH monotone | Wilson vs `statsmodels.proportion_confint`; BH vs `statsmodels.multipletests`; bootstrap coverage ≈ nominal on simulated data | — |
| `ece`, `aurc`, `recall_at_fpr`, `pair_*` | hand-built ROC/bins | perfect ⇒ extremum; label permutation destroys signal | ROC/AUROC vs `sklearn.metrics`; binned ECE vs a naive loop | — |
| matched controls | error itself excluded; same-class | standardized mean difference of matching variables ≤ `smd_max` | — | — |
| slice search | planted slice recovered with correct support | no outcome-derived columns used; FDR q respected | Fisher/hypergeometric p vs `scipy.stats` | random labels ⇒ ≈ no significant slice |
| interventions (O1–O7, `bgmask`, `occlude`, `N0.null`, `degrade`) | crop box coordinates exact; `bgmask`: inside pixels **bit-identical**, outside constant; `occlude` = complement; `sametok` token count equals the crop's ±1 | random controls same size and **IoU = 0** with the evidence box; `N0.null` bytes differ but PSNR ≥ `psnr_null_min_db`; `degrade` PSNR monotone in severity | — | — |
| token-grid mapping | `n_vis == t·h/2·w/2` on the real processor; box→cells→box round trip | cells cover ≥ 1 token for tiny boxes | — | — |
| `explain_recover` / `explain_necessity` / `explain_consistency` / `gate_ok` / `conf_ids` | table-driven synthetic recoveries incl. CI-straddle ⇒ `None`, worst control used | monotone in recovery; borderline errors never counted | — | **null fake model ⇒ false-positive rate ≤ `null_fpr_max` over `null_runs` seeded runs; planted effect ⇒ true-positive rate ≥ `planted_tpr_min` at `min_n`** |
| `decide` / SIG | each signature fires alone; `crop` only ⇒ `unexplained` | exhaustive enumeration of the diagnostic domain: never raises, deterministic | compatibility matrix snapshot `tests/snapshots/sig_compat.txt` (changing SIG means updating it knowingly) | — |
| attention maps | rows sum to 1; sink-excluded maps renormalize to 1; non-negative | empty/blocked positions get weight exactly 0 (eager) | maps with `eager` vs `sdpa` agree where both available **[VERIFY]** | — |
| `grad_x_input` | finite-difference directional derivative of `s` agrees within `finite_diff_rtol` | sign flips with `s` | — | — |
| deletion/insertion, explainer gate | GT-box map beats random order; inverted map reverses the order | rankings stable across fills | — | randomized-weights/labels ⇒ gate fails |
| probes | linearly separable synthetic features ⇒ ≥ 0.95; group CV by `pair_id` disjoint | label-shuffle accuracy ≈ chance | `LogisticRegression` vs closed form on a 2-D toy | no decodable signal ⇒ `underpowered`/not decodable |
| `knockout` | empty set ⇒ `s` identical to base | blocked attention weights exactly 0; hooks leave no state (second base run identical) | — | — |
| logit lens | final-layer lens == model logits | — | — | — |
| activation patching | `patch(clean→clean)` ⇒ no change; patch **everything** ⇒ counterfactual output (recovery 1); patch **nothing** ⇒ 0 | normalized recovery ∈ [0,1] on these anchors | — | — |
| forward passes | batch vs single forward equal within tolerance; two runs equal (determinism) | — | — | — |
| streaming `classify` and metrics | hand-built series covering every class (`ok/anticipated/silent/late/premature/spam`) | **causality:** perturbing future frames leaves earlier rows unchanged | — | — |
| `validate` | injected label noise never gets a confident cause; confusion matrix sums to the injected count | — | — | — |
| accounting | per-state error counts sum to the number of errors | — | — | — |

**Mutation checks (`tests/test_mutation.py`).** For each decision predicate (`explain_recover`, `explain_necessity`, `explain_consistency`, `gate_ok`, `conf_ids`, `decide`, `classify`) a parametrized test applies one deliberate bug (flip an inequality, drop the control, drop the gate, use the base re-run as control) with `monkeypatch` and asserts that **at least one** other test then fails. If none fails, the test suite is too weak and a test is added.

**Confound list per diagnostic.** Each diagnostic's docstring ends with *"Measures: … Does not measure: … Other explanations: …"* (e.g. `crop` recovery: resolution, clutter, context, label leak). The coder must not shorten it.

**Limits.** These tests establish that the **code does what the definitions say** and that diagnostics have measurable false-positive/true-positive rates on the fake model. They do not show that the diagnostics identify real causes in Qwen3-VL — that is `validate` (§5.8b).

---

## 8. Implementation order (stop at each checkpoint and run the listed check)

| Step | Build | Checkpoint |
|---|---|---|
| 1 | `data.py` (+`make_toy`), `prompts.py` | toy data renders; every prompt formats |
| 2 | `model.py` core (`VLM`, `prep`, `ids`, `score`, `groups`), `selfcheck` | GPU smoke tests 1–3; selfcheck prints config facts |
| 3 | `run.py` (T0 plan, runner, resume), `metrics.py`, `slices.py` | T0 on toy data with real model; metric unit tests; planted slice found |
| 4 | `triage.py`, `interventions.py` (O1–O6), T1/T2 plans | rows for all arms; wrong-facts control ≈ 0 recovery |
| 5 | `escalate.py`, `report.py`, `tests/fake.py` | **end-to-end test passes (four labels)** |
| 6 | Real adapters (as access arrives), knob grids from the checkpoint config | T0 on a real benchmark |
| 6b | `INJ.*`, `validate` (needs the real model) | confusion matrix printed; the `INJ.label` row has no confident cause; precision per cause recorded |
| 7 | `stream.py` | causal run on one video; decomposition tests |
| 8 | `attrib.py` | gate runs; planted-confounder test |
| 9 | `mech.py` (`Tap`, `knockout`, lens → probes → horizon → patching) | invariants 5–9 of §7.5; evidence-rule test |

---

## 9. Pitfalls checklist (read before coding)

1. **Threshold mix-ups.** Native (`s>0`) for errors/consistency/recovery; calibrated (`s>τ`) only for operating points and alerts. Never mix them in one table.
2. **Gold under interventions.** Always the original sample's gold. Controls are *expected* to show low recovery; wrong-facts is expected to *follow the wrong facts*.
3. **Polarity prompts.** `pos` must be the *violation* answer word. A wrong `pos` silently inverts those metrics — covered by the polarity unit test.
4. **Bootstrap unit.** Resample `sample_id`, not rows. Same for Fisher tests: one row per sample.
5. **Matched controls must change nothing but the intervention.** Same-token resize must match the crop's *area*; verify via `n_vis_tokens` (±5%) in the control rows.
6. **Crops change token counts** — always interpret O1 together with the same-token control (plan §7 caveat).
7. **Sink exclusion.** Any attention-mass number must exclude sink tokens and renormalize. Record the sink fraction in the output.
8. **DeepStack hooks.** Layer-output hooks on layers 0–2 miss the injection; patch from layer 3. Gradient saliency includes the DeepStack features (do not forget them).
9. **Token order.** Index `i` of visual features = `vis[i]` = row-major grid cell; pre-merger sites are 4 consecutive vectors per merged token. Both are asserted in `selfcheck`.
10. **Odd video frame counts.** Pad to even (temporal patch of 2). Preserve timestamp text across knobs.
11. **Eager vs sdpa.** T0–T2 use sdpa (speed). T3/T4 use eager (hooks). Results must agree to ±0.05 on `s`; test once and note it in the manifest.
12. **bf16 noise.** Treat |Δs| < 0.05 as no change. Do not interpret smaller margin differences.
13. **No hidden-state storage.** Pool in the hook; store float16 vectors only.
14. **Missing data.** Skip and count, never crash or impute silently.
15. **Don't "improve" the plan.** If something looks wrong in the plan, add it to the report's caveats and ask the human; do not silently change a definition.
16. **A deterministic re-run is not a control** (it recovers exactly 0). Use `N0.null`.
17. **Never decide recovery on borderline errors** (`|s_base| < conf_margin`); they flip under any perturbation.
18. **Equal enrichment ≠ H3.** Correct localization with a wrong answer is also what resolution-limited small-detail failures look like (2502.17422); H3 needs `crop=False ∧ hires=False`.
19. **O2 never names perception.** It only excludes rule-reasoning failure.
20. **"Prior-driven" only after the perception/knowledge gate** (`gate_ok`; 2406.08702).
21. **Check `mass`** before trusting any margin; flag `invalid_format` rows.
22. **Probe/horizon losses are only error-specific if matched corrects stay decodable** (information horizon is generic: 2512.07580).
23. **Never print "decisive"** unless `validate` supports it.
24. **Latent hazards: always run the cued vs uncued comparison and the twin false-alarm check.** Recall without twins is uninterpretable, and a cue that raises "Yes" on twins is bias, not recovery.
25. **Brevity is never a reason to skip a test, control, validity gate or input check.** A tool without its §7.6 verification card is not finished.

---

## 10. Rules for the coding agent

1. Implement in the order of §8. Do not start step *n+1* until step *n*'s checkpoint passes.
2. Do not add classes where a function works; do not add abstraction layers; do not add configuration options not listed in `config.yaml`.
3. Do not add a dependency outside `requirements.md` §3.2. Do not reimplement anything listed in §3.4 "Do not build".
4. A module may exceed its planning budget when correctness requires it (§0.1). Log it in `DEVIATIONS.md`. Beyond +50 %, first check whether a function can go **without removing a check, control, test or validity gate**; if not, continue and record why. Never trade a check for lines.
5. Every function that implements a requirement carries a `# FR-xxx` comment; every test names the requirement it covers.
6. When an assumption in `requirements.md` §9 fails, **stop and report the actual value** (print the module tree / config) instead of guessing.
7. Never fabricate benchmark formats. If a benchmark is not accessible, leave the adapter as a documented stub and use `make_toy` for development.
8. Numbers in the report must come from `runs.jsonl` only. No hand-entered results.

9. Never emit a cause as **decisive** unless `validate` reports precision ≥ `validate.min_precision` for it; otherwise print `candidate` with the validated precision and CI.
10. If more than one signature matches, print `ambiguous` with the candidate list. Never pick one.
11. **A step is finished only when its tools' verification cards (§7.6) pass** — known-answer, cross-check, null/planted, anchor and mutation tests — not when the code runs.
12. Record every budget overrun, departure from a definition and failed [VERIFY] item in `DEVIATIONS.md` with the actual values.