"""model.py — the ONLY module allowed to import transformers / touch model
internals (GC-2). Exposes a small API (~10 functions) to everything else.

Purpose:  load Qwen3-VL (FR-M-1), compute the margin s and answer mass (FR-M-2),
          expose the visual-token grid (FR-M-3), knobs + latency/memory
          metadata (FR-M-4), determinism tolerance (FR-M-5). Hook tools
          (Tap, knockout, attn_last, grad_x_input, sink_mask; FR-M-6..11)
          live here too but are introduced with T3/T4 (design §8 steps 8-9).
Implements plan sections: §6.1 (margin), §4.2 (model adapter).
Inputs:   rendered prompt text + PIL images / video frames.
Outputs:  dict(s=, mass=, n_vis=, ms=, mem=) per call; token-position groups.

[VERIFY] items below are pinned against transformers 5.17 + Qwen3-VL config
(patch 16, merge 2, deepstack_visual_indexes [5,11,17], 36 LLM layers,
hidden 2560) and asserted by `python -m ea selfcheck`.
"""

import time
from contextlib import contextmanager
from functools import lru_cache

import torch
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen3VLForConditionalGeneration)

# ------------------------------------------------------------ loading (FR-M-1)


class VLM:
    """Qwen3-VL wrapper: eval mode, greedy readout, batch size 1 (FR-M-1).

    quant: None | "8bit" | "4bit" (bitsandbytes). attn: "sdpa" for T0-T2,
    "eager" for T3/T4 (hooks; design §5.3). device_map defaults to the first
    CUDA device and falls back to CPU (justified deviation: 6 GB GPU cannot
    hold the bf16 weights; logged in DEVIATIONS.md).
    """

    def __init__(self, name, quant=None, min_pixels=None, max_pixels=None,
                 attn="sdpa", device_map=None):
        qc = {"8bit": BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"]),
              "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)}.get(quant)
        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device_map,
            quantization_config=qc, attn_implementation=attn).eval()   # [VERIFY] dtype= (5.x renamed torch_dtype)
        self.p = AutoProcessor.from_pretrained(name)
        self.attn = attn
        self._ids_cache: dict[tuple, list[int]] = {}
        if max_pixels or min_pixels:                        # FR-M-4 knob: resolution
            self.set_pixels(min_pixels, max_pixels)

    def set_pixels(self, min_pixels=None, max_pixels=None):
        ip = self.p.image_processor                         # [VERIFY] key names: size dict, verified in selfcheck
        if min_pixels is not None:
            ip.min_pixels = int(min_pixels)
            ip.size["shortest_edge"] = int(min_pixels)
        if max_pixels is not None:
            ip.max_pixels = int(max_pixels)
            ip.size["longest_edge"] = int(max_pixels)

    # ------------------------------------------------------------ prep

    def prep(self, text, images=None, frames=None):
        """PIL images / frame list -> model inputs (chat template + pixels)."""
        kind = [{"type": "image"}] * len(images or []) + ([{"type": "video"}] if frames else [])
        msgs = [{"role": "user", "content": kind + [{"type": "text", "text": text}]}]
        prompt = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.p(text=[prompt], images=images or None,
                      videos=[frames] if frames else None,
                      return_tensors="pt").to(self.m.device)   # [VERIFY] video fps/timestamps kwargs (selfcheck 7)

    # ------------------------------------------------------------ answer ids (FR-M-2)

    def ids(self, words):
        """Single-token ids for answer-word variants; drop multi-token, raise if empty.
        Instance cache (BUG-20 fix): avoids keeping self alive in global lru_cache."""
        key = tuple(words)
        if key in self._ids_cache:
            return self._ids_cache[key]
        out = {t[0] for w in words if len(t := self.p.tokenizer.encode(w, add_special_tokens=False)) == 1}
        assert out, f"no single-token variant for {words}"
        res = sorted(out)
        self._ids_cache[key] = res
        return res

    def _s(self, lg, pos, neg):
        """Margin s = logsumexp(logit[pos]) - logsumexp(logit[neg]) (§6.1)."""
        return (torch.logsumexp(lg[self.ids(tuple(pos))], 0)
                - torch.logsumexp(lg[self.ids(tuple(neg))], 0))

    # ------------------------------------------------------------ scoring (FR-M-2)

    def score_inp(self, inp, pos=("Yes",), neg=("No",)):
        """Forward on prepared inputs -> float margin. Used by T3/T4 so inputs
        are prepared once and hooks wrap the call (design §5.3)."""
        lg = self.m(**inp, logits_to_keep=1).logits[0, -1].float()
        return self._s(lg, pos, neg).item()

    @torch.no_grad()
    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        """One forward pass -> margin s, answer mass, token counts, latency, memory."""
        inp = self.prep(text, images, frames)
        t0 = time.perf_counter()
        lg = self.m(**inp, logits_to_keep=1).logits[0, -1].float()
        if self.m.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        mem = (torch.cuda.max_memory_allocated() / 2**30
               if self.m.device.type == "cuda" else 0.0)
        return dict(s=self._s(lg, pos, neg).item(),
                    mass=float(torch.softmax(lg, 0)[self.ids(tuple(pos)) + self.ids(tuple(neg))].sum()),
                    n_vis=int(self.groups(inp)["vis"].numel()), ms=ms, mem=mem)

    # ------------------------------------------------------------ token groups / grid (FR-M-3)

    def groups(self, inp):
        """Positions of visual / question / last tokens and the merged grid.

        Index i of any per-visual-token tensor <-> vis[i] <-> grid cell i
        (row-major per image, time-major per video) — the single fact that
        makes every spatial analysis certain (design §5.3).
        """
        ids = inp["input_ids"][0]
        if "image_grid_thw" in inp:
            key, vid = "image_grid_thw", self.m.config.image_token_id
        elif "video_grid_thw" in inp:
            key, vid = "video_grid_thw", self.m.config.video_token_id
        else:
            key, vid = None, None
        last = ids.numel() - 1
        if key is None:
            return dict(vis=ids[:0], q=torch.arange(0, last, device=ids.device),
                        last=torch.tensor([last], device=ids.device), grid=None)
        vis = (ids == vid).nonzero().squeeze(1)
        t, h, w = inp[key][0].tolist()
        return dict(vis=vis, q=torch.arange(int(vis.max()) + 1, last, device=ids.device),
                    last=torch.tensor([last], device=ids.device), grid=(t, h // 2, w // 2))


@contextmanager
def mask_tokens(vlm, keep_frac, seed=0):
    """Token-budget knob (random pruning, FR-T2-3).

    Zeroes visual features of a random subset at the visual-encoder merger
    output, removing information while keeping MRoPE positions intact.
    """
    def hook(mod, a, out):
        g = torch.Generator().manual_seed(int(seed))
        if isinstance(out, tuple):
            merged = out[0]
            drop = (torch.rand(merged.shape[0], generator=g) > keep_frac).to(merged.device)
            merged = merged.masked_fill(drop[:, None], 0)
            ds = [d.masked_fill(drop[:, None], 0) if isinstance(d, torch.Tensor) else d for d in out[1:]]
            return (merged, *ds)
        elif isinstance(out, torch.Tensor):
            drop = (torch.rand(out.shape[0], generator=g) > keep_frac).to(out.device)
            return out.masked_fill(drop[:, None], 0)
        return out

    if not hasattr(vlm, "m"):
        # Duck-typed fake/mock VLM in unit tests (e.g. _CorrectVLM)
        yield
        return

    target = getattr(getattr(getattr(vlm, "m", vlm), "visual", None), "merger",
                     getattr(getattr(vlm, "m", vlm), "visual", None))
    if target is None:
        raise RuntimeError("mask_tokens: visual merger / visual module not found on model (FR-T2-3)")
    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ------------------------------------------------------------ hooks & explainers (FR-M-6..11)

class Tap:
    """Context manager. llm: capture decoder-layer outputs. vit: capture visual sites.
       patch: {layer: fn(h)->h} replaces layer output (activation patching)."""
    def __init__(self, vlm, vit=False, patch=None):
        self.v, self.vit, self.patch = vlm, vit, patch or {}
        self.cap, self.site, self.hk = {}, {}, []

    def __enter__(self):
        m = getattr(self.v.m, "model", self.v.m)
        lm = getattr(m, "language_model", m)
        layers = getattr(lm, "layers", [])
        for i, L in enumerate(layers):
            self.hk.append(L.register_forward_hook(self._layer(i)))
        if self.vit and hasattr(m, "visual"):
            if hasattr(m.visual, "merger"):
                self.hk.append(m.visual.merger.register_forward_pre_hook(lambda mod, a: self.site.__setitem__("vit", a[0].detach())))
                self.hk.append(m.visual.merger.register_forward_hook(lambda mod, a, o: self.site.__setitem__("merger", o.detach())))
            if hasattr(m.visual, "deepstack_merger_list"):
                for k, dm in enumerate(m.visual.deepstack_merger_list):
                    self.hk.append(dm.register_forward_pre_hook(lambda mod, a, k=k: self.site.__setitem__(f"ds{k}", a[0].detach())))
        return self

    def _layer(self, i):
        def f(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            self.cap[i] = h.detach()
            if i in self.patch:
                h2 = self.patch[i](h)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return f

    def __exit__(self, *a):
        for h in self.hk:
            h.remove()


def per_token(site, x, n_vis):
    """Pre-merger sites: 4 consecutive vectors = 1 merged token."""
    return x.view(n_vis, -1, x.shape[-1]).mean(1) if site != "merger" else x


class KO:
    """Historical container for knockout metadata; kept for backwards compatibility."""
    layers: set = set()
    blocks: list = []


@contextmanager
def knockout(vlm, inp, spec, layers):
    """Knockout attention paths between groups (spec) across layers (FR-M-8).
    Re-entrant (BUG-20 fix): closure variables per context invocation, no global clobbering."""
    g = vlm.groups(inp)
    target_layers = set(layers)
    blocks = [(g[a], g[b]) for a, b in spec]
    hk = []
    m = getattr(vlm.m, "model", vlm.m)
    lm = getattr(m, "language_model", m)
    layers_list = getattr(lm, "layers", [])
    for i, L in enumerate(layers_list):
        def pre(mod, args, kwargs, i=i):
            if i not in target_layers:
                return
            mask = kwargs.get("attention_mask")
            if mask is None:
                # BUG-07 fix: a silent return would make the knockout a no-op
                # and report a fake "no causal role" effect of exactly zero.
                raise RuntimeError(
                    "knockout: self-attention received attention_mask=None "
                    "(sdpa path?). Load the model with attn='eager' for "
                    "knockout/horizon/knock_effect — a silent no-op must never "
                    "be mistaken for 'no causal effect' (FR-M-8, selfcheck 6).")
            if mask.dim() != 4:
                raise RuntimeError(f"knockout: expected a 4-D attention mask, "
                                   f"got {mask.dim()}-D (attn='eager' required)")
            mask = mask.clone()
            neg = torch.finfo(mask.dtype).min if mask.dtype.is_floating_point else False
            for frm, to in blocks:
                mask[:, :, to[:, None], frm[None, :]] = neg
            kwargs["attention_mask"] = mask
            return args, kwargs
        if hasattr(L, "self_attn"):
            hk.append(L.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in hk:
            h.remove()


def attn_last(vlm, inp):
    """Last-token attention row per layer -> [layers, heads, T_all] (FR-M-9)."""
    if getattr(vlm, "attn", "eager") != "eager":
        # BUG-08 fix: under sdpa the attention module returns o[1] = None;
        # fail with the fix in the message instead of a TypeError deep inside.
        raise ValueError("attn_last requires an eager-attention model instance "
                         "(VLM(..., attn='eager')); sdpa does not materialize "
                         "attention weights")
    out, hk = {}, []
    m = getattr(vlm.m, "model", vlm.m)
    lm = getattr(m, "language_model", m)
    layers_list = getattr(lm, "layers", [])
    for i, L in enumerate(layers_list):
        if hasattr(L, "self_attn"):
            hk.append(L.self_attn.register_forward_hook(
                lambda mod, a, kw, o, i=i: out.__setitem__(i, o[1][0, :, -1, :].float().cpu()),
                with_kwargs=True))
    with torch.no_grad():
        vlm.m(**inp, logits_to_keep=1)
    for h in hk:
        h.remove()
    return torch.stack([out[i] for i in sorted(out)])


def sink_mask(h_vis, ratio):
    """h_vis: [Nv, D] hidden states of visual tokens at a middle layer (FR-M-11)."""
    m = h_vis.abs().amax(-1)
    return m > ratio * m.median()


def grad_x_input(vlm, inp, pos, neg):
    """Gradients flow to features (merged + deepstack) -> [Nv] (FR-M-10)."""
    feats = []
    def hook(mod, a, out):
        if isinstance(out, tuple):
            merged, ds = out[0], out[1:]
            merged = merged.detach().requires_grad_(True)
            ds = [d.detach().requires_grad_(True) if isinstance(d, torch.Tensor) else d for d in ds]
            feats.extend([f for f in [merged, *ds] if isinstance(f, torch.Tensor)])
            return (merged, *ds)
        elif isinstance(out, torch.Tensor):
            t = out.detach().requires_grad_(True)
            feats.append(t)
            return t
        return out

    m = getattr(vlm.m, "model", vlm.m)
    vis = getattr(m, "visual", None)
    hk = vis.register_forward_hook(hook) if vis is not None else None
    with torch.enable_grad():
        lg = vlm.m(**inp, logits_to_keep=1).logits[0, -1].float()
        s = vlm._s(lg, pos, neg)
        s.backward()
    if hk:
        hk.remove()
    return sum((f.grad * f).abs().sum(-1) for f in feats if f.grad is not None).detach().float().cpu()


