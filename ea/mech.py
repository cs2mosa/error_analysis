"""mech.py — T4 mechanistic analysis: probes, knockout, logit lens, patching (FR-T4-1..10).

Purpose:  FR-T4-1: cached pooled site features (ViT, merger, DeepStack, LLM layers).
          FR-T4-2: linear probes per site/layer (balanced accuracy + selectivity).
          FR-T4-3: vision-only encoder ceiling (DINOv2).
          FR-T4-4: probe_verdict decision rule (encoder limit / H1 / H2 / H3).
          FR-T4-5: attention knockout sweep over layer windows.
          FR-T4-6: logit lens per layer.
          FR-T4-7: information horizon per sample/cluster.
          FR-T4-8: activation patching (counterfactual denoising/noising).
          FR-T4-9: visual-token similarity across depth.
          FR-T4-10: evidence rule (>= 2 methods: decodability + causal).
Implements: design §5.10, requirements §4.10.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ea.data import box_mask


# ----------------------------------------------------------------- FR-T4-1 feature caching

def cache_sample(vlm, s, prompt_text: str, box: list[float] | tuple[float, ...], out_dir: str | Path):
    """Cache pooled site features for one sample to disk (FR-T4-1; design §5.10)."""
    from ea.model import Tap, per_token
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    inp = vlm.prep(prompt_text, images=[Image.open(s.media)])
    g = vlm.groups(inp)
    n = g["vis"].numel()
    grid = g["grid"]
    gh, gw = (grid[1], grid[2]) if len(grid) == 3 else grid
    m_np = box_mask(box, (gh, gw))
    M = torch.from_numpy(m_np.ravel()).to(g["vis"].device)

    with Tap(vlm, vit=True) as t:
        vlm.m(**inp, logits_to_keep=1)

    f = {}
    for k, v in t.site.items():
        pooled = per_token(k, v, n)
        if M.any() and pooled.shape[0] == M.shape[0]:
            f[k] = pooled[M].mean(0)
        else:
            f[k] = pooled.mean(0)

    for i, h in t.cap.items():
        h_vis = h[0, g["vis"]]
        if M.any() and h_vis.shape[0] == M.shape[0]:
            f[f"L{i}_vis"] = h_vis[M].mean(0)
        else:
            f[f"L{i}_vis"] = h_vis.mean(0)
        f[f"L{i}_last"] = h[0, -1]
        f[f"L{i}_q"] = h[0, g["q"]].mean(0)

    np.savez(out_dir / f"{s.id}.npz", **{k: v.float().cpu().numpy().astype(np.float16) for k, v in f.items()})


# ----------------------------------------------------------------- FR-T4-3 vision-only ceiling

_DINO = {"m": None}


def dinov2_pool(img: "Image.Image", box: list[float], out_dim: int = 768) -> np.ndarray | None:
    """DINOv2 encoder-ceiling probe features (FR-T4-3): mean patch-token vector
    inside the box on the DINOv2 patch grid (patch 14). Returns None (counted
    skip by the caller) when the weights are unavailable offline."""
    import zlib
    import torch
    from PIL import Image as I
    if _DINO["m"] is None:
        try:
            from transformers import AutoModel
            _DINO["m"] = AutoModel.from_pretrained("facebook/dinov2-base")
        except Exception as e:
            print(f"[mech] dinov2 unavailable (offline?): {e!r} — encoder-ceiling "
                  "probe skipped and counted (NFR-5)")
            _DINO["m"] = False
    if _DINO["m"] is False:
        return None
    m = _DINO["m"].eval()
    w, h = img.size
    s = max(14, min(518, (min(w, h) // 14) * 14))
    im = img.convert("RGB").resize((s, s))
    with torch.no_grad():
        pt = m(torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float()).last_hidden_state[0]
    gh = gw = s // 14
    x0, y0, x1, y1 = box
    mm = np.zeros((gh, gw), bool)
    mm[int(np.floor(y0 * gh)):max(int(np.ceil(y1 * gh)), int(np.floor(y0 * gh)) + 1),
       int(np.floor(x0 * gw)):max(int(np.ceil(x1 * gw)), int(np.floor(x0 * gw)) + 1)] = True
    pooled = pt.reshape(gh, gw, -1)[mm].mean(0)
    return pooled.numpy().astype(np.float32)


# ----------------------------------------------------------------- FR-T4-2 & FR-T4-3 probes

def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, seed: int = 0):
    """Linear probe with 5-fold CV and permutation control (FR-T4-2; design §5.10).

    Returns:
      pred: cross-validated predictions
      acc: balanced accuracy score
      ctl_acc: balanced accuracy on permuted labels (selectivity = acc - ctl_acc)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    # Check minimum samples per class for StratifiedKFold
    n_classes = len(np.unique(y))
    if n_classes < 2 or len(y) < 10:
        # Fallback for tiny toy datasets
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = float(balanced_accuracy_score(y, pred))
        return pred, acc, 0.5

    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = float(balanced_accuracy_score(y, pred))
        return pred, acc, 0.5

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)

    pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    rng = np.random.default_rng(seed)
    ctl_y = rng.permutation(y)
    ctl = cross_val_predict(clf, X, ctl_y, cv=cv, groups=groups)

    acc = float(balanced_accuracy_score(y, pred))
    ctl_acc = float(balanced_accuracy_score(ctl_y, ctl))
    return pred, acc, ctl_acc


# ----------------------------------------------------------------- FR-T4-4 probe verdict

def probe_verdict(d: dict) -> str:
    """Implement T4 decision rule from site decodability map (FR-T4-4; design §5.10).

    d: {"dino": bool, "vit": bool, "merger": bool, "layers": {layer_idx: bool}}
    """
    if not d.get("vit", False):
        return "encoder/resolution limit" if d.get("dino", False) else "label/too-small: check data"
    if not d.get("merger", False):
        return "H1 merger bottleneck"

    layers = d.get("layers", {})
    lost = [l for l in sorted(layers) if not layers[l] and not any(layers[m] for m in layers if m > l)]
    if lost:
        return f"H2 depth fading (lost at layer {lost[0]})"
    return "H3 decodable throughout -> readout"


def t4_tag(verdict: str) -> str:
    """Extract standard t4 diagnostic key consumed by SIG ('H1'|'H2'|'H3'|'enc')."""
    if verdict.startswith("H1"):
        return "H1"
    if verdict.startswith("H2"):
        return "H2"
    if verdict.startswith("H3"):
        return "H3"
    if verdict.startswith("encoder"):
        return "enc"
    return "unresolved"


# ----------------------------------------------------------------- FR-T4-5 knockout sweep

def knock_effect(vlm, inp, pos=("Yes",), neg=("No",), spec=(("vis", "last"),), window=(0, 4)) -> float:
    """Attention knockout effect size = base margin - knocked margin (FR-T4-5)."""
    from ea.model import knockout
    base = vlm.score_inp(inp, pos, neg)
    with knockout(vlm, inp, spec, range(window[0], window[1])):
        s = vlm.score_inp(inp, pos, neg)
    return float(base - s)


# ----------------------------------------------------------------- FR-T4-6 logit lens

def lens(vlm, inp, pos=("Yes",), neg=("No",)) -> list[float]:
    """Read margin from each LLM layer's last-token state through final norm & head (FR-T4-6)."""
    from ea.model import Tap
    with Tap(vlm) as t:
        vlm.m(**inp, logits_to_keep=1)
    m = getattr(vlm.m, "model", vlm.m)
    lm = getattr(m, "language_model", m)
    norm = getattr(lm, "norm", None)
    head = getattr(vlm.m, "lm_head", None)
    if norm is None or head is None:
        return []
    res = []
    for i in sorted(t.cap):
        last_tok = t.cap[i][0, -1]
        lg = head(norm(last_tok)).float()
        res.append(float(vlm._s(lg, pos, neg).item()))
    return res


# ----------------------------------------------------------------- FR-T4-7 information horizon

def horizon(vlm, inp, pos=("Yes",), neg=("No",), L: int = 36, frac: float = 0.95) -> int | None:
    """Information-horizon layer: smallest layer where blocking visual flow retains >= frac of effect (FR-T4-7)."""
    from ea.model import knockout
    s_full = vlm.score_inp(inp, pos, neg)
    with knockout(vlm, inp, [("vis", "q"), ("vis", "last")], range(L)):
        s_blind = vlm.score_inp(inp, pos, neg)
    if abs(s_full - s_blind) < 1e-4:
        return None
    for l in range(L + 1):
        with knockout(vlm, inp, [("vis", "q"), ("vis", "last")], range(l, L)):
            s_l = vlm.score_inp(inp, pos, neg)
        if (s_l - s_blind) / (s_full - s_blind) >= frac:
            return l
    return L


# ----------------------------------------------------------------- FR-T4-8 activation patching

def patch_scan(vlm, clean_inp, corr_inp, pos=("Yes",), neg=("No",), L: int = 36,
               win: int = 4, groups=("vis_box", "vis_all", "q", "last"),
               box_mask_idx: torch.Tensor | None = None) -> dict[tuple[str, int], float]:
    """Activation patching normalized logit-difference recovery (FR-T4-8; design §5.10)."""
    from ea.model import Tap
    assert clean_inp["input_ids"].shape == corr_inp["input_ids"].shape, "token lengths differ"
    with Tap(vlm) as t:
        s_clean = vlm.score_inp(clean_inp, pos, neg)
    clean = {i: h.clone() for i, h in t.cap.items()}
    s_corr = vlm.score_inp(corr_inp, pos, neg)

    if abs(s_clean - s_corr) < 1e-4:
        return {(pg, l): 0.0 for pg in groups for l in range(3, L - win + 1, win)}

    g = vlm.groups(corr_inp)
    P = dict(
        vis_box=g["vis"][box_mask_idx] if box_mask_idx is not None else g["vis"],
        vis_all=g["vis"],
        q=g["q"],
        last=g["last"],
    )
    res = {}
    for pg in groups:
        for l in range(3, L - win + 1, win):
            def _make_patch(layer_idx, p_idx):
                return lambda h: h.index_copy(1, P[p_idx], clean[layer_idx][:, P[p_idx]])

            patch_dict = {i: _make_patch(i, pg) for i in range(l, l + win)}
            with Tap(vlm, patch=patch_dict):
                s_p = vlm.score_inp(corr_inp, pos, neg)
            res[(pg, l)] = float((s_p - s_corr) / (s_clean - s_corr))
    return res


# ----------------------------------------------------------------- FR-T4-9 token similarity

def visual_token_similarity(h_layers: dict[int, torch.Tensor], vis_idx: torch.Tensor) -> dict[int, float]:
    """Mean pairwise cosine similarity of visual tokens per layer across depth (FR-T4-9)."""
    sims = {}
    for l, h in h_layers.items():
        v = h[0, vis_idx].float()  # [Nv, D]
        norm = torch.nn.functional.normalize(v, p=2, dim=-1)
        sim_mat = torch.mm(norm, norm.t())
        n = sim_mat.shape[0]
        if n > 1:
            # Off-diagonal mean
            diag_mask = ~torch.eye(n, dtype=torch.bool, device=sim_mat.device)
            sims[l] = float(sim_mat[diag_mask].mean().item())
        else:
            sims[l] = 1.0
    return sims


# ----------------------------------------------------------------- FR-T4-10 evidence rule

def internal_label(evidence: list[dict]) -> str:
    """Evidence rule enforced in code: internal root-cause label emitted only when
    >= 2 independent methods agree (>= 1 decodability + >= 1 causal) (FR-T4-10).

    Each evidence item: dict(method=..., kind='decodability'|'causal', supports='H1'|'H2'|'H3')
    """
    labels = {e["supports"] for e in evidence if "supports" in e}
    by = {l: {e["kind"] for e in evidence if e.get("supports") == l} for l in labels}
    for l in sorted(by):
        kinds = by[l]
        if {"decodability", "causal"} <= kinds:
            return l
    return "unresolved (single method)"
