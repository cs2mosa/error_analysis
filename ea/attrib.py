"""attrib.py — T3 attribution: attention, gradient saliency, localization, deletion/insertion (FR-T3-1..8).

Purpose:  FR-T3-1: token-grid explainers: map_attn (sink-excluded) and map_grad.
          FR-T3-2: localization metrics (mass, rank, point, enrich, bg).
          FR-T3-3: background/scene attribution mass.
          FR-T3-4: 3-way reading (loc_equal_high / loc_lower_in_errors / loc_low_both).
          FR-T3-5: deletion/insertion AUC with ROAD-style noisy linear imputation (nli).
          FR-T3-6: explainer-validity gate (t3_gate.json).
          FR-T3-7: SpRAy-style clustering of error maps.
          FR-T3-8: planted-confounder check.
Implements: design §5.9, requirements §4.9.
"""
from __future__ import annotations

from pathlib import Path
import json
import zlib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import torch


# ----------------------------------------------------------------- FR-T3-1 explainers

def map_attn(vlm, inp, band: tuple[int, int] | None = None, ratio: float = 10.0) -> np.ndarray:
    """Raw attention explainer on the token grid, sink-excluded (FR-T3-1; design §5.9).

    band: tuple of (start_layer, end_layer). Defaults to upper half of LLM layers.
    """
    from ea.model import attn_last, Tap, sink_mask
    g = vlm.groups(inp)
    A = attn_last(vlm, inp)[:, :, g["vis"].cpu()]  # [layers, heads, Nv]
    n_layers = A.shape[0]
    if band is None:
        band = (n_layers // 2, n_layers)

    with Tap(vlm) as t:
        vlm.m(**inp, logits_to_keep=1)
    mid = len(t.cap) // 2
    if mid in t.cap and t.cap[mid].shape[1] > int(g["vis"].max()):
        h_mid = t.cap[mid][0, g["vis"]]
        sink = sink_mask(h_mid, ratio).cpu()
    else:
        sink = torch.zeros(A.shape[-1], dtype=torch.bool)

    a = A[band[0]:band[1]].mean((0, 1))
    a[sink] = 0.0
    s_sum = float(a.sum())
    if s_sum > 0:
        a = a / s_sum
    else:
        a = torch.full_like(a, 1.0 / max(1, a.numel()))

    grid = g["grid"]
    h, w = (grid[1], grid[2]) if len(grid) == 3 else grid
    return a.numpy().reshape(h, w)


def map_grad(vlm, inp, pos=("Yes",), neg=("No",)) -> np.ndarray:
    """Gradient x input explainer on the token grid (FR-T3-1; design §5.9)."""
    from ea.model import grad_x_input
    g = vlm.groups(inp)
    sal = grad_x_input(vlm, inp, pos, neg)
    grid = g["grid"]
    h, w = (grid[1], grid[2]) if len(grid) == 3 else grid
    return sal.numpy().reshape(h, w)


# ----------------------------------------------------------------- FR-T3-2 localization

def box_mask(box: list[float] | tuple[float, ...], grid_shape: tuple) -> np.ndarray:
    """Convert normalized [x0, y0, x1, y1] box to a boolean mask on the token
    grid — delegates to data.box_mask (BUG-06 fix).

    data.box_mask is the single coordinate convention of the codebase
    (floor on the low edges, ceil on the high edges; flat time-major mask for
    (t, h, w) video grids). The old attrib-local implementation used round()
    with a forced >= 1 cell, so T3 localization and T4 probe pooling measured
    a *different region* than the intervention arms for the same box (90% of
    random boxes differ). A 2-D grid returns an (h, w) mask; a (t, h, w)
    grid returns the flat token mask."""
    from ea.data import box_mask as _data_box_mask
    m = _data_box_mask(box, grid_shape)
    if len(grid_shape) == 2:
        return m.reshape(grid_shape)
    return m


def loc(amap: np.ndarray, box: list[float] | tuple[float, ...]) -> dict[str, float]:
    """Localization metrics against GT box on the token grid (FR-T3-2).

    Returns:
      mass: fraction of attribution mass inside box
      enrich: mass / area_fraction (enrichment, chance = 1.0)
      rank: mean rank accuracy of top-k cells where k = box_cells
      point: 1.0 if argmax is inside box else 0.0 (pointing game)
      bg: fraction of attribution mass outside box (1 - mass)
    """
    amap = np.asarray(amap, dtype=float)
    m = box_mask(box, amap.shape)
    a = np.clip(amap, 0, None)
    s = a.sum()
    if s > 0:
        a = a / s
    else:
        a = np.full_like(a, 1.0 / a.size)

    k = max(1, int(m.sum()))
    top = np.argsort(-a.ravel())[:k]
    mass = float(a[m].sum())
    area_frac = float(m.mean())
    enrich = float(mass / area_frac) if area_frac > 0 else 1.0
    rank = float(m.ravel()[top].mean())
    point = float(m.ravel()[a.argmax()])
    return dict(mass=mass, enrich=enrich, rank=rank, point=point, bg=float(1.0 - mass))


# ----------------------------------------------------------------- FR-T3-4 3-way reading

def three_way(enr_err: list[float] | np.ndarray, enr_cor: list[float] | np.ndarray,
              shortcut_enrichment: float = 1.5, equal_delta: float = 0.15) -> str:
    """3-way reading comparing enrichment of errors vs matched corrects (FR-T3-4, C8).

    Returns:
      - 'loc_low_both': both medians below shortcut threshold -> scene shortcut (H5)
      - 'loc_equal_high': error enrichment within equal_delta of correct -> localized-but-unused
      - 'loc_lower_in_errors': error enrichment lower than corrects -> perception/routing failure
    """
    enr_err = np.asarray(enr_err, dtype=float)
    enr_cor = np.asarray(enr_cor, dtype=float)
    if len(enr_err) == 0 or len(enr_cor) == 0:
        # Abstain instead of voting for the scene-shortcut reading: an empty
        # comparison is under-powered, and None never matches a SIG row.
        return None
    m_err = float(np.median(enr_err))
    m_cor = float(np.median(enr_cor))
    if max(m_err, m_cor) < shortcut_enrichment:
        return "loc_low_both"
    if m_err >= m_cor - equal_delta * m_cor:
        return "loc_equal_high"
    return "loc_lower_in_errors"


# ----------------------------------------------------------------- FR-T3-5 deletion / insertion

def nli(img_arr: np.ndarray, mask: np.ndarray,
        noise_std: float | None = None, seed: int | None = None,
        iters: int = 20) -> np.ndarray:
    """Noisy Linear Imputation (ROAD, 2202.00449 §5.2; design §5.9 [VERIFY] item 11).

    Bootstraps each masked pixel by linear interpolation between boundary pixels,
    then solves the 2D harmonic inpainting stencil (w_d = 1/6 for orthogonal
    neighbors, w_i = 1/12 for diagonal neighbors per ROAD Eq. 3-5). Additive
    Gaussian noise calibrated to unmasked variance is applied to masked pixels.

    Determinism (BUG-09 fix): the noise RNG is seeded explicitly; the default
    seed derives from the image bytes, so the same image+mask always yields
    the same fill (NFR-1) without a global-RNG dependency.
    """
    out = img_arr.copy()
    if not mask.any():
        return out
    unmasked = ~mask
    if not unmasked.any():
        return np.full_like(out, 128.0)

    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(img_arr).tobytes()) % 2**31
    rng = np.random.default_rng(int(seed))
    std_val = float(img_arr[unmasked].std()) if noise_std is None else float(noise_std)

    H, W = mask.shape
    # rows with at least one unmasked pixel can be filled by row-wise
    # interpolation; the remaining (fully masked) rows fall back to a column pass.
    row_fillable = mask & unmasked.any(axis=1)[:, None]
    remaining = mask & ~row_fillable
    idx_cols, idx_rows = np.arange(W), np.arange(H)
    for c in range(out.shape[2]):
        ch = out[:, :, c].astype(np.float64)
        for r in range(H):
            mrow = row_fillable[r]
            if not mrow.any():
                continue
            ku = idx_cols[~mask[r]]
            ch[r, mrow] = np.interp(idx_cols[mrow], ku, ch[r, ku])
        if remaining.any():                              # boundary = unmasked + row-filled
            known = ~remaining
            for col in range(W):
                mcol = remaining[:, col]
                if not mcol.any():
                    continue
                kcols = idx_rows[known[:, col]]
                ch[mcol, col] = np.interp(idx_rows[mcol], kcols, ch[kcols, col])

        # ROAD 2D harmonic inpainting (2202.00449 §5.2): 8-neighbor stencil relaxation
        for _ in range(iters):
            padded = np.pad(ch, 1, mode="edge")
            stencil = (1.0 / 6.0) * (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                                     padded[1:-1, :-2] + padded[1:-1, 2:]) + \
                      (1.0 / 12.0) * (padded[:-2, :-2] + padded[:-2, 2:] +
                                     padded[2:, :-2] + padded[2:, 2:])
            ch[mask] = stencil[mask]
        out[:, :, c] = ch

    noise = rng.normal(0, max(1.0, std_val * 0.5), size=out.shape)
    out[mask] = out[mask] + noise[mask]
    return out


def paint(img: Image.Image, cells: list[int] | np.ndarray, grid: tuple[int, int],
          how: str = "nli", seed: int | None = None) -> Image.Image:
    """Mask specified cells in image space with given fill strategy (FR-T3-5; design §5.9).

    seed: RNG seed for the nli fill (defaults to a hash of the image bytes —
    deterministic for a given image+mask, NFR-1)."""
    a = np.asarray(img, dtype=np.float32)
    H, W = a.shape[:2]
    h, w = grid
    m = np.zeros((H, W), dtype=bool)
    for c in cells:
        r, k = divmod(int(c), w)
        m[int(r * H / h):int((r + 1) * H / h), int(k * W / w):int((k + 1) * W / w)] = True

    if how == "gray":
        a[m] = 128.0
    elif how == "blur":
        blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(8)), dtype=np.float32)
        a[m] = blurred[m]
    else:
        a = nli(a, m, seed=seed)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def del_ins(score_fn, img: Image.Image, amap: np.ndarray, mode: str = "del",
            steps: int = 10, sign: int = 1, how: str = "nli", seed: int | None = None) -> float:
    """Deletion/insertion area-under-the-curve (FR-T3-5; design §5.9).

    score_fn: callable(PIL.Image) -> float margin s
    sign: 2 * gold - 1 (positive indicates margin toward gold)
    seed: forwarded to the nli fill for reproducibility (BUG-09 fix).
    """
    amap = np.asarray(amap, dtype=float)
    order = np.argsort(-amap.ravel())
    N = len(order)
    curve = []
    for f in np.linspace(0, 1, steps + 1):
        k = int(f * N)
        cells = order[:k] if mode == "del" else order[k:]
        curve.append(sign * score_fn(paint(img, cells, amap.shape, how=how, seed=seed)))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(curve, dx=1.0 / steps))



# ----------------------------------------------------------------- FR-T3-6 Explainer gate

def explainer_validity_gate(score_fn, img: Image.Image, amap: np.ndarray,
                            steps: int = 5, sign: int = 1, seed: int = 0,
                            ko_fn: callable | None = None) -> dict:
    """Explainer-validity gate verifying deletion/insertion properties (FR-T3-6).

    seed controls both the random comparison map and the nli fill, so the
    gate verdict is reproducible (BUG-09 fix; NFR-1).
    ko_fn: optional internal knockout evaluator checking internal agreement (FR-T3-6 e)."""
    amap = np.asarray(amap, dtype=float)
    rng = np.random.default_rng(seed)
    rand_map = rng.uniform(0, 1, size=amap.shape)

    d_expl = del_ins(score_fn, img, amap, mode="del", steps=steps, sign=sign, how="nli", seed=seed)
    d_rand = del_ins(score_fn, img, rand_map, mode="del", steps=steps, sign=sign, how="nli", seed=seed)

    i_expl = del_ins(score_fn, img, amap, mode="ins", steps=steps, sign=sign, how="nli", seed=seed)
    i_rand = del_ins(score_fn, img, rand_map, mode="ins", steps=steps, sign=sign, how="nli", seed=seed)

    # Faithful explainer should have lower deletion AUC and higher insertion AUC than random
    del_beats_rand = bool(d_expl <= d_rand + 1e-4)
    ins_beats_rand = bool(i_expl >= i_rand - 1e-4)

    # Fill stability: check del AUC under gray and blur
    d_gray = del_ins(score_fn, img, amap, mode="del", steps=steps, sign=sign, how="gray", seed=seed)
    d_blur = del_ins(score_fn, img, amap, mode="del", steps=steps, sign=sign, how="blur", seed=seed)
    fill_stable = bool(abs(d_gray - d_blur) < 5.0)

    # Internal agreement check (FR-T3-6 item e / arXiv:2509.04588)
    internal_agreement = None
    if ko_fn is not None:
        flat_order = np.argsort(-amap.ravel())
        top_k = flat_order[:max(1, len(flat_order) // 4)]
        rand_k = rng.choice(len(flat_order), size=len(top_k), replace=False)
        eff_top = float(ko_fn(top_k))
        eff_rand = float(ko_fn(rand_k))
        internal_agreement = bool(eff_top >= eff_rand)

    valid = bool(del_beats_rand and ins_beats_rand and fill_stable and
                 (internal_agreement is None or internal_agreement))
    return dict(
        valid=valid,
        del_expl=float(d_expl),
        del_rand=float(d_rand),
        ins_expl=float(i_expl),
        ins_rand=float(i_rand),
        d_gray=float(d_gray),
        d_blur=float(d_blur),
        del_beats_rand=del_beats_rand,
        ins_beats_rand=ins_beats_rand,
        fill_stable=fill_stable,
        internal_agreement=internal_agreement,
    )


# ----------------------------------------------------------------- FR-T3-7 & FR-T3-8 SpRAy & confounder

def spray_cluster(maps: list[np.ndarray], k: int = 4, seed: int = 0) -> np.ndarray:
    """SpRAy-style clustering on downsampled (8x8) attribution maps (FR-T3-7)."""
    if len(maps) == 0:
        return np.zeros(0, dtype=int)
    down = []
    for m in maps:
        # Resize/downsample to 8x8 using PIL
        im = Image.fromarray(m.astype(np.float32))
        res = np.asarray(im.resize((8, 8), Image.Resampling.BILINEAR))
        norm = res / max(1e-6, res.sum())
        down.append(norm.ravel())
    X = np.stack(down)
    k_eff = min(k, len(maps))
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
    return km.fit_predict(X)


def planted_confounder_check(amap: np.ndarray, bg_box: list[float] | tuple[float, ...],
                             min_bg_enrichment: float = 1.0) -> bool:
    """Check if explainer puts high mass/enrichment on a planted scene shortcut (FR-T3-8)."""
    res = loc(amap, bg_box)
    return bool(res["enrich"] >= min_bg_enrichment)
