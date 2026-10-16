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
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr, wilcoxon
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

# ROAD §5.2 neighbour weights: direct 1/6, diagonal 1/12 (sum to 1 for an interior pixel).
_STENCIL = ((-1, 0, 1 / 6), (1, 0, 1 / 6), (0, -1, 1 / 6), (0, 1, 1 / 6),
            (-1, -1, 1 / 12), (-1, 1, 1 / 12), (1, -1, 1 / 12), (1, 1, 1 / 12))
_MAX_EXACT_PIXELS = 150_000


def _harmonic_fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact solution of the ROAD linear system: every masked pixel equals the weighted
    mean of its 8 neighbours, known neighbours moved to the right-hand side. Weights are
    renormalised over in-image neighbours so border pixels stay convex combinations
    (maximum principle: the fill never leaves the range of the known pixels).
    img: float64 [H, W, C]; returns a copy with the masked pixels replaced."""
    H, W, C = img.shape
    n = int(mask.sum())
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[mask] = np.arange(n)
    ys, xs = np.nonzero(mask)
    wsum = np.zeros(n)
    rhs = np.zeros((n, C))
    rows, cols, vals = [], [], []
    for dy, dx, w in _STENCIL:
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        src = np.flatnonzero(ok)
        wsum[src] += w
        nb = idx[ny[ok], nx[ok]]
        unk = nb >= 0
        rows.append(src[unk]); cols.append(nb[unk]); vals.append(np.full(int(unk.sum()), -w))
        kn = src[~unk]
        rhs[kn] += w * img[ny[ok][~unk], nx[ok][~unk]]
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    A = (sparse.identity(n, format="csc") +
         sparse.coo_matrix((v / wsum[r], (r, c)), shape=(n, n)).tocsc())
    sol = np.asarray(spsolve(A, rhs / wsum[:, None])).reshape(n, C)
    out = img.copy()
    out[mask] = sol
    return out


def nli(img_arr: np.ndarray, mask: np.ndarray,
        noise_std: float | None = None, seed: int | None = None) -> np.ndarray:
    """Noisy Linear Imputation (ROAD, 2202.00449 §5.2; design §5.9 [VERIFY] item 11).

    Each masked pixel is the weighted mean of its 8 neighbours (w_d = 1/6, w_i = 1/12),
    solved exactly as a sparse linear system, as in the authors' reference code. Gaussian
    noise is then added so the linear dependency cannot be used to spot the filled region.

    noise_std: noise standard deviation in pixel units; None -> 0.5 x the std of the
    unmasked pixels. The paper text states sigma = 0.1 on an unstated pixel scale and the
    reference implementation defaults to 0.01, so the scale is a calibration choice here.
    Masks above ~150k pixels are solved on a subsampled grid and bilinearly upsampled
    (a fill that is converged but smoother than the full-resolution solution).

    Determinism: the noise RNG is seeded explicitly; the default seed derives from the
    image bytes, so the same image+mask always yields the same fill (NFR-1).
    """
    out = np.asarray(img_arr, dtype=np.float64).copy()
    if not mask.any():
        return out
    unmasked = ~mask
    if not unmasked.any():
        return np.full_like(out, 128.0)

    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(img_arr).tobytes()) % 2**31
    rng = np.random.default_rng(int(seed))
    sigma = (max(1.0, 0.5 * float(out[unmasked].std())) if noise_std is None
             else float(noise_std))

    n = int(mask.sum())
    if n <= _MAX_EXACT_PIXELS:
        filled = _harmonic_fill(out, mask)
    else:
        from scipy.ndimage import zoom
        f = int(np.ceil(np.sqrt(n / _MAX_EXACT_PIXELS)))
        small_m = mask[::f, ::f]
        small = _harmonic_fill(out[::f, ::f], small_m) if (~small_m).any() else out[::f, ::f]
        up = zoom(small, (f, f, 1), order=1)[:out.shape[0], :out.shape[1]]
        filled = out.copy()
        filled[mask] = up[mask]
    out[mask] = filled[mask]
    if sigma > 0:
        out[mask] += rng.normal(0.0, sigma, size=out.shape)[mask]
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

_FILLS = ("nli", "gray", "blur")


def map_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two attribution maps (0.0 if either is constant)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b)[0])


def explainer_validity_gate(score_fn, img: Image.Image, amap: np.ndarray,
                            steps: int = 5, sign: int = 1, seed: int = 0,
                            ko_fn: callable | None = None,
                            base_map: np.ndarray | None = None,
                            eps: float = 1e-4) -> dict:
    """Per-sample explainer-validity checks (FR-T3-6 a, b, e); aggregate with gate_aggregate.

    Deletion should score lower and insertion higher than a random-order map under the
    primary debiased fill. Strict inequality: a tie means the curves carry no evidence, so a
    model that ignores the image is reported `informative=False` and never valid. The
    ordering must be the same under every fill (nli, gray, blur): ROAD shows rankings
    depend on the fill, so a verdict that flips with the fill is not trusted. `sign` is the
    model's own predicted side (score_fn margin toward the explained decision).
    base_map: optional raw-attention baseline map to compare against (FR-T3-6 a).
    ko_fn: callable(flat_cell_idx) -> |s| reduction when those visual tokens are knocked
    out; internal agreement requires top-quartile cells to beat random cells.
    seed controls both the random comparison map and the nli fill (NFR-1).
    """
    amap = np.asarray(amap, dtype=float)
    rng = np.random.default_rng(seed)
    rand_map = rng.uniform(0, 1, size=amap.shape)

    del_by_fill = {}
    for how in _FILLS:
        d_e = del_ins(score_fn, img, amap, mode="del", steps=steps, sign=sign, how=how, seed=seed)
        d_r = del_ins(score_fn, img, rand_map, mode="del", steps=steps, sign=sign, how=how, seed=seed)
        del_by_fill[how] = (float(d_e), float(d_r))
    i_expl = float(del_ins(score_fn, img, amap, mode="ins", steps=steps, sign=sign, how="nli", seed=seed))
    i_rand = float(del_ins(score_fn, img, rand_map, mode="ins", steps=steps, sign=sign, how="nli", seed=seed))

    d_expl, d_rand = del_by_fill["nli"]
    beats = {how: bool(de < dr - eps) for how, (de, dr) in del_by_fill.items()}
    del_beats_rand = beats["nli"]
    ins_beats_rand = bool(i_expl > i_rand + eps)
    informative = bool(abs(d_expl - d_rand) > eps or abs(i_expl - i_rand) > eps)
    fill_stable = len(set(beats.values())) == 1

    d_base = i_base = None
    if base_map is not None:
        base_map = np.asarray(base_map, dtype=float)
        d_base = float(del_ins(score_fn, img, base_map, mode="del", steps=steps, sign=sign,
                               how="nli", seed=seed))
        i_base = float(del_ins(score_fn, img, base_map, mode="ins", steps=steps, sign=sign,
                               how="nli", seed=seed))

    internal_agreement = None
    if ko_fn is not None:
        flat_order = np.argsort(-amap.ravel())
        top_k = flat_order[:max(1, len(flat_order) // 4)]
        rand_k = rng.choice(len(flat_order), size=len(top_k), replace=False)
        internal_agreement = bool(float(ko_fn(top_k)) >= float(ko_fn(rand_k)))

    valid = bool(informative and del_beats_rand and ins_beats_rand and fill_stable and
                 (internal_agreement is None or internal_agreement))
    return dict(
        valid=valid, informative=informative,
        del_expl=d_expl, del_rand=d_rand, ins_expl=i_expl, ins_rand=i_rand,
        del_base=d_base, ins_base=i_base,
        d_gray=del_by_fill["gray"][0], d_blur=del_by_fill["blur"][0],
        del_beats_rand=del_beats_rand, ins_beats_rand=ins_beats_rand,
        del_beats_rand_by_fill=beats, fill_stable=fill_stable,
        internal_agreement=internal_agreement,
    )


def _paired_p(a, b, alternative: str, min_n: int):
    """One-sided paired Wilcoxon p-value of a vs b; None if fewer than min_n non-zero pairs."""
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    d = d[np.abs(d) > 1e-12]
    if len(d) < min_n:
        return None
    return float(wilcoxon(d, alternative=alternative).pvalue)


def gate_aggregate(details: list[dict], alpha: float = 0.05, min_pass_rate: float = 0.7,
                   min_n: int = 5, randomization_rho: list[float] | None = None,
                   robustness_rho: list[float] | None = None,
                   rand_rho_max: float = 0.5, noise_rho_min: float = 0.6) -> dict:
    """Dataset-level explainer gate (FR-T3-6; design §5.9).

    Requires: paired Wilcoxon p < alpha against the random-order map for deletion (lower)
    and insertion (higher); non-inferiority to the raw-attention baseline when it was
    computed (not significantly worse; sink filtering may legitimately tie it); a per-sample pass rate >= min_pass_rate; mean Spearman(map, map after layer
    re-initialization) < rand_rho_max (the map must depend on the learned weights); mean
    Spearman(map, map under small pixel noise) >= noise_rho_min. A check whose inputs were
    not supplied is listed in `skipped` and does not count as a pass or a failure.
    """
    n = len(details)
    n_valid = sum(bool(d.get("valid")) for d in details)
    pass_rate = n_valid / n if n else 0.0
    col = lambda k: [d.get(k) for d in details]
    p_del = _paired_p(col("del_expl"), col("del_rand"), "less", min_n)
    p_ins = _paired_p(col("ins_expl"), col("ins_rand"), "greater", min_n)
    has_base = any(d.get("del_base") is not None for d in details)
    # raw-attention baseline: superiority p-values are informational; the sink-filtered map is
    # raw attention minus sink tokens, so ties are expected. The requirement is non-inferiority:
    # no significant evidence that the explainer is WORSE than raw attention.
    p_del_b = _paired_p(col("del_expl"), col("del_base"), "less", min_n) if has_base else None
    p_ins_b = _paired_p(col("ins_expl"), col("ins_base"), "greater", min_n) if has_base else None
    worse_del = _paired_p(col("del_expl"), col("del_base"), "greater", min_n) if has_base else None
    worse_ins = _paired_p(col("ins_expl"), col("ins_base"), "less", min_n) if has_base else None
    base_ok = not any(p is not None and p < alpha for p in (worse_del, worse_ins))

    skipped, ok = [], True
    stat_ps = [p_del, p_ins]
    stats_ok = all(p is not None and p < alpha for p in stat_ps)
    ok &= stats_ok and base_ok and pass_rate >= min_pass_rate
    rand_ok = rob_ok = None
    if randomization_rho is None:
        skipped.append("randomization")
    else:
        rand_ok = bool(len(randomization_rho) and np.mean(randomization_rho) < rand_rho_max)
        ok &= rand_ok
    if robustness_rho is None:
        skipped.append("robustness")
    else:
        rob_ok = bool(len(robustness_rho) and np.mean(robustness_rho) >= noise_rho_min)
        ok &= rob_ok
    if not n:
        status = "no samples"
    elif any(p is None for p in stat_ps):
        status = f"under-powered (< {min_n} informative paired samples)"
    else:
        status = "computed (FR-T3-6)"
    return dict(valid=bool(ok and n), status=status, n_samples=n, n_valid=n_valid,
                pass_rate=round(pass_rate, 3), p_del_rand=p_del, p_ins_rand=p_ins,
                p_del_base=p_del_b, p_ins_base=p_ins_b, baseline_not_worse=base_ok,
                randomization=dict(ok=rand_ok, mean_rho=(None if randomization_rho is None or not len(randomization_rho) else float(np.mean(randomization_rho)))),
                robustness=dict(ok=rob_ok, mean_rho=(None if robustness_rho is None or not len(robustness_rho) else float(np.mean(robustness_rho)))),
                skipped=skipped)


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
