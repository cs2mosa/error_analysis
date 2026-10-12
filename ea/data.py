"""data.py — Sample schema, on-disk interface, toy generator, frames, binning.

Purpose:  the single data layer (FR-D-1): one Sample schema, one on-disk
          interface (data/<bench>/samples.jsonl), benchmark adapter stubs
          (FR-D-2), metadata binning (FR-D-5), causal video frame sampling
          (FR-D-6) and the synthetic dataset generator make_toy (FR-D-4).
Implements plan sections: §4.1 (data layer), §6.7 (matched controls input).
Inputs:   data/<bench>/samples.jsonl + media files (images; videos via cv2).
Outputs:  list[Sample]; (PIL frames, timestamps); toy dataset on disk.
No other module may read raw benchmark files (FR-D-1).

Toy semantics (make_toy): the colored square IS the required PPE item.
Green = item present (compliant, label 0); red = violation evidence, item
absent (label 1); blue border = vtype="absence" scene where the violation is
the absence itself (label 1, no square). `facts` always describe the image.
Latent-hazard scenes (§5.14, FR-H-1/2): a brown container whose lid bar is
orange (open, hazard present, label 1) or teal (closed, secured twin,
label 0); pair members share geometry, condition descriptors and pair_id.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- Sample


@dataclass
class Sample:
    """One evaluation unit (image or video). Schema: requirements.md §4.1."""

    id: str
    bench: str
    modality: str            # "image" | "video"
    media: str               # path as written in samples.jsonl
    label: int               # 1 = violation, 0 = safe
    cls: str                 # rule id
    rule_text: str | None = None
    item: str | None = None  # PPE item name or None (FR-P-3)
    facts: dict = field(default_factory=dict)   # {item: "present"|"absent"|"not_visible"}
    boxes: list = field(default_factory=list)   # [{role, xyxy normalized [0,1]}]
    pair_id: str | None = None
    meta: dict = field(default_factory=dict)
    video: dict | None = None                   # {fps, duration_s, onset_s, end_s} or None

    @classmethod
    def from_json(cls, d: dict) -> "Sample":
        # FR-D-3: every field except id/bench/modality/media/label/cls is
        # optional; unknown keys are ignored, explicit nulls fall back to the
        # field defaults so missing metadata never crashes a module.
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in known}
        for k in ("facts", "boxes", "meta"):
            if kw.get(k) is None:
                kw[k] = {}
        if kw.get("video") is None:
            kw["video"] = None
        return cls(**kw)


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def load(bench: str, root: str | Path = "data") -> list[Sample]:
    """Read data/<bench>/samples.jsonl (FR-D-1). The only raw-data reader."""
    path = Path(root) / bench / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run make_toy() or provide the benchmark via its adapter (FR-D-2).")
    return [Sample.from_json(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def to_df(samples: list[Sample]) -> pd.DataFrame:
    """One row per sample, meta flattened into columns (design §5.1).

    Missing metadata keys simply yield NaN (FR-D-3); the union of all keys
    becomes columns.
    """
    rows = []
    for s in samples:
        d = dict(id=s.id, bench=s.bench, modality=s.modality, media=s.media,
                 label=s.label, cls=s.cls, rule_text=s.rule_text, item=s.item,
                 pair_id=s.pair_id)
        d.update(s.meta)
        d["video"] = json.dumps(s.video, sort_keys=True) if s.video else None
        rows.append(d)
    return pd.DataFrame(rows)


def bin_meta(df: pd.DataFrame, cols, q: int = 3) -> pd.DataFrame:
    """Discretize continuous metadata into quantile bins (FR-D-5).

    Output columns `<col>_bin` are categorical; NaN stays NaN (no
    NaN-to-category leak); non-numeric/absent columns are skipped.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        out[c + "_bin"] = pd.qcut(out[c], q=q, duplicates="drop").astype("category")
    return out


# ---------------------------------------------------------------- geometry

def box_mask(box, grid) -> np.ndarray:
    """Normalized [x0,y0,x1,y1] -> flat bool mask over visual tokens.

    grid=(h,w) for images or (t,h,w) for video; row-major / time-major
    (design §5.1). This is the single coordinate convention of the codebase.
    """
    t, (h, w) = (grid[0], grid[1:]) if len(grid) == 3 else (1, grid)
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    m[int(np.floor(y0 * h)):int(np.ceil(y1 * h)), int(np.floor(x0 * w)):int(np.ceil(x1 * w))] = True
    return np.tile(m.ravel(), t)


def media_path(sample: Sample, root: str | Path = ".") -> Path:
    """Resolve a sample's media path relative to the workspace root."""
    p = Path(sample.media)
    return p if p.is_absolute() else Path(root) / p


# ---------------------------------------------------------------- video

def frames(path, fps: float, n: int, t_end: float, order: str = "fwd",
           single: bool = False, seed: int = 0):
    """Causally sample n PIL frames with timestamps <= t_end (FR-D-6).

    Returns (list[PIL.Image], np.ndarray timestamps). `order` permutes frames
    only; the timestamp multiset is unchanged. single=True -> exactly one
    frame (the one at t_end). An odd frame count is padded by repeating the
    last frame because Qwen3-VL merges 2 frames per temporal patch [VERIFY:
    preprocessor_config.json temporal_patch_size].
    """
    import cv2  # opencv-python (requirements §3.2); imported lazily
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    ts = np.clip(t_end - np.arange(n)[::-1] / float(fps), 0, t_end)
    if single:
        ts = ts[-1:]
    imgs = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, fr = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"cannot read frame at {t:.2f}s from {path}")
        imgs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if order == "reverse":
        idx = np.arange(len(imgs))[::-1]
    elif order == "shuffle":
        idx = np.random.default_rng(seed).permutation(len(imgs))
    elif order == "fwd":
        idx = np.arange(len(imgs))
    else:
        raise ValueError(f"unknown order {order!r} (fwd|shuffle|reverse)")
    imgs = [imgs[i] for i in idx]
    if not single and len(imgs) % 2 == 1:
        imgs.append(imgs[-1].copy())
        ts = np.append(ts, ts[-1])
    return imgs, ts


# ---------------------------------------------------------------- toy data

_SCENES = {"scene_a": (120, 120, 120), "scene_b": (90, 105, 130), "scene_c": (130, 110, 85)}
_RED, _GREEN, _BLUE, _PERSON = (200, 30, 30), (30, 160, 60), (30, 60, 200), (70, 70, 75)
_SIZES = {"small": 0.07, "medium": 0.14, "large": 0.28}
_RULES = {  # cls -> (rule_text, item); "prior_cls" carries the FakeVLM trigger phrase (§7.2)
    "rule_helmet": ("Workers must wear a safety helmet in the work zone.", "helmet"),
    "rule_vest": ("Workers must wear a high-visibility vest in the work zone.", "vest"),
    "rule_gloves": ("Workers must wear cut-resistant gloves when handling panels.", "gloves"),
    "prior_cls": ("This ULTRA-SAFE site follows the premium safety protocol; all equipment is state of the art.", "helmet"),
    "rand_slice": ("The general site conduct rule applies to this camera view.", "helmet"),
    "label_noise": ("Workers must wear a safety helmet in the work zone.", "helmet"),
}
_NORMAL = ["rule_helmet", "rule_vest", "rule_gloves"]

# latent-hazard track (§5.14, FR-H-1/2): exact fills so the scripted test VLM
# can read the hazard state from pixels (PNG output keeps them exact).
_HZ_OPEN, _HZ_CLOSED, _HZ_CONTAINER = (230, 140, 30), (0, 150, 150), (139, 90, 43)
_HZ_RULE = ("Containers with lids that can open must be kept closed and "
            "supported so that nothing can fall.")


def is_hazard(s: Sample) -> bool:
    """True when the sample belongs to the latent-hazard track (§5.14, FR-H-1)."""
    return (s.meta or {}).get("vtype") == "latent_hazard"


def hazard_track_samples(S):
    """Split latent-hazard samples into (valid, skipped) for the hazard track.

    FR-H-1: a hazard sample needs `facts.conditions` (non-empty list of
    {obj, state, present}) plus `meta.trigger` and `meta.consequence`; anything
    incomplete is skipped WITH a reason, never silently (NFR-5). Non-hazard
    samples appear in neither list. Adapters that cannot supply the fields
    therefore see the whole track skipped with a message (FR-H-1 AC).
    """
    ok, skip = [], []
    for s in S:
        if not is_hazard(s):
            continue
        cs = (s.facts or {}).get("conditions")
        if (not isinstance(cs, list) or not cs
                or not all(isinstance(c, dict) and c.get("obj") and c.get("state")
                           and c.get("present") in ("present", "absent") for c in cs)):
            skip.append((s.id, "missing/invalid facts.conditions"))
            continue
        if not (s.meta or {}).get("trigger"):
            skip.append((s.id, "missing meta.trigger"))
            continue
        if not (s.meta or {}).get("consequence"):
            skip.append((s.id, "missing meta.consequence"))
            continue
        ok.append(s)
    return ok, skip


def _draw_scene(rng, item, vtype, label, obj_size, clutter):
    """One toy image -> (PIL image, boxes, facts).

    Green square = item present (label 0); red square = violation evidence,
    item absent (label 1); absence scene = blue border, violation by absence.
    A gray 'person' box gives O5 arms and worker-count slices a target.
    Distractor squares are drawn only for the clutter slice (§7.2).
    """
    side = int(rng.choice([224, 256, 288, 320, 352, 384, 416, 448]))
    img = Image.new("RGB", (side, side), _SCENES[str(rng.choice(list(_SCENES)))])
    d = ImageDraw.Draw(img)
    if vtype == "absence":
        bw = max(4, side // 28)
        d.rectangle([0, 0, side - 1, side - 1], outline=_BLUE, width=bw)
    boxes, facts = [], {item: "present" if label == 0 else "absent"}
    px = rng.uniform(0.15, 0.85 - _SIZES[obj_size])
    if vtype == "absence" and label == 1:
        facts = {item: "absent"}                     # nothing to see: violation by absence
    else:
        s = _SIZES[obj_size] * side
        x0, y0 = px * side, px * side
        d.rectangle([x0, y0, x0 + s, y0 + s], fill=_RED if label == 1 else _GREEN)
        boxes.append(dict(role="violator", xyxy=[px, px, px + _SIZES[obj_size], px + _SIZES[obj_size]]))
    pw = rng.uniform(0.05, 0.09)                     # one 'person' rectangle (box role: person)
    px2, py2 = rng.uniform(0.05, 0.85), rng.uniform(0.05, 0.85)
    d.rectangle([px2 * side, py2 * side, (px2 + pw) * side, (py2 + 2.2 * pw) * side], fill=_PERSON)
    boxes.append(dict(role="person", xyxy=[px2, py2, px2 + pw, py2 + 2.2 * pw]))
    if clutter:
        for _ in range(int(rng.integers(2, 5))):
            cs = rng.uniform(0.04, 0.10)
            cx, cy = rng.uniform(0.02, 0.9), rng.uniform(0.02, 0.9)
            col = tuple(int(v) for v in rng.integers(60, 240, 3))
            d.rectangle([cx * side, cy * side, (cx + cs) * side, (cy + cs) * side], fill=col)
    return img, boxes, facts


def _draw_hazard(rng, hazard: bool):
    """One latent-hazard toy image -> (img, boxes) (§5.14, FR-H-2).

    A brown container with a lid bar: open (orange, raised) = the hazard
    conditions hold; closed (teal, flat) = the secured twin, hazard removed.
    Pair members share geometry and condition descriptors, so Q_P asks the
    same questions of both; only the truth differs.
    """
    side = int(rng.choice([224, 256, 288, 320]))
    img = Image.new("RGB", (side, side), _SCENES[str(rng.choice(list(_SCENES)))])
    d = ImageDraw.Draw(img)
    cw, ch = 0.30, 0.20
    cx, cy = rng.uniform(0.12, 0.55), rng.uniform(0.40, 0.62)
    d.rectangle([cx * side, cy * side, (cx + cw) * side, (cy + ch) * side], fill=_HZ_CONTAINER)
    if hazard:
        d.rectangle([(cx + cw - 0.03) * side, (cy - 0.12) * side, (cx + cw) * side, cy * side],
                    fill=_HZ_OPEN)
    else:
        d.rectangle([cx * side, (cy - 0.025) * side, (cx + cw) * side, cy * side], fill=_HZ_CLOSED)
    pw = rng.uniform(0.05, 0.09)                     # one 'person' rectangle (O5 uniformity)
    px2, py2 = rng.uniform(0.05, 0.85), rng.uniform(0.05, 0.85)
    d.rectangle([px2 * side, py2 * side, (px2 + pw) * side, (py2 + 2.2 * pw) * side], fill=_PERSON)
    boxes = [dict(role="violator", xyxy=[cx, max(0.0, cy - 0.13), cx + cw, cy + ch]),
             dict(role="person", xyxy=[px2, py2, px2 + pw, py2 + 2.2 * pw])]
    return img, boxes


def _hz_conditions(present: bool):
    """Condition descriptors shared by a hazard/twin pair; `present` is the
    per-sample truth of the condition (the design's {obj, state, support})."""
    p = "present" if present else "absent"
    return [{"obj": "container", "state": "lid open", "present": p},
            {"obj": "container", "state": "unsupported", "present": p}]


def make_toy(n: int = 200, root: str | Path = "data/toy", seed: int = 0) -> list[Sample]:
    """Synthetic dataset generator (FR-D-4): colored shapes on scene backgrounds.

    Composition follows the FakeVLM mechanisms of design §7.2/§7.3: matched
    pairs (red<->green), absence scenes (blue border), the prior-rule class,
    a clutter slice (distractor squares), a random-error slice,
    flipped-gold (label-noise) samples, and latent-hazard pairs with their
    secured twins (§5.14). Writes <root>/samples.jsonl + PNGs; returns the
    list of Samples. Deterministic for a fixed seed (NFR-1).
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    kinds = rng.choice(["pair", "absence", "prior", "clutter", "rand",
                        "label_noise", "plain", "hazard"], size=int(n * 1.5),
                       p=[0.24, 0.12, 0.09, 0.14, 0.09, 0.06, 0.14, 0.12])
    samples, i = [], 0

    def meta_of(obj_size, scene, vtype, **extra):
        m = dict(scene=scene, viewpoint=str(rng.choice(["wide", "close"])),
                 vtype=vtype, n_workers=int(rng.integers(1, 9)),
                 feature_share=float(rng.uniform(0.05, 0.5)), obj_size=obj_size,
                 dist=float(rng.uniform(2, 40)), occlusion=str(rng.choice(["none", "partial"])),
                 blur=float(rng.uniform(0, 1)), brightness=float(rng.uniform(0.3, 1.0)))
        m.update(extra)
        return m

    def emit(img, label, cls, rule, item, facts, boxes, meta, pair_id=None):
        nonlocal i
        sid = f"toy_{i:04d}"
        img.save(root / f"{sid}.png")
        samples.append(Sample(id=sid, bench="toy", modality="image",
                              media=f"{Path(root).as_posix()}/{sid}.png", label=label,
                              cls=cls, rule_text=rule, item=item, facts=facts,
                              boxes=boxes, pair_id=pair_id, meta=meta, video=None))
        i += 1

    for kind in kinds:
        if len(samples) >= n:
            break
        obj_size = str(rng.choice(list(_SIZES)))
        scene = str(rng.choice(list(_SCENES)))
        if kind == "hazard":                          # hazard + secured twin pair (FR-H-1/2)
            pid = f"hzpair_{i:04d}"
            agreement = round(float(rng.uniform(0.55, 1.0)), 2)   # FR-H-6: straddles min_agreement
            for lab in (1, 0):
                img, boxes = _draw_hazard(rng, hazard=(lab == 1))
                emit(img, lab, "hz_container", _HZ_RULE, None,
                     dict(conditions=_hz_conditions(lab == 1)), boxes,
                     meta_of("medium", scene, "latent_hazard", trigger="wind",
                             consequence="the container falling", hazard_gold=lab,
                             agreement=agreement),
                     pair_id=pid)
            continue
        if kind == "pair":                            # matched violation/safe pair (FR-B-5/11)
            cls = str(rng.choice(_NORMAL))
            rule, item = _RULES[cls]
            pid = f"pair_{i:04d}"
            px, ps = rng.uniform(0.15, 0.85 - _SIZES[obj_size]), _SIZES[obj_size]
            for lab in (1, 0):                        # identical geometry, red <-> green
                img, boxes, facts = _draw_scene(rng, item, "presence", lab, obj_size, False)
                emit(img, lab, cls, rule, item, facts, boxes,
                     meta_of(obj_size, scene, "presence"), pair_id=pid)
            continue
        cls = {"absence": str(rng.choice(["rule_helmet", "rule_vest"])), "prior": "prior_cls",
               "clutter": str(rng.choice(_NORMAL)), "rand": "rand_slice",
               "label_noise": "label_noise", "plain": str(rng.choice(_NORMAL))}[kind]
        rule, item = _RULES[cls]
        vtype = "absence" if kind == "absence" else "presence"
        if kind == "label_noise":                     # compliant input, gold flipped (C15)
            img, boxes, facts = _draw_scene(rng, item, "presence", 0, obj_size, False)
            emit(img, 1, cls, rule, item, facts, boxes,
                 meta_of(obj_size, scene, "presence", label_flipped=True))
            continue
        label = 1 if kind == "absence" else int(rng.integers(0, 2))
        if kind == "prior":
            label = 1 if rng.random() < 0.7 else 0
        img, boxes, facts = _draw_scene(rng, item, vtype, label, obj_size,
                                        clutter=(kind == "clutter"))
        emit(img, label, cls, rule, item, facts, boxes,
             meta_of(obj_size, scene, vtype, clutter=kind == "clutter"))
    samples = samples[:n]
    write_jsonl(root / "samples.jsonl", [asdict(s) for s in samples])
    return samples


# ---------------------------------------------------------------- adapters (FR-D-2)

def _stub(name: str, verify_items: str):
    raise NotImplementedError(
        f"adapter {name}: needs access/format. [VERIFY] items: {verify_items}")


def adapter_construction_site10k(root="data"):
    """ConstructionSite10k -> samples.jsonl.
    [VERIFY] native annotation format & label semantics (plan §4); license;
    split definition; whether boxes/facts per violation exist; image sizes."""
    _stub("construction_site10k", "format, labels, license, splits, boxes/facts availability")


def adapter_isafetybench(root="data"):
    """iSafetyBench -> samples.jsonl.
    [VERIFY] image/video subtask split; MCQ vs yes/no answer format (single-token
    answers needed, FR-M-2); license; class taxonomy mapping to `cls`."""
    _stub("isafetybench", "subtask split, answer format, license, taxonomy")


def adapter_fieldworkarena(root="data"):
    """FieldWorkArena -> samples.jsonl.
    [VERIFY] task types (detection vs VQA); annotation availability of facts;
    license; video fps/duration metadata."""
    _stub("fieldworkarena", "task types, facts availability, license, video metadata")


def adapter_pasbench_video(root="data"):
    """PaSBench-Video -> samples.jsonl.
    [VERIFY] frame-level risk-onset annotation format (FR-ST-3); accident
    boundaries; no-risk (twin) clips for false alarms; license."""
    _stub("pasbench_video", "risk onset format, boundaries, twins, license")


def adapter_ovo_bench_rt(root="data"):
    """OVO-Bench real-time subset -> samples.jsonl.
    [VERIFY] RT subset definition & timestamps; query format (single-token
    answers); license."""
    _stub("ovo_bench_rt", "RT subset definition, timestamps, query format, license")
