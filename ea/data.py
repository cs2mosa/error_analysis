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


# ---------------------------------------------------------------- dataset registry & auto-download

# Maps bench name -> (hf_repo_id, converter_fn_name).
# converter_fn receives (hf_dataset_dict, out_dir: Path) and writes samples.jsonl + media.
# None converter means the bench is supported via make_toy (synthetic) or a manual adapter.
_BENCH_REGISTRY: dict[str, tuple[str, str]] = {
    "cs10k":        ("LouisChen15/ConstructionSite",      "_convert_cs10k"),
    "isafetybench": ("raiyaanabdullah/isafety-bench",      "_convert_isafetybench"),
    "fieldworkarena":("Fujitsu/FieldWorkArena_Dataset",   "_convert_fieldworkarena"),
    "pasbench":     ("beingbetter11643/PaSBench-Video",   "_convert_pasbench"),
    "ovo_bench_rt": ("JoeLeelyf/OVO-Bench",               "_convert_ovo_bench_rt"),
    "toy":          (None,                                 None),  # synthetic
}


def download_bench(bench: str, root: str | Path = "data") -> list["Sample"]:
    """Download a known benchmark from HF Hub, convert to samples.jsonl, persist, and return.

    Mirrors the VLM model-resolution pattern: download once, cache locally,
    use the local copy on every subsequent run (FR-D-2).

    bench names recognised: toy, cs10k, isafetybench, fieldworkarena, pasbench,
    ovo_bench_rt.  Any other name raises FileNotFoundError.
    """
    root = Path(root)
    out_dir = root / bench
    out_dir.mkdir(parents=True, exist_ok=True)

    target_jsonl = out_dir / "samples.jsonl"
    if target_jsonl.exists():
        print(f"[data] using existing cached benchmark -> {target_jsonl}")
        return [Sample.from_json(json.loads(l)) for l in open(target_jsonl, encoding="utf-8") if l.strip()]

    if bench == "toy":
        print(f"[data] auto-generating synthetic 'toy' dataset -> {out_dir}")
        return make_toy(200, root=out_dir)

    if bench not in _BENCH_REGISTRY:
        known = ", ".join(sorted(_BENCH_REGISTRY))
        raise FileNotFoundError(
            f"No samples.jsonl found at {out_dir} and '{bench}' is not in the known registry "
            f"(known: {known}). Provide your own samples.jsonl or use a known bench name.")

    hf_repo, converter_name = _BENCH_REGISTRY[bench]
    print(f"[data] '{bench}' not found locally; downloading from HF Hub '{hf_repo}' -> {out_dir}")
    print(f"[data] This will be cached at {out_dir}/samples.jsonl for future offline runs.")

    try:
        import datasets as hf_datasets  # huggingface datasets library
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for auto-download. "
            "Install it with: pip install datasets"
        ) from e

    ds = hf_datasets.load_dataset(hf_repo, trust_remote_code=False)
    converter = globals()[converter_name]
    samples = converter(ds, out_dir)
    write_jsonl(out_dir / "samples.jsonl", [__import__("dataclasses").asdict(s) for s in samples])
    print(f"[data] wrote {len(samples)} samples -> {out_dir / 'samples.jsonl'}")
    return samples


def load(bench: str, root: str | Path = "data") -> list["Sample"]:
    """Read data/<bench>/samples.jsonl (FR-D-1). The only raw-data reader.

    If samples.jsonl is not present and the bench name is in the known registry,
    the dataset is downloaded from HF Hub, converted, and persisted automatically
    (same lazy-download pattern as the VLM weights).
    """
    path = Path(root) / bench / "samples.jsonl"
    if not path.exists():
        # Attempt auto-download for known benches (mirrors model _model_ref pattern).
        return download_bench(bench, root=root)
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


# ---------------------------------------------------------------- converters (FR-D-2)
# These convert HF Hub datasets to the Sample schema / samples.jsonl format.
# Each converter receives (hf_dataset_dict, out_dir: Path) and returns list[Sample].
# Media files are saved under out_dir/; samples.jsonl is written by download_bench().


def _stub(name: str, verify_items: str):
    raise NotImplementedError(
        f"adapter {name}: needs access/format. [VERIFY] items: {verify_items}")


def _convert_cs10k(ds, out_dir: Path) -> list["Sample"]:
    """ConstructionSite10k (HF: LouisChen15/ConstructionSite) -> list[Sample].

    Schema mapping (from Parquet inspection of the HF dataset):
    - Uses the 'test' split (3,004 rows) for benchmarking.
    - 'image' column: PIL Image.
    - 'answer' column: free-form text; presence of safety-negative keywords -> label=1.
    - 'question' column: used as rule_text.
    - bounding boxes: stored as [ymin, xmin, ymax, xmax] normalized; re-ordered to xyxy.
    [VERIFY] Parquet schema, bbox field names, label extraction heuristic.
    """
    import hashlib
    split = "test" if "test" in ds else next(iter(ds))
    rows = ds[split]
    samples = []
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    _NEG_KEYWORDS = {
        "violation", "missing", "not wearing", "no helmet", "no vest",
        "unsafe", "hazard", "danger", "absent", "without",
    }

    for i, row in enumerate(rows):
        # Save image
        img = row.get("image") or row.get("img")
        if img is None:
            continue
        sid = f"cs10k_{i:05d}"
        img_path = img_dir / f"{sid}.jpg"
        if not img_path.exists():
            img.convert("RGB").save(img_path, quality=90)

        # Derive label from answer text heuristic
        answer_raw = str(row.get("answer") or row.get("caption") or "")
        answer_lower = answer_raw.lower()
        label = int(any(kw in answer_lower for kw in _NEG_KEYWORDS))

        # Rule text from question field
        rule_text = str(row.get("question") or "Workers must follow construction site safety rules.")

        # Bounding boxes: [ymin,xmin,ymax,xmax] -> xyxy
        boxes = []
        raw_boxes = row.get("bboxes") or row.get("bbox") or []
        if isinstance(raw_boxes, list):
            for b in raw_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    ymin, xmin, ymax, xmax = b
                    boxes.append(dict(role="violator", xyxy=[xmin, ymin, xmax, ymax]))

        samples.append(Sample(
            id=sid, bench="cs10k", modality="image",
            media=str(img_path.as_posix()),
            label=label,
            cls="rule_safety",
            rule_text=rule_text,
            item=None,
            facts={},
            boxes=boxes,
            meta={"source": "CS10K", "split": split, "answer_raw": answer_raw[:200]},
        ))
    return samples


def _convert_pasbench(ds, out_dir: Path) -> list["Sample"]:
    """PaSBench-Video (HF: beingbetter11643/PaSBench-Video) -> list[Sample].

    Schema mapping:
    - 'video' column: video bytes or path.
    - 'label' column: 1 = risk, 0 = safe (or derived from category).
    - 'onset_s' / 'risk_start' column: risk onset timestamp.
    - 'end_s' / 'risk_end' column: risk end timestamp.
    [VERIFY] Exact column names, video storage format, timestamp units.
    """
    split = "test" if "test" in ds else next(iter(ds))
    rows = ds[split]
    samples = []
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(exist_ok=True)

    for i, row in enumerate(rows):
        sid = f"pasbench_{i:05d}"

        # Video: may be bytes, path string, or a dict with 'path'
        video_data = row.get("video") or row.get("video_path")
        if video_data is None:
            continue
        vid_path = vid_dir / f"{sid}.mp4"
        if isinstance(video_data, bytes) and not vid_path.exists():
            vid_path.write_bytes(video_data)
        elif isinstance(video_data, str):
            vid_path = Path(video_data)
        elif isinstance(video_data, dict) and "path" in video_data:
            vid_path = Path(video_data["path"])

        label = int(row.get("label", row.get("risk", 1)))
        onset_s = float(row.get("onset_s") or row.get("risk_start") or 0.0)
        end_s = float(row.get("end_s") or row.get("risk_end") or onset_s + 5.0)
        fps = float(row.get("fps", 8.0))
        duration_s = float(row.get("duration_s") or end_s + 1.0)

        samples.append(Sample(
            id=sid, bench="pasbench", modality="video",
            media=str(vid_path.as_posix()),
            label=label,
            cls="streaming_hazard",
            rule_text="Proactive safety: identify risk events before they escalate.",
            item=None,
            facts={},
            boxes=[],
            video=dict(fps=fps, duration_s=duration_s, onset_s=onset_s, end_s=end_s),
            meta={"source": "PaSBench-Video", "split": split,
                  "category": str(row.get("category") or "")},
        ))
    return samples


def _convert_isafetybench(ds, out_dir: Path) -> list["Sample"]:
    """iSafetyBench (HF: raiyaanabdullah/isafety-bench) -> list[Sample].
    [VERIFY] MCQ answer format, column names, video fps/duration fields.
    """
    _stub("isafetybench",
          "MCQ to Yes/No translation, action-to-cls mapping, video fps/duration")


def _convert_fieldworkarena(ds, out_dir: Path) -> list["Sample"]:
    """FieldWorkArena (HF: Fujitsu/FieldWorkArena_Dataset) -> list[Sample].
    [VERIFY] Task split (hazard vs procedural), facts extraction, video metadata.
    """
    _stub("fieldworkarena",
          "task types, facts availability, license, video metadata")


def _convert_ovo_bench_rt(ds, out_dir: Path) -> list["Sample"]:
    """OVO-Bench RT subset (HF: JoeLeelyf/OVO-Bench) -> list[Sample].
    [VERIFY] RT subset filtering, timestamp format, yes/no query framing.
    """
    _stub("ovo_bench_rt",
          "RT subset definition, timestamps, query format, license")


def adapter_construction_site10k(root="data"):
    """ConstructionSite10k (CS10K) -> samples.jsonl.
    Source: Chen & Zou, 'Are Large Pre-trained Vision Language Models Effective
            Construction Safety Inspectors?' (Data-Centric Engineering / arXiv:2508.11011).
    HF Hub: datasets.load_dataset('LouisChen15/ConstructionSite')
    Scale:  10,013 images (7,009 train, 3,004 test in Parquet).
    Tasks:  Image captioning, safety rule violation VQA, visual grounding (bboxes).
    [VERIFY] items: Parquet schema mapping to Sample (bboxes [ymin,xmin,ymax,xmax]
             normalized to xyxy [x0,y0,x1,y1]), rule text extraction into `rule_text`,
             and mapping positive violation answers to label=1, compliant to label=0.
    """
    _stub("construction_site10k", "format, labels, license, splits, boxes/facts availability")


def adapter_isafetybench(root="data"):
    """iSafetyBench -> samples.jsonl.
    Source: Abdullah et al., 'iSafetyBench: A Video-Language Benchmark for Safety in
            Industrial Environments' (ICCV VISION Workshop / arXiv:2508.00399).
    GitHub: https://github.com/iSafetyBench/data
    HF Hub: datasets.load_dataset('raiyaanabdullah/isafety-bench')
    Scale:  1,100 industrial video clips; 98 routine + 67 hazardous actions.
    [VERIFY] items: MCQ to single-token Yes/No translation (FR-M-2 margin readout);
             mapping action tags to `cls` taxonomy; video duration and fps extraction.
    """
    _stub("isafetybench", "subtask split, answer format, license, taxonomy")


def adapter_fieldworkarena(root="data"):
    """FieldWorkArena -> samples.jsonl.
    Source: Fujitsu Research, 'FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks'
            (arXiv:2505.19662).
    GitHub: https://github.com/FujitsuResearch/FieldWorkArena
    HF Hub: datasets.load_dataset('Fujitsu/FieldWorkArena_Dataset')
    Scale:  Multi-modal enterprise data across manufacturing, warehouse, and retail.
    [VERIFY] items: Task split (hazard detection vs procedural manual QA); extracting
             facts; video timestamp/fps metadata.
    """
    _stub("fieldworkarena", "task types, facts availability, license, video metadata")


def adapter_pasbench_video(root="data"):
    """PaSBench-Video (Proactive Safety Bench) -> samples.jsonl.
    Source: 'PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning'
            (arXiv:2606.02443).
    HF Hub: datasets.load_dataset('beingbetter11643/PaSBench-Video')
    Scale:  740 streaming videos (481 risk videos, 259 no-risk safe control videos)
            across Driving, Daily Life, Healthcare, and Industrial production.
    [VERIFY] items: Extracting `onset_s` (risk onset) and `end_s` (accident boundary)
             into Sample.video dict; mapping 259 no-risk clips as twin controls (label 0).
    """
    _stub("pasbench_video", "risk onset format, boundaries, twins, license")


def adapter_ovo_bench_rt(root="data"):
    """OVO-Bench real-time subset -> samples.jsonl.
    Source: Lee et al., 'OVO-Bench: Online Video Benchmark for Video-LLMs'
            (CVPR / arXiv:2501.05510).
    GitHub: https://github.com/JoeLeelyf/OVO-Bench
    HF Hub: datasets.load_dataset('JoeLeelyf/OVO-Bench')
    Scale:  644 streaming videos; 3,000+ timestamp-anchored QA tasks.
    [VERIFY] items: Filtering the 'Real-Time Understanding' task subset; mapping
             timestamps to query points (t_end); framing yes/no margin verification queries.
    """
    _stub("ovo_bench_rt", "RT subset definition, timestamps, query format, license")


def adapter_safebuild_bench(root="data"):
    """SafeBuild-Bench -> samples.jsonl.
    Source: 'SafeBuild-Bench: Evaluating MLLMs on Industrial Construction Safety'
            (arXiv:2608.00068).
    Scale:  3,314 task instances from 3,000+ expert-verified images via GEMS pipeline.
    Tasks:  Multiple-choice hazard identification, free-form hazard description.
    [VERIFY] items: Mapping multiple-choice hazard categories into rule_text and cls;
             mapping hazard present (1) vs safe compliant (0) into binary margin queries.
    """
    _stub("safebuild_bench", "GEMS instance format, hazard identification mapping, license")
