"""interventions.py — the INTERV registry: O1–O7 arms with matched controls,
the noise-floor arm N0.null (C4), and the INJ.* known-cause injections (C15).

Purpose:  every intervention is a pure function (sample, **params) ->
          (images | None, sample_override | None) registered in one dict
          (GC-5); each arm carries its named matched control; injections are
          used only by `validate` (§5.8b).
Implements plan sections: §7 T2 table; §5.14 HZ arms arrive with the
          latent-hazard track; O7 is handled in frames() params (no function).
Inputs:   Sample objects; PIL images; config values (crop margin, null jpeg).
Outputs:  images/frames lists or None (text-only), optional sample override.

Measures / Does-not-measure cards (FR-Q-1) are per-arm docstrings in the
registry table below; property tests (FR-Q-5) live in tests/test_interventions.py.
"""

import io
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ea.data import media_path

INTERV = {}


def reg(name, control=None, kind="recover"):
    def d(f):
        INTERV[name] = dict(fn=f, control=control, kind=kind)
        return f
    return d


def _seed(s, extra=""):
    """Deterministic per-(sample, arm) seed (NFR-1): NOT Python hash(), which
    is randomized per process — crc32 of id + arm name instead."""
    return zlib.crc32(f"{s.id}|{extra}".encode()) % 2**31


def _seed_bytes(data: bytes, extra=""):
    return (zlib.crc32(data) ^ zlib.crc32(extra.encode())) % 2**31


def _img(s, root="."):
    """Load a sample's image (RGB); video samples contribute their frame at
    the sample's onset/end midpoint (single frame) — O7 covers temporal arms."""
    from PIL import Image as I
    if s.modality == "video":
        from ea.data import frames
        dur = (s.video or {}).get("duration_s", 1.0)
        on = (s.video or {}).get("onset_s")
        t_end = min(dur, (on if on is not None else dur) or dur)
        imgs, _ = frames(media_path(s, root), fps=2, n=1, t_end=max(t_end, 0.0), single=True)
        return imgs[0]
    return I.open(media_path(s, root)).convert("RGB")


def _violator_box(s):
    for b in s.boxes:
        if b["role"] == "violator":
            return b["xyxy"]
    return None


def _expand(box, margin):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return [max(0, x0 - margin * w), max(0, y0 - margin * h),
            min(1, x1 + margin * w), min(1, y1 + margin * h)]


def _crop(img, box):
    W, H = img.size
    return img.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))


def _rand_box(s, rng, side_frac):
    """Random box with the same AREA as the evidence box+margin, IoU=0 with it."""
    x0, y0, x1, y1 = _violator_box(s)
    a = (x1 - x0) * (y1 - y0)
    w = h = a ** 0.5
    for _ in range(100):
        x = rng.uniform(0, 1 - w); y = rng.uniform(0, 1 - h)
        if x + w < x0 or x > x1 or y + h < y0 or y > y1:    # disjoint (IoU=0)
            return [x, y, x + w, y + h]
    return [0.0, 0.0, w, h] if x0 > w else [1 - w, 1 - h, 1.0, 1.0]




# ---------------------------------------------------------------- O1 crop ladder

@reg("O1.gtcrop", control="O1.randcrop")
def o1_gtcrop(s, root=".", margin=0.25, **kw):
    """GT-box crop expanded by crop.margin. Measures: recovery when the
    evidence is enlarged. Does not measure: resolution (needs hires), clutter
    (needs bgmask). Other explanations: token-count change (control O1.sametok)."""
    img = _img(s, root)
    b = _violator_box(s)
    return [_crop(img, _expand(b, margin))], None


@reg("O1.tightcrop", control="O1.randcrop")
def o1_tightcrop(s, root=".", **kw):
    b = _violator_box(s)
    return [_crop(_img(s, root), b)], None


@reg("O1.randcrop", control=None, kind="control")
def o1_randcrop(s, root=".", margin=0.25, **kw):
    """Control: random same-size crop (seeded by sample id; pitfall 5)."""
    img = _img(s, root)
    b = _expand(_violator_box(s), margin)
    rng = np.random.default_rng(_seed(s, "randcrop"))
    W, H = img.size
    w, h = (b[2] - b[0]) * W, (b[3] - b[1]) * H
    x = int(rng.uniform(0, W - w)); y = int(rng.uniform(0, H - h))
    return [img.crop((x, y, x + int(w), y + int(h)))], None


@reg("O1.sametok", control=None, kind="control")
def o1_sametok(s, root=".", margin=0.25, **kw):
    """Control: full frame resized to the crop's pixel AREA (same token count ±1,
    FR-T2-5: crops change token count — always compare with the same-token control)."""
    img = _img(s, root)
    b = _expand(_violator_box(s), margin)
    W, H = img.size
    a = int((b[2] - b[0]) * W) * int((b[3] - b[1]) * H)
    side = int(a ** 0.5)
    return [img.resize((max(side, 8), max(side, 8)))], None


@reg("O1.attncrop", control="O1.randcrop", kind="recover")
def o1_attncrop(s, root=".", box=None, **kw):
    """P1: crop around the argmax cell of the attn_last map (needs T3 pass;
    the map's box in normalized coords is passed by the caller)."""
    img = _img(s, root)
    return [_crop(img, box)], None


# ---------------------------------------------------------------- O1.bgmask / O3.occlude (C3)

def _inside_rect(box, W, H):
    """Pixel rectangle of a normalized box using the box_mask convention:
    floor on the low edges, ceil on the high edges (design §5.1)."""
    return (int(np.floor(box[0] * W)), int(np.floor(box[1] * H)),
            int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H)))


@reg("O1.bgmask", control="O1.randmask")
def o1_bgmask(s, root=".", margin=0.25, color=(128, 128, 128), **kw):
    """Full frame at NATIVE resolution, everything outside the evidence box +
    margin grayed -> same token count. Measures: clutter/context recovery at
    equal tokens (separates H8/H5 from H1). Does not measure: resolution.
    Other explanations: the gray itself (control O1.randmask)."""
    img_arr = np.asarray(_img(s, root)).copy()
    H, W = img_arr.shape[:2]
    x0, y0, x1, y1 = _expand(_violator_box(s), margin)
    ix0, iy0, ix1, iy1 = _inside_rect([x0, y0, x1, y1], W, H)
    out = np.full_like(img_arr, color)
    out[iy0:iy1, ix0:ix1] = img_arr[iy0:iy1, ix0:ix1]
    return [Image.fromarray(out)], None


@reg("O1.randmask", control=None, kind="control")
def o1_randmask(s, root=".", margin=0.25, color=(128, 128, 128), **kw):
    """Control: same-size box at a random location kept, rest grayed (IoU=0)."""
    img = _img(s, root).copy()
    W, H = img.size
    b = _rand_box(s, np.random.default_rng(_seed(s, "randmask")), 0)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=color)
    d.rectangle([int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)],
                fill=tuple(np.asarray(_img(s, root))[int(b[1] * H):int(b[3] * H),
                                                             int(b[0] * W):int(b[2] * W)]
                           .reshape(-1, 3).mean(0).astype(int)))
    return [img], None


@reg("O3.occlude", control="O3.randoccl", kind="necessity")
def o3_occlude(s, root=".", color=(128, 128, 128), **kw):
    """Gray the GT evidence box only, native resolution. Measures: whether the
    answer depends on the evidence REGION (necessity), not just on an image
    being present. Does not measure: global priors (that is O3.blank).
    Other explanations: any same-size patch (control O3.randoccl)."""
    img = _img(s, root).copy()
    W, H = img.size
    b = _violator_box(s)
    d = ImageDraw.Draw(img)
    d.rectangle([int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)], fill=color)
    return [img], None


@reg("O3.randoccl", control=None, kind="control")
def o3_randoccl(s, root=".", color=(128, 128, 128), **kw):
    """Control: gray a random same-size box (IoU=0 with the evidence box)."""
    img = _img(s, root).copy()
    W, H = img.size
    b = _rand_box(s, np.random.default_rng(_seed(s, "randoccl")), 0)
    d = ImageDraw.Draw(img)
    d.rectangle([int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)], fill=color)
    return [img], None


# ---------------------------------------------------------------- O2 facts (exclusion test, C2)

@reg("O2.facts", control="O2.wrongfacts")
def o2_facts(s, root=".", **kw):
    """Text-only rule + oracle facts (prompt qr; images=None). EXCLUSION test:
    qr_ok=True says nothing about perception; only qr_ok=False is informative
    (C2). Measures: whether the model can apply the rule given the facts."""
    return None, s                                          # sample override -> qr prompt renders facts


@reg("O2.wrongfacts", control=None, kind="control")
def o2_wrongfacts(s, root=".", **kw):
    """Control: flip each fact present<->absent; expected ≈0 recovery toward
    the ORIGINAL gold (follows the wrong facts instead; pitfall 2)."""
    import copy
    s2 = copy.deepcopy(s)
    s2.facts = {k: ("absent" if v == "present" else "present") if v in ("present", "absent")
                else v for k, v in s.facts.items()}
    return None, s2


# ---------------------------------------------------------------- O3 blind/noise/swap (necessity)

def _gray(w=224, h=224, v=128):
    return Image.new("RGB", (w, h), (v, v, v))


@reg("O3.blank", control=None, kind="necessity")
def o3_blank(s, root=".", **kw):
    """Uniform gray image. Measures (with O3.occlude): image insensitivity ->
    prior-driven. Gate: FR-T1-4 (perception+rule must pass first)."""
    return [_gray()], None


@reg("O3.noise", control=None, kind="necessity")
def o3_noise(s, root=".", seed=None, **kw):
    rng = np.random.default_rng(_seed(s, "noise") if seed is None else seed)
    a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return [Image.fromarray(a)], None


@reg("O3.swap", control=None, kind="necessity")
def o3_swap(s, root=".", other=None, **kw):
    """Image of a random other-scene sample with the opposite label (scene swap).
    `other` is a path supplied by the runner (plan §7 O3)."""
    img = Image.open(other).convert("RGB") if other else _gray()
    return [img], None


# ---------------------------------------------------------------- N0.null (C4)

@reg("N0.null", control=None, kind="control")
def n0_null(s, root=".", jpeg_q=95, shift_px=1, **kw):
    """Imperceptible perturbation: JPEG re-encode + 1px translation. The
    noise-floor control for knobs and O6. A deterministic base re-run is NOT
    a control (it recovers exactly 0 by construction)."""
    img = _img(s, root)
    a = np.asarray(img)
    px = int(shift_px)
    if px != 0:
        rolled = np.roll(a, (px, px), axis=(0, 1))
        out = np.clip(0.7 * a.astype(float) + 0.3 * rolled.astype(float), 0, 255).astype(np.uint8)
    else:
        out = a
    b = io.BytesIO()
    Image.fromarray(out).save(b, "JPEG", quality=int(jpeg_q), subsampling=0)
    return [Image.open(b).convert("RGB")], None


# ---------------------------------------------------------------- O5 per-person (H8)

@reg("O5.personcrop", control="O5.randperson")
def o5_personcrop(s, root=".", idx=0, **kw):
    """Crop each person box (idx from the job params). Measures: binding
    (right person picked out). Other explanations: crop size (same-token control)."""
    persons = [b["xyxy"] for b in s.boxes if b["role"] == "person"]
    img = _img(s, root)
    if not persons:
        return [img], None                                   # counted as skip by the runner? no: keep image
    return [_crop(img, persons[idx % len(persons)])], None


@reg("O5.overlay", control="O5.randoverlay")
def o5_overlay(s, root=".", idx=0, color=(255, 200, 0), width=6, **kw):
    """Draw a rectangle on each person box. Measures: pointing/grounding."""
    img = _img(s, root).copy()
    persons = [b["xyxy"] for b in s.boxes if b["role"] == "person"]
    if persons:
        b = persons[idx % len(persons)]
        W, H = img.size
        ImageDraw.Draw(img).rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H],
                                      outline=color, width=width)
    return [img], None


@reg("O5.randperson", control=None, kind="control")
def o5_randperson(s, root=".", idx=0, **kw):
    """Control: crop a random OTHER region of the same size (IoU=0 with persons)."""
    persons = [b["xyxy"] for b in s.boxes if b["role"] == "person"]
    img = _img(s, root)
    if not persons:
        return [img], None
    b = persons[idx % len(persons)]
    w, h = b[2] - b[0], b[3] - b[1]
    rng = np.random.default_rng(_seed(s, "randperson"))
    for _ in range(100):
        x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        nb = [x, y, x + w, y + h]
        if all(nb[2] < p[0] or nb[0] > p[2] or nb[3] < p[1] or nb[1] > p[3] for p in persons):
            return [_crop(img, nb)], None
    return [_crop(img, b)], None


@reg("O5.randoverlay", control=None, kind="control")
def o5_randoverlay(s, root=".", idx=0, color=(255, 200, 0), width=6, **kw):
    """Control: same rectangle at a random non-person location."""
    img = _img(s, root).copy()
    persons = [b["xyxy"] for b in s.boxes if b["role"] == "person"]
    b = persons[idx % len(persons)] if persons else [0.1, 0.1, 0.3, 0.3]
    w, h = b[2] - b[0], b[3] - b[1]
    rng = np.random.default_rng(_seed(s, "randoverlay"))
    W, H = img.size
    x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    ImageDraw.Draw(img).rectangle([x * W, y * H, (x + w) * W, (y + h) * H],
                                  outline=color, width=width)
    return [img], None


# ---------------------------------------------------------------- O6 quality (H9)

def degrade(img, kind, sev):
    """Corruption operator, severity 1..5 (config knobs.quality). Seeded from
    the image bytes so the same image+kind+severity is reproducible (NFR-1)."""
    a = np.asarray(img, np.float32) / 255
    rng = np.random.default_rng(_seed_bytes(np.asarray(img).tobytes(), kind))
    if kind == "blur":
        b = img.filter(ImageFilter.GaussianBlur(1.5 * sev))
        out = (np.asarray(b, float) / 255.0) * (1 - 0.02 * sev)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    if kind == "lowlight":
        out = np.clip((a ** (1 + .4 * sev)) * (1 - .1 * sev)
                      + rng.normal(0, .01 * sev, a.shape), 0, 1)
        return Image.fromarray((out * 255).astype(np.uint8))
    if kind == "noise":
        out = np.clip(a + rng.normal(0, .03 * sev, a.shape), 0, 1)
        return Image.fromarray((out * 255).astype(np.uint8))
    if kind == "haze":
        out = a * (1 - .15 * sev) + .15 * sev
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    if kind == "jpeg":
        b = io.BytesIO()
        img.save(b, "JPEG", quality=max(5, 100 - 18 * sev))
        return Image.open(b).convert("RGB")
    if kind == "resample":
        w, h = img.size
        f = 1 / (1 + sev)
        return img.resize((int(w * f), int(h * f)), Image.NEAREST).resize((w, h), Image.NEAREST)
    if kind == "motion":
        k = np.zeros((9, 9)); k[4, :] = 1 / 9
        return img.filter(ImageFilter.Kernel((9, 9), k.ravel().tolist()))
    raise ValueError(f"unknown degrade kind {kind!r}")


@reg("O6.degrade", control="N0.null")
def o6_degrade(s, root=".", kind="blur", sev=1, **kw):
    """Severity sweep on the full image. Measures: quality dependence.
    Control: N0.null (never a base re-run, C4)."""
    return [degrade(_img(s, root), kind, sev)], None


@reg("O6.sharpen", control="N0.null")
def o6_sharpen(s, root=".", **kw):
    """UnsharpMask + autocontrast on errors. Signature: input quality (H9)."""
    img = _img(s, root).filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    from PIL import ImageOps
    return [ImageOps.autocontrast(img)], None


# ---------------------------------------------------------------- HZ latent-hazard arms (§5.14, FR-H-4)

@reg("HZ.cue", control="HZ.cue_twin", kind="recover")
def hz_cue(s, root=".", **kw):
    """Cued latent-hazard arm: the cued prompt `hz_cue` names the consequence
    and trigger on purpose (splits cannot-infer from does-not-volunteer,
    §5.14); the image is unchanged. Measures: whether the model reports the
    hazard when cued. Control: HZ.cue_twin — a cue that also raises 'Yes' on
    the secured twins is prompt bias, not hazard awareness (pitfall 24). The
    three-valued diagnostic (`cue`) is computed by escalate.explain_cue."""
    return [_img(s, root)], None


@reg("HZ.cue_twin", control=None, kind="control")
def hz_cue_twin(s, root=".", **kw):
    """Control: identity transform on the secured twin scene (the runner points
    the job at the twin sample). Its false-alarm rate under the cued prompt
    gates the `cue` diagnostic (FR-H-4): twin FA upper bound must stay
    ≤ exit.control for cue=True."""
    return [_img(s, root)], None


# ---------------------------------------------------------------- INJ.* (C15, §5.8b) — used only by validate

@reg("INJ.small", control=None, kind="inject")
def inj_small(s, root=".", k=3, **kw):
    """Put the image in the centre of a k-times larger reflect-padded canvas
    and rescale the box: with max_pixels fixed the object occupies fewer
    tokens. Expected cause: pixel/token-budget limit (H1/H4)."""
    img = _img(s, root)
    W, H = img.size
    canvas = Image.new("RGB", (W * k, H * k))
    canvas.paste(img, ((W * k - W) // 2, (H * k - H) // 2))
    import copy
    s2 = copy.deepcopy(s)
    s2.boxes = [{**b, "xyxy": [(x + (k - 1) / 2) / k for x in b["xyxy"]]}
                for b in s.boxes if b["role"] == "violator"] + \
               [b for b in s.boxes if b["role"] != "violator"]
    return [canvas], s2


@reg("INJ.clutter", control=None, kind="inject")
def inj_clutter(s, root=".", m=4, others=None, **kw):
    """Paste m person/object crops from other samples around the violator at
    native scale and update boxes. Expected cause: clutter/binding (H8/H5)."""
    img = _img(s, root).copy()
    import copy
    rng = np.random.default_rng(_seed(s, "injclutter"))
    W, H = img.size
    vb = _violator_box(s)
    s2 = copy.deepcopy(s)
    new_boxes = list(s.boxes)
    for j in range(int(m)):
        p = others[j % len(others)] if others else None
        patch = _img(p, root) if p is not None else _gray(64, 64)
        pw = rng.uniform(0.06, 0.12)
        side = int(pw * W)
        patch = patch.resize((side, side))
        for _ in range(100):
            x, y = rng.uniform(0, 1 - pw), rng.uniform(0, 1 - pw)
            bb = [x, y, x + pw, y + pw]
            if bb[2] < vb[0] or bb[0] > vb[2] or bb[3] < vb[1] or bb[1] > vb[3]:
                break
        img.paste(patch, (int(x * W), int(y * H)))
        new_boxes.append(dict(role="person", xyxy=bb))
    s2.boxes = new_boxes
    return [img], s2


@reg("INJ.degrade", control=None, kind="inject")
def inj_degrade(s, root=".", kind="blur", sev=4, **kw):
    """Degrade the full image (reuses O6.degrade). Expected cause: input
    quality (H9)."""
    return [degrade(_img(s, root), kind, sev)], None


@reg("INJ.lead", control=None, kind="inject")
def inj_lead(s, root=".", **kw):
    """Prefix d0 with a misleading framing (frm_pos on safe, frm_neg on
    violations). Expected cause: polarity/framing bias (H6/H7). Shares the
    O4 prompts: validates the signature logic, not O4's sensitivity."""
    import copy
    s2 = copy.deepcopy(s)
    lead = "This site has an excellent safety record. " if s.label == 1 \
        else "This scene very likely contains a violation. "
    s2.rule_text = lead + "Rule: " + (s.rule_text or "")
    return None, s2


@reg("INJ.label", control=None, kind="inject")
def inj_label(s, root=".", **kw):
    """Label-noise null: gold flipped for a fraction of the pool, INPUT
    unchanged. Must end unexplained/audit; any confident cause is a false
    attribution (FR-V-2). Implemented at the validate level by flipping gold;
    the arm itself is a pass-through."""
    return None, None
