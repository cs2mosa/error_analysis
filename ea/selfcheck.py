"""selfcheck.py — startup assertions for the [VERIFY] items (requirements §9, NFR-3).

Purpose:  fail loudly, with the actual value printed, when an assumption about
          the checkpoint / library / processor is wrong. Two modes:
          default = structure + tokenizer checks (no weights loaded, ~seconds);
          --load  = real forward-pass checks (items 3, 4, 6, 10) on a toy image.
Item status: PASS / FAIL (exit 1) / DEFERRED (checked by a later phase's tests).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import yaml


def _model_ref(cfg, override=None):
    """Resolve the model to load: explicit override > MODEL_PATH env > cfg.model.path > HF_HOME snapshot > repo name."""
    if override:
        return override
    env_p = __import__("os").environ.get("MODEL_PATH")
    if env_p:
        ep = Path(env_p)
        if ep.exists() or (not env_p.startswith("/") and not env_p.startswith(".") and ":" not in env_p):
            return env_p
    if cfg and cfg.get("model", {}).get("path"):
        p = Path(cfg["model"]["path"])
        if p.exists():
            return str(p)
    name = (cfg.get("model", {}).get("name") if cfg else None) or "Qwen/Qwen3-VL-4B-Instruct"
    hf_home = Path(str(__import__("os").environ.get("HF_HOME", "")))
    local = hf_home / "hub" / f"models--{name.replace('/', '--')}"
    if local.exists():
        snaps = sorted((local / "snapshots").glob("*"))
        if snaps:
            return str(snaps[-1])
    return name


def _meta_model(mp):
    """Structural probe without loading the 9 GB of weights."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    acfg = AutoConfig.from_pretrained(mp)
    with torch.device("meta"):
        m = Qwen3VLForConditionalGeneration._from_config(acfg)
    return acfg, m


def checks_default():
    """Items 1, 2, 5, 7 (config part), 8, 9 — no weights required."""
    out = []
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
    mp = _model_ref(cfg)
    try:
        acfg, m = _meta_model(mp)
    except Exception as e:                                   # item 1/2: actionable failure
        out.append(("1/2 checkpoint+modules", "FAIL", f"cannot build model from {mp}: {e!r}"))
        return out, cfg
    tc, vc = acfg.text_config, acfg.vision_config
    # item 1: identity + architecture numbers
    detail = (f"model_type={acfg.model_type} layers={tc.num_hidden_layers} hidden={tc.hidden_size} "
              f"heads={tc.num_attention_heads} kv_heads={tc.num_key_value_heads} "
              f"vit_depth={vc.depth} deepstack={getattr(vc, 'deepstack_visual_indexes', None)} "
              f"patch={vc.patch_size} merge={vc.spatial_merge_size} "
              f"shortest_edge={None} name={cfg['model']['name']} ({'Instruct' if 'Thinking' not in mp else 'Thinking'})")
    ok = (vc.patch_size == 16 and vc.spatial_merge_size == 2
          and tc.num_hidden_layers == 36 and getattr(vc, "deepstack_visual_indexes", None) == [5, 11, 17])
    out.append(("1 checkpoint identity/arch", "PASS" if ok else "FAIL",
                detail + "  (expected: patch 16, 2x2 merge -> 1 token per 32x32 px)"))
    # item 2: module paths used by model.py
    missing = []
    for path in ("model.visual", "model.visual.blocks", "model.visual.merger",
                 "model.visual.deepstack_merger_list", "model.language_model.layers",
                 "model.language_model.norm", "lm_head"):
        obj = m
        for part in path.split("."):
            if not hasattr(obj, part):
                missing.append(path); break
            obj = getattr(obj, part)
    if missing:
        tree = [n for n, _ in m.model.named_children()]
        out.append(("2 module paths", "FAIL", f"missing {missing}; actual children: {tree}"))
    else:
        out.append(("2 module paths", "PASS",
                    "visual.blocks/.merger/.deepstack_merger_list, language_model.layers/.norm, lm_head"))
    # item 5: DeepStack injection outside decoder-layer modules
    lm_children = [n for n, _ in m.model.language_model.named_children()]
    ds_outside = not any("deepstack" in c for c in lm_children)
    out.append(("5 DeepStack placement", "PASS" if ds_outside else "WARN",
                f"language_model children: {lm_children}; layer hooks 0-2 see states WITHOUT the "
                f"injection (design §5.3) -> patching windows start at layer >= 3"))
    # item 7 (config part): video processor temporal patch
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(mp)
    tps = getattr(p.image_processor, "temporal_patch_size", None)
    out.append(("7 video processor (config)", "PASS" if tps == 2 else "FAIL",
                f"temporal_patch_size={tps} (odd frame lists must be padded to even)"))
    # item 8: answer tokens single
    tok = p.tokenizer
    bad = {w: tok.encode(w, add_special_tokens=False)
           for w in ("Yes", "yes", "No", "no", " Yes", " No")
           if len(tok.encode(w, add_special_tokens=False)) != 1}
    out.append(("8 answer tokens single", "PASS" if not bad else "FAIL",
                f"multi-token variants: {bad}" if bad else "Yes/yes/No/no/' Yes'/' No' all single-token"))
    # item 9: benchmark [VERIFY] items live in adapter docstrings
    from ea import data as D
    out.append(("9 benchmark formats", "DEFERRED",
                "recorded in adapter docstrings (data.py); adapters stay stubs until access exists"))
    return out, cfg


def checks_load(mp):
    """Items 3, 4, 6, 7 (behaviour), 10 — one real forward on a toy image/video."""
    out = []
    import numpy as np
    from PIL import Image
    from ea.model import VLM
    from ea.prompts import PROMPTS
    with tempfile.TemporaryDirectory() as td:
        from ea.data import make_toy
        S = make_toy(4, root=Path(td) / "toy", seed=0)
        vlm = VLM(mp, quant="8bit", attn="eager")            # eager: needed by item 6
        img = Image.open(S[0].media)
        # item 6: eager attention passes a materialized 4-D mask kwarg and returns weights
        seen = {}
        L = vlm.m.model.language_model.layers[0]
        def pre(mod, args, kwargs):
            seen["mask"] = kwargs.get("attention_mask")
            seen["positional"] = "attention_mask" not in kwargs and len(args) > 1
            return args, kwargs
        h = L.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
        captured = {}
        def hook(mod, a, kw, o):
            captured["out"] = o
        h2 = L.self_attn.register_forward_hook(hook, with_kwargs=True)
        inp = vlm.prep(PROMPTS["d0"]["text"].format(rule_text=S[0].rule_text), images=[img])
        with torch.no_grad():
            vlm.m(**inp, logits_to_keep=1)
        h.remove(); h2.remove()
        mask = seen.get("mask"); wts = captured.get("out")
        w = wts[1] if isinstance(wts, tuple) and len(wts) > 1 else None
        ok6 = (mask is not None and mask.dim() == 4 and w is not None)
        out.append(("6 eager mask+weights", "PASS" if ok6 else "FAIL",
                    f"mask: {'4D' if mask is not None and mask.dim() == 4 else type(mask).__name__ + str(None if mask is None else mask.dim()) + 'D'}, "
                    f"attn weights returned: {w is not None}"))
        # item 3: visual-token order == positions of image_token_id, row-major
        g = vlm.groups(inp)
        ids = inp["input_ids"][0]
        ok3 = torch.equal(g["vis"], (ids == vlm.m.config.image_token_id).nonzero().squeeze(1))
        t, hh, ww = g["grid"]
        out.append(("3 token order", "PASS" if ok3 and g["vis"].numel() == t * hh * ww else "FAIL",
                    f"vis positions == image_token_id positions: {bool(ok3)}; n_vis={g['vis'].numel()} == t*h/2*w/2 = {t*hh*ww}"))
        # item 4: pre-merger count == 4x merged count (Tap arrives with mech.py, phase 9)
        try:
            from ea.model import Tap
            with Tap(vlm, vit=True) as tap:
                with torch.no_grad():
                    vlm.m(**inp, logits_to_keep=1)
            pre_n = tap.site["vit"].shape[0]
            ok4 = pre_n == 4 * g["vis"].numel()
            out.append(("4 pre-merger 4x", "PASS" if ok4 else "FAIL",
                        f"pre-merger {pre_n} vs 4*n_vis {4 * g['vis'].numel()}"))
        except ImportError:
            out.append(("4 pre-merger 4x", "DEFERRED", "Tap arrives with mech.py (phase 9)"))
        # item 7 (behaviour): video prep with a frame list + timestamps survive
        try:
            arr = [np.asarray(img.resize((64, 64)))] * 2
            vin = vlm.prep("dummy", frames=arr)
            g2 = vlm.groups(vin)
            out.append(("7 video prep", "PASS" if g2["grid"] is not None else "FAIL",
                        f"video grid {g2['grid']}"))
        except Exception as e:
            out.append(("7 video prep", "FAIL", repr(e)))
        # item 10: answer mass >= min_mass on a calibration sample per group
        import yaml as _y
        min_mass = _y.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))["answer"]["min_mass"]
        fails = []
        for pid, pr in PROMPTS.items():
            txt = P_render(pr, S[0])
            if txt is None:
                continue
            media = {} if not pr["img"] else dict(images=[img])
            r = vlm.score(txt, **media, pos=pr["pos"], neg=pr["neg"])
            if r["mass"] < min_mass:
                fails.append((pid, r["mass"]))
        out.append(("10 answer mass", "PASS" if not fails else "FAIL",
                    f"mass >= {min_mass} on calibration sample for all prompt groups"
                    if not fails else f"low mass: {fails}"))
    return out


def P_render(pr, s):
    from ea.prompts import render
    return render(pr, s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true",
                    help="also run the forward-pass checks (items 3,4,6,7,10); needs GPU/CPU weights")
    args = ap.parse_args(argv)
    out, _cfg = checks_default()
    if args.load:
        cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
        out += checks_load(_model_ref(cfg))
    else:
        out += [("3 token order", "DEFERRED", "needs a forward pass: rerun with --load"),
                ("4 pre-merger 4x", "DEFERRED", "needs a forward pass: rerun with --load"),
                ("6 eager mask+weights", "DEFERRED", "needs a forward pass: rerun with --load"),
                ("10 answer mass", "DEFERRED", "needs a forward pass: rerun with --load"),
                ("11 nli port vs ROAD", "DEFERRED", "validated with T3 (design §8 step 8)"),
                ("12 upper-half band / q routing", "DEFERRED", "checked with attn_last/knockout on 2 real images (T3)")]
    bad = 0
    print(f"{'item':34} {'status':9} detail")
    for name, status, detail in out:
        print(f"{name:34} {status:9} {detail}")
        bad += status == "FAIL"
    print(("selfcheck FAILED (%d)" % bad) if bad else "selfcheck OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
