"""tests/test_model_gpu.py — Phase 2 GPU smoke tests (marker `gpu`; design §7.5).

Cover FR-M-2/3/5 on the real checkpoint: margin == manual logit difference;
answer mass in (0,1]; groups() count == t*(h/2)*(w/2); determinism |Δs|<0.05;
prompt ends with the assistant generation prefix (selfcheck item 10).
Run:  pytest -m gpu tests/test_model_gpu.py
Skipped automatically without a CUDA device.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("torch")
if not torch.cuda.is_available():                     # graceful skip on CPU-only hosts
    pytest.skip("no CUDA device", allow_module_level=True)

from ea.data import make_toy
from ea.model import VLM
from ea.prompts import PROMPTS, render

import os
from pathlib import Path
import yaml
from ea.selfcheck import _model_ref


def _resolve_test_model():
    if env_p := os.environ.get("MODEL_PATH"):
        p = Path(env_p)
        if p.exists() or (not env_p.startswith("/") and not env_p.startswith(".") and not ":" in env_p):
            return env_p
    cfg_p = Path(__file__).resolve().parents[1] / "ea" / "config.yaml"
    if cfg_p.exists():
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        return _model_ref(cfg)
    return "Qwen/Qwen3-VL-4B-Instruct"


MP = _resolve_test_model()


@pytest.fixture(scope="module")
def vlm():
    return VLM(MP, quant="8bit")                      # 8-bit: 6 GB GPU cannot hold bf16


@pytest.fixture(scope="module")
def toy():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        S = make_toy(4, root=Path(td) / "toy", seed=0)
        yield S


def test_margin_equals_manual_logit_diff(vlm, toy):   # FR-M-2 AC
    from PIL import Image
    s0 = next(s for s in toy if s.item and s.rule_text)
    txt = render(PROMPTS["d0"], s0)
    r = vlm.score(txt, images=[Image.open(s0.media)], pos=PROMPTS["d0"]["pos"], neg=PROMPTS["d0"]["neg"])
    inp = vlm.prep(txt, images=[Image.open(s0.media)])
    with torch.no_grad():
        lg = vlm.m(**inp, logits_to_keep=1).logits[0, -1].float()
    manual = (torch.logsumexp(lg[vlm.ids(("Yes", "yes"))], 0)
              - torch.logsumexp(lg[vlm.ids(("No", "no"))], 0)).item()
    assert abs(r["s"] - manual) < 1e-3
    assert 0.0 < r["mass"] <= 1.0


def test_groups_count_and_grid(vlm, toy):             # FR-M-3 AC
    from PIL import Image
    s0 = next(s for s in toy if s.item and s.rule_text)
    inp = vlm.prep(render(PROMPTS["d0"], s0), images=[Image.open(s0.media)])
    g = vlm.groups(inp)
    t, h, w = g["grid"]
    assert g["vis"].numel() == t * h * w
    assert g["q"].numel() + g["vis"].numel() + 2 <= inp["input_ids"].numel()  # vis + q + last (+start/end tags)


def test_determinism(vlm, toy):                       # FR-M-5 AC
    from PIL import Image
    s0 = next(s for s in toy if s.item and s.rule_text)
    img = Image.open(s0.media)
    txt = render(PROMPTS["d0"], s0)
    r1 = vlm.score(txt, images=[img], pos=PROMPTS["d0"]["pos"], neg=PROMPTS["d0"]["neg"])
    r2 = vlm.score(txt, images=[img], pos=PROMPTS["d0"]["pos"], neg=PROMPTS["d0"]["neg"])
    assert abs(r1["s"] - r2["s"]) < 0.05              # bf16 tolerance (pitfall 12)
    assert r1["ms"] > 0


def test_prompt_ends_with_generation_prefix(vlm, toy):  # selfcheck item 10 / FR-P-1
    from PIL import Image
    s0 = next(s for s in toy if s.item and s.rule_text)
    txt = render(PROMPTS["d0"], s0)
    ptxt = vlm.p.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}],
        tokenize=False, add_generation_prompt=True)
    assert ptxt.rstrip().endswith("<|im_start|>assistant")   # first generated token is the answer


def test_t0_on_real_model(tmp_path, vlm, toy):            # milestone M1 gate
    from ea import run as R
    cfg = {"answer": {"min_mass": 0.5}, "model": {"name": "Qwen3-VL-4B-Instruct"}}
    S = toy[:2]
    n = R.run(R.plan_t0(cfg, S), S, cfg, out=tmp_path / "runs.jsonl",
              vlm_factory=lambda mc: vlm)
    assert n == 6                                         # 2 samples x 3 prompts
    df = R.load_runs(tmp_path / "runs.jsonl")
    assert df.valid.all() and df.n_vis_tokens.gt(0).all() and df.latency_ms.gt(0).all()
    man = R.manifest(cfg, path=tmp_path / "manifest.json")  # FR-R-4
    assert man["prompt_version"] and man["gpu"]
