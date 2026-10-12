"""tests/test_run.py — Phase 3 tests for run.py (FR-R-*).

FakeVLM-free (a trivial scripted model stands in): resumability/idempotence
(FR-R-2), row schema (FR-R-3), failure isolation (FR-R-5), mass validity flag
(FR-R-7), gold-under-intervention semantics, majority-vote errors table,
matched controls (excludes the error itself, same cls), manifest.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from ea import data as D
from ea import run as R
from ea.prompts import PROMPTS


class ScriptedVLM:
    """Minimal stand-in with the VLM.score interface: s depends only on the
    prompt's pos/neg and the sample's gold (sign flipped for arm='O3.blank')."""

    def __init__(self, s=1.5, mass=0.99):
        self.s, self.mass = s, mass

    def score(self, text, images=None, frames=None, pos=("Yes",), neg=("No",)):
        s = self.s if images is not None else self.s + 10.0   # text-only (qr) is easy
        return dict(s=s, mass=self.mass, n_vis=64, ms=1.0, mem=0.0)


def test_plan_t0_shape():
    S = D.make_toy(10, seed=0)
    jobs = R.plan_t0({"model": {"name": "x"}}, S)
    assert len(jobs) == 30                              # 3 prompts x 10 samples
    assert {j.prompt for j in jobs} == {"d0", "d1", "d2"}
    assert all(j.arm == "base" for j in jobs)


def test_run_resumable_idempotent(tmp_path):
    S = D.make_toy(8, root=tmp_path / "toy", seed=0)
    out = tmp_path / "runs.jsonl"
    n1 = R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.5}}, out=out,
               vlm_factory=lambda mc: ScriptedVLM())
    assert n1 == 24
    n2 = R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.5}}, out=out,
               vlm_factory=lambda mc: ScriptedVLM())     # re-run: all keys known
    assert n2 == 0
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 24 and len({r["key"] for r in rows}) == 24   # no duplicates


def test_row_schema(tmp_path):
    S = D.make_toy(4, root=tmp_path / "toy", seed=1)
    out = tmp_path / "runs.jsonl"
    R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.5}}, out=out,
          vlm_factory=lambda mc: ScriptedVLM())
    row = json.loads(open(out, encoding="utf-8").readline())
    need = {"key", "sample_id", "prompt_id", "group", "arm", "knob", "value",
            "model_cfg", "s", "pred", "gold", "correct", "n_vis_tokens",
            "latency_ms", "t", "extra", "valid"}
    assert need <= set(row), need - set(row)
    assert row["extra"] and json.loads(row["extra"])["mass"] == 0.99


def test_failure_isolated(tmp_path):                     # FR-R-5
    S = D.make_toy(4, root=tmp_path / "toy", seed=2)
    S[1].rule_text = None                               # d0 needs rule_text -> render None -> skip
    out, fails = tmp_path / "runs.jsonl", tmp_path / "failures.jsonl"
    n = R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.5}}, out=out,
              vlm_factory=lambda mc: ScriptedVLM())
    assert n == 9                                        # 3 samples x 3 prompts
    assert fails.exists() and sum(1 for _ in open(fails)) == 3
    runs = pd.read_json(out, lines=True)
    assert (runs.sample_id == S[1].id).sum() == 0


class BadMediaVLM:
    def score(self, text=None, **kw):
        raise RuntimeError("boom")


def test_exception_logged_not_fatal(tmp_path):
    S = D.make_toy(2, root=tmp_path / "toy", seed=3)
    out, fails = tmp_path / "runs.jsonl", tmp_path / "failures.jsonl"
    n = R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.5}}, out=out,
              vlm_factory=lambda mc: BadMediaVLM())
    assert n == 0
    assert all("boom" in l for l in open(fails, encoding="utf-8"))


def test_low_mass_invalid(tmp_path):                     # FR-R-7 / C16
    S = D.make_toy(4, root=tmp_path / "toy", seed=4)
    out = tmp_path / "runs.jsonl"
    R.run(R.plan_t0({}, S), S, {"answer": {"min_mass": 0.995}}, out=out,
          vlm_factory=lambda mc: ScriptedVLM(mass=0.5))
    df = pd.read_json(out, lines=True)
    assert (df.valid == False).all()                     # mass 0.5 < 0.995 -> invalid


def test_errors_table_majority(tmp_path):
    rows = []
    for i in range(6):
        preds = [1, 1, 0] if i < 2 else [0, 0, 0] if i < 4 else [1, 1, 1]
        for pid, pr in zip(("d0", "d1", "d2"), preds):
            g = 1 if i % 2 == 0 else 0
            rows.append(dict(sample_id=f"s{i}", prompt_id=pid, group="D", arm="base",
                             knob=None, value=None, s=float(pr) or -1, pred=pr, gold=g,
                             correct=int(pr == g), valid=True, n_vis_tokens=8,
                             latency_ms=1, t=None, extra="{}"))
    runs = pd.DataFrame(rows)
    S = [D.Sample(id=f"s{i}", bench="toy", modality="image", media="x.png",
                  label=i % 2, cls="c", rule_text="r", item="helmet")
         for i in range(6)]
    et = R.errors_table(runs, S)
    assert len(et) == 6
    # majority: s0,s1 vote 1; s2,s3 vote 0; s4,s5 vote 1
    assert et.pred.tolist() == [1, 1, 0, 0, 1, 1]
    assert et.err.tolist() == [0, 1, 1, 0, 0, 1]        # vs gold 1,0,1,0,1,0


def test_match_excludes_self_and_matches_cls():
    cor = pd.DataFrame(dict(sample_id=[f"c{i}" for i in range(10)],
                            cls=["a"] * 5 + ["b"] * 5,
                            obj_size_bin=["small"] * 5 + ["large"] * 5))
    err = pd.DataFrame([dict(sample_id="e0", cls="a", obj_size_bin="small")])
    ids, fb = R.match(err, cor, k=2, seed=0)
    assert "e0" not in ids
    assert all(i.startswith("c") and int(i[1:]) < 5 for i in ids)   # same cls+bin
    # fallback when the exact bin is empty
    err2 = pd.DataFrame([dict(sample_id="e1", cls="b", obj_size_bin="small")])
    ids2, fb2 = R.match(err2, cor, k=1, seed=0)
    assert fb2 == 1 and all(i in {f"c{i}" for i in range(5, 10)} for i in ids2)


def test_manifest_written(tmp_path):
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).parents[1] / "ea" / "config.yaml", encoding="utf-8"))
    man = R.manifest(cfg, path=tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").exists()
    assert man["model"] == cfg["model"]["name"] and man["seed"] == cfg["seed"]
    assert len(man["prompt_version"]) == 8
