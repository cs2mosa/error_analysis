"""tests/test_plans.py — Phase 4 tests for the T1/T2 job plans (FR-T2-1/2).

Every arm appears with its control; both errors and matched corrects are
covered (symmetry); N0.null rows exist; O2 uses the qr prompt; quality sweep
expands from config; shared controls dedupe through job keys.
"""
from collections import Counter

import pytest

from ea.data import make_toy
from ea.interventions import INTERV
from ea.run import Job, plan_t1, plan_t2, key, BASE_MCFG


@pytest.fixture(scope="module")
def toy():
    import tempfile
    from pathlib import Path
    td = tempfile.mkdtemp()
    S = make_toy(40, root=Path(td) / "toy", seed=0)
    return S


def test_plan_t1_prompts_on_error_ids(toy):
    S = toy
    err_ids = {s.id for s in S[:5]}
    jobs = plan_t1({}, S, err_ids)
    triage_prompts = {"qp", "qp_abs", "qr", "pol_pos", "pol_neg", "pol_miss",
                      "frm_neu", "frm_pos", "frm_neg"}
    got = {j.prompt for j in jobs if j.arm == "base"}
    assert got == triage_prompts                        # FR-T1-1: rows exist for all
    assert all(j.sid in err_ids for j in jobs if j.arm == "base")
    blank = [j for j in jobs if j.arm == "O3.blank"]
    assert {j.prompt for j in blank} == {"d0", "d1", "d2"}   # flip-rate baseline, ALL samples
    assert len(blank) == 3 * len(S)


def test_plan_t2_all_arms_with_controls(toy):
    S = toy
    ids = {s.id for s in S[:4]}
    jobs = plan_t2({"knobs": {"quality": {"kinds": ["blur", "jpeg"], "severities": [1, 3]}}}, S, ids)
    arms = Counter(j.arm for j in jobs)
    # every non-INJ arm with a plan presence and its control present
    for name, spec in INTERV.items():
        if name.startswith(("INJ.", "HZ.")) or name in ("O3.swap", "O1.attncrop"):
            continue                                      # INJ: validate only; HZ: plan_hazard (§5.14); swap/attncrop need extras
        if name == "O6.degrade":
            assert arms[name] == 4 * 2 * 2               # 4 ids x 2 kinds x 2 severities
        else:
            assert arms[name] >= 1, name
        if spec["control"] and spec["control"] != "N0.null":
            assert arms[spec["control"]] >= 1, (name, spec["control"])
    assert arms["N0.null"] == 4                           # ONE noise-floor row per sample (C4)
    assert "O1.attncrop" not in arms                      # P1 arm: needs the T3 attention pass
    assert all(j.sid in ids for j in jobs)               # symmetry: errors ∪ matched corrects only


def test_plan_t2_o2_uses_qr_prompt(toy):
    S = toy
    jobs = plan_t2({"knobs": {"quality": {"kinds": [], "severities": []}}}, S, {S[0].id})
    o2 = [j for j in jobs if j.arm == "O2.facts"]
    assert o2 and all(j.prompt == "qr" for j in o2)      # Q_R and O2 are the same prompt (C2)
    wf = [j for j in jobs if j.arm == "O2.wrongfacts"]
    assert wf and all(j.prompt == "qr" for j in wf)


def test_plan_t2_quality_sweep_from_config(toy):
    S = toy
    cfg = {"knobs": {"quality": {"kinds": ["haze"], "severities": [2, 4]}}}
    jobs = plan_t2(cfg, S, {S[0].id})
    dg = {(j.params["kind"], j.params["sev"]) for j in jobs if j.arm == "O6.degrade"}
    assert dg == {("haze", 2), ("haze", 4)}


def test_shared_control_dedupes_by_key(toy):
    S = toy
    ids = {s.id for s in S[:3]}
    jobs = plan_t2({"knobs": {"quality": {"kinds": [], "severities": []}}}, S, ids)
    keys = [key(j) for j in jobs]
    assert len(keys) == len(set(keys))                   # idempotent runner requires unique keys


def test_swap_gets_other_media(toy):
    S = toy
    jobs = plan_t2({"knobs": {"quality": {"kinds": [], "severities": []}}}, S, {S[0].id})
    sw = [j for j in jobs if j.arm == "O3.swap"]
    assert sw and sw[0].params.get("other")              # deterministic opposite-label media
    other = next(x for x in S if x.media == sw[0].params["other"])
    assert other.label != S[0].label
