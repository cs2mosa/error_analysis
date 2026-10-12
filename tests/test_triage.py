"""tests/test_triage.py — Phase 4 tests for triage.py (FR-T1-2, FR-T1-4).

Exhaustive 8-case unit test + the None case (AC); perception/knowledge gate.
"""
import pandas as pd
import pytest

from ea.triage import triage, gate_ok, NEXT


def test_triage_exhaustive_8_cases():                    # FR-T1-2 AC (design §5.8 literal table)
    # (qp_ok, qr_ok, qd_ok) -> reading
    cases = {
        (True, True, True): "ok",                        # Q_R irrelevant when Q_D+Q_P correct
        (True, False, True): "ok",
        (False, True, True): "right_for_wrong_reason",   # Q_D correct but perception failed
        (False, False, True): "right_for_wrong_reason",
        (False, True, False): "perception_failure",      # Q_R irrelevant
        (False, False, False): "perception_failure",
        (True, True, False): "process_failure",          # 2609.22588 split
        (True, False, False): "rule_reasoning_failure",
    }
    for (qp, qr, qd), expected in cases.items():
        assert triage(qp, qr, qd) == expected, (qp, qr, qd)


def test_triage_untriaged_no_item():                     # FR-T1-2 AC (C7)
    assert triage(None, True, False) == "untriaged (no item)"
    assert triage(None, None, None) == "untriaged (no item)"
    assert "untriaged (no item)" in NEXT                  # counted, never dropped


def test_triage_is_provisional_only():
    # the reading must not be called a cause: the signature table
    # (escalate.decide) is the only cause assigner (FR-E-5)
    assert "PROVISIONAL" in triage.__doc__.upper()


def _runs_for_gate(qp, qp_abs, qr, sid="s0"):
    rows = []
    for pid, correct in (("qp", qp), ("qp_abs", qp_abs), ("qr", qr)):
        rows.append(dict(sample_id=sid, prompt_id=pid, group="QP" if pid.startswith("qp") else "QR",
                         arm="base", s=1.0 if correct else -1.0, pred=int(correct),
                         gold=1 if correct else 0, correct=int(correct), valid=True))
    return pd.DataFrame(rows)


def test_gate_ok_all_pass():                             # FR-T1-4
    assert gate_ok(_runs_for_gate(True, True, True), "s0") is True


def test_gate_fails_on_perception():                     # C6: perception failure ≠ prior
    assert gate_ok(_runs_for_gate(False, True, True), "s0") is False
    assert gate_ok(_runs_for_gate(True, False, True), "s0") is False   # negation partner fails


def test_gate_fails_on_rule_knowledge():
    assert gate_ok(_runs_for_gate(True, True, False), "s0") is False


def test_gate_fails_when_prompts_missing():              # incomplete rows -> gate closed
    r = _runs_for_gate(True, True, True)
    assert gate_ok(r[r.prompt_id != "qr"], "s0") is False
    assert gate_ok(r[r.prompt_id != "qp_abs"], "s0") is False


def test_gate_ignores_invalid_rows():                    # low-mass rows cannot support the gate
    r = _runs_for_gate(True, True, True)
    r.loc[r.prompt_id == "qr", "valid"] = False
    assert gate_ok(r, "s0") is False
