"""triage.py — the T1 Q_P/Q_R/Q_D reading table (FR-T1-2) and the
perception/knowledge gate (FR-T1-4, C6).

Purpose:  map (qp_ok, qr_ok, qd_ok) to the plan's provisional reading:
          perception failure / process failure / rule-reasoning failure /
          right-for-wrong-reason / ok; `untriaged (no item)` when Q_P is
          undefined (proximity/zone violations without an `item`).
          The reading is a PROVISIONAL LABEL only — a cause comes from the
          signature table (escalate.decide, FR-E-5).
Implements plan sections: §7 T1; perception-vs-process split of 2609.22588.
Inputs:   three-valued booleans or None (qp_ok, qr_ok, qd_ok).
Outputs:  reading string; suggested next tier ("Next" column, FR-T1-3).
"""

NEXT = dict(perception_failure="O1,O6 → T4",
            process_failure="O1.bgmask, O1.gtcrop and hires must NOT recover → then T3/T4",
            rule_reasoning_failure="prompt/rule representation",
            right_for_wrong_reason="audit",
            ok="—",
            **{"untriaged (no item)": "per-person/zone arms (O5) where boxes exist"})


def triage(qp_ok, qr_ok, qd_ok):
    """The plan's 8-case table + the None case (FR-T1-2 AC: exhaustive).
    PROVISIONAL label only — a cause comes from the signature table (FR-E-5).

    qp_ok = qp ∧ qp_abs both correct (C7). `untriaged (no item)` when qp_ok
    is None (Q_P undefined: no PPE item for proximity/zone violations) —
    counted, never silently dropped.
    """
    if qp_ok is None:
        return "untriaged (no item)"
    if qd_ok:
        return "ok" if qp_ok else "right_for_wrong_reason"
    if not qp_ok:
        return "perception_failure"          # Q_R irrelevant: cannot read the rule without the item
    return "process_failure" if qr_ok else "rule_reasoning_failure"


def gate_ok(runs, sample_id, prompts=("qp", "qp_abs", "qr")):
    """Perception/knowledge gate (FR-T1-4, C6): a 'prior-driven' (necessity)
    diagnosis is only issued for samples where Q_P (both polarities) and Q_R
    are correct — otherwise a perception failure is mistaken for a prior
    (2406.08702). Gated-out samples are counted by the caller (NFR-5)."""
    sub = runs[(runs.sample_id == sample_id) & (runs.arm == "base") & runs.valid.fillna(True)]
    r = sub[sub.prompt_id.isin(prompts)]
    return len(r) == len(prompts) and bool(r.correct.all())
