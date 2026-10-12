"""tests/test_prompts.py — Phase 1 tests for prompts.py (FR-P-*).

Groups, single-Yes/No answerability, computed version hash, render skips
(FR-P-3), polarity/gold semantics (C7, §5.2 conventions).
"""
import pytest

import ea.prompts as P
from ea.data import make_toy


def test_groups_present():                                      # FR-P-2
    groups = {p["group"] for p in P.PROMPTS.values()}
    assert {"D", "POL", "FRM", "QP", "QR"} <= groups
    assert sum(p["group"] == "D" for p in P.PROMPTS.values()) >= 3   # >= 3 paraphrases
    for p in P.PROMPTS.values():
        assert p["pos"] and p["neg"] and callable(p["gold"])    # single Yes/No answerable
        assert ("Yes" in p["text"] and "No" in p["text"]) or p["group"] == "QR"


def test_prompt_version_computed_not_hand_bumped():             # FR-P-1
    v1 = P.prompt_version()
    assert v1 == P.PROMPT_VERSION and len(v1) == 8
    modified = {k: dict(v) for k, v in P.PROMPTS.items()}
    modified["d0"] = dict(modified["d0"], text=modified["d0"]["text"] + " ")
    assert P.prompt_version(modified) != v1                     # template change -> hash change
    flipped = {k: dict(v) for k, v in P.PROMPTS.items()}
    flipped["qp"] = dict(flipped["qp"], pos=["No"], neg=["Yes"])
    assert P.prompt_version(flipped) != v1                     # polarity change -> hash change


def test_render_skips_missing_fields():                         # FR-P-3
    S = make_toy(30, seed=2)
    no_item = [s for s in S if s.item is None]
    if not no_item:                                             # toy always has an item; synthesize one
        s0 = S[0]
        s0.item = None
        no_item = [s0]
    s = no_item[0]
    assert P.render(P.PROMPTS["qp"], s) is None                 # {item} missing -> skipped, not errored
    assert P.render(P.PROMPTS["pol_pos"], s) is None
    assert P.render(P.PROMPTS["qr"], s) is not None or s.rule_text is None
    s2 = next(s for s in S if s.item and s.rule_text)
    for pid in ("d0", "d1", "d2", "pol_pos", "pol_neg", "pol_miss",
                "frm_neu", "frm_pos", "frm_neg", "qp", "qp_abs", "qr"):
        assert P.render(P.PROMPTS[pid], s2) is not None, pid    # all render on a complete sample


def test_qr_is_text_only():                                     # FR-P-2, O2 (img=False); hz_r likewise (FR-H-3)
    assert P.PROMPTS["qr"]["img"] is False
    assert P.PROMPTS["hz_r"]["img"] is False
    assert all(p["img"] for pid, p in P.PROMPTS.items() if pid not in ("qr", "hz_r"))


def test_polarity_convention():                                 # §4: s>0 always asserts the positive class
    S = make_toy(30, seed=2)
    s = next(x for x in S if x.item)
    # pol_pos asks "wearing?" -> violation answer is "No" -> pos = No variants
    assert P.PROMPTS["pol_pos"]["pos"] == P.N and P.PROMPTS["pol_pos"]["gold"](s) == s.label
    assert P.PROMPTS["pol_neg"]["pos"] == P.Y                   # "NOT wearing" -> violation answer is "Yes"
    qp, qp_abs = P.PROMPTS["qp"], P.PROMPTS["qp_abs"]
    assert qp["gold"](s) + qp_abs["gold"](s) == 1               # C7: negation partner, both pos=Y
    assert qp["pos"] == P.Y and qp_abs["pos"] == P.Y


def test_present_treats_not_visible_as_absent():                # §5.2 note
    class S:
        item = "helmet"
    S.facts = {"helmet": "not_visible"}
    assert P.present(S) == 0                                    # not_visible -> not present
    S.facts = {"helmet": "present"}
    assert P.present(S) == 1
    S.facts = {}
    assert P.present(S) == 0                                    # missing facts never crash
