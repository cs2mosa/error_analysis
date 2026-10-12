"""prompts.py — the versioned prompt dict and renderer.

Purpose:  one source of truth for prompts (FR-P-1/2): groups D, POL, FRM, QP,
          QR; render() skips prompts whose fields are missing (FR-P-3);
          PROMPT_VERSION is computed from the templates, never hand-bumped.
Implements plan sections: §6.1 margin convention, §7 T0/T1 prompt groups.
Inputs:   Sample objects (data.py).
Outputs:  rendered prompt text or None; prompt ids; version hash.

Convention (design §4): pos/neg are answer-word lists; for prompts where the
violation answer is "No", pos is the "No" variant, so **s > 0 always means
"model asserts the prompt's positive class"** and gold(sample) is defined
accordingly. `qp_ok` = qp AND qp_abs both correct (C7, negation weakness
2501.09425). The latent-hazard prompts (FR-H-3) are added with §5.14.
"""

import json
import re
from hashlib import md5

Y, N = ["Yes", "yes"], ["No", "no"]
viol = lambda s: int(s.label)
present = lambda s: int(s.facts.get(s.item, "not_visible") == "present")  # not_visible -> not present (§5.2)
D = "Safety rule: {rule_text}\n"

# Latent-hazard track (§5.14, FR-H-3): the design's `hz_s_<k>` prompts are
# static slots for the first HZ_CONDITION_SLOTS conditions; raise the constant
# to widen (logged in DEVIATIONS.md). hz_gold: for hazard samples `label` IS
# hazard_gold (FR-H-1), so the uncued/cued decision gold is the sample label.
HZ_CONDITION_SLOTS = 2


def _hz_cond(s, k):
    cs = s.facts.get("conditions") if isinstance(s.facts, dict) else None
    return cs[k] if isinstance(cs, list) and k < len(cs) and isinstance(cs[k], dict) else None


def _hz_present(k):
    def g(s):
        c = _hz_cond(s, k)
        return int(bool(c) and c.get("present") == "present")
    return g


def _hz_all_present(s):
    """hz_r gold: the consequence can occur iff every oracle condition holds
    (physics chain intact); derived from the TEXT facts, not the label."""
    cs = s.facts.get("conditions") if isinstance(s.facts, dict) else None
    return int(bool(cs) and all(isinstance(c, dict) and c.get("present") == "present" for c in cs))


hz_gold = lambda s: int(s.label)

_PROMPT_LIST = [
    # ---- D: >= 3 paraphrases of the violation question (FR-P-2)
    dict(id="d0", group="D", text=D + "Does this image show a violation of the rule? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    dict(id="d1", group="D", text="Is the following rule being violated in this image: {rule_text}? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    dict(id="d2", group="D", text="As a site safety inspector, would you flag this scene for: {rule_text}? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    # ---- POL: polarity flips (H6). pol_pos: "No" asserts the violation -> pos=N
    dict(id="pol_pos", group="POL", text="Is the worker wearing {item}? Answer Yes or No.",
         pos=N, neg=Y, gold=viol, img=True),
    dict(id="pol_neg", group="POL", text="Is the worker NOT wearing {item}? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    dict(id="pol_miss", group="POL", text="Is any required protective equipment missing? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    # ---- FRM: leading-positive / neutral / leading-negative framing (H7)
    dict(id="frm_neu", group="FRM", text=D + "Is the rule violated? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    dict(id="frm_pos", group="FRM", text="This scene very likely contains a violation. " + D + "Is the rule violated? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    dict(id="frm_neg", group="FRM", text="This site has an excellent safety record. " + D + "Is the rule violated? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=True),
    # ---- QP: perception sub-question in BOTH polarities (C7)
    dict(id="qp", group="QP", text="Is {item} clearly visible on the worker? Answer Yes or No.",
         pos=Y, neg=N, gold=present, img=True),
    dict(id="qp_abs", group="QP", text="Is the worker missing {item}? Answer Yes or No.",
         pos=Y, neg=N, gold=lambda s: 1 - present(s), img=True),
    # ---- QR: rule + oracle facts in text, no image (O2; exclusion test only, FR-T2-6)
    dict(id="qr", group="QR", text="Rule: {rule_text}\nFacts: {facts_text}\nDoes this situation violate the rule? Answer Yes or No.",
         pos=Y, neg=N, gold=viol, img=False),
    # ---- HZ: latent-hazard track (§5.14, FR-H-3). hz_d = uncued Q_D (never
    # names the trigger/consequence); hz_cue = cued arm (leaks them on purpose
    # to split cannot-infer from does-not-volunteer); hz_r = text-only Q_R
    # (exclusion test only, FR-T2-6); hz_s<k>(_abs) = per-condition Q_P in both
    # polarities (C7); qp_ok for hazards = ALL of them correct (conjunction).
    dict(id="hz_d", group="HZ",
         text="Does this scene contain a condition that could foreseeably lead to an incident "
              "(for example something falling) under ordinary conditions such as wind, vibration "
              "or contact? Answer Yes or No.",
         pos=Y, neg=N, gold=hz_gold, img=True),
    dict(id="hz_cue", group="HZ",
         text="Could {consequence} happen here under {trigger}? Answer Yes or No.",
         pos=Y, neg=N, gold=hz_gold, img=True),
    dict(id="hz_r", group="HZR",
         text="Facts: {hz_conditions_text}\nTrigger: {trigger}\nCould {consequence} occur? Answer Yes or No.",
         pos=Y, neg=N, gold=_hz_all_present, img=False),
]
for _k in range(HZ_CONDITION_SLOTS):
    _PROMPT_LIST.append(dict(id=f"hz_s{_k}", group="HZP",
                             text=f"Is {{obj_{_k}}} {{state_{_k}}}? Answer Yes or No.",
                             pos=Y, neg=N, gold=_hz_present(_k), img=True))
    _PROMPT_LIST.append(dict(id=f"hz_s{_k}_abs", group="HZP",
                             text=f"Is {{obj_{_k}}} NOT {{state_{_k}}}? Answer Yes or No.",
                             pos=Y, neg=N, gold=lambda s, _k=_k: 1 - _hz_present(_k)(s), img=True))

PROMPTS = {p["id"]: p for p in _PROMPT_LIST}


def render(p, s):
    """Render prompt p for sample s; None if a required field is missing (FR-P-3)."""
    f = dict(rule_text=s.rule_text, item=s.item,
             facts_text="; ".join(f"{k}: {v}" for k, v in s.facts.items()) or None)
    cs = s.facts.get("conditions") if isinstance(s.facts, dict) else None
    cs = cs if isinstance(cs, list) else []
    for k in range(HZ_CONDITION_SLOTS):                     # hazard-track fields (§5.14)
        c = cs[k] if k < len(cs) and isinstance(cs[k], dict) else None
        f[f"obj_{k}"] = c.get("obj") if c else None
        f[f"state_{k}"] = c.get("state") if c else None
    f["trigger"] = (s.meta or {}).get("trigger")
    f["consequence"] = (s.meta or {}).get("consequence")
    f["hz_conditions_text"] = ("; ".join(f"{c.get('obj')} {c.get('state')}: {c.get('present')}"
                                         for c in cs if isinstance(c, dict)) or None)
    need = re.findall(r"{(\w+)}", p["text"])
    return None if any(f.get(k) is None for k in need) else p["text"].format(**f)


def prompt_version(prompts=None) -> str:
    """Hash of all templates + answer words (FR-P-1): changing any text or
    polarity changes the version; it is computed, not hand-bumped."""
    src = PROMPTS if prompts is None else prompts
    blob = json.dumps({k: (v["text"], v["pos"], v["neg"]) for k, v in src.items()}, sort_keys=True)
    return md5(blob.encode("utf-8")).hexdigest()[:8]


PROMPT_VERSION = prompt_version()
