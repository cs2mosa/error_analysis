"""metrics.py — all behavioral metrics as pure functions of DataFrames (GC-4).

Purpose:  FR-B-0..18: bootstrap CIs (cluster = sample_id), recall/FPR and
          operating points, cost, worst-slice/pair metrics, polarity and
          paraphrase consistency, visual necessity (KL), framing shift,
          calibration (ECE/Brier), selective prediction (AURC/coverage),
          knob-wise recompute support, recovery/breakage, quality curves,
          temporal gains, margin-shift reporting (C5); latent-hazard track
          metrics and the hz_report checklist (FR-H-5/7, §5.14).
Implements plan sections: §6.2, §6.7.
Inputs:   pandas DataFrames with columns sample_id, s, pred, gold, correct,
          arm, prompt_id, valid, ... (runs.jsonl rows).
Outputs:  floats / DataFrames; every stratified report includes n.

Measures/Does-not-measure cards per FR-Q-1 are in the docstrings of each
function; the cross-checks (FR-Q-2) live in tests/test_metrics.py.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def p(s):
    """Two-token probability: p_pos = sigmoid(s) (design §4)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))


def boot_ci(fn, df, n=1000, seed=0, a=0.05):
    """Cluster bootstrap over sample_id (FR-B-0): resample whole sample ids,
    never rows (multiple prompts/arms per sample are not independent)."""
    ids = df.sample_id.unique()
    rng = np.random.default_rng(seed)
    g = dict(tuple(df.groupby("sample_id")))
    v = [fn(pd.concat([g[i] for i in rng.choice(ids, len(ids))])) for _ in range(n)]
    return tuple(np.nanpercentile(v, [100 * a / 2, 100 * (1 - a / 2)]))


def _clean(df):
    """Analysis functions ignore invalid rows (mass gate; FR-R-7) and NaN s."""
    if "valid" in df.columns:
        df = df[df.valid.fillna(True)]
    return df[df.s.notna()]


# ------------------------------------------------------------ operating points

def per_class(df, by=None):
    """Per-class recall and FPR on safe samples (FR-B-1), stratifiable."""
    df = _clean(df)
    if by:
        return pd.concat([per_class(d)[0] for _, d in df.groupby(by)])
    pos, neg = df[df.gold == 1], df[df.gold == 0]
    out = pd.Series(dict(
        recall=float(pos.correct.mean()) if len(pos) else np.nan,
        fpr=float((neg.pred == 1).mean()) if len(neg) else np.nan,
        n_pos=len(pos), n_neg=len(neg)))
    return out, df


def recall_at_fpr(s, y, fpr):
    """Recall at a fixed FPR from the ROC of s (FR-B-2)."""
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def fpr_at_recall(s, y, rec):
    """FPR at a fixed recall (FR-B-2): the SMALLEST FPR whose recall still
    meets the target. roc_curve keeps the trivial last point (fpr=1, tpr=1)
    and drops collinear ones, so plain np.interp(rec, t, f) would return 1.0
    for a perfect scorer; we scan the un-dropped curve for the first
    threshold achieving rec (deviation from the design sketch, logged)."""
    f, t, _ = roc_curve(y, s, drop_intermediate=False)
    ok = np.asarray(t) >= rec
    return float(f[ok][0]) if ok.any() else 1.0


def expected_cost(pred, y, cm, cf):
    """Cost-weighted error (FR-B-3): cm*C(miss) + cf*C(false alarm)."""
    return (cm * ((y == 1) & (pred == 0)).mean() + cf * ((y == 0) & (pred == 1)).mean())


def min_cost_threshold(s, y, cm, cf):
    """Threshold minimizing the expected cost (FR-B-3).
    Vectorized O(n log n) sweep for large arrays (BUG-20 fix)."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    if len(s) == 0:
        return 0.0, 0.0
    u_thresh = np.unique(s)
    if len(u_thresh) <= 200:
        return min((expected_cost((s > t).astype(int), y, cm, cf), float(t)) for t in u_thresh)
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    n_pos = int((y == 1).sum())
    n = len(y)
    idx_right = np.searchsorted(s_sorted, u_thresh, side="right") - 1
    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)
    misses = cum_pos[idx_right]
    false_alarms = (n - n_pos) - cum_neg[idx_right]
    costs = (cm * misses + cf * false_alarms) / n
    best_i = int(np.argmin(costs))
    return float(costs[best_i]), float(u_thresh[best_i])


# ------------------------------------------------------------ pairs (FR-B-5)

def pair_acc(df):
    """Both-items-correct accuracy over pair_id groups (one prompt)."""
    df = _clean(df).dropna(subset=["pair_id"])
    if df.empty:
        return np.nan
    g = df.groupby("pair_id").correct.min()          # both members correct
    return float(g.mean())


def pair_rank_acc(df):
    """Threshold-free pair ranking: s_violation > s_safe (FR-B-5)."""
    df = _clean(df).dropna(subset=["pair_id"])
    if df.empty:
        return np.nan
    w = df.pivot_table(index="pair_id", columns="gold", values="s", aggfunc="mean")
    if 0 not in w.columns or 1 not in w.columns:
        return np.nan
    return float((w[1] > w[0]).mean())


# ------------------------------------------------------------ consistency (FR-B-6/7/9)

def flip_rate(df):
    """Share of samples whose pred differs across the rows given (paraphrase /
    polarity flip baseline: pass the D-group or POL-group base rows)."""
    df = _clean(df)
    if df.empty:
        return np.nan
    return float((df.groupby("sample_id").pred.nunique() > 1).mean())


def polarity_consistency(df):
    """All POL prompts agree on the same answer per sample (FR-B-6)."""
    df = _clean(df)[_clean(df).groupby("sample_id").prompt_id.transform("nunique") > 1]
    if df.empty:
        return np.nan
    piv = df.pivot_table(index="sample_id", columns="prompt_id", values="pred", aggfunc="first")
    return float((piv.nunique(axis=1) == 1).mean())


def yes_rate(df, yes_is_pos):
    """P(model answers Yes) per prompt -> affirmation bias (FR-B-6).

    yes_is_pos: {prompt_id: bool} — True when the Yes token asserts the
    prompt's positive class (pos == Yes variants).
    """
    df = _clean(df)
    rows = []
    for pid, g in df.groupby("prompt_id"):
        is_pos = yes_is_pos.get(pid, True)
        yes_pred = (g.pred == 1) if is_pos else (g.pred == 0)
        rows.append(dict(prompt_id=pid, yes_rate=float(yes_pred.mean()), n=len(g)))
    return pd.DataFrame(rows).set_index("prompt_id")


def framing_shift(df):
    """Mean Δs of leading framings vs neutral (FR-B-9)."""
    df = _clean(df)
    w = df.pivot_table(index="sample_id", columns="prompt_id", values="s", aggfunc="mean")
    if {"frm_pos", "frm_neu", "frm_neg"} - set(w.columns):
        return dict(pos=np.nan, neg=np.nan)
    return dict(pos=float((w.frm_pos - w.frm_neu).mean()),
                neg=float((w.frm_neg - w.frm_neu).mean()))


def flip_report(df):
    """FR-B-7: paraphrase flip rate is only printed next to the same prompts on
    the blank frame (arm O3.blank). Enforced helper returning both."""
    d = _clean(df)
    real = flip_rate(d[(d.arm == "base") & (d.group == "D")])
    blank = flip_rate(d[(d.arm == "O3.blank") & (d.group == "D")])
    if np.isnan(real) or np.isnan(blank):
        raise ValueError("flip_report needs D-group rows under both arm=base and arm=O3.blank (FR-B-7)")
    return dict(flip_real=float(real), flip_blank=float(blank), n=int(d[d.group == "D"].sample_id.nunique()))


# ------------------------------------------------------------ necessity (FR-B-8)

def kl_bern(a, b, e=1e-6):
    a, b = np.clip(a, e, 1 - e), np.clip(b, e, 1 - e)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def visual_necessity(real, other):
    """mean KL(Bern(σ(s_real)) || Bern(σ(s_other))) for aligned Series."""
    real, other = np.asarray(real, float), np.asarray(other, float)
    return float(np.mean(kl_bern(p(real), p(other))))


# ------------------------------------------------------------ calibration (FR-B-12)

def ece(prob, y, bins=15):
    """Expected calibration error over equal-width bins (FR-B-12)."""
    prob, y = np.asarray(prob, float), np.asarray(y, float)
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)


def brier(prob, y):
    return float(np.mean((np.asarray(prob, float) - np.asarray(y, float)) ** 2))


def reliability(prob, y, bins=15):
    """Reliability-diagram data: (mean_conf, acc, n) per bin."""
    prob, y = np.asarray(prob, float), np.asarray(y, float)
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    rows = [dict(bin=b, conf=float(prob[idx == b].mean()), acc=float(y[idx == b].mean()),
                 n=int((idx == b).sum())) for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------ selective prediction (FR-B-13)

def risk_coverage(conf, err):
    """Risk-coverage curve; conf = |s|, err = 1 if native decision wrong."""
    conf, err = np.asarray(conf, float), np.asarray(err, float)
    o = np.argsort(-conf)
    k = np.arange(1, len(o) + 1)
    return k / len(o), np.cumsum(err[o]) / k


def aurc(conf, err):
    return float(risk_coverage(conf, err)[1].mean())


def coverage_at_risk(conf, err, r):
    c, rk = risk_coverage(conf, err)
    ok = rk <= r
    return float(c[ok].max()) if ok.any() else 0.0


def coverage_at_recall(conf, s, y, rec):
    """Coverage achievable while keeping recall >= target among accepted
    positives (FR-B-13). Max accepted fraction over ALL prefixes (recall can
    dip and recover, so an early break would under-report coverage)."""
    conf, s, y = np.asarray(conf, float), np.asarray(s, float), np.asarray(y)
    order = np.argsort(-conf)
    pred_pos = s > 0
    total_pos = (y == 1).sum()
    out, tp = 0.0, 0
    for k in range(1, len(order) + 1):
        sel = order[:k]
        tp += int(pred_pos[sel[k - 1]] and y[sel[k - 1]] == 1)
        recall = (tp / total_pos) if total_pos else 1.0
        if recall >= rec:
            out = k / len(order)
    return out


# ------------------------------------------------------------ recovery (FR-B-15/18)

def _arm_filter(df, arm):
    if isinstance(arm, tuple):
        knob, value = arm
        return df.knob.eq(knob) & df.value.eq(value)
    return df.arm == arm


def arm_recovery(df, ids, arm, gold_col="gold", majority=True):
    """(k, n) of confident-error recovery under an arm: native decision correct
    on ≥ half the prompts (majority) — always the ORIGINAL gold (pitfall 2)."""
    df_clean = _clean(df)
    df_arm = df_clean[_arm_filter(df_clean, arm) & df_clean.sample_id.isin(ids)]
    v = df_arm.groupby("sample_id").correct.mean().ge(0.5)
    return int(v.sum()), len(v)


def arm_breakage(df, ids, arm):
    """Matched-correct breakage under the same arm: n - recovered."""
    k, n = arm_recovery(df, ids, arm)
    return n - k, n


def margin_shift(df, ids, arm, ci=True):
    """Mean Δs signed toward gold = mean( (s_arm - s_base) * (2*gold - 1) ) (C5).

    Returns (mean, boot CI over sample_id, n). Reported next to recovery for
    every (cluster, arm) — sign flips alone under-report effects (2605.22903).
    """
    df = _clean(df)

    def stat(d):
        b = d[(d.arm == "base") & d.sample_id.isin(ids)].groupby("sample_id").agg(
            s_base=("s", "mean"), gold=("gold", "first"))
        a = d[_arm_filter(d, arm) & d.sample_id.isin(ids)].groupby("sample_id").s.mean()
        j = b.join(a.to_frame("s_arm"), how="inner").dropna()
        if j.empty:
            return np.nan
        return float((((j.s_arm - j.s_base) * (2 * j.gold - 1))).mean())

    mask_base = df.arm == "base"
    mask_arm = _arm_filter(df, arm)
    sub = df[df.sample_id.isin(ids) & (mask_base | mask_arm)]
    mean = stat(sub)
    if not np.isfinite(mean):
        return np.nan, (np.nan, np.nan), 0
    boot = boot_ci(stat, sub) if ci else (np.nan, np.nan)
    return mean, boot, int(sub[_arm_filter(sub, arm)].sample_id.nunique())



# ------------------------------------------------------------ curves (FR-B-16/17)

def quality_curve(df, sev_col="sev", kind_col="kind"):
    """Accuracy / mean |s| vs severity per corruption kind (FR-B-16)."""
    df = _clean(df)
    g = df.groupby([kind_col, sev_col]).agg(acc=("correct", "mean"), mean_abs_s=("s", lambda x: np.abs(x).mean()),
                                            n=("sample_id", "nunique"))
    return g.reset_index()


def temporal_gains(acc):
    """multi-frame / order / reverse gains from a dict of accuracies (FR-B-17)."""
    return dict(multi_frame=acc["full"] - acc["single"],
                order=acc["full"] - acc["shuffle"],
                reverse=acc["full"] - acc["reverse"])


# ------------------------------------------------------------ stratified helper (FR-B-14)

def by_knob(df, fns):
    """Recompute any metric(s) per (knob, value) — FR-B-14: calibration/AURC
    per knob value; output has a knob,value index."""
    rows = []
    for (knob, value), g in df[df.knob.notna()].groupby(["knob", "value"]):
        row = dict(knob=knob, value=value, n=int(g.sample_id.nunique()))
        for name, fn in fns.items():
            row[name] = fn(g)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["knob", "value"])


# ------------------------------------------------------------ latent hazard (FR-H-5/7, §5.14)

def hazard_metrics(runs, S=None, tau=0.0):
    """Latent-hazard track metrics (FR-H-5).

    Recall of hazard samples (gold=1) at the calibrated τ is always reported
    WITH the false-alarm rate on the secured twins (gold=0) — recall without
    twins is not reported (FR-H-2, 2606.02443: recall and FPR are tightly
    coupled). Headline pair metrics hazard-vs-twin reuse pair_acc /
    pair_rank_acc over pair_id (S supplies the id→pair_id join).

    Measures: uncued and cued hazard reporting with its false-alarm twin.
    Does not measure: timing (hazards are single-frame; §5.11 does not apply).
    """
    d = _clean(runs)
    hz = d[(d.prompt_id == "hz_d") & (d.arm == "base")]
    if S is not None:
        pid = {s.id: s.pair_id for s in S}
        hz = hz.assign(pair_id=hz.sample_id.map(pid))
    pos, tw = hz[hz.gold == 1], hz[hz.gold == 0]
    cue = d[(d.prompt_id == "hz_cue") & (d.arm == "HZ.cue") & (d.gold == 1)]
    return dict(
        recall=float((pos.s > tau).mean()) if len(pos) else np.nan,
        n_recall=int(len(pos)),
        twin_fa=float((tw.s > tau).mean()) if len(tw) else np.nan,
        n_twin=int(len(tw)),
        pair_acc=pair_acc(hz), pair_rank_acc=pair_rank_acc(hz),
        cued_recall=float((cue.s > tau).mean()) if len(cue) else np.nan,
        n_cued=int(len(cue)),
    )


_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "under", "could",
         "can", "be", "is", "are", "not", "with", "this", "that"}


def _terms(text):
    return [w for w in re.findall(r"[a-z]+", str(text).lower()) if len(w) > 3 and w not in _STOP]


def hz_checklist(text, s):
    """FR-H-7: lenient keyword/synonym checklist for generated hazard reports.

    Checks that the text names the object, the condition and the consequence,
    using content-word prefixes from facts.conditions / meta (lenient by
    design: 'fall' hits 'falling'/'falls'); always followed by the human audit
    sample. Returns dict(object, condition, consequence, complete).
    """
    cs = s.facts.get("conditions") if isinstance(getattr(s, "facts", None), dict) else None
    cs = [c for c in (cs or []) if isinstance(c, dict)]
    objs = {t for c in cs for t in _terms(c.get("obj", ""))}
    states = {t for c in cs for t in _terms(c.get("state", ""))}
    cons = set(_terms((getattr(s, "meta", None) or {}).get("consequence", "")))
    words = re.findall(r"[a-z]+", str(text).lower())

    def hit(terms):
        return any(w.startswith(t) or t.startswith(w) for w in words for t in terms)

    obj, cond, con = hit(objs), hit(states), hit(cons)
    return dict(object=obj, condition=cond, consequence=con,
                complete=bool(obj and cond and con))
