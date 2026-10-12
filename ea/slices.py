"""slices.py — metadata slice discovery (FR-S).

Purpose:  enumerate conjunctions of up to max_order categorical metadata
          columns; report error rate, support, Wilson CI, gap, one-sided
          Fisher p and BH q-values (FR-S-1); run separately for misses and
          false alarms (FR-S-2); return predicates usable by escalate.py
          (FR-S-3). P3 embedding slices are NOT built (descope §3.4).
Implements plan sections: §6.3.
Inputs:   one-row-per-sample DataFrame with `err` column (1=error) + metadata.
Outputs:  DataFrame ranked by the Wilson lower bound; slices as dict
          col->value predicates.

Measures: elevated error rate in a metadata-defined subgroup.
Does not measure: coherence or causality — the audit sample carries a human
coherence verdict (2203.14960); embedding-based slices (P3, descoped).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# candidate columns (design §5.7); any column with > 50% missing is skipped
COLUMNS = ["cls", "scene", "viewpoint", "vtype", "n_workers_bin", "feature_share",
           "obj_size_bin", "dist_bin", "occlusion", "blur_bin", "brightness_bin", "bitrate_bin"]

err_col = "err"


def slice_search(df, cols=None, err="err", max_order=2, min_support=30, alpha=0.05, fdr=None, **kw):
    """FR-S-1. One row per (significant) slice; NaN keys are dropped — bin
    first (bin_meta). Ranking: Wilson lower bound, descending."""
    if fdr is not None:
        alpha = fdr
    cols = cols or [c for c in COLUMNS if c in df.columns]
    cols = [c for c in cols if df[c].notna().mean() > 0.5]     # skip >50% missing
    N, E = len(df), int(df[err].sum())
    if N == 0 or E == 0:
        return pd.DataFrame()
    base = E / N
    rows = []
    for k in range(1, max_order + 1):
        for cs in combinations(cols, k):
            for key, g in df.groupby(list(cs), observed=True, dropna=True):
                n, e = len(g), int(g[err].sum())
                if n < min_support:
                    continue
                lo, hi = proportion_confint(e, n, method="wilson")
                pv = fisher_exact([[e, n - e], [E - e, N - n - (E - e)]],
                                  alternative="greater")[1]
                rows.append(dict(slice=dict(zip(cs, key if isinstance(key, tuple) else (key,))),
                                 n=n, err=e / n, lo=lo, hi=hi, gap=e / n - base, p=pv))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out.p, method="fdr_bh")[1]
    return out[out.q < alpha].sort_values("lo", ascending=False).reset_index(drop=True)


def slice_ids(df, pred):
    """FR-S-3 round trip: predicate dict col->value -> sample ids."""
    m = pd.Series(True, index=df.index)
    for c, v in pred.items():
        m &= df[c] == v
    return list(df.id[m])


def misses_frame(df_samples, decision="pred", gold_col="label"):
    """One row per positive sample with err = miss (FR-S-2, majority vote of
    the native decision when several rows exist per sample)."""
    pos = df_samples[df_samples[gold_col] == 1].copy()
    pos[err_col] = (pos[decision] == 0).astype(int)
    return pos


def false_alarms_frame(df_samples, decision="pred", gold_col="label"):
    """One row per negative sample with err = false alarm (FR-S-2)."""
    neg = df_samples[df_samples[gold_col] == 0].copy()
    neg[err_col] = (neg[decision] == 1).astype(int)
    return neg
