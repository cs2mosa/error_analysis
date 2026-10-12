"""stream.py — streaming track: online step evaluation & offline trigger policies (FR-ST-1..8).

Purpose:  FR-ST-1: causal online evaluation over video margin series.
          FR-ST-2: trigger policies applied offline (k-of-n, tau).
          FR-ST-3: timing metrics (onset error, warning latency, asymmetric score,
                   spam rate, silent-miss rate, premature-alarm rate).
          FR-ST-4: timing-error decomposition per video (ok / anticipated /
                   silent / late [policy_late vs perception_late] / premature / spam).
          FR-ST-5: window-level vs stream-level metric gap (H11 test).
          FR-ST-6: persistence/recovery after degraded frame.
          FR-ST-7: latency profile (p50/p95 forward time, per-frame cost).
          FR-ST-8: protocol disclosure header on every streaming result file.
Implements: design §5.11, requirements §4.11.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------------------------------------------- FR-ST-8 header

def protocol_header(window_n: int = 4, fps: float = 2.0, k: int = 2, n: int = 3,
                    boundary: str = "frames <= t", trigger: str = "k-of-n") -> str:
    """Protocol disclosure block header required by FR-ST-8."""
    return f"# evidence boundary: {boundary}; history: window-recompute ({window_n},{fps}); trigger: {trigger} ({k}-of-{n})\n"


# ----------------------------------------------------------------- FR-ST-2 trigger policy

def alerts(S: np.ndarray | list[float], tau: float = 0.0, k: int = 2, n: int = 3) -> np.ndarray:
    """k-of-n persistence on calibrated decisions (design §5.11 Step 2).

    Rising edges (transition from 0 to 1) are alert events.
    """
    S = np.asarray(S, dtype=float)
    if len(S) == 0:
        return np.zeros(0, dtype=bool)
    hit = (S > tau).astype(int)
    roll = np.convolve(hit, np.ones(n, dtype=int), mode="full")[:len(S)]
    on = roll >= k
    return on & ~np.r_[False, on[:-1]]


# ----------------------------------------------------------------- FR-ST-4 decomposition

def classify(ts: np.ndarray | list[float], S: np.ndarray | list[float], tau: float,
             onset: float | None, end: float | None, tol: float, k: int, n: int,
             ant: float = 0.0) -> str:
    """Decomposition of streaming alert timing into normative categories (FR-ST-4).

    Returns one of:
      - 'ok': alert fired on time within [onset - tol, onset + tol]
      - 'anticipated': alert fired early in [onset - tol - ant, onset - tol) (requires ant > 0)
      - 'silent': risk clip, but no alert fired in valid window and no premature alert
      - 'premature': alert fired before onset - tol - ant, none in valid window
      - 'policy_late': window margin exceeded tau early, but persistence delayed the alert
      - 'perception_late': window margin was below tau until after onset + tol
      - 'spam': no-risk clip (onset is None) where an alert fired
    """
    ts = np.asarray(ts, dtype=float)
    S = np.asarray(S, dtype=float)
    ev = ts[alerts(S, tau, k, n)]

    if onset is None:
        return "spam" if len(ev) else "ok"  # no-risk clip

    early = ev[ev < onset - tol - ant]
    end_bound = end if end is not None else (onset + tol)
    inwin = ev[(ev >= onset - tol - ant) & (ev <= end_bound)]

    if len(inwin) == 0:
        return "premature" if len(early) else "silent"

    first = inwin.min()
    if first < onset - tol:
        return "anticipated"  # reachable only when ant > 0
    if first - onset <= tol:
        return "ok"

    win_pos = ts[(S > tau) & (ts >= onset - tol)]
    if len(win_pos) and win_pos.min() < first - tol:
        return "policy_late"
    return "perception_late"


# ----------------------------------------------------------------- FR-ST-3 metrics

def onset_error(ts: np.ndarray | list[float], S: np.ndarray | list[float], tau: float,
                onset: float | None, k: int = 2, n: int = 3) -> float | None:
    """Time delta between first alert event and ground-truth onset (FR-ST-3)."""
    if onset is None:
        return None
    ts = np.asarray(ts, dtype=float)
    ev = ts[alerts(S, tau, k, n)]
    if len(ev) == 0:
        return None
    return float(ev[0] - onset)


def warning_latency(ts: np.ndarray | list[float], S: np.ndarray | list[float], tau: float,
                    onset: float | None, k: int = 2, n: int = 3) -> float | None:
    """Latency from risk onset to the first alert firing on/after onset (FR-ST-3)."""
    if onset is None:
        return None
    ts = np.asarray(ts, dtype=float)
    ev = ts[alerts(S, tau, k, n)]
    post = ev[ev >= onset]
    if len(post) == 0:
        return None
    return float(post[0] - onset)


def asymmetric_score(ts: np.ndarray | list[float], S: np.ndarray | list[float], tau: float,
                     onset: float | None, end: float | None, tol: float, k: int, n: int,
                     c_miss: float = 1.0, c_early: float = 0.3, decay_s: float = 5.0,
                     ant: float = 0.0) -> float:
    """Asymmetric early/late reward score (FR-ST-3; design §5.11).

    miss -> -c_miss
    early / premature / spam -> -c_early
    on-time / late -> exp(-(t_alert - onset) / decay_s)
    """
    cl = classify(ts, S, tau, onset, end, tol, k, n, ant=ant)
    if cl in ("silent",):
        return -float(c_miss)
    if cl in ("premature", "spam"):
        return -float(c_early)
    if onset is None:
        return 0.0  # safe clip, no spam
    ts = np.asarray(ts, dtype=float)
    ev = ts[alerts(S, tau, k, n)]
    inwin = ev[ev >= onset - tol - ant]
    if len(inwin) == 0:
        return -float(c_miss)
    first = inwin.min()
    delta = max(0.0, float(first - onset))
    return float(np.exp(-delta / float(decay_s)))


def spam_rate(ts: np.ndarray | list[float], S: np.ndarray | list[float], tau: float,
              onset: float | None, end: float | None, tol: float, k: int, n: int,
              ant: float = 0.0) -> float:
    """Alert events per minute in dead time (FR-ST-3; design §5.11)."""
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return 0.0
    ev = ts[alerts(S, tau, k, n)]
    if onset is None:
        duration_s = max(1.0, ts[-1] - ts[0])
        return float(len(ev) / (duration_s / 60.0))

    end_s = end if end is not None else onset + tol
    dead_events = ev[(ev < onset - tol - ant) | (ev > end_s + tol)]
    dead_time_s = max(0.0, (onset - tol - ant) - ts[0]) + max(0.0, ts[-1] - (end_s + tol))
    if dead_time_s <= 0.0:
        return 0.0
    return float(len(dead_events) / (dead_time_s / 60.0))


# ----------------------------------------------------------------- FR-ST-6 persistence/recovery

def recovery_steps(S: np.ndarray | list[float], tau: float = 0.0, degrade_idx: int = 0) -> int:
    """Count steps after a degraded frame until margin returns above tau (FR-ST-6)."""
    S = np.asarray(S, dtype=float)
    for i in range(degrade_idx, len(S)):
        if S[i] > tau:
            return i - degrade_idx
    return len(S) - degrade_idx


# ----------------------------------------------------------------- FR-ST-5 & FR-ST-7 profile / gap

def window_vs_stream_gap(runs_df: pd.DataFrame, samples: list, cfg: dict, tau: float = 0.0) -> dict:
    """Compute window-level vs stream-level metric gap for H11 trigger-policy test (FR-ST-5)."""
    st_cfg = cfg.get("stream", {})
    tol = float(st_cfg.get("tolerance_s", 2.0))
    k = int(st_cfg.get("k", 2))
    n = int(st_cfg.get("n", 3))
    ant = float(st_cfg.get("anticipation_s", 0.0))

    sample_map = {s.id: s for s in samples}
    df = runs_df[runs_df.t.notna()].copy()
    if df.empty:
        return dict(window_recall=0.0, stream_recall=0.0, gap=0.0, stream_policy=False)

    win_tp, win_fn = 0, 0
    for sid, group in df.groupby("sample_id"):
        s = sample_map.get(sid)
        if not s or s.modality != "video":
            continue
        v = s.video or {}
        onset = v.get("onset_s")
        end = v.get("end_s", onset + tol if onset else None)
        if onset is None:
            continue
        for _, row in group.iterrows():
            t = float(row["t"])
            if onset <= t <= end:
                if row["s"] > tau:
                    win_tp += 1
                else:
                    win_fn += 1

    win_recall = win_tp / max(1, win_tp + win_fn)

    stream_tp, stream_fn = 0, 0
    policy_late_count = 0
    total_risk = 0

    for sid, group in df.groupby("sample_id"):
        s = sample_map.get(sid)
        if not s or s.modality != "video":
            continue
        v = s.video or {}
        onset = v.get("onset_s")
        end = v.get("end_s", onset + tol if onset else None)
        if onset is None:
            continue
        total_risk += 1
        group = group.sort_values("t")
        ts = group["t"].to_numpy()
        S_vals = group["s"].to_numpy()
        cl = classify(ts, S_vals, tau, onset, end, tol, k, n, ant=ant)
        if cl in ("ok", "anticipated"):
            stream_tp += 1
        else:
            stream_fn += 1
            if cl == "policy_late":
                policy_late_count += 1

    stream_recall = stream_tp / max(1, total_risk)
    gap = win_recall - stream_recall
    policy_failure = bool(gap >= 0.2 or (policy_late_count / max(1, total_risk) >= 0.3))

    return dict(
        window_recall=float(win_recall),
        stream_recall=float(stream_recall),
        gap=float(gap),
        policy_late_rate=float(policy_late_count / max(1, total_risk)),
        stream_policy=policy_failure,
    )


def stream_latency_profile(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Latency profile per knob configuration (FR-ST-7)."""
    df = runs_df[runs_df.t.notna()].copy()
    if df.empty or "latency_ms" not in df.columns:
        return pd.DataFrame(columns=["knob", "value", "p50_ms", "p95_ms", "n_frames"])
    rows = []
    group_cols = ["knob", "value"] if "knob" in df.columns else []
    if group_cols and df["knob"].notna().any():
        for (knob, val), grp in df.groupby(group_cols):
            lat = grp["latency_ms"].dropna()
            rows.append(dict(
                knob=knob, value=val,
                p50_ms=float(lat.median()) if len(lat) else 0.0,
                p95_ms=float(lat.quantile(0.95)) if len(lat) else 0.0,
                n_frames=len(lat),
            ))
    else:
        lat = df["latency_ms"].dropna()
        rows.append(dict(
            knob="all", value="all",
            p50_ms=float(lat.median()) if len(lat) else 0.0,
            p95_ms=float(lat.quantile(0.95)) if len(lat) else 0.0,
            n_frames=len(lat),
        ))
    return pd.DataFrame(rows)


def evaluate_stream(runs_df: pd.DataFrame, samples: list, cfg: dict,
                    tau: float = 0.0, out_dir: str | Path = "outputs") -> dict:
    """Full streaming track evaluation pipeline (FR-ST-1..8)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st_cfg = cfg.get("stream", {})
    tol = float(st_cfg.get("tolerance_s", 2.0))
    k = int(st_cfg.get("k", 2))
    n = int(st_cfg.get("n", 3))
    ant = float(st_cfg.get("anticipation_s", 0.0))
    c_miss = float(st_cfg.get("c_miss", 1.0))
    c_early = float(st_cfg.get("c_early", 0.3))
    decay_s = float(st_cfg.get("decay_s", 5.0))
    window = int(st_cfg.get("window", 4))
    fps = float(st_cfg.get("fps", 2.0))

    sample_map = {s.id: s for s in samples}
    df = runs_df[runs_df.t.notna()].copy()
    video_records = []

    for sid, group in df.groupby("sample_id"):
        s = sample_map.get(sid)
        if not s:
            continue
        v = s.video or {}
        onset = v.get("onset_s")
        end = v.get("end_s", onset + tol if onset else None)
        group = group.sort_values("t")
        ts = group["t"].to_numpy()
        S_vals = group["s"].to_numpy()

        cat = classify(ts, S_vals, tau, onset, end, tol, k, n, ant=ant)
        on_err = onset_error(ts, S_vals, tau, onset, k, n)
        w_lat = warning_latency(ts, S_vals, tau, onset, k, n)
        asym = asymmetric_score(ts, S_vals, tau, onset, end, tol, k, n,
                                c_miss=c_miss, c_early=c_early, decay_s=decay_s, ant=ant)
        sp = spam_rate(ts, S_vals, tau, onset, end, tol, k, n, ant=ant)

        video_records.append(dict(
            sample_id=sid, label=s.label, onset=onset, end=end,
            category=cat, onset_error=on_err, warning_latency=w_lat,
            asymmetric_score=asym, spam_rate=sp,
        ))

    vdf = pd.DataFrame(video_records)

    gap_res = window_vs_stream_gap(runs_df, samples, cfg, tau=tau)
    lat_df = stream_latency_profile(runs_df)

    hdr = protocol_header(window_n=window, fps=fps, k=k, n=n)

    sum_path = out_dir / "stream_summary.csv"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(hdr)
        vdf.to_csv(f, index=False)

    lat_path = out_dir / "stream_latency.csv"
    with open(lat_path, "w", encoding="utf-8") as f:
        f.write(hdr)
        lat_df.to_csv(f, index=False)

    return dict(
        summary_df=vdf,
        gap=gap_res,
        latency_df=lat_df,
        summary_path=sum_path,
        latency_path=lat_path,
    )
