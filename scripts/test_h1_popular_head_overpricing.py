#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 reset test: is the most market-supported line's head overbought?

This intentionally ignores all downstream strategy logic (score gaps, candidate tiers,
OH_HIGH, running style, compression, odds threshold, and bet selection).

For each race with full strategy context and a complete trifecta board:
1. Detect the most market-supported non-solo line using the existing v0.1b joint mass.
2. Let its line head be the target.
3. Convert all trifecta odds into normalized inverse-odds weights across the mutually
   exclusive outcome board. This removes the common within-race overround/takeout scale.
4. Derive market-implied target probabilities for WIN and TOP3.
5. Compare those probabilities with realized outcomes.

Positive mean gap (market - realized) = evidence of overpricing.
Negative mean gap = evidence of underpricing.

Results are diagnostic/exploratory. No ticket is selected and no result field is used
before the target and market probabilities are fixed.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "keirin_data"
CTX = DATA / "strategy_context"
OUT = CTX / "h1_popular_head_overpricing_summary.json"
DETAIL = CTX / "h1_popular_head_overpricing_details.csv"
MONTHS = [f"2025_{m:02d}" for m in range(1, 13)] + [f"2026_{m:02d}" for m in range(1, 7)]


def actual_rank(g: pd.DataFrame, target: int):
    r = g[pd.to_numeric(g.banum, errors="coerce") == int(target)]
    if r.empty:
        return None
    raw = str(r.iloc[0].get("rank", "")).strip()
    try:
        return int(raw)
    except Exception:
        return None


def bootstrap_ci(values, seed=20260902, draws=10000):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = rng.choice(a, size=len(a), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def load_month(month: str):
    bp = DATA / f"{month}_keirin.csv"
    cp = CTX / f"{month}_races.csv"
    op = CTX / f"{month}_odds_3rentan.csv"
    if not (bp.exists() and cp.exists() and op.exists()):
        return None
    b = pd.read_csv(bp, encoding="utf-8-sig", dtype={"race_id": str})
    c = pd.read_csv(cp, encoding="utf-8-sig", dtype={"race_id": str})
    o = pd.read_csv(op, encoding="utf-8-sig", dtype={"race_id": str})
    for d in (b, c, o):
        d["race_id"] = d.race_id.astype(str)
    c = c.drop_duplicates("race_id", keep="last")
    return b, c, o


def analyze_race(month, rid, pre_full, cr, og):
    lines = base.parse_true_line(cr.get("true_line"))
    if not lines:
        return None, "line_unresolved"
    pre = pre_full[["race_id", "banum", "race_score"]].copy()
    riders = base.make_riders(pre, lines)
    if len(riders) < 4:
        return None, "score_or_entry_missing"
    tri = base.odds_map(og)
    expected = len(riders) * (len(riders) - 1) * (len(riders) - 2)
    if len(tri) != expected:
        return None, "odds_board_incomplete"

    # Exact v0.1b popular-line detector: non-solo line with max joint inverse-odds mass.
    pop = v01b.detect_popular_line_joint_mass(riders, tri)
    if pop is None:
        return None, "popular_line_unresolved"
    pop_line, target, joint_mass, _ = pop

    inv = {k: 1.0 / v for k, v in tri.items() if v > 0}
    z = sum(inv.values())
    if z <= 0:
        return None, "zero_market_mass"
    q_win = sum(w for (a, b, c), w in inv.items() if a == target.frame_no) / z
    q_top3 = sum(w for combo, w in inv.items() if target.frame_no in combo) / z

    rank = actual_rank(pre_full, target.frame_no)
    if rank is None:
        return None, "result_rank_missing"
    y_win = 1 if rank == 1 else 0
    y_top3 = 1 if 1 <= rank <= 3 else 0

    members = base.line_members(riders)[pop_line]
    return {
        "month": month,
        "race_id": rid,
        "target": int(target.frame_no),
        "target_score": float(target.race_score),
        "popular_line_size": int(len(members)),
        "joint_market_mass_raw": float(joint_mass),
        "market_win_p": float(q_win),
        "actual_win": y_win,
        "win_gap_market_minus_actual": float(q_win - y_win),
        "market_top3_p": float(q_top3),
        "actual_top3": y_top3,
        "top3_gap_market_minus_actual": float(q_top3 - y_top3),
        "actual_rank": int(rank),
        "price_quality": str(cr.get("price_quality", "")),
    }, None


def fixed_bins(df, col, cuts):
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        x = df[(df[col] >= lo) & (df[col] < hi if hi < 1.000001 else df[col] <= hi)]
        if x.empty:
            continue
        actual_col = "actual_win" if col == "market_win_p" else "actual_top3"
        out.append({
            "range": f"[{lo:.2f},{hi:.2f}{')' if hi < 1 else ']'}",
            "n": int(len(x)),
            "market_mean": float(x[col].mean()),
            "actual_rate": float(x[actual_col].mean()),
            "gap_market_minus_actual": float((x[col] - x[actual_col]).mean()),
        })
    return out


def summarize(df: pd.DataFrame, skipped: dict, context_counts: dict):
    win_gap = df.market_win_p - df.actual_win
    top3_gap = df.market_top3_p - df.actual_top3
    monthly = []
    for m, x in df.groupby("month", sort=True):
        monthly.append({
            "month": m,
            "n": int(len(x)),
            "market_win_mean": float(x.market_win_p.mean()),
            "actual_win_rate": float(x.actual_win.mean()),
            "win_gap": float((x.market_win_p - x.actual_win).mean()),
            "market_top3_mean": float(x.market_top3_p.mean()),
            "actual_top3_rate": float(x.actual_top3.mean()),
            "top3_gap": float((x.market_top3_p - x.actual_top3).mean()),
        })
    by_line_size = []
    for s, x in df.groupby("popular_line_size", sort=True):
        by_line_size.append({
            "line_size": int(s), "n": int(len(x)),
            "market_win_mean": float(x.market_win_p.mean()),
            "actual_win_rate": float(x.actual_win.mean()),
            "win_gap": float((x.market_win_p - x.actual_win).mean()),
            "market_top3_mean": float(x.market_top3_p.mean()),
            "actual_top3_rate": float(x.actual_top3.mean()),
            "top3_gap": float((x.market_top3_p - x.actual_top3).mean()),
        })
    return {
        "hypothesis": "H1: most market-supported line head is overpriced relative to realized win/top3 frequency",
        "scope": "2025-01..2026-06 available full context at run time",
        "selection": "v0.1b popular-line joint inverse-odds mass; non-solo lines only; no downstream strategy rules",
        "market_probability": "normalized inverse trifecta odds across complete mutually-exclusive outcome board",
        "n": int(len(df)),
        "context_counts": context_counts,
        "skipped": skipped,
        "overall": {
            "market_win_mean": float(df.market_win_p.mean()),
            "actual_win_rate": float(df.actual_win.mean()),
            "win_gap_market_minus_actual": float(win_gap.mean()),
            "win_gap_95pct_bootstrap_ci": bootstrap_ci(win_gap),
            "market_top3_mean": float(df.market_top3_p.mean()),
            "actual_top3_rate": float(df.actual_top3.mean()),
            "top3_gap_market_minus_actual": float(top3_gap.mean()),
            "top3_gap_95pct_bootstrap_ci": bootstrap_ci(top3_gap, seed=20260903),
            "actual_bust_rate": float(1.0 - df.actual_top3.mean()),
            "market_bust_mean": float(1.0 - df.market_top3_p.mean()),
        },
        "win_calibration_bins": fixed_bins(df, "market_win_p", [0,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.000001]),
        "top3_calibration_bins": fixed_bins(df, "market_top3_p", [0,.20,.30,.40,.50,.60,.70,.80,.90,1.000001]),
        "by_popular_line_size": by_line_size,
        "monthly": monthly,
    }


def main():
    rows = []
    skipped = {}
    context_counts = {"context_rows": 0, "full_rows": 0}
    for month in MONTHS:
        loaded = load_month(month)
        if loaded is None:
            continue
        b, c, o = loaded
        context_counts["context_rows"] += int(len(c))
        use = c.copy()
        if "context_quality" in use:
            use = use[use.context_quality.astype(str) == "full"]
        if "price_usable" in use:
            use = use[use.price_usable.astype(str).str.lower().isin({"true", "1"})]
        context_counts["full_rows"] += int(len(use))
        bby = {str(k): g for k, g in b.groupby("race_id", sort=False)}
        oby = {str(k): g for k, g in o.groupby("race_id", sort=False)}
        for cr in use.to_dict("records"):
            rid = str(cr["race_id"])
            pre = bby.get(rid); og = oby.get(rid)
            if pre is None or og is None:
                key = "base_or_odds_missing"; skipped[key] = skipped.get(key, 0) + 1; continue
            row, why = analyze_race(month, rid, pre, cr, og)
            if row is None:
                skipped[why] = skipped.get(why, 0) + 1
            else:
                rows.append(row)
    if not rows:
        raise SystemExit("No H1 rows")
    df = pd.DataFrame(rows).sort_values(["month", "race_id"])
    df.to_csv(DETAIL, index=False, encoding="utf-8-sig")
    payload = summarize(df, skipped, context_counts)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"detail={DETAIL}")
    print(f"summary={OUT}")


if __name__ == "__main__":
    main()
