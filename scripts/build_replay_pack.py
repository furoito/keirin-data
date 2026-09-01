# -*- coding: utf-8 -*-
"""Build a blind pre-race replay pack with all result columns physically excluded."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA = Path("keirin_data")
CTX = DATA / "strategy_context"
OUT = DATA / "replay"

# Strict allowlist: result/payout fields are intentionally impossible to pass through.
PLAYER_COLS = [
    "race_id", "venue_slug", "date", "race_no", "banum", "player_name", "pref",
    "age", "term", "player_class", "running_style", "gear", "race_score",
    "mark", "mark_num", "win_rate_4m", "top2_rate_4m", "top3_rate_4m",
    "nige_4m", "maku_4m", "close_time",
]
ODDS_COLS = [
    "race_id", "date", "venue_slug", "race_no", "b1", "b2", "b3",
    "odds_decimal", "snapshot_at", "snapshot_kind", "source_url",
]
CTX_COLS = [
    "race_id", "date", "venue_slug", "race_no", "classes", "n_players",
    "expected_3rentan", "true_line", "n_lines", "line_quality", "odds_quality",
    "price_quality", "price_usable", "context_quality", "bet_close_time",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Replay date YYYY-MM-DD")
    ap.add_argument("--classes", default="A1,A2")
    ap.add_argument("--seven-only", action="store_true")
    args = ap.parse_args()

    ym = args.date[:7].replace("-", "_")
    base_path = DATA / f"{ym}_keirin.csv"
    ctx_path = CTX / f"{ym}_races.csv"
    odds_path = CTX / f"{ym}_odds_3rentan.csv"
    for p in (base_path, ctx_path, odds_path):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    base = read_csv(base_path)
    ctx = read_csv(ctx_path)
    odds = read_csv(odds_path)

    classes = {x.strip() for x in args.classes.split(",") if x.strip()}
    c = ctx[(ctx["date"].astype(str) == args.date) & (ctx["context_quality"] == "full")].copy()
    c = c[c["classes"].fillna("").apply(lambda s: set(str(s).split(",")).issubset(classes))]
    if args.seven_only:
        c = c[c["n_players"] == 7]
    c = c[[x for x in CTX_COLS if x in c.columns]].drop_duplicates("race_id")
    race_ids = set(c["race_id"].astype(str))
    if not race_ids:
        raise SystemExit(f"No eligible blind replay races for {args.date}")

    # Use only the explicit pre-race allowlist; no rank, finish_type, margins or payout columns.
    missing = [x for x in PLAYER_COLS if x not in base.columns]
    if missing:
        raise SystemExit(f"Base CSV missing expected pre-race columns: {missing}")
    p = base[base["race_id"].astype(str).isin(race_ids)][PLAYER_COLS].copy()
    p = p.merge(c[["race_id", "true_line", "n_lines", "n_players"]], on="race_id", how="left")
    p = p.sort_values(["venue_slug", "race_no", "banum"])

    o = odds[odds["race_id"].astype(str).isin(race_ids)][[x for x in ODDS_COLS if x in odds.columns]].copy()
    o = o.sort_values(["venue_slug", "race_no", "b1", "b2", "b3"])

    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / args.date
    p.to_csv(f"{prefix}_players.csv", index=False, encoding="utf-8-sig")
    o.to_csv(f"{prefix}_odds_3rentan.csv", index=False, encoding="utf-8-sig")
    c.sort_values(["venue_slug", "race_no"]).to_csv(
        f"{prefix}_races.csv", index=False, encoding="utf-8-sig"
    )

    print(f"date={args.date}")
    print(f"races={len(c)} player_rows={len(p)} odds_rows={len(o)}")
    print("race list:")
    print(c[["venue_slug", "race_no", "true_line", "n_players"]].to_string(index=False))
    print("Blind pack contains no result/payout columns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
