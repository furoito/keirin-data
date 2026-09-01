# -*- coding: utf-8 -*-
"""Extract exact trifecta odds from blind replay data only. Never reads result files."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--tickets", required=True, help="comma separated b1-b2-b3")
    args = ap.parse_args()

    path = Path("keirin_data/replay") / f"{args.date}_odds_3rentan.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
    df = df[df["race_id"].astype(str) == str(args.race_id)].copy()

    for ticket in args.tickets.split(","):
        b1, b2, b3 = map(int, ticket.strip().split("-"))
        hit = df[(df.b1 == b1) & (df.b2 == b2) & (df.b3 == b3)]
        if len(hit) != 1:
            raise SystemExit(f"Expected one odds row for {ticket}, got {len(hit)}")
        row = hit.iloc[0]
        print(f"{ticket}={row.odds_decimal} snapshot={row.snapshot_at} kind={row.snapshot_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
