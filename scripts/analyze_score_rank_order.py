#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Order qualifying 3-rider groups by exact race_score-sum rank.

Fixed structural conditions:
- exactly 3 distinct true lines and selected lines are multi-rider lines
- every selected rider is running_style='両' OR line_pos=2
- race top3 riders by race_score are spread across 3 distinct true lines

No percentile band is imposed. Instead, report exact group-score rank 1,2,3,...
within each race. The bet view is the single cheapest posted trifecta order within
each qualifying unordered trio.

Primary goal: see whether the prior 20-40% effect is better explained by a simple
absolute score-rank position rather than an arbitrary percentile band.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import analyze_core_top40_rank1_robustness as rb
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'formation_box_chain_expanded_groups.csv'
OUT = CTX / 'score_rank_order_summary.json'
DETAIL = CTX / 'score_rank_order_tickets.csv'
SEED = 20260906


def race_group_counts(df: pd.DataFrame) -> dict[tuple[str,str], int]:
    keys = set(zip(df.month.astype(str), df.race_id.astype(str)))
    out = {}
    for month in sorted({m for m, _ in keys}):
        loaded = h1.load_month(month)
        if loaded is None:
            continue
        base, _, _ = loaded
        for rid, g in base.groupby('race_id', sort=False):
            k = (str(month), str(rid))
            if k not in keys:
                continue
            try:
                n = int(pd.to_numeric(g.banum, errors='coerce').dropna().nunique())
            except Exception:
                continue
            if n >= 3:
                out[k] = math.comb(n, 3)
    return out


def summarize(x: pd.DataFrame, seed: int):
    return rb.summarize_slice(x, seed)


def main():
    if not SRC.exists():
        raise SystemExit(f'missing input: {SRC}')
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str, 'trio_key': str})
    required = {
        'month','race_id','trio_key','group_score_percentile','role_ok','top3_spread',
        'group_market_p','rank1_conditional_share','group_hit','rank1_exact_hit'
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')

    cand = df[(df.role_ok == 1) & (df.top3_spread == 1)].copy()
    counts = race_group_counts(cand)
    cand['group_count_in_race'] = [counts.get((str(m), str(r))) for m, r in zip(cand.month, cand.race_id)]
    cand = cand[cand.group_count_in_race.notna()].copy()
    cand['group_count_in_race'] = cand.group_count_in_race.astype(int)
    cand['group_score_rank'] = (cand.group_score_percentile * cand.group_count_in_race).round().astype(int)

    tickets, odds_skipped = rb.add_posted_rank1_odds(cand)
    if tickets.empty:
        raise SystemExit('no qualifying tickets')
    tickets['year'] = tickets.month.str[:4]
    tickets.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    exact = {}
    for i, (rank, g) in enumerate(tickets.groupby('group_score_rank', sort=True)):
        exact[str(int(rank))] = summarize(g, SEED + i)

    # Simple pooled ranges only as stability aids; exact-rank table is primary.
    def rank_bucket(r: int) -> str:
        if r <= 3: return 'R1_3'
        if r <= 6: return 'R4_6'
        if r <= 10: return 'R7_10'
        if r <= 15: return 'R11_15'
        if r <= 20: return 'R16_20'
        return 'R21_PLUS'

    tickets['rank_bucket'] = tickets.group_score_rank.map(rank_bucket)
    pooled = {}
    for i, (label, g) in enumerate(tickets.groupby('rank_bucket', sort=False)):
        pooled[str(label)] = summarize(g, SEED + 100 + i)

    exact_2025 = {}
    y25 = tickets[tickets.year == '2025']
    for i, (rank, g) in enumerate(y25.groupby('group_score_rank', sort=True)):
        exact_2025[str(int(rank))] = summarize(g, SEED + 200 + i)

    pooled_2025 = {}
    for i, (label, g) in enumerate(y25.groupby('rank_bucket', sort=False)):
        pooled_2025[str(label)] = summarize(g, SEED + 300 + i)

    payload = {
        'status': 'exploratory_exact_group_score_rank_order',
        'candidate_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            'race top3 riders by race_score are spread across 3 distinct true lines',
            'NO score-percentile cutoff',
            'bet the single cheapest posted 3rentan permutation within each qualifying unordered trio',
        ],
        'rank_semantics': '1 = highest race_score sum among all unordered 3-rider groups in that race; larger rank = weaker group',
        'odds_recovery_skipped': odds_skipped,
        'all_available': {
            'tickets': int(len(tickets)),
            'races': int(tickets.race_uid.nunique()),
            'exact_rank': exact,
            'pooled_rank_ranges': pooled,
        },
        'year_2025_latest_backfill': {
            'tickets': int(len(y25)),
            'races': int(y25.race_uid.nunique()),
            'exact_rank': exact_2025,
            'pooled_rank_ranges': pooled_2025,
        },
        'warning': 'Exploratory same-data decomposition. Exact ranks can be sparse; pooled ranges are descriptive stability aids, not pre-registered thresholds.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
