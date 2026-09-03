#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-history version of the current candidate probe.

Fixed candidate:
- exactly 3 distinct true lines
- each selected rider is 両 or line_pos=2 (bante)
- 0.20 < group_score_percentile <= 0.40
- race top3 riders by race_score spread across 3 distinct true lines
- buy the single cheapest posted 3rentan permutation

Window: 2022-01 through 2026-08, using only months with complete base/context/odds inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_formation_box_chain_expanded as exp
import analyze_core_top40_rank1_robustness as rb

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'core_20_40_top3spread_long_history_summary.json'
DETAIL = CTX / 'core_20_40_top3spread_long_history_tickets.csv'
SEED = 20260905

MONTHS = (
    [f'2022_{m:02d}' for m in range(1, 13)] +
    [f'2023_{m:02d}' for m in range(1, 13)] +
    [f'2024_{m:02d}' for m in range(1, 13)] +
    [f'2025_{m:02d}' for m in range(1, 13)] +
    [f'2026_{m:02d}' for m in range(1, 9)]
)


def summarize(x: pd.DataFrame):
    payload = {'overall': rb.summarize_slice(x, SEED)}

    by_year = {}
    for i, (k, g) in enumerate(x.groupby(x.month.str[:4], sort=True)):
        by_year[str(k)] = rb.summarize_slice(g, SEED + 100 + i)
    payload['by_year'] = by_year

    x = x.copy()
    x['half_year'] = x.month.map(rb.half_year_label)
    by_half = {}
    for i, (k, g) in enumerate(x.groupby('half_year', sort=True)):
        by_half[str(k)] = rb.summarize_slice(g, SEED + 200 + i)
    payload['by_half_year'] = by_half

    x['odds_band'] = x.rank1_posted_odds.map(rb.odds_band)
    by_odds = {}
    for i, label in enumerate(['LT30','30_TO_50','50_TO_100','100_TO_200','GE200']):
        by_odds[label] = rb.summarize_slice(x[x.odds_band == label], SEED + 300 + i)
    payload['by_posted_odds_band'] = by_odds

    target_30_100 = x[(x.rank1_posted_odds >= 30) & (x.rank1_posted_odds < 100)]
    payload['focus_30_to_100'] = rb.summarize_slice(target_30_100, SEED + 400)
    return payload


def main():
    exp.MONTHS = MONTHS
    groups, build_context = exp.build_groups()
    if groups.empty:
        raise SystemExit('no expanded groups built')

    cand = groups[
        (groups.role_ok == 1) &
        (groups.top3_spread == 1) &
        (groups.group_score_percentile > 0.20) &
        (groups.group_score_percentile <= 0.40)
    ].copy()

    tickets, odds_skipped = rb.add_posted_rank1_odds(cand)
    if tickets.empty:
        raise SystemExit('no candidate tickets with posted odds')

    tickets.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    payload = {
        'status': 'exploratory_long_history_core_20_40_top3spread_probe',
        'months_requested': MONTHS,
        'build_context': build_context,
        'odds_recovery_skipped': odds_skipped,
        'candidate_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            '0.20 < group_score_percentile <= 0.40',
            'race top3 riders by race_score are spread across 3 distinct true lines',
            'bet the single cheapest posted 3rentan permutation within each qualifying unordered trio',
        ],
        'candidate_exposure': {
            'tickets': int(len(tickets)),
            'races': int(tickets.race_uid.nunique()),
            'wins': int(tickets.rank1_exact_hit.sum()),
        },
        **summarize(tickets),
        'warning': 'Same-data exploratory refinement. The longer history increases power but does not make this unbiased OOS evidence.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
