#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the proposed candidate:
3 lines x all selected riders 両/番手 x score percentile 20-40 x race top3 scores on 3 distinct lines.
Bet the single cheapest posted trifecta order within each qualifying unordered trio.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_core_top40_rank1_robustness as rb

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'core_top40_rank1_tickets.csv'
OUT = CTX / 'core_20_40_top3spread_summary.json'
SEED = 20260904


def summarize_slices(x: pd.DataFrame):
    out = {'overall': rb.summarize_slice(x, SEED)}

    periods = {}
    for i, (k, g) in enumerate(x.groupby('half_year', sort=True)):
        periods[str(k)] = rb.summarize_slice(g, SEED + 100 + i)
    out['by_half_year'] = periods

    odds = {}
    order = ['LT30','30_TO_50','50_TO_100','100_TO_200','GE200']
    for i, k in enumerate(order):
        odds[k] = rb.summarize_slice(x[x.odds_band == k], SEED + 200 + i)
    out['by_posted_odds_band'] = odds

    cumulative = {}
    for i, t in enumerate([30, 50, 100, 200]):
        cumulative[f'GE{t}'] = rb.summarize_slice(x[x.rank1_posted_odds >= t], SEED + 300 + i)
    out['by_cumulative_odds_threshold'] = cumulative
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f'missing input: {SRC}')
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str})
    required = {'group_score_percentile','top3_spread','role_ok','rank1_posted_odds','rank1_exact_hit','race_uid','half_year','odds_band'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')

    cand = df[
        (df.role_ok == 1) &
        (df.top3_spread == 1) &
        (df.group_score_percentile > 0.20) &
        (df.group_score_percentile <= 0.40)
    ].copy()

    payload = {
        'status': 'exploratory_core_20_40_top3spread_probe',
        'candidate_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            '0.20 < group_score_percentile <= 0.40',
            'race top3 riders by race_score are spread across 3 distinct true lines',
            'bet the single cheapest posted 3rentan permutation within every qualifying unordered trio',
        ],
        'candidate_exposure': {
            'tickets': int(len(cand)),
            'races': int(cand.race_uid.nunique()),
            'wins': int(cand.rank1_exact_hit.sum()),
        },
        **summarize_slices(cand),
        'comparison_note': 'Compare mainly against the prior TOP40 core probe; this isolates the effect of restricting to score 20-40 and restoring the top3-score-spread race filter.',
        'warning': 'Exploratory same-data refinement, not fresh out-of-sample evidence.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
