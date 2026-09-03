#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic for a two-layer market-mispricing mechanism.

Hypothesis under test:
1) Formation-style betting underweights some unordered three-rider combinations.
2) Conditional on that trio actually filling the podium, the market's ordering
   across the six exact trifecta permutations is comparatively well calibrated.

Input: canonical 2024-01..2026-06 ticket output. Candidate generation is unchanged.
This is a mechanism diagnostic on explored data, not fresh OOS validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'h1_canonical_three_line_ryo_bante_strong_extended_2024_2026_tickets.csv'
OUT = CTX / 'formation_demand_mechanism_summary.json'
DETAIL = CTX / 'formation_demand_mechanism_groups.csv'

CUTS = {
    'STRUCTURAL_ONLY': 1.00,
    'TOP50_SCORE_SUM': 0.50,
    'TOP40_SCORE_SUM': 0.40,
    'TOP30_SCORE_SUM': 0.30,
}
GROUP_ODDS_BINS = [
    ('LT_5', 0.0, 5.0),
    ('5_TO_10', 5.0, 10.0),
    ('10_TO_20', 10.0, 20.0),
    ('20_TO_30', 20.0, 30.0),
    ('30_TO_50', 30.0, 50.0),
    ('50_TO_100', 50.0, 100.0),
    ('100_TO_200', 100.0, 200.0),
    ('GE_200', 200.0, float('inf')),
]


def trio_key(ticket: str) -> str:
    return '-'.join(map(str, sorted(int(x) for x in str(ticket).split('-'))))


def group_agg(g: pd.DataFrame):
    n = int(len(g))
    if not n:
        return {'candidate_trios': 0, 'races': 0, 'actual_group_hits': 0,
                'market_expected_group_hits': 0.0, 'group_calibration_ratio': None,
                'mean_group_market_p': None, 'median_effective_group_odds': None}
    hits = int(g.group_hit.sum())
    exp = float(g.group_market_p.sum())
    return {
        'candidate_trios': n,
        'races': int(g.race_id.nunique()),
        'actual_group_hits': hits,
        'market_expected_group_hits': exp,
        'group_calibration_ratio': float(hits / exp) if exp > 0 else None,
        'mean_group_market_p': float(g.group_market_p.mean()),
        'median_effective_group_odds': float(g.effective_group_odds.median()),
    }


def conditional_order_diag(t: pd.DataFrame):
    # Restrict to trios that actually filled the podium. This isolates order
    # calibration conditional on the trio event having occurred.
    hit_keys = t.groupby(['race_id', 'trio_key'], sort=False).actual_hit.sum()
    hit_keys = hit_keys[hit_keys == 1].index
    if not len(hit_keys):
        return {'group_hits': 0, 'ranks': {}}

    x = t.set_index(['race_id', 'trio_key']).loc[hit_keys].reset_index()
    x['group_market_p'] = x.groupby(['race_id', 'trio_key']).market_p.transform('sum')
    x['conditional_market_p'] = x.market_p / x.group_market_p
    x = x.sort_values(['race_id', 'trio_key', 'odds', 'ticket'], kind='mergesort')
    x['odds_rank_ascending'] = x.groupby(['race_id', 'trio_key']).cumcount() + 1

    out = {}
    for rank in range(1, 7):
        r = x[x.odds_rank_ascending == rank]
        actual = int(r.actual_hit.sum())
        expected = float(r.conditional_market_p.sum())
        out[f'RANK_{rank}'] = {
            'actual_winning_orders': actual,
            'conditional_market_expected_wins': expected,
            'actual_over_conditional_market': float(actual / expected) if expected > 0 else None,
            'actual_share_of_group_hits_pct': float(100 * actual / len(hit_keys)),
            'market_expected_share_pct': float(100 * expected / len(hit_keys)),
            'median_exact_odds': float(r.odds.median()),
        }

    # Basic rank-quality summaries: lower is better; market is useful if the
    # actual order tends to appear near rank 1 more often than uniform 1/6.
    winners = x[x.actual_hit == 1]
    return {
        'group_hits': int(len(hit_keys)),
        'actual_winner_mean_odds_rank': float(winners.odds_rank_ascending.mean()),
        'actual_winner_median_odds_rank': float(winners.odds_rank_ascending.median()),
        'uniform_random_expected_mean_rank': 3.5,
        'ranks': out,
    }


def main():
    t = pd.read_csv(SRC)
    req = {'race_id', 'ticket', 'odds', 'market_p', 'actual_hit', 'group_score_percentile', 'period'}
    missing = req - set(t.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {sorted(missing)}')

    t = t.copy()
    t['trio_key'] = t.ticket.map(trio_key)
    sizes = t.groupby(['race_id', 'trio_key']).size()
    if not (sizes == 6).all():
        raise SystemExit(f'Expected exactly six exact orders per trio; bad={int((sizes != 6).sum())}')

    groups = t.groupby(['race_id', 'trio_key'], as_index=False).agg(
        group_market_p=('market_p', 'sum'),
        group_hit=('actual_hit', 'sum'),
        group_score_percentile=('group_score_percentile', 'first'),
        period=('period', 'first'),
    )
    groups['effective_group_odds'] = 1.0 / groups.group_market_p
    groups.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    variants = {}
    for name, cut in CUTS.items():
        g = groups[groups.group_score_percentile <= cut].copy()
        keys = g[['race_id', 'trio_key']]
        x = t.merge(keys, on=['race_id', 'trio_key'], how='inner')

        by_group_odds = {}
        for bname, lo, hi in GROUP_ODDS_BINS:
            z = g[(g.effective_group_odds >= lo) & (g.effective_group_odds < hi)]
            by_group_odds[bname] = group_agg(z)

        variants[name] = {
            'group_level_all': group_agg(g),
            'group_level_by_effective_odds': by_group_odds,
            'group_level_by_period': {str(p): group_agg(z) for p, z in g.groupby('period', sort=True)},
            'within_six_order_conditional_on_group_hit': conditional_order_diag(x),
        }

    payload = {
        'status': 'exploratory_formation_demand_two_layer_mechanism_test',
        'source': str(SRC.relative_to(ROOT)),
        'hypothesis': {
            'layer_1_group_selection': 'formation-style demand underweights some canonical cross-line unordered trios, producing group calibration > 1, potentially stronger at higher effective group odds',
            'layer_2_ordering': 'conditional on the trio hitting, relative market weights across its six exact orders are substantially better calibrated than the group-selection layer',
        },
        'predeclared_interpretation': {
            'supports_mechanism': 'group calibration > 1 (especially rising in higher effective-group-odds bands) while within-six rank calibration is materially closer to 1 and actual winners are concentrated toward low odds ranks',
            'weakens_mechanism': 'no group-level underpricing, or order-rank conditional calibration is equally/more distorted with no useful concentration toward low odds ranks',
        },
        'warning': 'Mechanism diagnostic on previously explored historical data; not fresh OOS and does not identify bettor behavior directly.',
        'variants': variants,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
