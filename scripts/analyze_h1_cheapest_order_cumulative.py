#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the user's practical staking simplification on the canonical H1 candidates.

Input is the canonical ticket set produced by
analyze_h1_canonical_three_line_ryo_bante_strong.py, which already enforces:
- exactly 3 distinct multi-rider true lines
- each selected rider is running_style='両' OR line_pos=2
- top 3 riders by race_score in the race are spread across 3 true lines

For every unordered candidate trio, its six exact trifecta permutations are sorted
by quoted odds ascending. We then evaluate cumulative selections:
  LOWEST_1, LOWEST_2, ..., LOWEST_6
where every selected ticket receives one equal stake unit (the real-world 100-yen
minimum analogue).

Strength variants are fixed from the canonical run:
  STRUCTURAL_ONLY, TOP50_SCORE_SUM, TOP40_SCORE_SUM, TOP30_SCORE_SUM.

This directly tests whether dropping the high-odds permutations solves the minimum
stake / over-allocation problem, with the user's predeclared prediction that
LOWEST_1 will be best.

Same-data exploratory test, not OOS. Odds rank is part of the betting rule here, not
an ex-post descriptive slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'h1_canonical_three_line_ryo_bante_strong_tickets.csv'
OUT = CTX / 'h1_cheapest_order_cumulative_summary.json'
DETAIL = CTX / 'h1_cheapest_order_cumulative_tickets.csv'

CUTS = {
    'STRUCTURAL_ONLY': 1.00,
    'TOP50_SCORE_SUM': 0.50,
    'TOP40_SCORE_SUM': 0.40,
    'TOP30_SCORE_SUM': 0.30,
}
BOOTSTRAPS = 2000
SEED = 20260903


def trio_key(ticket: str) -> str:
    xs = sorted(int(x) for x in str(ticket).split('-'))
    return '-'.join(map(str, xs))


def agg(x: pd.DataFrame):
    n = int(len(x))
    if not n:
        return {
            'tickets': 0, 'races': 0, 'candidate_trios': 0,
            'actual_hits': 0, 'normalized_market_expected_hits': 0.0,
            'actual_over_normalized_market': None,
            'gross_return_units': 0.0, 'gross_roi_pct': None,
            'avg_odds': None, 'median_odds': None,
        }
    hits = int(x.actual_hit.sum())
    exp = float(x.market_p.sum())
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum())
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()),
        'candidate_trios': int(x.groupby(['race_id', 'trio_key']).ngroups),
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'gross_return_units': gross,
        'gross_roi_pct': float(100.0 * gross / n),
        'avg_odds': float(x.odds.mean()),
        'median_odds': float(x.odds.median()),
    }


def bootstrap_race_ci(x: pd.DataFrame):
    if x.empty:
        return {'bootstrap_reps': 0, 'calibration_ratio_ci95': [None, None], 'gross_roi_pct_ci95': [None, None]}
    y = x.copy()
    y['gross'] = np.where(y.actual_hit == 1, y.odds, 0.0)
    rg = y.groupby('race_id', sort=False).agg(
        tickets=('actual_hit', 'size'),
        hits=('actual_hit', 'sum'),
        exp=('market_p', 'sum'),
        gross=('gross', 'sum'),
    ).reset_index(drop=True)
    n = len(rg)
    if n < 2:
        return {'bootstrap_reps': 0, 'calibration_ratio_ci95': [None, None], 'gross_roi_pct_ci95': [None, None]}
    arr = rg[['tickets', 'hits', 'exp', 'gross']].to_numpy(float)
    rng = np.random.default_rng(SEED + n + int(len(x)))
    ratios = np.empty(BOOTSTRAPS, dtype=float)
    rois = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        s = arr[idx].sum(axis=0)
        ratios[i] = s[1] / s[2] if s[2] > 0 else np.nan
        rois[i] = 100.0 * s[3] / s[0] if s[0] > 0 else np.nan
    return {
        'bootstrap_reps': BOOTSTRAPS,
        'calibration_ratio_ci95': [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
        'gross_roi_pct_ci95': [float(np.nanpercentile(rois, 2.5)), float(np.nanpercentile(rois, 97.5))],
    }


def view(x: pd.DataFrame):
    out = agg(x)
    out['race_bootstrap'] = bootstrap_race_ci(x)
    return out


def main():
    df = pd.read_csv(SRC)
    required = {
        'month', 'period', 'race_id', 'ticket', 'odds', 'market_p', 'actual_hit',
        'group_score_percentile', 'unordered_role_set'
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['trio_key'] = df.ticket.map(trio_key)
    # Stable deterministic tie-breaker uses ticket text only; no outcome information.
    df = df.sort_values(['race_id', 'trio_key', 'odds', 'ticket'], kind='mergesort')
    df['odds_rank_ascending'] = df.groupby(['race_id', 'trio_key'], sort=False).cumcount() + 1
    group_sizes = df.groupby(['race_id', 'trio_key']).size()
    bad = group_sizes[group_sizes != 6]
    if len(bad):
        raise SystemExit(f'Expected exactly six permutations per candidate trio; bad groups={len(bad)}')

    # Diagnostics for tied quoted odds, relevant because LOWEST_1 needs a deterministic
    # tie break even though equal odds imply equal market price.
    tied_groups = 0
    tied_at_min_groups = 0
    for _, g in df.groupby(['race_id', 'trio_key'], sort=False):
        if g.odds.duplicated(keep=False).any():
            tied_groups += 1
        if int((g.odds == g.odds.min()).sum()) > 1:
            tied_at_min_groups += 1

    df.to_csv(DETAIL, index=False)

    variants = {}
    for vname, cut in CUTS.items():
        base = df[df.group_score_percentile <= cut].copy()
        ks = {}
        for k in range(1, 7):
            x = base[base.odds_rank_ascending <= k].copy()
            ks[f'LOWEST_{k}'] = {
                'all': view(x),
                'periods': {str(p): view(g) for p, g in x.groupby('period', sort=True)},
                'role_sets': {str(r): agg(g) for r, g in x.groupby('unordered_role_set', sort=True)},
            }
        variants[vname] = ks

    payload = {
        'status': 'exploratory_cheapest_exact_order_cumulative_test',
        'predeclared_user_prediction': 'LOWEST_1 will have the best practical performance.',
        'source_candidate_rule': {
            'exactly_three_distinct_multi_rider_true_lines': True,
            "selected_role": "running_style='両' OR line_pos=2",
            'strong_riders_spread_condition': 'top 3 riders by race_score occupy 3 distinct true_line IDs',
            'strength_ladder': CUTS,
        },
        'bet_rule': 'Within each unordered candidate trio, rank all six exact trifecta permutations by quoted odds ascending; buy the lowest k at one equal stake unit each.',
        'why_this_test': 'If equal 100-yen stakes on all six permutations over-allocate to high-odds tickets relative to equal-payout staking, progressively removing the high-odds tail should improve practical ROI. LOWEST_1 is the simplest extreme.',
        'odds_rank_tie_policy': 'stable sort by odds ascending, then ticket text; tie-break uses no outcome information',
        'tie_diagnostics': {
            'candidate_trios_total': int(df.groupby(['race_id', 'trio_key']).ngroups),
            'groups_with_any_equal_odds': int(tied_groups),
            'groups_with_tied_minimum_odds': int(tied_at_min_groups),
        },
        'primary_metrics': ['actual_over_normalized_market', 'gross_roi_pct'],
        'warning': 'Same discovery data; odds rank is intentionally part of the proposed live betting rule. Period splits are stability diagnostics, not OOS.',
        'variants': variants,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
