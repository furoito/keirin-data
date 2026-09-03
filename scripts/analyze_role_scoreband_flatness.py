#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the core three-line 両/番手 hypothesis by disjoint score bands.

Fixed core conditions:
- exactly three distinct true lines
- each selected line has at least two riders
- every selected rider is running_style='両' OR line_pos=2 (bante)
- no race-level top3-score-dispersion requirement

Only the unordered trio strength band changes:
TOP10, 10-20, 20-30, 30-40, 40-50, 50-100 by group_score_percentile.

For each band report:
1) unordered 3-rider group calibration: actual group hits / sum(group market p)
2) cheapest exact-order calibration, both unconditional and conditional on group hit
3) flatness quartiles within the band, outcome-blind, with the same calibrations

Input is the already-generated expanded group detail, so this is a cheap diagnostic
rather than another full raw-data rebuild. Exploratory / same-data, not fresh OOS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'formation_box_chain_expanded_groups.csv'
OUT = CTX / 'role_scoreband_flatness_summary.json'

BANDS = [
    ('TOP10', 0.00, 0.10),
    ('10_TO_20', 0.10, 0.20),
    ('20_TO_30', 0.20, 0.30),
    ('30_TO_40', 0.30, 0.40),
    ('40_TO_50', 0.40, 0.50),
    ('50_TO_100', 0.50, 1.00),
]


def safe_ratio(a: float, b: float):
    return float(a / b) if b > 0 else None


def calibrations(x: pd.DataFrame):
    groups = int(len(x))
    races = int(x.race_id.nunique()) if groups else 0

    group_actual = int(x.group_hit.sum()) if groups else 0
    group_expected = float(x.group_market_p.sum()) if groups else 0.0

    # Cheapest exact order in each unordered trio. Its normalized exact-board
    # probability equals group_market_p * rank1_conditional_share.
    rank1_exact_p = x.group_market_p * x.rank1_conditional_share
    rank1_actual = int(x.rank1_exact_hit.sum()) if groups else 0
    rank1_expected_uncond = float(rank1_exact_p.sum()) if groups else 0.0

    hit = x[x.group_hit == 1]
    rank1_expected_cond = float(hit.rank1_conditional_share.sum()) if len(hit) else 0.0

    return {
        'groups': groups,
        'races': races,
        'median_group_odds': float(x.effective_group_odds.median()) if groups else None,
        'mean_entropy_norm': float(x.entropy_norm.mean()) if groups else None,
        'group_market_calibration': {
            'actual_hits': group_actual,
            'expected_hits': group_expected,
            'actual_over_market': safe_ratio(group_actual, group_expected),
        },
        'rank1_unconditional_calibration': {
            'actual_hits': rank1_actual,
            'expected_hits': rank1_expected_uncond,
            'actual_over_market': safe_ratio(rank1_actual, rank1_expected_uncond),
        },
        'rank1_conditional_on_group_hit': {
            'group_hits': int(len(hit)),
            'rank1_actual_wins': rank1_actual,
            'rank1_expected_wins': rank1_expected_cond,
            'actual_over_conditional_market': safe_ratio(rank1_actual, rank1_expected_cond),
            'actual_share_pct': float(100.0 * rank1_actual / len(hit)) if len(hit) else None,
            'market_expected_share_pct': float(100.0 * rank1_expected_cond / len(hit)) if len(hit) else None,
        },
    }


def add_flatness_quartiles(x: pd.DataFrame):
    z = x.copy()
    if len(z) < 4:
        z['flatness_quartile'] = None
        return z
    z['flatness_quartile'] = pd.qcut(
        z.entropy_norm.rank(method='first'),
        4,
        labels=['Q1_LEAST_FLAT', 'Q2', 'Q3', 'Q4_MOST_FLAT'],
    )
    return z


def summarize_band(df: pd.DataFrame, name: str, lo: float, hi: float):
    # Percentile is rank/N, so 0 < pct <= 1. Use (lo, hi].
    x = df[(df.group_score_percentile > lo) & (df.group_score_percentile <= hi)].copy()
    qdf = add_flatness_quartiles(x)
    quartiles = {}
    if 'flatness_quartile' in qdf and qdf.flatness_quartile.notna().any():
        for q, g in qdf.groupby('flatness_quartile', observed=True, sort=True):
            d = calibrations(g)
            d['median_rank1_conditional_share_pct'] = float(100.0 * g.rank1_conditional_share.median())
            d['median_rank1_to_rank6_share_ratio'] = float(g.rank1_to_rank6_share_ratio.median())
            quartiles[str(q)] = d

    out = calibrations(x)
    out.update({
        'band': name,
        'percentile_range': {'gt': lo, 'le': hi},
        'median_rank1_conditional_share_pct': float(100.0 * x.rank1_conditional_share.median()) if len(x) else None,
        'median_rank1_to_rank6_share_ratio': float(x.rank1_to_rank6_share_ratio.median()) if len(x) else None,
        'flatness_quartiles_within_band': quartiles,
    })
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f'missing input: {SRC}')
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'race_id': str})

    required = {
        'race_id','group_score_percentile','role_ok','group_market_p','effective_group_odds',
        'entropy_norm','rank1_conditional_share','rank1_to_rank6_share_ratio',
        'group_hit','rank1_exact_hit'
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')

    # Core role condition is fixed. The source itself already contains only
    # three-distinct-line, multi-rider-line groups.
    core = df[df.role_ok == 1].copy()

    bands = {name: summarize_band(core, name, lo, hi) for name, lo, hi in BANDS}
    payload = {
        'status': 'exploratory_core_role_scoreband_flatness_test',
        'decision_question': 'Within the already-supported three-line 両/番手 structure, which trio-strength bands carry group-level and cheapest-order value, and is that value related to flatness?',
        'fixed_core_conditions': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            'no top3-score-spread race filter',
        ],
        'strength_bands': [{'name': n, 'gt': lo, 'le': hi} for n, lo, hi in BANDS],
        'core_pool': {
            'groups': int(len(core)),
            'races': int(core.race_id.nunique()),
            'group_hits': int(core.group_hit.sum()),
            'rank1_exact_hits': int(core.rank1_exact_hit.sum()),
        },
        'bands': bands,
        'interpretation_note': 'Primary comparison is across disjoint strength bands. Flatness quartiles are secondary diagnostics defined independently inside each strength band, so strength and flatness are not mechanically conflated.',
        'warning': 'Exploratory same-data analysis; not fresh OOS validation.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
