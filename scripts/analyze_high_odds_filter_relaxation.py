#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'core_top40_rank1_tickets.csv'
OUT = CTX / 'high_odds_filter_relaxation_summary.json'


def summarize(x: pd.DataFrame):
    n = len(x)
    gross = float((x['rank1_posted_odds'] * x['rank1_exact_hit']).sum()) if n else 0.0
    exact_exp = float(x['rank1_exact_market_p'].sum()) if n else 0.0
    group_exp = float(x['group_market_p'].sum()) if n else 0.0
    return {
        'tickets': int(n),
        'races': int(x['race_uid'].nunique()) if n else 0,
        'exact_wins': int(x['rank1_exact_hit'].sum()) if n else 0,
        'group_hits': int(x['group_hit'].sum()) if n else 0,
        'gross_return_units': gross,
        'posted_odds_roi': gross / n if n else None,
        'exact_expected_hits_normalized_market': exact_exp,
        'exact_actual_over_market': (float(x['rank1_exact_hit'].sum()) / exact_exp) if exact_exp > 0 else None,
        'group_expected_hits_normalized_market': group_exp,
        'group_actual_over_market': (float(x['group_hit'].sum()) / group_exp) if group_exp > 0 else None,
        'median_posted_odds': float(x['rank1_posted_odds'].median()) if n else None,
    }


def main():
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str})
    base = df[(df['role_ok'] == 1) & (df['group_score_percentile'] > 0.20) & (df['group_score_percentile'] <= 0.40)].copy()

    variants = {
        'TOP3_SPREAD_REQUIRED': base[base['top3_spread'] == 1],
        'TOP3_SPREAD_REMOVED_ALL': base,
        'TOP3_NOT_SPREAD_ONLY': base[base['top3_spread'] == 0],
    }

    out = {
        'status': 'exploratory_high_odds_filter_relaxation',
        'fixed_conditions': [
            'exactly 3 distinct true lines',
            'each selected rider is running_style=両 OR line_pos=2',
            '0.20 < group_score_percentile <= 0.40',
            'bet cheapest posted 3rentan order within group',
        ],
        'changed_condition': 'race top3 riders by race_score spread across 3 lines: required vs removed',
        'variants': {},
        'warning': 'Same-data exploratory sensitivity test; do not optimize odds cutoff from these outcomes.',
    }

    for name, g in variants.items():
        out['variants'][name] = {
            'all_odds': summarize(g),
            'ge100': summarize(g[g['rank1_posted_odds'] >= 100]),
            '100_to_200': summarize(g[(g['rank1_posted_odds'] >= 100) & (g['rank1_posted_odds'] < 200)]),
            'ge200': summarize(g[g['rank1_posted_odds'] >= 200]),
            'by_year_ge100': {
                str(year): summarize(yg[yg['rank1_posted_odds'] >= 100])
                for year, yg in g.groupby(g['month'].str[:4], sort=True)
            },
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
