#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intersection diagnostic for the current ticket-level H1 candidate.

Base candidate is already fixed in h1_ticket_order_patterns_details.csv:
- group score top 45% within race
- exactly 3 lines represented
- no rider at line_pos >= 3

This script compares:
1) winner_running_style == '両'
2) line_pos_order == '1-2-2'
3) intersection of both

Each view is evaluated at quoted trifecta odds >=50, >=100, >=200 using flat 1 unit per ticket.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
DETAIL = CTX / 'h1_ticket_order_patterns_details.csv'
OUT = CTX / 'h1_ticket_order_intersection_summary.json'
CUTS = [50, 100, 200]


def agg(x: pd.DataFrame) -> dict:
    n = int(len(x))
    stake = float(n)
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum()) if n else 0.0
    exp = float(x.market_p.sum()) if n else 0.0
    hits = int(x.actual_hit.sum()) if n else 0
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()) if n else 0,
        'stake_units': stake,
        'gross_return_units': gross,
        'gross_roi_pct': float(100.0 * gross / stake) if stake else None,
        'net_roi_pct': float(100.0 * (gross - stake) / stake) if stake else None,
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'avg_ticket_odds': float(x.odds.mean()) if n else None,
        'median_ticket_odds': float(x.odds.median()) if n else None,
    }


def summarize(z: pd.DataFrame) -> dict:
    return {
        'all_odds': agg(z),
        'min_ticket_odds': {str(c): agg(z[z.odds >= c]) for c in CUTS},
        'non_overlapping_bins': {
            '50-100': agg(z[(z.odds >= 50) & (z.odds < 100)]),
            '100-200': agg(z[(z.odds >= 100) & (z.odds < 200)]),
            '200-plus': agg(z[z.odds >= 200]),
        },
    }


def main():
    if not DETAIL.exists():
        raise SystemExit(f'Missing detail file: {DETAIL}')
    df = pd.read_csv(DETAIL, encoding='utf-8-sig', dtype={'race_id': str})

    views = {
        'WINNER_RYO': df[df.winner_running_style.astype(str) == '両'].copy(),
        'LINE_POS_1_2_2': df[df.line_pos_order.astype(str) == '1-2-2'].copy(),
        'WINNER_RYO_AND_LINE_POS_1_2_2': df[
            (df.winner_running_style.astype(str) == '両') &
            (df.line_pos_order.astype(str) == '1-2-2')
        ].copy(),
    }

    payload = {
        'status': 'exploratory_same_data_intersection_diagnostic',
        'base_candidate': 'group_score_top45pct AND exactly_3_lines AND NO_THIRD',
        'warning': 'Intersection conditions were discovered on the same data; treat results as Discovery only until re-tested on added/backfilled data.',
        'stake_model': 'flat 1 unit per ordered trifecta ticket',
        'odds_cuts': CUTS,
        'views': {name: summarize(z) for name, z in views.items()},
    }

    # Within the intersection, expose score-order distribution without optimizing on it.
    inter = views['WINNER_RYO_AND_LINE_POS_1_2_2']
    score_orders = {}
    for so, g in inter.groupby('score_order', sort=True):
        score_orders[str(so)] = {
            'tickets': int(len(g)),
            'actual_hits': int(g.actual_hit.sum()),
            'min_ticket_odds': {str(c): agg(g[g.odds >= c]) for c in CUTS},
        }
    payload['intersection_score_order_diagnostic'] = score_orders

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
