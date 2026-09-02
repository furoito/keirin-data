#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure how solo riders affect the current ticket-level H1 candidate.

Base detail file is produced by analyze_h1_ticket_order_patterns.py and already fixes:
- group score top 45% within race
- exactly 3 lines represented
- no rider at line_pos >= 3

This diagnostic reconstructs line size from true_line and compares:
- any solo rider in trio vs none (all base tickets)
- winner solo vs winner is head of a multi-rider line
for WINNER_RYO, LINE_POS_1_2_2, and their intersection.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
DETAIL = CTX / 'h1_ticket_order_patterns_details.csv'
OUT = CTX / 'h1_ticket_solo_effect_summary.json'
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
    }


def line_size_maps() -> dict[tuple[str, str], dict[int, int]]:
    out = {}
    for month in h1.MONTHS:
        cp = CTX / f'{month}_races.csv'
        if not cp.exists():
            continue
        c = pd.read_csv(cp, encoding='utf-8-sig', dtype={'race_id': str}).drop_duplicates('race_id', keep='last')
        for r in c.to_dict('records'):
            rid = str(r['race_id'])
            lines = base.parse_true_line(r.get('true_line'))
            if not lines:
                continue
            sizes = {}
            for g in lines:
                for fn in g:
                    sizes[int(fn)] = len(g)
            out[(month, rid)] = sizes
    return out


def main():
    if not DETAIL.exists():
        raise SystemExit(f'Missing detail file: {DETAIL}')
    df = pd.read_csv(DETAIL, encoding='utf-8-sig', dtype={'race_id': str})
    maps = line_size_maps()

    solo_count = []
    winner_is_solo = []
    mapped = []
    for r in df.itertuples(index=False):
        sizes = maps.get((str(r.month), str(r.race_id)))
        if not sizes:
            solo_count.append(None); winner_is_solo.append(None); mapped.append(0); continue
        try:
            trio = [int(x) for x in str(r.trio).split('-')]
            winner = int(str(r.ticket).split('-')[0])
        except Exception:
            solo_count.append(None); winner_is_solo.append(None); mapped.append(0); continue
        if any(fn not in sizes for fn in trio) or winner not in sizes:
            solo_count.append(None); winner_is_solo.append(None); mapped.append(0); continue
        solo_count.append(sum(1 for fn in trio if sizes[fn] == 1))
        winner_is_solo.append(int(sizes[winner] == 1))
        mapped.append(1)

    df['solo_count'] = solo_count
    df['winner_is_solo'] = winner_is_solo
    df['solo_mapped'] = mapped
    df = df[df.solo_mapped == 1].copy()
    df['solo_count'] = df.solo_count.astype(int)
    df['winner_is_solo'] = df.winner_is_solo.astype(int)

    views = {}

    # Broad effect in the whole fixed base candidate.
    views['BASE_ANY_SOLO'] = summarize(df[df.solo_count >= 1])
    views['BASE_NO_SOLO'] = summarize(df[df.solo_count == 0])

    subsets = {
        'WINNER_RYO': df[df.winner_running_style.astype(str) == '両'],
        'LINE_POS_1_2_2': df[df.line_pos_order.astype(str) == '1-2-2'],
        'WINNER_RYO_AND_LINE_POS_1_2_2': df[
            (df.winner_running_style.astype(str) == '両') &
            (df.line_pos_order.astype(str) == '1-2-2')
        ],
    }
    for name, z in subsets.items():
        views[f'{name}_WINNER_SOLO'] = summarize(z[z.winner_is_solo == 1])
        views[f'{name}_WINNER_MULTI_LINE_HEAD'] = summarize(z[z.winner_is_solo == 0])
        views[f'{name}_ANY_SOLO'] = summarize(z[z.solo_count >= 1])
        views[f'{name}_NO_SOLO'] = summarize(z[z.solo_count == 0])

    # Distribution only; no optimization on solo count.
    solo_count_distribution = {
        str(k): summarize(g)
        for k, g in df.groupby('solo_count', sort=True)
    }

    payload = {
        'status': 'exploratory_same_data_solo_effect_diagnostic',
        'base_candidate': 'group_score_top45pct AND exactly_3_lines AND NO_THIRD',
        'solo_definition': 'rider belongs to a true_line group of size 1',
        'warning': 'Solo effect is explored on the same discovery data; do not canonize until re-tested after backfill.',
        'mapped_tickets': int(len(df)),
        'views': views,
        'solo_count_distribution': solo_count_distribution,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
