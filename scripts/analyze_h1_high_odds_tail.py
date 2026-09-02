#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory high-odds-tail diagnostic for H1.

Focuses on whether the distortion strengthens above effective fair odds 200.
Reports non-overlapping bins and cumulative cuts for BASELINE and NO_THIRD,
across group-score top 40/45/50/55/60%.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_high_odds_tail_summary.json'
PCTS = [0.40, 0.45, 0.50, 0.55, 0.60]
BINS = [(100, 150), (150, 200), (200, 300), (300, 500), (500, 1000), (1000, None)]
CUM = [100, 150, 200, 300, 500, 1000]


def period(month: str) -> str:
    y, m = map(int, month.split('_'))
    if y == 2025 and m <= 6: return '2025_H1'
    if y == 2025: return '2025_H2'
    if y == 2026 and m <= 6: return '2026_H1'
    return 'other'


def agg(x: pd.DataFrame) -> dict:
    e = float(x.market_p.sum()) if len(x) else 0.0
    h = int(x.actual_hit.sum()) if len(x) else 0
    return {
        'groups': int(len(x)),
        'races': int(x.race_id.nunique()) if len(x) else 0,
        'expected_hits': e,
        'actual_hits': h,
        'actual_over_market': float(h/e) if e > 0 else None,
    }


def main():
    rows = []
    skipped = Counter()
    usable_by_month = {}
    for month in h1.MONTHS:
        loaded = h1.load_month(month)
        if loaded is None: continue
        b, c, o = loaded
        use = c.copy()
        if 'context_quality' in use:
            use = use[use.context_quality.astype(str) == 'full']
        if 'price_usable' in use:
            use = use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        use = use.drop_duplicates('race_id', keep='last')
        usable_by_month[month] = int(len(use))
        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}
        oby = {str(k): g for k, g in o.groupby('race_id', sort=False)}
        for cr in use.to_dict('records'):
            rid = str(cr['race_id'])
            pre = bby.get(rid); og = oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing'] += 1; continue
            rr, why = h1.race_rows(month, rid, pre, cr, og)
            if rr is None:
                skipped[why] += 1; continue
            lines = base.parse_true_line(cr.get('true_line'))
            pos = {}
            for group in lines:
                for i, fn in enumerate(group, 1): pos[int(fn)] = i
            for q in rr:
                trio = tuple(int(x) for x in q['trio'].split('-'))
                q['contains_line3plus'] = int(any(pos.get(fn, 99) >= 3 for fn in trio))
                q['period'] = period(month)
                rows.append(q)

    df = pd.DataFrame(rows)
    df = df[df.line_span >= 3].copy()
    payload = {
        'status': 'exploratory_high_odds_tail_current_available_data',
        'quality_gate': 'context_quality=full AND price_usable=true',
        'score_percentiles_tested': [40,45,50,55,60],
        'non_overlapping_bins': [[lo, hi] for lo, hi in BINS],
        'cumulative_cuts': CUM,
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'results': {},
    }
    for pct in PCTS:
        pkey = str(int(round(pct * 100)))
        basez = df[df.group_score_percentile <= pct]
        payload['results'][pkey] = {}
        for name, z in {
            'BASELINE': basez,
            'NO_THIRD': basez[basez.contains_line3plus == 0],
        }.items():
            out = {'bins': {}, 'cumulative': {}, 'periods': {}}
            for lo, hi in BINS:
                key = f'{lo}-{hi if hi is not None else "plus"}'
                x = z[z.effective_fair_odds >= lo]
                if hi is not None:
                    x = x[x.effective_fair_odds < hi]
                out['bins'][key] = agg(x)
            for cut in CUM:
                out['cumulative'][str(cut)] = agg(z[z.effective_fair_odds >= cut])
            for per, g in z.groupby('period', sort=True):
                out['periods'][str(per)] = {
                    '200': agg(g[g.effective_fair_odds >= 200]),
                    '300': agg(g[g.effective_fair_odds >= 300]),
                    '500': agg(g[g.effective_fair_odds >= 500]),
                }
            payload['results'][pkey][name] = out

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
