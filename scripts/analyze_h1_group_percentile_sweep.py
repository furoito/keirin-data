#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory sweep of group-score percentile cutoffs for H1.

Sweeps top 40/45/50/55/60% among unordered 3-rider groups within each race.
Keeps H1 market construction unchanged and compares BASELINE / NO_ESCAPE /
NO_THIRD / NO_ESCAPE_NO_THIRD over effective-odds >=30/50/100/200.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
OUT = CTX / 'h1_group_percentile_sweep_summary.json'
PCTS = [0.40, 0.45, 0.50, 0.55, 0.60]
ODDS_CUTS = [30, 50, 100, 200]


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
            styles = {}
            for r in pre.itertuples(index=False):
                try: fn = int(float(r.banum))
                except Exception: continue
                styles[fn] = str(getattr(r, 'running_style', '')).strip()
            for q in rr:
                trio = tuple(int(x) for x in q['trio'].split('-'))
                q['contains_escape'] = int(any(styles.get(fn, '') == '逃' for fn in trio))
                q['contains_line3plus'] = int(any(pos.get(fn, 99) >= 3 for fn in trio))
                q['period'] = period(month)
                rows.append(q)

    df = pd.DataFrame(rows)
    df = df[df.line_span >= 3].copy()
    payload = {
        'status': 'exploratory_percentile_sweep_current_available_data',
        'score_percentiles_tested': [40,45,50,55,60],
        'effective_odds_cuts': ODDS_CUTS,
        'quality_gate': 'context_quality=full AND price_usable=true',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'results': {},
    }
    variants = {
        'BASELINE': lambda x: x,
        'NO_ESCAPE': lambda x: x[x.contains_escape == 0],
        'NO_THIRD': lambda x: x[x.contains_line3plus == 0],
        'NO_ESCAPE_NO_THIRD': lambda x: x[(x.contains_escape == 0) & (x.contains_line3plus == 0)],
    }
    for pct in PCTS:
        pkey = str(int(round(pct*100)))
        z0 = df[df.group_score_percentile <= pct]
        payload['results'][pkey] = {}
        for name, fn in variants.items():
            z = fn(z0)
            v = {'odds': {}, 'periods': {}}
            for cut in ODDS_CUTS:
                v['odds'][str(cut)] = agg(z[z.effective_fair_odds >= cut])
            for per, g in z.groupby('period', sort=True):
                v['periods'][str(per)] = {
                    str(cut): agg(g[g.effective_fair_odds >= cut]) for cut in [50,100]
                }
            payload['results'][pkey][name] = v

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
