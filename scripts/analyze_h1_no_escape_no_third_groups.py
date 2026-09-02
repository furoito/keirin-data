#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory filter diagnostic on frozen H1 group-level hypothesis.

Baseline is unchanged:
- unordered 3-rider groups
- spans exactly/at least 3 lines (with current 7-rider data, line_span>=3)
- group race_score sum percentile <= 50% within race
- no upper effective-odds cap

Compare BASELINE / NO_ESCAPE / NO_THIRD / NO_ESCAPE_NO_THIRD.
This is diagnostic only; it does not change the frozen hypothesis.
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
OUT = CTX / 'h1_no_escape_no_third_groups_summary.json'
CUTS = [10, 20, 30, 50, 100, 200]
MONTHS = h1.MONTHS


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
    for month in MONTHS:
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
    base_df = df[(df.line_span >= 3) & (df.group_score_percentile <= 0.50)].copy()
    variants = {
        'BASELINE': base_df,
        'NO_ESCAPE': base_df[base_df.contains_escape == 0],
        'NO_THIRD': base_df[base_df.contains_line3plus == 0],
        'NO_ESCAPE_NO_THIRD': base_df[(base_df.contains_escape == 0) & (base_df.contains_line3plus == 0)],
    }
    payload = {
        'status': 'exploratory_filter_diagnostic_current_available_data',
        'baseline_definition': 'three-line spanning unordered group × group-score top50%; no upper effective-odds cap',
        'filter_definition': {
            'NO_ESCAPE': "group contains no rider whose running_style is exactly '逃'",
            'NO_THIRD': 'group contains no rider at line_pos >= 3 in true_line',
            'NO_ESCAPE_NO_THIRD': 'both filters',
        },
        'quality_gate': 'context_quality=full AND price_usable=true',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'variants': {},
    }
    for name, z in variants.items():
        v = {'all_effective_odds': agg(z), 'min_effective_odds': {}, 'periods': {}, 'months_at_30_50': {}}
        for cut in CUTS:
            x = z[z.effective_fair_odds >= cut]
            v['min_effective_odds'][str(cut)] = agg(x)
        for p, g in z.groupby('period', sort=True):
            v['periods'][str(p)] = {
                '30': agg(g[g.effective_fair_odds >= 30]),
                '50': agg(g[g.effective_fair_odds >= 50]),
            }
        for m, g in z.groupby('month', sort=True):
            v['months_at_30_50'][str(m)] = {
                '30': agg(g[g.effective_fair_odds >= 30]),
                '50': agg(g[g.effective_fair_odds >= 50]),
            }
        payload['variants'][name] = v
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
