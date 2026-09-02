#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory diagnostic: does requiring exactly 3 lines add value?

Fix group-score top 45%. Compare exact line_span=1/2/3 with and without
NO_THIRD. Main cuts are effective fair odds >=50, >=100, >=200.
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
OUT = CTX / 'h1_line_span_45_summary.json'
PCT = 0.45
CUTS = [50, 100, 200]


def agg(x: pd.DataFrame) -> dict:
    e = float(x.market_p.sum()) if len(x) else 0.0
    h = int(x.actual_hit.sum()) if len(x) else 0
    return {
        'groups': int(len(x)),
        'races': int(x.race_id.nunique()) if len(x) else 0,
        'expected_hits': e,
        'actual_hits': h,
        'actual_over_market': float(h / e) if e > 0 else None,
    }


def main():
    rows = []
    skipped = Counter()
    usable_by_month = {}

    for month in h1.MONTHS:
        loaded = h1.load_month(month)
        if loaded is None:
            continue
        b, c, o = loaded
        use = c.copy()
        if 'context_quality' in use:
            use = use[use.context_quality.astype(str) == 'full']
        if 'price_usable' in use:
            use = use[use.price_usable.astype(str).str.lower().isin({'true', '1'})]
        use = use.drop_duplicates('race_id', keep='last')
        usable_by_month[month] = int(len(use))

        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}
        oby = {str(k): g for k, g in o.groupby('race_id', sort=False)}

        for cr in use.to_dict('records'):
            rid = str(cr['race_id'])
            pre = bby.get(rid)
            og = oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing'] += 1
                continue
            rr, why = h1.race_rows(month, rid, pre, cr, og)
            if rr is None:
                skipped[why] += 1
                continue

            lines = base.parse_true_line(cr.get('true_line'))
            pos = {}
            for group in lines:
                for i, fn in enumerate(group, 1):
                    pos[int(fn)] = i

            for q in rr:
                trio = tuple(int(x) for x in q['trio'].split('-'))
                q['contains_line3plus'] = int(any(pos.get(fn, 99) >= 3 for fn in trio))
                rows.append(q)

    df = pd.DataFrame(rows)
    df = df[df.group_score_percentile <= PCT].copy()

    payload = {
        'status': 'exploratory_line_span_diagnostic_current_available_data',
        'group_score_top_pct': 45,
        'quality_gate': 'context_quality=full AND price_usable=true',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'results': {},
    }

    filters = {
        'ALL': df,
        'NO_THIRD': df[df.contains_line3plus == 0],
    }

    for filt_name, z in filters.items():
        fout = {}
        for span in [1, 2, 3]:
            s = z[z.line_span == span]
            sout = {'all_odds': agg(s), 'cuts': {}, 'bins': {}}
            for cut in CUTS:
                sout['cuts'][str(cut)] = agg(s[s.effective_fair_odds >= cut])
            sout['bins']['50-100'] = agg(s[(s.effective_fair_odds >= 50) & (s.effective_fair_odds < 100)])
            sout['bins']['100-200'] = agg(s[(s.effective_fair_odds >= 100) & (s.effective_fair_odds < 200)])
            sout['bins']['200-plus'] = agg(s[s.effective_fair_odds >= 200])
            fout[f'line_span_{span}'] = sout

        # Direct contrast that tests the current 3-line restriction.
        two = z[z.line_span == 2]
        three = z[z.line_span == 3]
        fout['two_vs_three'] = {
            str(cut): {
                'line_span_2': agg(two[two.effective_fair_odds >= cut]),
                'line_span_3': agg(three[three.effective_fair_odds >= cut]),
            }
            for cut in CUTS
        }
        payload['results'][filt_name] = fout

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
