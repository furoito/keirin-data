#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ticket-level validation for the current H1 candidate.

Candidate construction remains group based:
- group score top 45% within race
- exactly 3 lines represented
- optional NO_THIRD filter

Evaluation is then at the actual ordered trifecta ticket level. Each qualifying
unordered trio is expanded to its six ordered trifecta permutations. Tickets are
filtered by quoted trifecta odds >= 50x / 100x / 200x.

Reports:
- flat 1-unit stake per ticket gross ROI and profit
- exact-ticket hits
- normalized market expected hits and calibration ratio
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_ticket_level_45_summary.json'
DETAIL = CTX / 'h1_ticket_level_45_details.csv'
PCT = 0.45
ODDS_CUTS = [50, 100, 200]


def actual_ordered_top3(pre: pd.DataFrame):
    vals = []
    for r in pre.itertuples(index=False):
        try:
            pos = int(str(r.rank).strip())
            fn = int(r.banum)
        except Exception:
            continue
        if 1 <= pos <= 3:
            vals.append((pos, fn))
    vals.sort()
    if [p for p, _ in vals] != [1, 2, 3]:
        return None
    return tuple(fn for _, fn in vals)


def agg(x: pd.DataFrame) -> dict:
    n = int(len(x))
    stake = float(n)
    gross_return = float(x.loc[x.actual_hit == 1, 'odds'].sum()) if n else 0.0
    profit = gross_return - stake
    exp = float(x.market_p.sum()) if n else 0.0
    hits = int(x.actual_hit.sum()) if n else 0
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()) if n else 0,
        'stake_units': stake,
        'gross_return_units': gross_return,
        'profit_units': profit,
        'gross_roi_pct': float(100.0 * gross_return / stake) if stake > 0 else None,
        'net_roi_pct': float(100.0 * profit / stake) if stake > 0 else None,
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'avg_ticket_odds': float(x.odds.mean()) if n else None,
        'median_ticket_odds': float(x.odds.median()) if n else None,
    }


def main():
    ticket_rows = []
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

            group_rows, why = h1.race_rows(month, rid, pre, cr, og)
            if group_rows is None:
                skipped[why] += 1
                continue

            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1
                continue

            tri = base.odds_map(og)
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                skipped['zero_mass'] += 1
                continue

            lines = base.parse_true_line(cr.get('true_line'))
            pos = {}
            for group in lines:
                for i, fn in enumerate(group, 1):
                    pos[int(fn)] = i

            # Keep only the fixed current candidate construction: top45 and 3 lines.
            candidates = [
                q for q in group_rows
                if q['group_score_percentile'] <= PCT and q['line_span'] == 3
            ]

            for q in candidates:
                trio = tuple(int(x) for x in q['trio'].split('-'))
                contains_third = int(any(pos.get(fn, 99) >= 3 for fn in trio))
                for perm in itertools.permutations(trio):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    p = (1.0 / float(od)) / z
                    ticket_rows.append({
                        'month': month,
                        'race_id': rid,
                        'trio': '-'.join(map(str, trio)),
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'contains_line3plus': contains_third,
                        'group_score_percentile': float(q['group_score_percentile']),
                        'group_effective_fair_odds': float(q['effective_fair_odds']),
                    })

    df = pd.DataFrame(ticket_rows)
    if df.empty:
        raise SystemExit('No ticket rows')
    df = df.sort_values(['month', 'race_id', 'trio', 'ticket'])
    df.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    payload = {
        'status': 'exploratory_ticket_level_current_available_data',
        'candidate': 'group_score_top45pct AND exactly_3_lines; ticket-level quoted odds filter',
        'quality_gate': 'context_quality=full AND price_usable=true',
        'stake_model': 'flat 1 unit per qualifying ordered trifecta ticket',
        'odds_semantics': 'quoted ordered trifecta odds, not group effective fair odds',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'views': {},
    }

    views = {
        'THREE_LINES_TOP45_ALL': df,
        'THREE_LINES_TOP45_NO_THIRD': df[df.contains_line3plus == 0],
    }
    for name, z in views.items():
        out = {'all_tickets': agg(z), 'min_ticket_odds': {}, 'non_overlapping_bins': {}}
        for cut in ODDS_CUTS:
            out['min_ticket_odds'][str(cut)] = agg(z[z.odds >= cut])
        out['non_overlapping_bins']['50-100'] = agg(z[(z.odds >= 50) & (z.odds < 100)])
        out['non_overlapping_bins']['100-200'] = agg(z[(z.odds >= 100) & (z.odds < 200)])
        out['non_overlapping_bins']['200-plus'] = agg(z[z.odds >= 200])
        payload['views'][name] = out

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'detail={DETAIL}')
    print(f'summary={OUT}')


if __name__ == '__main__':
    main()
