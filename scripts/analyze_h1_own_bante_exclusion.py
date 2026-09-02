#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test whether excluding a RYO line-head's own bante is itself associated with value.

Discovery filters kept fixed:
- unordered trio group-score top 45% in race
- no rider at line_pos >= 3

Ticket anchor:
- 1st rider is running_style == '両'
- 1st rider is line_pos == 1
- 1st rider belongs to a multi-rider line with an identifiable own bante (line_pos == 2)

Primary split:
- OWN_BANTE_IN_TRIO
- OWN_BANTE_EXCLUDED

Mechanism split:
- positions 2 and 3 are both line_pos == 2 (bante-bante)
  * own bante included => two-line structure
  * own bante excluded => three-line collapse structure

Flat 1-unit ticket ROI at quoted trifecta odds >=50 / >=100 / >=200.
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
OUT = CTX / 'h1_own_bante_exclusion_summary.json'
CUTS = [50, 100, 200]
PCT = 0.45


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
            line_of = {}
            line_pos = {}
            own_bante_of_head = {}
            for li, g in enumerate(lines, 1):
                if len(g) >= 2:
                    own_bante_of_head[int(g[0])] = int(g[1])
                for pos, fn in enumerate(g, 1):
                    line_of[int(fn)] = li
                    line_pos[int(fn)] = pos

            style = {}
            for r in pre.itertuples(index=False):
                try:
                    fn = int(r.banum)
                except Exception:
                    continue
                style[fn] = str(getattr(r, 'running_style', '')).strip()

            for q in group_rows:
                if float(q['group_score_percentile']) > PCT:
                    continue
                trio = tuple(int(x) for x in q['trio'].split('-'))
                if any(line_pos.get(fn, 99) >= 3 for fn in trio):
                    continue

                for perm in itertools.permutations(trio):
                    a, b2, c3 = perm
                    if style.get(a) != '両' or line_pos.get(a) != 1:
                        continue
                    own_bante = own_bante_of_head.get(a)
                    if own_bante is None:
                        continue
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue

                    own_in = int(own_bante in trio)
                    both_minor_bante = int(line_pos.get(b2) == 2 and line_pos.get(c3) == 2)
                    p = (1.0 / float(od)) / z
                    rows.append({
                        'month': month,
                        'race_id': rid,
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'own_bante_in_trio': own_in,
                        'both_second_third_are_bante': both_minor_bante,
                        'line_span': int(q['line_span']),
                        'group_score_percentile': float(q['group_score_percentile']),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No matching tickets')

    views = {
        'RYO_HEAD_OWN_BANTE_IN_TRIO': summarize(df[df.own_bante_in_trio == 1]),
        'RYO_HEAD_OWN_BANTE_EXCLUDED': summarize(df[df.own_bante_in_trio == 0]),
        'RYO_HEAD_BANTE_BANTE_OWN_BANTE_IN_TRIO': summarize(df[(df.both_second_third_are_bante == 1) & (df.own_bante_in_trio == 1)]),
        'RYO_HEAD_BANTE_BANTE_OWN_BANTE_EXCLUDED': summarize(df[(df.both_second_third_are_bante == 1) & (df.own_bante_in_trio == 0)]),
    }

    span_diag = {}
    for own_label, own_value in [('IN', 1), ('OUT', 0)]:
        zdf = df[df.own_bante_in_trio == own_value]
        span_diag[own_label] = {str(span): summarize(g) for span, g in zdf.groupby('line_span', sort=True)}

    payload = {
        'status': 'exploratory_same_data_own_bante_exclusion_diagnostic',
        'base_filter': 'group_score_top45pct AND NO_THIRD; ordered ticket winner is multi-rider-line head with running_style=両',
        'primary_question': 'Does excluding the winner head own bante from the candidate trio improve ticket-level value?',
        'warning': 'Same discovery data. Own-bante inclusion is structurally related to line span, especially in the bante-bante subset; treat as mechanism evidence, not causal proof.',
        'stake_model': 'flat 1 unit per ordered trifecta ticket',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'views': views,
        'line_span_diagnostic': span_diag,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
