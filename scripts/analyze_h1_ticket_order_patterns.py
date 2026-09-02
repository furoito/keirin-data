#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore which ordered trifecta patterns retain value inside the H1 candidate.

Candidate is fixed for this diagnostic:
- unordered group score top 45% within race
- exactly 3 lines represented
- no rider at line_pos >= 3

Every candidate trio is expanded to all six ordered trifecta tickets. We then
classify the ordered ticket by pre-race information only:
- score_order: permutation of within-trio race_score ranks (1=highest score)
- winner_score_rank
- line_pos_order and winner_line_pos
- running_style_order and winner_running_style
- escape_count / winner_is_escape
- winner_score_rank_x_line_pos

Evaluation uses quoted ordered trifecta odds and flat 1-unit stake at >=50/100/200.
This is exploratory selection work, not validation.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_ticket_order_patterns_summary.json'
DETAIL = CTX / 'h1_ticket_order_patterns_details.csv'
PCT = 0.45
ODDS_CUTS = [50, 100, 200]


def actual_ordered_top3(pre: pd.DataFrame):
    vals = []
    for r in pre.itertuples(index=False):
        try:
            pos = int(str(r.rank).strip())
            fn = int(float(r.banum))
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


def summarize_view(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for value, z in df.groupby(col, dropna=False, sort=True):
        key = str(value)
        d = {'all_odds': agg(z), 'min_ticket_odds': {}}
        for cut in ODDS_CUTS:
            d['min_ticket_odds'][str(cut)] = agg(z[z.odds >= cut])
        out[key] = d
    return out


def main():
    rows = []
    skipped = Counter()
    usable_by_month = {}
    style_values = Counter()

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
            line_pos = {}
            for group in lines:
                for i, fn in enumerate(group, 1):
                    line_pos[int(fn)] = i

            score = {}
            style = {}
            for r in pre.itertuples(index=False):
                try:
                    fn = int(float(r.banum))
                except Exception:
                    continue
                try:
                    score[fn] = float(r.race_score)
                except Exception:
                    pass
                s = str(getattr(r, 'running_style', '')).strip()
                if not s or s.lower() == 'nan':
                    s = 'UNKNOWN'
                style[fn] = s
                style_values[s] += 1

            candidates = [
                q for q in group_rows
                if q['group_score_percentile'] <= PCT and q['line_span'] == 3
            ]
            for q in candidates:
                trio = tuple(int(x) for x in q['trio'].split('-'))
                if any(line_pos.get(fn, 99) >= 3 for fn in trio):
                    continue
                if any(fn not in score for fn in trio):
                    skipped['candidate_score_missing'] += 1
                    continue

                ranked = sorted(trio, key=lambda fn: (-score[fn], fn))
                score_rank = {fn: i + 1 for i, fn in enumerate(ranked)}
                escape_count = sum(1 for fn in trio if style.get(fn) == '逃')

                for perm in itertools.permutations(trio):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    p = (1.0 / float(od)) / z
                    score_order = '-'.join(str(score_rank[fn]) for fn in perm)
                    lp_order = '-'.join(str(line_pos.get(fn, 99)) for fn in perm)
                    style_order = '-'.join(style.get(fn, 'UNKNOWN') for fn in perm)
                    winner_sr = score_rank[perm[0]]
                    winner_lp = line_pos.get(perm[0], 99)
                    winner_style = style.get(perm[0], 'UNKNOWN')
                    rows.append({
                        'month': month,
                        'race_id': rid,
                        'trio': '-'.join(map(str, trio)),
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'score_order': score_order,
                        'winner_score_rank': int(winner_sr),
                        'line_pos_order': lp_order,
                        'winner_line_pos': int(winner_lp),
                        'running_style_order': style_order,
                        'winner_running_style': winner_style,
                        'escape_count': int(escape_count),
                        'winner_is_escape': int(winner_style == '逃'),
                        'winner_score_rank_x_line_pos': f'{winner_sr}x{winner_lp}',
                        'group_score_percentile': float(q['group_score_percentile']),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No ticket rows')
    df = df.sort_values(['month', 'race_id', 'trio', 'ticket'])
    df.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    views = [
        'score_order',
        'winner_score_rank',
        'line_pos_order',
        'winner_line_pos',
        'winner_running_style',
        'escape_count',
        'winner_is_escape',
        'winner_score_rank_x_line_pos',
    ]
    payload = {
        'status': 'exploratory_ticket_order_pattern_current_available_data',
        'candidate': 'group_score_top45pct AND exactly_3_lines AND NO_THIRD',
        'quality_gate': 'context_quality=full AND price_usable=true',
        'stake_model': 'flat 1 unit per qualifying ordered trifecta ticket',
        'odds_cuts': ODDS_CUTS,
        'style_values_seen': dict(style_values),
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'overall': {
            'all_odds': agg(df),
            'min_ticket_odds': {str(c): agg(df[df.odds >= c]) for c in ODDS_CUTS},
        },
        'views': {col: summarize_view(df, col) for col in views},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'detail={DETAIL}')
    print(f'summary={OUT}')


if __name__ == '__main__':
    main()
