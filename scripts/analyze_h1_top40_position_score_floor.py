#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test whether race-score lower-tail exclusion should apply only to early finish positions.

Fixed base hypothesis:
- exact trifecta ticket
- unordered trio spans exactly 3 distinct true_line IDs
- all selected riders belong to multi-rider lines
- each selected rider is running_style='両' OR line_pos=2
- unordered trio race_score-sum percentile <= 40% within race

Compare four ordered-ticket views:
- TOP40_BASE: no individual score floor
- WINNER_FLOOR: 1st-position rider is not in race-score bottom 30%
- TOP2_FLOOR: 1st and 2nd-position riders are not in bottom 30%
- ALL3_FLOOR: all three riders are not in bottom 30%

Individual lower-tail definition:
- rank race_score descending within race
- midrank percentile=(rank-0.5)/n
- bottom30 iff percentile > 0.70

Same discovery data. Diagnostic only; not OOS validation.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_top40_position_score_floor_summary.json'

ODDS_BINS = [
    ('<50', 0.0, 50.0),
    ('50-100', 50.0, 100.0),
    ('100-200', 100.0, 200.0),
    ('200-500', 200.0, 500.0),
    ('500-1000', 500.0, 1000.0),
    ('1000+', 1000.0, float('inf')),
]


def actual_ordered_top3(pre):
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


def individual_midrank_percentiles(score):
    ordered = sorted(score, key=lambda fn: (-score[fn], fn))
    n = len(ordered)
    return {fn: (i + 0.5) / n for i, fn in enumerate(ordered)}


def agg(x):
    n = int(len(x))
    stake = float(n)
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum()) if n else 0.0
    hits = int(x.actual_hit.sum()) if n else 0
    exp = float(x.market_p.sum()) if n else 0.0
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


def odds_slices(df):
    return {name: agg(df[(df.odds >= lo) & (df.odds < hi)]) for name, lo, hi in ODDS_BINS}


def full_view(df):
    return {
        'all_odds': agg(df),
        'ticket_odds_bins': odds_slices(df),
        'min_ticket_odds': {str(c): agg(df[df.odds >= c]) for c in [50, 100, 200, 500]},
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

            lines = base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved'] += 1
                continue
            frames = sorted({int(x) for g in lines for x in g})

            tri = base.odds_map(og)
            expected = len(frames) * (len(frames) - 1) * (len(frames) - 2)
            if len(tri) != expected:
                skipped['odds_board_incomplete'] += 1
                continue
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                skipped['zero_mass'] += 1
                continue

            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1
                continue

            line_of = {}
            pos_of = {}
            line_size_of = {}
            for li, g in enumerate(lines, 1):
                g2 = [int(x) for x in g]
                for pos, fn in enumerate(g2, 1):
                    line_of[fn] = li
                    pos_of[fn] = pos
                    line_size_of[fn] = len(g2)

            score = {}
            style = {}
            for r in pre.itertuples(index=False):
                try:
                    fn = int(float(r.banum))
                except Exception:
                    continue
                try:
                    sc = float(r.race_score)
                    if np.isfinite(sc):
                        score[fn] = sc
                except Exception:
                    pass
                s = str(getattr(r, 'running_style', '')).strip()
                if not s or s.lower() == 'nan':
                    s = 'UNKNOWN'
                style[fn] = s

            if set(frames) - set(score):
                skipped['score_missing'] += 1
                continue

            indiv_pct = individual_midrank_percentiles(score)
            all_groups = list(itertools.combinations(frames, 3))
            score_sums = {g: float(sum(score[x] for x in g)) for g in all_groups}
            ordered_groups = sorted(all_groups, key=lambda g: (-score_sums[g], g))
            rank_of = {g: i + 1 for i, g in enumerate(ordered_groups)}
            n_groups = len(ordered_groups)

            for trio_u in all_groups:
                if len({line_of.get(x) for x in trio_u}) != 3:
                    continue
                if any(line_size_of.get(x, 0) <= 1 for x in trio_u):
                    continue
                if any(not (style.get(x) == '両' or pos_of.get(x) == 2) for x in trio_u):
                    continue
                group_pct = rank_of[trio_u] / n_groups
                if group_pct > 0.40:
                    continue

                for perm in itertools.permutations(trio_u):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    p = (1.0 / float(od)) / z
                    p1, p2, p3 = [float(indiv_pct[x]) for x in perm]
                    rows.append({
                        'month': month,
                        'race_id': rid,
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'group_score_percentile': float(group_pct),
                        'p1_individual_score_percentile': p1,
                        'p2_individual_score_percentile': p2,
                        'p3_individual_score_percentile': p3,
                        'p1_bottom30': bool(p1 > 0.70),
                        'p2_bottom30': bool(p2 > 0.70),
                        'p3_bottom30': bool(p3 > 0.70),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No qualifying tickets')

    views = {
        'TOP40_BASE': df,
        'WINNER_FLOOR': df[~df.p1_bottom30],
        'TOP2_FLOOR': df[(~df.p1_bottom30) & (~df.p2_bottom30)],
        'ALL3_FLOOR': df[(~df.p1_bottom30) & (~df.p2_bottom30) & (~df.p3_bottom30)],
    }

    hitdf = df[df.actual_hit == 1].copy()
    hit_position_diag = {
        'base_hits': int(len(hitdf)),
        'p1_bottom30_hits': int(hitdf.p1_bottom30.sum()),
        'p2_bottom30_hits': int(hitdf.p2_bottom30.sum()),
        'p3_bottom30_hits': int(hitdf.p3_bottom30.sum()),
        'p1_or_p2_bottom30_hits': int((hitdf.p1_bottom30 | hitdf.p2_bottom30).sum()),
        'p3_bottom30_while_top2_not_bottom30_hits': int(((~hitdf.p1_bottom30) & (~hitdf.p2_bottom30) & hitdf.p3_bottom30).sum()),
    }

    payload = {
        'status': 'exploratory_top40_position_specific_score_floor_test',
        'question': 'Does the individual race_score lower-tail exclusion belong on positions 1-2 while allowing a weaker rider to sneak into 3rd?',
        'fixed_filters': [
            'exactly 3 distinct true_line IDs',
            'all selected riders from multi-rider lines',
            "each selected rider: running_style='両' OR line_pos=2",
            'unordered trio race_score-sum percentile <= 40%',
        ],
        'individual_floor_definition': 'race_score descending midrank percentile=(rank-0.5)/n; bottom30 iff percentile>0.70',
        'views': ['TOP40_BASE', 'WINNER_FLOOR', 'TOP2_FLOOR', 'ALL3_FLOOR'],
        'warning': 'Same discovery data. Position-specific floor idea was proposed after observing prior diagnostics; not OOS validation.',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'actual_hit_bottom30_by_position': hit_position_diag,
        'results': {k: full_view(v) for k, v in views.items()},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
