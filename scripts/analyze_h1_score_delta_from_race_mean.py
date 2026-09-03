#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose score strength using distance from each race's mean race_score.

Purpose
-------
Test whether absolute score distance from the race mean explains the current
three-line role hypothesis better than within-race rank/percentile cuts.

Structural base only (NO score threshold):
- exact trifecta ticket
- unordered trio spans exactly 3 distinct true_line IDs
- all selected riders belong to multi-rider lines
- each selected rider is running_style='両' OR line_pos=2

For every qualifying unordered trio:
- race_mean_score = mean race_score of all starters with valid scores
- member_delta_i = selected rider race_score - race_mean_score
- trio_mean_delta = mean(member_delta_i)
- trio_min_delta = min(member_delta_i)
- trio_score_sum_percentile is also retained only as a reference to TOP40

Predeclared coarse bands (no threshold optimization):
  <-3, -3..-1, -1..+1, +1..+3, >=+3

Outputs:
- structural BASE and existing TOP40 reference
- 1D bands for trio_mean_delta
- 1D bands for trio_min_delta
- 2D trio_mean_delta x trio_min_delta cells
Each cell reports all odds, >=200x, and >=500x summaries.

Same discovery data. Exploratory diagnostic, not OOS validation.
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
OUT = CTX / 'h1_score_delta_from_race_mean_summary.json'

BANDS = [
    ('LT_-3', float('-inf'), -3.0),
    ('-3_TO_-1', -3.0, -1.0),
    ('-1_TO_+1', -1.0, 1.0),
    ('+1_TO_+3', 1.0, 3.0),
    ('GE_+3', 3.0, float('inf')),
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
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'avg_ticket_odds': float(x.odds.mean()) if n else None,
        'median_ticket_odds': float(x.odds.median()) if n else None,
        'avg_trio_mean_delta': float(x.trio_mean_delta.mean()) if n else None,
        'avg_trio_min_delta': float(x.trio_min_delta.mean()) if n else None,
    }


def compact_view(df):
    return {
        'all_odds': agg(df),
        'odds_ge_200': agg(df[df.odds >= 200.0]),
        'odds_ge_500': agg(df[df.odds >= 500.0]),
    }


def band_mask(s, lo, hi):
    return (s >= lo) & (s < hi)


def band_views(df, col):
    out = {}
    for name, lo, hi in BANDS:
        out[name] = compact_view(df[band_mask(df[col], lo, hi)])
    return out


def two_dimensional_views(df):
    out = {}
    for mean_name, mean_lo, mean_hi in BANDS:
        row = {}
        mm = band_mask(df.trio_mean_delta, mean_lo, mean_hi)
        for min_name, min_lo, min_hi in BANDS:
            m = mm & band_mask(df.trio_min_delta, min_lo, min_hi)
            row[min_name] = compact_view(df[m])
        out[mean_name] = row
    return out


def main():
    rows = []
    skipped = Counter()
    usable_races = 0

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
        usable_races += int(len(use))

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
                st = str(getattr(r, 'running_style', '')).strip()
                if not st or st.lower() == 'nan':
                    st = 'UNKNOWN'
                style[fn] = st

            if set(frames) - set(score):
                skipped['score_missing'] += 1
                continue

            race_mean = float(np.mean([score[x] for x in frames]))
            delta = {x: float(score[x] - race_mean) for x in frames}

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

                ds = [delta[x] for x in trio_u]
                trio_mean_delta = float(np.mean(ds))
                trio_min_delta = float(min(ds))
                group_pct = float(rank_of[trio_u] / n_groups)

                for perm in itertools.permutations(trio_u):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    p = (1.0 / float(od)) / z
                    rows.append({
                        'race_id': rid,
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'group_score_percentile': group_pct,
                        'race_mean_score': race_mean,
                        'trio_mean_delta': trio_mean_delta,
                        'trio_min_delta': trio_min_delta,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No qualifying tickets')

    payload = {
        'status': 'exploratory_score_delta_from_race_mean_diagnostic',
        'question': 'Does distance from the race mean explain score quality better than relative rank/percentile?',
        'structural_base': [
            'exactly 3 distinct true_line IDs',
            'all selected riders from multi-rider lines',
            "each selected rider: running_style='両' OR line_pos=2",
            'no score threshold in BASE',
        ],
        'definitions': {
            'member_delta': 'rider race_score - mean race_score of all starters in that race',
            'trio_mean_delta': 'mean of the three selected member_delta values',
            'trio_min_delta': 'minimum of the three selected member_delta values',
            'coarse_bands': ['<-3', '-3..-1', '-1..+1', '+1..+3', '>=+3'],
        },
        'warning': 'Same discovery data and post-hoc diagnostic. Coarse bands are descriptive and not OOS validation; do not freeze the best-looking cell as a rule.',
        'usable_context_rows': usable_races,
        'skipped': dict(skipped),
        'reference': {
            'STRUCTURAL_BASE': compact_view(df),
            'TOP40_SCORE_SUM': compact_view(df[df.group_score_percentile <= 0.40]),
        },
        'trio_mean_delta_bands': band_views(df, 'trio_mean_delta'),
        'trio_min_delta_bands': band_views(df, 'trio_min_delta'),
        'mean_delta_x_min_delta': two_dimensional_views(df),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
