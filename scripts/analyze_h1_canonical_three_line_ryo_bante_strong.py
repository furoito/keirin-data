#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical exploratory test of the current high-level hypothesis.

Hypothesis:
  Pick one rider from each of three distinct multi-rider lines; every selected rider
  is either running_style='両' or line_pos=2 (bante); the race's three highest
  race_score riders must themselves be spread across three distinct true_line IDs;
  then prefer trios that are strong relative to all unordered trios in the race.

Candidate discovery is unordered. Evaluation expands each trio into all six exact
trifecta orders and uses normalized exact-board market probabilities.

Strength ladder is predeclared for this run:
  STRUCTURAL_ONLY, TOP50_SCORE_SUM, TOP40_SCORE_SUM, TOP30_SCORE_SUM.
No odds threshold is used to generate candidates. Odds bands are descriptive only.

Same-data exploratory test, not OOS. Period splits and race-level bootstrap are for
stability/uncertainty diagnostics, not fresh validation.
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
OUT = CTX / 'h1_canonical_three_line_ryo_bante_strong_summary.json'
DETAIL = CTX / 'h1_canonical_three_line_ryo_bante_strong_tickets.csv'

CUTS = {
    'STRUCTURAL_ONLY': 1.00,
    'TOP50_SCORE_SUM': 0.50,
    'TOP40_SCORE_SUM': 0.40,
    'TOP30_SCORE_SUM': 0.30,
}
ODDS_BINS = [
    ('LT_50', 0.0, 50.0),
    ('50_TO_100', 50.0, 100.0),
    ('100_TO_200', 100.0, 200.0),
    ('200_TO_500', 200.0, 500.0),
    ('500_TO_1000', 500.0, 1000.0),
    ('GE_1000', 1000.0, float('inf')),
]
BOOTSTRAPS = 2000
SEED = 20260903


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


def period_of(month: str):
    y, m = month.split('_')
    y = int(y); m = int(m)
    if y == 2025 and m <= 6:
        return '2025_H1'
    if y == 2025:
        return '2025_H2'
    if y == 2026 and m <= 6:
        return '2026_H1'
    return f'{y}_OTHER'


def agg(x: pd.DataFrame):
    n = int(len(x))
    if not n:
        return {
            'tickets': 0, 'races': 0, 'actual_hits': 0,
            'normalized_market_expected_hits': 0.0,
            'actual_over_normalized_market': None,
            'gross_return_units': 0.0, 'gross_roi_pct': None,
            'avg_odds': None, 'median_odds': None,
        }
    hits = int(x.actual_hit.sum())
    exp = float(x.market_p.sum())
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum())
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()),
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'gross_return_units': gross,
        'gross_roi_pct': float(100.0 * gross / n),
        'avg_odds': float(x.odds.mean()),
        'median_odds': float(x.odds.median()),
    }


def bootstrap_race_ci(x: pd.DataFrame):
    if x.empty:
        return {'bootstrap_reps': 0, 'calibration_ratio_ci95': [None, None], 'gross_roi_pct_ci95': [None, None]}
    rg = x.groupby('race_id', sort=False).agg(
        tickets=('actual_hit', 'size'),
        hits=('actual_hit', 'sum'),
        exp=('market_p', 'sum'),
        gross=('return_units', 'sum'),
    ).reset_index(drop=True)
    n = len(rg)
    if n < 2:
        return {'bootstrap_reps': 0, 'calibration_ratio_ci95': [None, None], 'gross_roi_pct_ci95': [None, None]}
    arr = rg[['tickets', 'hits', 'exp', 'gross']].to_numpy(float)
    rng = np.random.default_rng(SEED + n)
    ratios = np.empty(BOOTSTRAPS, dtype=float)
    rois = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        s = arr[idx].sum(axis=0)
        ratios[i] = s[1] / s[2] if s[2] > 0 else np.nan
        rois[i] = 100.0 * s[3] / s[0] if s[0] > 0 else np.nan
    return {
        'bootstrap_reps': BOOTSTRAPS,
        'calibration_ratio_ci95': [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
        'gross_roi_pct_ci95': [float(np.nanpercentile(rois, 2.5)), float(np.nanpercentile(rois, 97.5))],
    }


def view(x: pd.DataFrame):
    out = agg(x)
    out['race_bootstrap'] = bootstrap_race_ci(x)
    return out


def main():
    rows = []
    skipped = Counter()
    usable_by_month = {}

    for month in h1.MONTHS:
        loaded = h1.load_month(month)
        if loaded is None:
            skipped['month_missing'] += 1
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
            pre = bby.get(rid); og = oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing'] += 1
                continue
            lines = base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved'] += 1
                continue
            frames = sorted({int(x) for g in lines for x in g})
            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1
                continue

            tri = base.odds_map(og)
            expected = len(frames) * (len(frames) - 1) * (len(frames) - 2)
            if len(tri) != expected:
                skipped['odds_board_incomplete'] += 1
                continue
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                skipped['zero_market_mass'] += 1
                continue

            line_of, pos_of, size_of = {}, {}, {}
            for li, g in enumerate(lines, 1):
                gg = [int(x) for x in g]
                for pos, fn in enumerate(gg, 1):
                    line_of[fn] = li
                    pos_of[fn] = pos
                    size_of[fn] = len(gg)

            score, style = {}, {}
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

            # Added canonical race-level condition: the three strongest riders in
            # the race must be distributed across three different true lines.
            score_order = sorted(frames, key=lambda x: (-score[x], x))
            top3_score_riders = score_order[:3]
            if len(top3_score_riders) < 3 or len({line_of[x] for x in top3_score_riders}) != 3:
                skipped['top3_scores_not_spread_across_3_lines'] += 1
                continue

            all_groups = list(itertools.combinations(frames, 3))
            score_sums = {g: float(sum(score[x] for x in g)) for g in all_groups}
            ordered = sorted(all_groups, key=lambda g: (-score_sums[g], g))
            rank_of = {g: i + 1 for i, g in enumerate(ordered)}
            ng = len(ordered)
            race_mean = float(np.mean([score[x] for x in frames]))

            for trio_u in all_groups:
                if len({line_of.get(x) for x in trio_u}) != 3:
                    continue
                if any(size_of.get(x, 0) <= 1 for x in trio_u):
                    continue
                if any(not (style.get(x) == '両' or pos_of.get(x) == 2) for x in trio_u):
                    continue
                pct = float(rank_of[trio_u] / ng)
                trio_mean_delta = float(np.mean([score[x] - race_mean for x in trio_u]))
                role_set = tuple(sorted('RYO' if style.get(x) == '両' else 'BANTE' for x in trio_u))

                for perm in itertools.permutations(trio_u):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    hit = int(tuple(perm) == actual)
                    rows.append({
                        'month': month,
                        'period': period_of(month),
                        'race_id': rid,
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float((1.0 / od) / z),
                        'actual_hit': hit,
                        'return_units': float(od) if hit else 0.0,
                        'group_score_percentile': pct,
                        'trio_mean_delta': trio_mean_delta,
                        'unordered_role_set': '+'.join(role_set),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No qualifying tickets')
    df.to_csv(DETAIL, index=False)

    variants = {}
    for name, cut in CUTS.items():
        x = df[df.group_score_percentile <= cut].copy()
        variants[name] = {
            'all_odds': view(x),
            'periods': {p: view(g) for p, g in x.groupby('period', sort=True)},
            'odds_bins_descriptive': {
                bn: agg(x[(x.odds >= lo) & (x.odds < hi)]) for bn, lo, hi in ODDS_BINS
            },
            'role_sets': {str(k): agg(g) for k, g in x.groupby('unordered_role_set', sort=True)},
            'trio_mean_delta_distribution': {
                'mean': float(x.trio_mean_delta.mean()) if len(x) else None,
                'median': float(x.trio_mean_delta.median()) if len(x) else None,
                'q25': float(x.trio_mean_delta.quantile(0.25)) if len(x) else None,
                'q75': float(x.trio_mean_delta.quantile(0.75)) if len(x) else None,
            },
        }

    payload = {
        'status': 'exploratory_canonical_three_line_ryo_bante_strong_with_top_score_dispersion',
        'hypothesis': 'Choose three riders from three distinct multi-rider lines; each is running_style=両 or line_pos=2; require the race three highest race_score riders to be on three distinct lines; then test whether stronger candidate trio score-sum improves value.',
        'candidate_generation': {
            'unordered_discovery_then_all_6_exact_orders': True,
            'exactly_three_distinct_true_lines': True,
            'selected_lines_must_be_multi_rider': True,
            "selected_role": "running_style='両' OR line_pos=2",
            'strong_riders_spread_condition': 'top 3 riders by race_score in the race occupy 3 distinct true_line IDs',
            'strength_ladder': CUTS,
            'odds_used_for_candidate_selection': False,
            'not_added': ['bante>head condition', 'head-bante score-gap condition'],
        },
        'primary_metrics': ['actual_over_normalized_market', 'gross_roi_pct'],
        'stability_checks': ['2025_H1', '2025_H2', '2026_H1', 'race-level bootstrap 95% CI'],
        'warning': 'Same discovery data; thresholds and role idea have been explored previously. Period splits are stability diagnostics, not OOS. Same-race tickets are correlated, so race-level bootstrap is reported.',
        'coverage': {
            'usable_races_by_month': usable_by_month,
            'skipped': dict(skipped),
        },
        'variants': variants,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
