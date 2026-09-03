#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Directly bet the race-score top 3 when they are spread across 3 true lines.

User hypothesis:
- identify the three highest race_score riders in the race
- require those three riders to belong to three distinct true_line IDs
- use those exact three riders as the candidate trio
- allow every role/style, including 逃; no 両/番手 restriction

Evaluate the six exact 3-ren-tan permutations with equal stakes and cumulative
lowest-odds selections LOWEST_1..LOWEST_6. This is exploratory on the existing
2025-01..2026-06 backfilled sample, not fresh OOS validation.
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
OUT = CTX / 'top3_score_spread_direct_trio_summary.json'
DETAIL = CTX / 'top3_score_spread_direct_trio_tickets.csv'
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
    return f'{y}_H1' if m <= 6 else f'{y}_H2'


def role_label(fn, style, pos_of, size_of):
    st = style.get(fn, 'UNKNOWN')
    pos = pos_of.get(fn)
    size = size_of.get(fn, 0)
    if size == 1:
        return 'SOLO'
    if pos == 1:
        if st == '逃':
            return 'ESCAPE_HEAD'
        if st == '両':
            return 'RYO_HEAD'
        return 'OTHER_HEAD'
    if pos == 2:
        return 'BANTE'
    if pos and pos >= 3:
        return 'THIRD_PLUS'
    return 'UNKNOWN'


def agg(x: pd.DataFrame):
    n = int(len(x))
    if not n:
        return {'tickets': 0, 'races': 0, 'actual_hits': 0,
                'normalized_market_expected_hits': 0.0,
                'actual_over_normalized_market': None,
                'gross_return_units': 0.0, 'gross_roi_pct': None,
                'avg_odds': None, 'median_odds': None}
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
    y = x.copy()
    y['gross'] = np.where(y.actual_hit == 1, y.odds, 0.0)
    rg = y.groupby('race_id', sort=False).agg(
        tickets=('actual_hit', 'size'), hits=('actual_hit', 'sum'),
        exp=('market_p', 'sum'), gross=('gross', 'sum'),
    ).reset_index(drop=True)
    n = len(rg)
    if n < 2:
        return {'bootstrap_reps': 0, 'calibration_ratio_ci95': [None, None], 'gross_roi_pct_ci95': [None, None]}
    arr = rg[['tickets', 'hits', 'exp', 'gross']].to_numpy(float)
    rng = np.random.default_rng(SEED + n + len(x))
    ratios = np.empty(BOOTSTRAPS); rois = np.empty(BOOTSTRAPS)
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


def view(x):
    out = agg(x)
    out['race_bootstrap'] = bootstrap_race_ci(x)
    return out


def main():
    rows = []
    race_rows = []
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
                skipped['base_or_odds_missing'] += 1; continue
            lines = base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved'] += 1; continue
            frames = sorted({int(x) for g in lines for x in g})
            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1; continue
            tri = base.odds_map(og)
            expected = len(frames) * (len(frames)-1) * (len(frames)-2)
            if len(tri) != expected:
                skipped['odds_board_incomplete'] += 1; continue
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                skipped['zero_market_mass'] += 1; continue

            line_of, pos_of, size_of = {}, {}, {}
            for li, g in enumerate(lines, 1):
                gg = [int(x) for x in g]
                for pos, fn in enumerate(gg, 1):
                    line_of[fn] = li; pos_of[fn] = pos; size_of[fn] = len(gg)

            score, style = {}, {}
            for r in pre.itertuples(index=False):
                try: fn = int(float(r.banum))
                except Exception: continue
                try:
                    sc = float(r.race_score)
                    if np.isfinite(sc): score[fn] = sc
                except Exception: pass
                st = str(getattr(r, 'running_style', '')).strip()
                style[fn] = st if st and st.lower() != 'nan' else 'UNKNOWN'
            if set(frames) - set(score):
                skipped['score_missing'] += 1; continue

            top3 = tuple(sorted(frames, key=lambda x: (-score[x], x))[:3])
            if len(top3) != 3 or len({line_of[x] for x in top3}) != 3:
                skipped['top3_scores_not_spread_across_3_lines'] += 1; continue

            actual_set_hit = int(set(actual) == set(top3))
            roles = tuple(sorted(role_label(x, style, pos_of, size_of) for x in top3))
            role_set = '+'.join(roles)
            race_rows.append({
                'month': month, 'period': period_of(month), 'race_id': rid,
                'top3_score_trio': '-'.join(map(str, sorted(top3))),
                'unordered_top3_finish_hit': actual_set_hit,
                'role_set': role_set,
            })

            for perm in itertools.permutations(top3):
                od = tri.get(tuple(perm))
                if od is None or od <= 0: continue
                hit = int(tuple(perm) == actual)
                rows.append({
                    'month': month, 'period': period_of(month), 'race_id': rid,
                    'trio_key': '-'.join(map(str, sorted(top3))),
                    'ticket': '-'.join(map(str, perm)), 'odds': float(od),
                    'market_p': float((1.0/od)/z), 'actual_hit': hit,
                    'return_units': float(od) if hit else 0.0,
                    'role_set': role_set,
                })

    df = pd.DataFrame(rows)
    rdf = pd.DataFrame(race_rows)
    if df.empty: raise SystemExit('No qualifying tickets')
    df = df.sort_values(['race_id', 'trio_key', 'odds', 'ticket'], kind='mergesort')
    df['odds_rank_ascending'] = df.groupby(['race_id', 'trio_key'], sort=False).cumcount() + 1
    sizes = df.groupby(['race_id', 'trio_key']).size()
    bad = sizes[sizes != 6]
    if len(bad): raise SystemExit(f'Expected 6 permutations per trio; bad groups={len(bad)}')
    df.to_csv(DETAIL, index=False)

    all6 = view(df)
    cumulative = {}
    for k in range(1, 7):
        x = df[df.odds_rank_ascending <= k].copy()
        cumulative[f'LOWEST_{k}'] = {
            'all': view(x),
            'periods': {str(p): view(g) for p, g in x.groupby('period', sort=True)},
        }

    group_total = int(len(rdf))
    group_hits = int(rdf.unordered_top3_finish_hit.sum())
    role_sets = {}
    for r, g in rdf.groupby('role_set', sort=True):
        role_sets[str(r)] = {
            'races': int(len(g)), 'unordered_top3_finish_hits': int(g.unordered_top3_finish_hit.sum()),
            'unordered_top3_finish_hit_rate_pct': float(100.0*g.unordered_top3_finish_hit.mean()),
        }

    payload = {
        'status': 'exploratory_direct_top3_score_spread_trio',
        'candidate_rule': {
            'selected_riders': 'the three highest race_score riders in the race',
            'require_three_distinct_true_lines': True,
            'role_or_running_style_filter': None,
            'escape_allowed': True,
        },
        'unordered_trio_result': {
            'qualifying_races': group_total,
            'top3_score_trio_finished_as_top3_unordered_hits': group_hits,
            'unordered_hit_rate_pct': float(100.0*group_hits/group_total) if group_total else None,
        },
        'all_six_exact_orders_equal_stake': all6,
        'lowest_odds_cumulative_equal_stake': cumulative,
        'unordered_role_sets': role_sets,
        'coverage': {'usable_races_by_month': usable_by_month, 'skipped': dict(skipped)},
        'warning': 'Same-data exploratory diagnostic. LOWEST_k is a pre-race odds rule, but the hypothesis was formed after prior exploration. Period splits are stability diagnostics, not fresh OOS.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
