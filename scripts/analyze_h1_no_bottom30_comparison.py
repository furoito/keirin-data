#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare score-sum quality vs an individual lower-tail exclusion.

Core question:
Is the useful score condition really "high group/pair score sum", or is it mainly
"do not include a weak rider"?

Four predeclared views for BOTH markets:
  BASE                  structural filters only
  TOP40_SUM             group/pair score-sum percentile <= 40%
  NO_BOTTOM30           every selected rider is outside the race-score bottom 30%
  NO_BOTTOM30_TOP40     both conditions

Individual lower-tail definition:
  order riders by race_score descending within each race;
  individual_midrank_percentile = (rank - 0.5) / n_riders;
  bottom 30% means percentile > 0.70.
This gives ranks 6-7 as bottom 30% in a 7-rider race and ranks 7-9 in a 9-rider race.

2-sha-tan uses the already-fetched deterministic 10% probe ticket file; no network.
3-ren-tan uses the full existing discovery data and the current structural hypothesis:
  exactly three distinct lines, all selected riders are '両' or line_pos=2,
  and all selected riders belong to multi-rider lines.

Same discovery period. Exploratory diagnostic, not OOS proof.
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
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
OUT = CTX / 'h1_no_bottom30_comparison_summary.json'
TWO_TICKETS = CTX / 'h1_2shatan_ryo_other_bante_probe_tickets.csv'

THREE_ODDS_BINS = [
    ('<50', 0.0, 50.0), ('50-100', 50.0, 100.0), ('100-200', 100.0, 200.0),
    ('200-500', 200.0, 500.0), ('500-1000', 500.0, 1000.0), ('1000+', 1000.0, float('inf')),
]
TWO_ODDS_BINS = [
    ('<5', 0.0, 5.0), ('5-10', 5.0, 10.0), ('10-20', 10.0, 20.0),
    ('20-50', 20.0, 50.0), ('50-100', 50.0, 100.0), ('100+', 100.0, float('inf')),
]


def actual_ordered_top3(pre):
    vals = []
    for r in pre.itertuples(index=False):
        try:
            pos = int(str(r.rank).strip()); fn = int(float(r.banum))
        except Exception:
            continue
        if 1 <= pos <= 3:
            vals.append((pos, fn))
    vals.sort()
    if [p for p, _ in vals] != [1, 2, 3]:
        return None
    return tuple(fn for _, fn in vals)


def agg(x):
    n = int(len(x)); stake = float(n)
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum()) if n else 0.0
    hits = int(x.actual_hit.sum()) if n else 0
    exp = float(x.market_p.sum()) if n else 0.0
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()) if n else 0,
        'stake_units': stake,
        'gross_return_units': gross,
        'gross_roi_pct': float(100 * gross / stake) if stake else None,
        'net_roi_pct': float(100 * (gross - stake) / stake) if stake else None,
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'avg_ticket_odds': float(x.odds.mean()) if n else None,
        'median_ticket_odds': float(x.odds.median()) if n else None,
    }


def odds_slices(df, bins):
    return {name: agg(df[(df.odds >= lo) & (df.odds < hi)]) for name, lo, hi in bins}


def full_view(df, bins, mincuts):
    return {
        'all_odds': agg(df),
        'ticket_odds_bins': odds_slices(df, bins),
        'min_ticket_odds': {str(c): agg(df[df.odds >= c]) for c in mincuts},
    }


def individual_midrank_percentiles(score):
    ordered = sorted(score, key=lambda fn: (-score[fn], fn))
    n = len(ordered)
    return {fn: (i + 1 - 0.5) / n for i, fn in enumerate(ordered)}


def build_base_lookup(months):
    by_race = {}
    for month in months:
        bp = DATA / f'{month}_keirin.csv'
        if not bp.exists():
            continue
        b = pd.read_csv(bp, encoding='utf-8-sig', dtype={'race_id': str})
        b['race_id'] = b.race_id.astype(str)
        for rid, g in b.groupby('race_id', sort=False):
            score = {}
            for r in g.itertuples(index=False):
                try:
                    fn = int(float(r.banum)); sc = float(r.race_score)
                except Exception:
                    continue
                if np.isfinite(sc):
                    score[fn] = sc
            if score:
                by_race[str(rid)] = individual_midrank_percentiles(score)
    return by_race


def analyze_2shatan():
    if not TWO_TICKETS.exists():
        return {'status': 'missing_probe_ticket_file'}
    df = pd.read_csv(TWO_TICKETS, encoding='utf-8-sig', dtype={'race_id': str})
    if df.empty:
        return {'status': 'empty_probe_ticket_file'}
    months = sorted(set(df.month.astype(str)))
    pct_lookup = build_base_lookup(months)

    b1pct = []
    b2pct = []
    for r in df.itertuples(index=False):
        p = pct_lookup.get(str(r.race_id), {})
        b1pct.append(p.get(int(r.b1), np.nan))
        b2pct.append(p.get(int(r.b2), np.nan))
    df['b1_individual_score_percentile'] = b1pct
    df['b2_individual_score_percentile'] = b2pct
    df = df[np.isfinite(df.b1_individual_score_percentile) & np.isfinite(df.b2_individual_score_percentile)].copy()
    df['no_bottom30'] = (df.b1_individual_score_percentile <= 0.70) & (df.b2_individual_score_percentile <= 0.70)

    views = {
        'BASE': df,
        'TOP40_SUM': df[df.pair_score_percentile <= 0.40],
        'NO_BOTTOM30': df[df.no_bottom30],
        'NO_BOTTOM30_TOP40': df[df.no_bottom30 & (df.pair_score_percentile <= 0.40)],
    }
    return {
        'status': 'ok',
        'source': 'existing deterministic 10% 2-sha-tan probe; no refetch',
        'views': {k: full_view(v, TWO_ODDS_BINS, [5, 10, 20, 50]) for k, v in views.items()},
    }


def analyze_3rentan():
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
            rid = str(cr['race_id']); pre = bby.get(rid); og = oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing'] += 1; continue
            lines = base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved'] += 1; continue
            frames = sorted({int(x) for g in lines for x in g})
            tri = base.odds_map(og)
            expected = len(frames) * (len(frames) - 1) * (len(frames) - 2)
            if len(tri) != expected:
                skipped['odds_board_incomplete'] += 1; continue
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                skipped['zero_mass'] += 1; continue
            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1; continue

            line_of = {}; pos_of = {}; line_size_of = {}
            for li, g in enumerate(lines, 1):
                g2 = [int(x) for x in g]
                for pos, fn in enumerate(g2, 1):
                    line_of[fn] = li; pos_of[fn] = pos; line_size_of[fn] = len(g2)

            score = {}; style = {}
            for r in pre.itertuples(index=False):
                try:
                    fn = int(float(r.banum))
                except Exception:
                    continue
                try:
                    sc = float(r.race_score)
                    if np.isfinite(sc): score[fn] = sc
                except Exception:
                    pass
                s = str(getattr(r, 'running_style', '')).strip()
                if not s or s.lower() == 'nan': s = 'UNKNOWN'
                style[fn] = s
            if set(frames) - set(score):
                skipped['score_missing'] += 1; continue

            indiv_pct = individual_midrank_percentiles(score)
            all_groups = list(itertools.combinations(frames, 3))
            score_sums = {g: float(sum(score[x] for x in g)) for g in all_groups}
            ordered_groups = sorted(all_groups, key=lambda g: (-score_sums[g], g))
            group_rank = {g: i + 1 for i, g in enumerate(ordered_groups)}
            n_groups = len(ordered_groups)

            for trio_u in all_groups:
                if len({line_of.get(x) for x in trio_u}) != 3:
                    continue
                if any(line_size_of.get(x, 0) <= 1 for x in trio_u):
                    continue
                if any(not (style.get(x) == '両' or pos_of.get(x) == 2) for x in trio_u):
                    continue
                group_pct = group_rank[trio_u] / n_groups
                no_bottom30 = all(indiv_pct[x] <= 0.70 for x in trio_u)
                for perm in itertools.permutations(trio_u):
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue
                    p = (1.0 / float(od)) / z
                    rows.append({
                        'month': month, 'race_id': rid, 'ticket': '-'.join(map(str, perm)),
                        'odds': float(od), 'market_p': float(p), 'actual_hit': int(tuple(perm) == actual),
                        'group_score_percentile': float(group_pct), 'no_bottom30': bool(no_bottom30),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return {'status': 'no_qualifying_tickets', 'skipped': dict(skipped)}
    views = {
        'BASE': df,
        'TOP40_SUM': df[df.group_score_percentile <= 0.40],
        'NO_BOTTOM30': df[df.no_bottom30],
        'NO_BOTTOM30_TOP40': df[df.no_bottom30 & (df.group_score_percentile <= 0.40)],
    }
    return {
        'status': 'ok',
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'views': {k: full_view(v, THREE_ODDS_BINS, [50, 100, 200, 500]) for k, v in views.items()},
    }


def main():
    payload = {
        'status': 'exploratory_no_bottom30_vs_sum_top40_comparison',
        'question': 'Is score-sum quality useful because it avoids weak riders, rather than because the selected group total is high?',
        'individual_floor_definition': 'Within each race, race_score descending midrank percentile=(rank-0.5)/n; exclude selected rider if percentile>0.70.',
        'views': ['BASE', 'TOP40_SUM', 'NO_BOTTOM30', 'NO_BOTTOM30_TOP40'],
        'warning': 'Same discovery data; bottom30 threshold was proposed after prior score-sum exploration. Diagnostic, not OOS validation.',
        'two_shatan': analyze_2shatan(),
        'three_rentan': analyze_3rentan(),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

# trigger: compare lower-tail exclusion against score-sum filter
