#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose score order, race-score ranks and score gaps inside the current
30-40 percentile ticket candidate region.

This is SAME-DATA DISCOVERY ONLY.

Fixed pre-race candidate surface:
- NO_THIRD retained
- group-score percentile >30% and <=40%
- ordered ticket 1st rider is running_style=両 and line_pos=1
- winner line has >=2 riders
- winner's own bante is excluded from the ticket trio
- at least one other-line bante in the race has race_score > own-bante race_score

We do NOT choose score/rank/gap thresholds here. We only localize the signal.
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
OUT = CTX / 'h1_ticket_score_structure_30_40_summary.json'
HITS_OUT = CTX / 'h1_ticket_score_structure_30_40_hits.csv'

ODDS_BINS = [
    ('<50', 0.0, 50.0),
    ('50-100', 50.0, 100.0),
    ('100-200', 100.0, 200.0),
    ('200-500', 200.0, 500.0),
    ('500-1000', 500.0, 1000.0),
    ('1000+', 1000.0, float('inf')),
]

GAP_BINS = [
    ('<-4', -float('inf'), -4.0),
    ('-4~-2', -4.0, -2.0),
    ('-2~0', -2.0, 0.0),
    ('0~2', 0.0, 2.0),
    ('2~4', 2.0, 4.0),
    ('4+', 4.0, float('inf')),
]

SPREAD_BINS = [
    ('<2', 0.0, 2.0),
    ('2-4', 2.0, 4.0),
    ('4-6', 4.0, 6.0),
    ('6+', 6.0, float('inf')),
]

PCT_VIEWS = [
    ('30-40', 0.30, 0.40),
    ('30-35', 0.30, 0.35),
    ('35-40', 0.35, 0.40),
]


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


def race_score_ranks(score: dict[int, float]) -> dict[int, int]:
    # Competition rank: 1 + number of riders with a strictly higher score.
    return {
        fn: 1 + sum(1 for other in score.values() if other > sc)
        for fn, sc in score.items()
    }


def score_order_pattern(s1: float, s2: float, s3: float) -> str:
    vals = [('1st', s1), ('2nd', s2), ('3rd', s3)]
    vals.sort(key=lambda x: (-x[1], x[0]))
    groups = []
    for label, sc in vals:
        if not groups or abs(groups[-1][0] - sc) > 1e-9:
            groups.append([sc, [label]])
        else:
            groups[-1][1].append(label)
    return '>'.join('='.join(labels) for _, labels in groups)


def rank_bucket(rank: int) -> str:
    if rank <= 2:
        return 'R1-2'
    if rank <= 4:
        return 'R3-4'
    return 'R5+'


def gap_bin(v: float) -> str:
    for name, lo, hi in GAP_BINS:
        if lo <= v < hi:
            return name
    raise AssertionError(v)


def spread_bin(v: float) -> str:
    for name, lo, hi in SPREAD_BINS:
        if lo <= v < hi:
            return name
    raise AssertionError(v)


def agg(x: pd.DataFrame) -> dict:
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


def odds_slices(df: pd.DataFrame) -> dict:
    out = {}
    for name, lo, hi in ODDS_BINS:
        out[name] = agg(df[(df.odds >= lo) & (df.odds < hi)])
    return out


def category_summary(df: pd.DataFrame, col: str, order: list[str] | None = None) -> dict:
    vals = list(order) if order is not None else sorted(df[col].dropna().astype(str).unique().tolist())
    out = {}
    for val in vals:
        z = df[df[col].astype(str) == str(val)]
        if z.empty:
            continue
        out[str(val)] = {
            'all_ticket_odds': agg(z),
            'ticket_odds_bins': odds_slices(z),
        }
    return out


def exact_rank_tuple_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, z in df.groupby('race_rank_tuple', sort=False):
        a = agg(z)
        rows.append({'race_rank_tuple': str(key), **a})
    rows.sort(key=lambda r: (-r['tickets'], r['race_rank_tuple']))
    return rows


def numeric_position_rank_summary(df: pd.DataFrame, col: str) -> dict:
    out = {}
    ranks = sorted(int(x) for x in df[col].dropna().unique())
    for rank in ranks:
        out[str(rank)] = {
            'all_ticket_odds': agg(df[df[col] == rank]),
            'ticket_odds_bins': odds_slices(df[df[col] == rank]),
        }
    return out


def summarize_view(df: pd.DataFrame) -> dict:
    order_patterns = [
        '1st>2nd>3rd', '1st>3rd>2nd',
        '2nd>1st>3rd', '2nd>3rd>1st',
        '3rd>1st>2nd', '3rd>2nd>1st',
    ]
    seen_ties = sorted(x for x in df.score_order_pattern.unique() if '=' in str(x))
    order_patterns += seen_ties

    gap_order = [x[0] for x in GAP_BINS]
    spread_order = [x[0] for x in SPREAD_BINS]

    return {
        'overall': {
            'all_ticket_odds': agg(df),
            'ticket_odds_bins': odds_slices(df),
        },
        'score_order_pattern': category_summary(df, 'score_order_pattern', order_patterns),
        'race_rank_by_ticket_position': {
            'first_place_rider': numeric_position_rank_summary(df, 'rank1'),
            'second_place_rider': numeric_position_rank_summary(df, 'rank2'),
            'third_place_rider': numeric_position_rank_summary(df, 'rank3'),
        },
        'race_rank_bucket_pattern': category_summary(df, 'race_rank_bucket_pattern'),
        'exact_race_rank_tuples_by_ticket_count': exact_rank_tuple_summary(df),
        'score_gap_first_minus_second': category_summary(df, 'gap12_bin', gap_order),
        'score_gap_second_minus_third': category_summary(df, 'gap23_bin', gap_order),
        'score_gap_first_minus_third': category_summary(df, 'gap13_bin', gap_order),
        'selected_three_score_spread_max_minus_min': category_summary(df, 'spread_bin', spread_order),
        'descriptive_means': {
            'all_tickets': {
                'score1': float(df.score1.mean()),
                'score2': float(df.score2.mean()),
                'score3': float(df.score3.mean()),
                'rank1': float(df.rank1.mean()),
                'rank2': float(df.rank2.mean()),
                'rank3': float(df.rank3.mean()),
                'gap12': float(df.gap12.mean()),
                'gap23': float(df.gap23.mean()),
                'gap13': float(df.gap13.mean()),
                'spread': float(df.score_spread.mean()),
            },
            'actual_hits_only': ({
                'score1': float(df.loc[df.actual_hit == 1, 'score1'].mean()),
                'score2': float(df.loc[df.actual_hit == 1, 'score2'].mean()),
                'score3': float(df.loc[df.actual_hit == 1, 'score3'].mean()),
                'rank1': float(df.loc[df.actual_hit == 1, 'rank1'].mean()),
                'rank2': float(df.loc[df.actual_hit == 1, 'rank2'].mean()),
                'rank3': float(df.loc[df.actual_hit == 1, 'rank3'].mean()),
                'gap12': float(df.loc[df.actual_hit == 1, 'gap12'].mean()),
                'gap23': float(df.loc[df.actual_hit == 1, 'gap23'].mean()),
                'gap13': float(df.loc[df.actual_hit == 1, 'gap13'].mean()),
                'spread': float(df.loc[df.actual_hit == 1, 'score_spread'].mean()),
            } if int(df.actual_hit.sum()) else None),
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

            groups, why = h1.race_rows(month, rid, pre, cr, og)
            if groups is None:
                skipped[why] += 1
                continue
            actual = actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing'] += 1
                continue

            tri = base.odds_map(og)
            zmass = sum(1.0 / od for od in tri.values() if od > 0)
            if zmass <= 0:
                skipped['zero_mass'] += 1
                continue

            lines = base.parse_true_line(cr.get('true_line'))
            line_of = {}
            pos_of = {}
            members = {}
            for li, g in enumerate(lines, 1):
                members[li] = [int(x) for x in g]
                for pos, fn in enumerate(g, 1):
                    line_of[int(fn)] = li
                    pos_of[int(fn)] = pos

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

            ranks = race_score_ranks(score)

            for q in groups:
                pct = float(q['group_score_percentile'])
                if not (pct > 0.30 and pct <= 0.40):
                    continue
                trio = tuple(int(x) for x in q['trio'].split('-'))
                if any(pos_of.get(fn, 99) >= 3 for fn in trio):
                    continue

                for perm in itertools.permutations(trio):
                    a, b2, c3 = perm
                    if style.get(a) != '両' or pos_of.get(a) != 1:
                        continue
                    li = line_of.get(a)
                    lm = members.get(li, [])
                    if len(lm) < 2:
                        continue
                    own = int(lm[1])
                    if own in trio or own not in score:
                        continue

                    other_bantes = []
                    for oli, ogroup in members.items():
                        if oli == li or len(ogroup) < 2:
                            continue
                        fn = int(ogroup[1])
                        if fn in score:
                            other_bantes.append(fn)
                    race_higher = sum(1 for fn in other_bantes if score[fn] > score[own])
                    if race_higher < 1:
                        continue

                    if any(fn not in score or fn not in ranks for fn in perm):
                        skipped['ticket_score_missing'] += 1
                        continue
                    od = tri.get(tuple(perm))
                    if od is None or od <= 0:
                        continue

                    s1, s2, s3 = (float(score[a]), float(score[b2]), float(score[c3]))
                    r1, r2, r3 = (int(ranks[a]), int(ranks[b2]), int(ranks[c3]))
                    g12 = s1 - s2
                    g23 = s2 - s3
                    g13 = s1 - s3
                    spread = max(s1, s2, s3) - min(s1, s2, s3)
                    p = (1.0 / float(od)) / zmass

                    rows.append({
                        'month': month,
                        'race_id': rid,
                        'ticket': '-'.join(map(str, perm)),
                        'odds': float(od),
                        'market_p': float(p),
                        'actual_hit': int(tuple(perm) == actual),
                        'group_score_percentile': pct,
                        'own_bante': own,
                        'race_higher_other_bantes': int(race_higher),
                        'score1': s1,
                        'score2': s2,
                        'score3': s3,
                        'rank1': r1,
                        'rank2': r2,
                        'rank3': r3,
                        'score_order_pattern': score_order_pattern(s1, s2, s3),
                        'race_rank_tuple': f'{r1}-{r2}-{r3}',
                        'race_rank_bucket_pattern': f'{rank_bucket(r1)}-{rank_bucket(r2)}-{rank_bucket(r3)}',
                        'gap12': g12,
                        'gap23': g23,
                        'gap13': g13,
                        'gap12_bin': gap_bin(g12),
                        'gap23_bin': gap_bin(g23),
                        'gap13_bin': gap_bin(g13),
                        'score_spread': spread,
                        'spread_bin': spread_bin(spread),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No qualifying tickets')

    views = {}
    for name, lo, hi in PCT_VIEWS:
        z = df[(df.group_score_percentile > lo) & (df.group_score_percentile <= hi)].copy()
        views[name] = summarize_view(z)

    hit_cols = [
        'month','race_id','ticket','odds','market_p','group_score_percentile',
        'own_bante','race_higher_other_bantes',
        'score1','score2','score3','rank1','rank2','rank3',
        'score_order_pattern','race_rank_tuple','race_rank_bucket_pattern',
        'gap12','gap23','gap13','gap12_bin','gap23_bin','gap13_bin',
        'score_spread','spread_bin',
    ]
    df.loc[df.actual_hit == 1, hit_cols].sort_values(['month','race_id']).to_csv(HITS_OUT, index=False)

    payload = {
        'status': 'exploratory_ticket_score_order_rank_gap_diagnostic',
        'question': 'Within the current 30-40 percentile structural candidate, what score order, race-score rank pattern and score-gap shapes characterize exact ordered trifecta tickets?',
        'candidate_generation': 'NO_THIRD; group-score percentile >30% <=40%; ticket first rider=両 multi-rider line head; own bante excluded; >=1 other-line bante in race has higher score than own bante',
        'warning': 'Same discovery data after observing the 30-40 region. Do not promote any score/rank/gap cut to a betting rule without locked OOS validation.',
        'score_order_definition': 'Positions 1st/2nd/3rd are the ordered trifecta ticket positions, sorted by race_score descending; equality is retained as =.',
        'race_rank_definition': 'Competition rank within all riders in the race by pre-race race_score: 1 + count of riders with strictly higher score.',
        'gap_definition': {
            'gap12': 'score(1st ticket rider) - score(2nd ticket rider)',
            'gap23': 'score(2nd ticket rider) - score(3rd ticket rider)',
            'gap13': 'score(1st ticket rider) - score(3rd ticket rider)',
            'spread': 'max(selected three scores) - min(selected three scores)',
        },
        'ticket_odds_nonoverlap_bins': [x[0] for x in ODDS_BINS],
        'gap_bins': [x[0] for x in GAP_BINS],
        'spread_bins': [x[0] for x in SPREAD_BINS],
        'percentile_views': [x[0] for x in PCT_VIEWS],
        'usable_races_by_month': usable_by_month,
        'skipped': dict(skipped),
        'views': views,
        'hit_detail_csv': str(HITS_OUT.relative_to(ROOT)),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
