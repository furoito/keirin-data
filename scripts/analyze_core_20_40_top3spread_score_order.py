#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold unordered-group selection fixed and change only exact-order selection.

Group selection (fixed):
- exactly 3 distinct true lines; selected lines are multi-rider lines
- every selected rider is running_style='両' OR line_pos=2
- 0.20 < group_score_percentile <= 0.40
- race top3 riders by race_score are spread across 3 distinct true lines

Treatment:
- bet one exact trifecta in descending race_score order within the selected trio.

Control:
- bet the cheapest posted exact order within the same selected trio.

This isolates whether race_score is useful for ORDERING after the unordered trio has
already been selected. Exploratory same-data test, not OOS validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_core_top40_rank1_robustness as rb
import analyze_formation_box_chain_expanded as exp
import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'formation_box_chain_expanded_groups.csv'
OUT = CTX / 'core_20_40_top3spread_score_order_summary.json'
DETAIL = CTX / 'core_20_40_top3spread_score_order_tickets.csv'
SEED = 20260908


def add_score_order_ticket(cand: pd.DataFrame):
    parts = []
    skipped = {}
    for month, g in cand.groupby('month', sort=True):
        loaded = h1.load_month(str(month))
        if loaded is None:
            skipped[str(month)] = {'month_missing': int(len(g))}
            continue
        b, _, odds = loaded
        bby = {str(k): x for k, x in b.groupby('race_id', sort=False)}
        oby = {str(k): x for k, x in odds.groupby('race_id', sort=False)}
        rows = []
        miss = 0
        for r in g.itertuples(index=False):
            rid = str(r.race_id)
            pre = bby.get(rid); og = oby.get(rid)
            if pre is None or og is None:
                miss += 1; continue
            tri = base.odds_map(og)
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if z <= 0:
                miss += 1; continue
            score = {}
            for q in pre.itertuples(index=False):
                try:
                    fn = int(float(q.banum)); sc = float(q.race_score)
                except Exception:
                    continue
                if np.isfinite(sc): score[fn] = sc
            actual = exp.actual_ordered_top3(pre)
            if actual is None:
                miss += 1; continue
            try:
                trio = tuple(int(x) for x in str(r.trio_key).split('-'))
            except Exception:
                miss += 1; continue
            if any(x not in score for x in trio):
                miss += 1; continue
            chosen = tuple(sorted(trio, key=lambda x: (-score[x], x)))
            od = tri.get(chosen)
            if od is None or od <= 0:
                miss += 1; continue
            exact_market_p = float((1.0 / od) / z)
            group_p = float(r.group_market_p)
            d = r._asdict()
            d['rank1_posted_odds'] = float(od)
            d['rank1_order'] = '-'.join(map(str, chosen))
            d['rank1_exact_market_p'] = exact_market_p
            d['rank1_conditional_share'] = float(exact_market_p / group_p) if group_p > 0 else np.nan
            d['rank1_exact_hit'] = int(tuple(actual) == chosen)
            d['race_uid'] = f'{month}|{rid}'
            d['half_year'] = rb.half_year_label(str(month))
            d['odds_band'] = rb.odds_band(float(od))
            rows.append(d)
        if miss:
            skipped[str(month)] = {'unusable_rows': int(miss)}
        if rows:
            parts.append(pd.DataFrame(rows))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), skipped


def slices(x: pd.DataFrame, seed: int):
    out = {'overall': rb.summarize_slice(x, seed), 'by_half_year': {}, 'by_posted_odds_band': {}}
    for i, (k, g) in enumerate(x.groupby('half_year', sort=True)):
        out['by_half_year'][str(k)] = rb.summarize_slice(g, seed + 100 + i)
    for i, k in enumerate(['LT30','30_TO_50','50_TO_100','100_TO_200','GE200']):
        out['by_posted_odds_band'][k] = rb.summarize_slice(x[x.odds_band == k], seed + 200 + i)
    return out


def main():
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str, 'trio_key': str})
    cand = df[
        (df.role_ok == 1) &
        (df.top3_spread == 1) &
        (df.group_score_percentile > 0.20) &
        (df.group_score_percentile <= 0.40)
    ].copy()

    score_order, skipped = add_score_order_ticket(cand)
    if score_order.empty:
        raise SystemExit('no score-order tickets')
    score_order.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    cheapest, cheapest_skipped = rb.add_posted_rank1_odds(cand)
    if not cheapest.empty:
        cheapest['half_year'] = cheapest.month.map(rb.half_year_label)
        cheapest['odds_band'] = cheapest.rank1_posted_odds.map(rb.odds_band)

    payload = {
        'status': 'exploratory_fixed_group_score_order_probe',
        'group_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            '0.20 < group_score_percentile <= 0.40',
            'race top3 riders by race_score are spread across 3 distinct true lines',
        ],
        'treatment': 'one exact ticket ordered by race_score descending within each selected trio',
        'control': 'one exact ticket with the lowest posted odds within the same selected trio',
        'score_order_skipped': skipped,
        'cheapest_order_skipped': cheapest_skipped,
        'score_order': slices(score_order, SEED),
        'cheapest_order_same_groups': slices(cheapest, SEED + 1000) if not cheapest.empty else None,
        'warning': 'Same-data exploratory order-selection comparison; group selection was already developed on this history.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
