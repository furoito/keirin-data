#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe individual rider race_score rank windows for the current structural candidate.

Fixed structural conditions:
- unordered trio spans exactly 3 distinct true lines
- each selected line has >=2 riders (inherited from expanded group pool)
- every selected rider is running_style='両' OR line_pos=2
- race top3 riders by race_score are spread across 3 distinct true lines

Primary candidate:
- every selected rider's individual race_score rank is in 2..5 inclusive

Adjacent controls:
- 1..4
- 3..6

No group-score-percentile filter is used. Bet view is the single cheapest posted
3rentan order within each qualifying unordered trio.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_core_top40_rank1_robustness as rb
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'formation_box_chain_expanded_groups.csv'
OUT = CTX / 'individual_score_rank_2_5_summary.json'
DETAIL = CTX / 'individual_score_rank_2_5_tickets.csv'
SEED = 20260907

WINDOWS = {
    'RANK_1_TO_4': (1, 4),
    'RANK_2_TO_5': (2, 5),
    'RANK_3_TO_6': (3, 6),
}


def rider_rank_map(month: str, race_ids: set[str]) -> dict[tuple[str, str], dict[int, int]]:
    loaded = h1.load_month(month)
    if loaded is None:
        return {}
    base, _, _ = loaded
    out = {}
    for rid, g in base.groupby('race_id', sort=False):
        srid = str(rid)
        if srid not in race_ids:
            continue
        rows = []
        for r in g.itertuples(index=False):
            try:
                fn = int(float(r.banum))
                sc = float(r.race_score)
            except Exception:
                continue
            if pd.notna(sc):
                rows.append((fn, sc))
        rows.sort(key=lambda x: (-x[1], x[0]))
        if rows:
            out[(month, srid)] = {fn: i + 1 for i, (fn, _) in enumerate(rows)}
    return out


def add_rider_ranks(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for month, g in df.groupby('month', sort=True):
        race_ids = set(g.race_id.astype(str))
        maps = rider_rank_map(str(month), race_ids)
        rows = []
        for r in g.itertuples(index=False):
            rid = str(r.race_id)
            rm = maps.get((str(month), rid))
            if not rm:
                continue
            try:
                trio = tuple(int(x) for x in str(r.trio_key).split('-'))
            except Exception:
                continue
            ranks = [rm.get(x) for x in trio]
            if any(x is None for x in ranks):
                continue
            d = r._asdict()
            ranks = sorted(int(x) for x in ranks)
            d['selected_rider_score_ranks'] = '-'.join(map(str, ranks))
            d['selected_min_score_rank'] = min(ranks)
            d['selected_max_score_rank'] = max(ranks)
            rows.append(d)
        if rows:
            parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_window(tickets: pd.DataFrame, lo: int, hi: int, seed: int) -> dict:
    x = tickets[(tickets.selected_min_score_rank >= lo) & (tickets.selected_max_score_rank <= hi)].copy()
    out = {
        'window': [lo, hi],
        'overall': rb.summarize_slice(x, seed),
        'by_half_year': {},
        'by_posted_odds_band': {},
        'rank_compositions': {},
    }
    for i, (label, g) in enumerate(x.groupby('half_year', sort=True)):
        out['by_half_year'][str(label)] = rb.summarize_slice(g, seed + 100 + i)
    for i, label in enumerate(['LT30','30_TO_50','50_TO_100','100_TO_200','GE200']):
        out['by_posted_odds_band'][label] = rb.summarize_slice(x[x.odds_band == label], seed + 200 + i)
    for comp, g in x.groupby('selected_rider_score_ranks', sort=True):
        out['rank_compositions'][str(comp)] = {
            'tickets': int(len(g)),
            'races': int(g.race_uid.nunique()),
            'wins': int(g.rank1_exact_hit.sum()),
            'group_hits': int(g.group_hit.sum()),
            'posted_odds_return_ratio': float((g.rank1_posted_odds * g.rank1_exact_hit).sum() / len(g)) if len(g) else None,
            'rank1_actual_over_normalized_market': float(g.rank1_exact_hit.sum() / g.rank1_exact_market_p.sum()) if g.rank1_exact_market_p.sum() > 0 else None,
            'group_actual_over_normalized_market': float(g.group_hit.sum() / g.group_market_p.sum()) if g.group_market_p.sum() > 0 else None,
        }
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f'missing input: {SRC}')
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str, 'trio_key': str})
    required = {'month','race_id','trio_key','role_ok','top3_spread','group_market_p','rank1_conditional_share','group_hit','rank1_exact_hit'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')

    structural = df[(df.role_ok == 1) & (df.top3_spread == 1)].copy()
    ranked = add_rider_ranks(structural)
    if ranked.empty:
        raise SystemExit('no ranked structural groups')

    tickets, odds_skipped = rb.add_posted_rank1_odds(ranked)
    if tickets.empty:
        raise SystemExit('no tickets with posted odds')
    tickets['half_year'] = tickets.month.map(rb.half_year_label)
    tickets['odds_band'] = tickets.rank1_posted_odds.map(rb.odds_band)
    tickets.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    results = {}
    for i, (name, (lo, hi)) in enumerate(WINDOWS.items()):
        results[name] = summarize_window(tickets, lo, hi, SEED + i * 1000)

    primary = tickets[(tickets.selected_min_score_rank >= 2) & (tickets.selected_max_score_rank <= 5)].copy()
    y25 = primary[primary.month.str.startswith('2025_')]

    payload = {
        'status': 'exploratory_individual_race_score_rank_window_probe',
        'candidate_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            'race top3 riders by race_score are spread across 3 distinct true lines',
            'NO group-score-percentile filter',
            'primary: all 3 selected riders have individual race_score rank between 2 and 5 inclusive',
            'bet the single cheapest posted 3rentan permutation inside each qualifying unordered trio',
        ],
        'rank_semantics': 'individual rider rank within race by race_score descending; 1 = highest-scoring rider',
        'adjacent_controls': ['RANK_1_TO_4','RANK_3_TO_6'],
        'odds_recovery_skipped': odds_skipped,
        'structural_pool': {'groups': int(len(tickets)), 'races': int(tickets.race_uid.nunique())},
        'results': results,
        'primary_2025_latest_backfill': rb.summarize_slice(y25, SEED + 9000),
        'warning': 'Exploratory same-data refinement; adjacent windows are controls, not fresh OOS validation.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
