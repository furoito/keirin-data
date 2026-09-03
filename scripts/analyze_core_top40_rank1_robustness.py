#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness probe for the current practical keirin candidate.

Candidate fixed before this probe:
- unordered trio spans exactly 3 distinct true lines
- every selected rider is running_style='両' OR line_pos=2 (bante)
- selected lines are multi-rider lines (inherited from expanded group pool)
- trio race_score percentile <= 40%
- bet exactly the cheapest posted 3rentan order within each qualifying trio

This script measures:
- posted-odds flat-stake return ratio (1 unit per qualifying exact ticket)
- normalized market actual/expected calibration
- group-level calibration and conditional cheapest-order calibration
- calendar period robustness
- posted-odds band robustness
- score-band contribution inside TOP40
- race-level bootstrap intervals, preserving within-race ticket dependence

Exploratory same-data robustness check, not fresh out-of-sample validation.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'formation_box_chain_expanded_groups.csv'
OUT = CTX / 'core_top40_rank1_robustness_summary.json'
DETAIL = CTX / 'core_top40_rank1_tickets.csv'

BOOTSTRAPS = 5000
SEED = 20260903


def safe_ratio(a: float, b: float):
    return float(a / b) if b > 0 else None


def add_posted_rank1_odds(core: pd.DataFrame):
    """Recover the posted odds of the cheapest exact order for each trio.

    The expanded group file stores the normalized market shares but not raw posted
    odds. Re-read each month's odds board and use the exact same ordering rule as
    analyze_formation_box_chain_expanded.py: sort by (posted_odds, permutation).
    """
    parts = []
    skipped = {}

    for month, g in core.groupby('month', sort=True):
        loaded = h1.load_month(str(month))
        if loaded is None:
            skipped[str(month)] = {'month_missing': int(len(g))}
            continue
        _, _, odds = loaded
        oby = {str(k): x for k, x in odds.groupby('race_id', sort=False)}
        month_rows = []
        missing_race = 0
        incomplete_group = 0

        for r in g.itertuples(index=False):
            rid = str(r.race_id)
            og = oby.get(rid)
            if og is None:
                missing_race += 1
                continue
            tri = base.odds_map(og)
            try:
                trio = tuple(int(x) for x in str(r.trio_key).split('-'))
            except Exception:
                incomplete_group += 1
                continue
            perms = list(itertools.permutations(trio))
            if any(p not in tri or tri[p] <= 0 for p in perms):
                incomplete_group += 1
                continue
            rank1_perm = sorted(perms, key=lambda p: (tri[p], p))[0]
            d = r._asdict()
            d['rank1_posted_odds'] = float(tri[rank1_perm])
            d['rank1_order'] = '-'.join(map(str, rank1_perm))
            d['rank1_exact_market_p'] = float(r.group_market_p * r.rank1_conditional_share)
            d['race_uid'] = f'{month}|{rid}'
            month_rows.append(d)

        if missing_race or incomplete_group:
            skipped[str(month)] = {
                'race_odds_missing': int(missing_race),
                'group_odds_incomplete': int(incomplete_group),
            }
        if month_rows:
            parts.append(pd.DataFrame(month_rows))

    if not parts:
        return pd.DataFrame(), skipped
    return pd.concat(parts, ignore_index=True), skipped


def metrics(x: pd.DataFrame):
    if x.empty:
        return {
            'tickets': 0,
            'races': 0,
            'wins': 0,
            'posted_odds_return_ratio': None,
            'profit_units': None,
            'rank1_actual_over_normalized_market': None,
            'group_actual_over_normalized_market': None,
            'rank1_conditional_actual_over_market': None,
        }

    tickets = int(len(x))
    races = int(x.race_uid.nunique())
    wins = int(x.rank1_exact_hit.sum())
    gross = float((x.rank1_posted_odds * x.rank1_exact_hit).sum())
    exact_exp = float(x.rank1_exact_market_p.sum())
    group_actual = int(x.group_hit.sum())
    group_exp = float(x.group_market_p.sum())
    hit = x[x.group_hit == 1]
    cond_exp = float(hit.rank1_conditional_share.sum()) if len(hit) else 0.0

    return {
        'tickets': tickets,
        'races': races,
        'wins': wins,
        'hit_rate_pct': float(100.0 * wins / tickets),
        'posted_odds_gross_return_units': gross,
        'posted_odds_return_ratio': float(gross / tickets),
        'profit_units': float(gross - tickets),
        'median_ticket_posted_odds': float(x.rank1_posted_odds.median()),
        'median_winner_posted_odds': float(x.loc[x.rank1_exact_hit == 1, 'rank1_posted_odds'].median()) if wins else None,
        'rank1_normalized_market': {
            'actual_hits': wins,
            'expected_hits': exact_exp,
            'actual_over_market': safe_ratio(wins, exact_exp),
        },
        'group_normalized_market': {
            'actual_hits': group_actual,
            'expected_hits': group_exp,
            'actual_over_market': safe_ratio(group_actual, group_exp),
        },
        'rank1_conditional_on_group_hit': {
            'group_hits': int(len(hit)),
            'actual_rank1_wins': wins,
            'expected_rank1_wins': cond_exp,
            'actual_over_market': safe_ratio(wins, cond_exp),
        },
    }


def race_bootstrap(x: pd.DataFrame, seed: int):
    if x.empty or x.race_uid.nunique() < 2:
        return {
            'bootstrap_reps': 0,
            'posted_odds_return_ratio_ci95': [None, None],
            'posted_odds_return_ratio_prob_gt_1_pct': None,
            'rank1_market_ratio_ci95': [None, None],
            'rank1_market_ratio_prob_gt_1_pct': None,
            'group_market_ratio_ci95': [None, None],
            'group_market_ratio_prob_gt_1_pct': None,
        }

    z = x.copy()
    z['tickets_contrib'] = 1.0
    z['return_contrib'] = z.rank1_posted_odds * z.rank1_exact_hit
    z['rank1_actual_contrib'] = z.rank1_exact_hit.astype(float)
    z['rank1_expected_contrib'] = z.rank1_exact_market_p.astype(float)
    z['group_actual_contrib'] = z.group_hit.astype(float)
    z['group_expected_contrib'] = z.group_market_p.astype(float)

    cols = [
        'tickets_contrib','return_contrib',
        'rank1_actual_contrib','rank1_expected_contrib',
        'group_actual_contrib','group_expected_contrib',
    ]
    rg = z.groupby('race_uid', sort=False)[cols].sum()
    arr = rg.to_numpy(float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    rois, rank1_ratios, group_ratios = [], [], []

    for _ in range(BOOTSTRAPS):
        s = arr[rng.integers(0, n, n)].sum(axis=0)
        if s[0] > 0:
            rois.append(float(s[1] / s[0]))
        if s[3] > 0:
            rank1_ratios.append(float(s[2] / s[3]))
        if s[5] > 0:
            group_ratios.append(float(s[4] / s[5]))

    def pack(vals):
        if not vals:
            return [None, None], None
        a = np.asarray(vals, dtype=float)
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))], float(100.0 * np.mean(a > 1.0))

    roi_ci, roi_prob = pack(rois)
    rank1_ci, rank1_prob = pack(rank1_ratios)
    group_ci, group_prob = pack(group_ratios)
    return {
        'bootstrap_reps': int(min(len(rois), len(rank1_ratios), len(group_ratios))),
        'posted_odds_return_ratio_ci95': roi_ci,
        'posted_odds_return_ratio_prob_gt_1_pct': roi_prob,
        'rank1_market_ratio_ci95': rank1_ci,
        'rank1_market_ratio_prob_gt_1_pct': rank1_prob,
        'group_market_ratio_ci95': group_ci,
        'group_market_ratio_prob_gt_1_pct': group_prob,
    }


def summarize_slice(x: pd.DataFrame, seed: int):
    d = metrics(x)
    d['race_bootstrap'] = race_bootstrap(x, seed)
    return d


def half_year_label(month: str):
    y, m = map(int, str(month).split('_'))
    if y == 2026 and m >= 7:
        return '2026_JUL_AUG'
    return f'{y}_H{1 if m <= 6 else 2}'


def odds_band(v: float):
    if v < 30:
        return 'LT30'
    if v < 50:
        return '30_TO_50'
    if v < 100:
        return '50_TO_100'
    if v < 200:
        return '100_TO_200'
    return 'GE200'


def score_band(p: float):
    if p <= 0.10:
        return 'TOP10'
    if p <= 0.20:
        return '10_TO_20'
    if p <= 0.30:
        return '20_TO_30'
    return '30_TO_40'


def main():
    if not SRC.exists():
        raise SystemExit(f'missing input: {SRC}')

    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str, 'trio_key': str})
    required = {
        'month','race_id','trio_key','group_score_percentile','role_ok',
        'group_market_p','rank1_conditional_share','group_hit','rank1_exact_hit'
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')

    core = df[(df.role_ok == 1) & (df.group_score_percentile <= 0.40)].copy()
    tickets, odds_skipped = add_posted_rank1_odds(core)
    if tickets.empty:
        raise SystemExit('no TOP40 core tickets with recovered posted odds')

    tickets['year'] = tickets.month.str.slice(0, 4)
    tickets['half_year'] = tickets.month.map(half_year_label)
    tickets['odds_band'] = tickets.rank1_posted_odds.map(odds_band)
    tickets['score_band'] = tickets.group_score_percentile.map(score_band)

    # Exposure diagnostics before slicing.
    per_race = tickets.groupby('race_uid').size()
    exposure = {
        'qualifying_tickets': int(len(tickets)),
        'qualifying_races': int(tickets.race_uid.nunique()),
        'mean_tickets_per_qualifying_race': float(per_race.mean()),
        'median_tickets_per_qualifying_race': float(per_race.median()),
        'max_tickets_in_one_race': int(per_race.max()),
        'races_with_1_ticket': int((per_race == 1).sum()),
        'races_with_2plus_tickets': int((per_race >= 2).sum()),
    }

    overall = summarize_slice(tickets, SEED)

    yearly = {}
    for i, (label, g) in enumerate(tickets.groupby('year', sort=True)):
        yearly[str(label)] = summarize_slice(g, SEED + 100 + i)

    half_year = {}
    for i, (label, g) in enumerate(tickets.groupby('half_year', sort=True)):
        half_year[str(label)] = summarize_slice(g, SEED + 200 + i)

    score_bands = {}
    score_order = ['TOP10','10_TO_20','20_TO_30','30_TO_40']
    for i, label in enumerate(score_order):
        score_bands[label] = summarize_slice(tickets[tickets.score_band == label], SEED + 300 + i)

    odds_bands = {}
    odds_order = ['LT30','30_TO_50','50_TO_100','100_TO_200','GE200']
    for i, label in enumerate(odds_order):
        odds_bands[label] = summarize_slice(tickets[tickets.odds_band == label], SEED + 400 + i)

    cumulative_odds = {}
    for i, threshold in enumerate([30, 50, 100, 200]):
        g = tickets[tickets.rank1_posted_odds >= threshold]
        cumulative_odds[f'GE{threshold}'] = summarize_slice(g, SEED + 500 + i)

    payload = {
        'status': 'exploratory_core_top40_rank1_robustness_probe',
        'candidate_definition': [
            'exactly 3 distinct true lines',
            'selected lines have >=2 riders',
            "each selected rider is running_style='両' OR line_pos=2",
            'group_score_percentile <= 0.40',
            'bet the single cheapest posted 3rentan permutation inside every qualifying unordered trio',
        ],
        'measurement_semantics': {
            'posted_odds_return_ratio': 'sum(posted_odds for winning tickets) / number_of_tickets; 1 unit flat stake per ticket',
            'normalized_market_ratio': 'actual hits / sum normalized exact-market probabilities',
            'bootstrap': 'resample whole races with replacement so multiple qualifying tickets from one race remain clustered',
        },
        'source': str(SRC.relative_to(ROOT)),
        'odds_recovery_skipped': odds_skipped,
        'exposure': exposure,
        'overall': overall,
        'by_year': yearly,
        'by_half_year': half_year,
        'by_score_band_within_top40': score_bands,
        'by_posted_odds_band': odds_bands,
        'by_cumulative_posted_odds_threshold': cumulative_odds,
        'decision_use': {
            'positive_pattern': 'overall return and normalized calibration above 1, not dominated by a single period, with similar direction across score/odds slices',
            'warning_pattern': 'profit concentrated in one sparse period or extreme-odds winners while broad normalized calibration is near/below 1',
            'next_gate_if_positive': 'freeze this candidate definition and test on genuinely later out-of-sample races before treating it as production betting evidence',
        },
        'warning': 'Same historical data were used during hypothesis development. This probe measures robustness, not unbiased out-of-sample profitability.',
    }

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tickets.to_csv(DETAIL, index=False, encoding='utf-8-sig')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
