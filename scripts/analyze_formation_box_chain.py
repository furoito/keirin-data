#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic for a two-step bettor-demand mechanism.

Hypothesis chain:
A) Formation-style ticket construction tends to omit three-line combinations,
   so comparable three-line groups receive less market mass / longer group odds.
B) As a selected trio becomes more of a longshot, BOX-style equal-stake buying
   becomes more prevalent, flattening market weights across its six exact orders.
   If that flattening overstates unlikely orderings, the lowest-odds order can be
   relatively underweighted inside the six.

This script does not observe bettor tickets directly. It tests observable proxies:
1) matched market-price comparison of line-span 3 vs line-span 2 groups at similar
   pre-race group-score percentiles;
2) within-six market-share flatness vs effective group odds;
3) conditional calibration of the lowest-odds order vs flatness / group-odds bands.

Historical exploratory mechanism test; not fresh OOS.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
CANON = CTX / 'h1_canonical_three_line_ryo_bante_strong_extended_2024_2026_tickets.csv'
OUT = CTX / 'formation_box_chain_summary.json'
DETAIL = CTX / 'formation_box_chain_groups.csv'

MONTHS = [f'2024_{m:02d}' for m in range(1, 13)] + [f'2025_{m:02d}' for m in range(1, 13)] + [f'2026_{m:02d}' for m in range(1, 7)]
SCORE_BANDS = [
    ('TOP10', 0.00, 0.10),
    ('10_TO_25', 0.10, 0.25),
    ('25_TO_50', 0.25, 0.50),
]
GROUP_ODDS_BINS = [
    ('LT_10', 0.0, 10.0),
    ('10_TO_20', 10.0, 20.0),
    ('20_TO_30', 20.0, 30.0),
    ('30_TO_50', 30.0, 50.0),
    ('50_TO_100', 50.0, 100.0),
    ('100_TO_200', 100.0, 200.0),
    ('GE_200', 200.0, float('inf')),
]


def trio_key(ticket: str) -> str:
    return '-'.join(map(str, sorted(int(x) for x in str(ticket).split('-'))))


def med(x):
    return float(np.median(x)) if len(x) else None


def spearman(a: pd.Series, b: pd.Series):
    x = pd.concat([pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')], axis=1).dropna()
    if len(x) < 3:
        return None
    return float(x.iloc[:, 0].corr(x.iloc[:, 1], method='spearman'))


def build_all_group_market():
    rows = []
    skipped = Counter()
    for month in MONTHS:
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
        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}
        oby = {str(k): g for k, g in o.groupby('race_id', sort=False)}
        for cr in use.to_dict('records'):
            rid = str(cr['race_id'])
            pre = bby.get(rid)
            og = oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing'] += 1
                continue
            rr, why = h1.race_rows(month, rid, pre, cr, og)
            if rr is None:
                skipped[str(why)] += 1
            else:
                rows.extend(rr)
    return pd.DataFrame(rows), dict(skipped)


def formation_leakage_proxy(allg: pd.DataFrame):
    # Primary proxy: compare 3-line vs 2-line groups within the same race and
    # the same pre-fixed score-percentile band. This controls race-level pool
    # conditions and roughly controls group strength.
    paired = []
    band_summary = {}
    for bname, lo, hi in SCORE_BANDS:
        z = allg[(allg.group_score_percentile > lo) & (allg.group_score_percentile <= hi)].copy()
        for rid, rg in z.groupby('race_id', sort=False):
            a = rg[rg.line_span == 3].effective_fair_odds.to_numpy(float)
            b = rg[rg.line_span == 2].effective_fair_odds.to_numpy(float)
            if not len(a) or not len(b):
                continue
            m3, m2 = float(np.median(a)), float(np.median(b))
            paired.append({'race_id': rid, 'score_band': bname, 'median_odds_3line': m3, 'median_odds_2line': m2, 'odds_ratio_3line_over_2line': m3 / m2})
        p = pd.DataFrame([x for x in paired if x['score_band'] == bname])
        if p.empty:
            band_summary[bname] = {'paired_races': 0}
        else:
            band_summary[bname] = {
                'paired_races': int(len(p)),
                'median_3line_over_2line_odds_ratio': float(p.odds_ratio_3line_over_2line.median()),
                'mean_log_odds_ratio': float(np.log(p.odds_ratio_3line_over_2line).mean()),
                'share_races_3line_longer_pct': float(100.0 * (p.odds_ratio_3line_over_2line > 1).mean()),
                'median_3line_group_odds': float(p.median_odds_3line.median()),
                'median_2line_group_odds': float(p.median_odds_2line.median()),
            }
    p = pd.DataFrame(paired)
    overall = {'paired_race_band_cells': int(len(p))}
    if len(p):
        overall.update({
            'median_3line_over_2line_odds_ratio': float(p.odds_ratio_3line_over_2line.median()),
            'share_cells_3line_longer_pct': float(100.0 * (p.odds_ratio_3line_over_2line > 1).mean()),
        })
    return {'overall': overall, 'score_bands': band_summary}


def build_canonical_groups():
    t = pd.read_csv(CANON)
    req = {'race_id', 'ticket', 'odds', 'market_p', 'actual_hit', 'group_score_percentile', 'unordered_role_set'}
    missing = req - set(t.columns)
    if missing:
        raise SystemExit(f'Missing canonical columns: {sorted(missing)}')
    t = t.copy()
    t['trio_key'] = t.ticket.map(trio_key)
    sizes = t.groupby(['race_id', 'trio_key']).size()
    if not (sizes == 6).all():
        raise SystemExit(f'Expected six orders per trio; bad={int((sizes != 6).sum())}')

    grows = []
    for (rid, key), g in t.groupby(['race_id', 'trio_key'], sort=False):
        g = g.sort_values(['odds', 'ticket'], kind='mergesort').copy()
        gp = float(g.market_p.sum())
        if gp <= 0:
            continue
        shares = (g.market_p / gp).to_numpy(float)
        shares = shares / shares.sum()
        entropy = float(-np.sum(shares * np.log(shares)) / math.log(6.0))
        hhi = float(np.sum(shares ** 2))
        top = float(shares[0])
        bottom = float(shares[-1])
        winner_rank = None
        if int(g.actual_hit.sum()) == 1:
            winner_rank = int(np.argmax(g.actual_hit.to_numpy()) + 1)
        grows.append({
            'race_id': str(rid), 'trio_key': key,
            'group_market_p': gp, 'effective_group_odds': 1.0 / gp,
            'group_hit': int(g.actual_hit.sum()),
            'group_score_percentile': float(g.group_score_percentile.iloc[0]),
            'unordered_role_set': str(g.unordered_role_set.iloc[0]),
            'entropy_norm': entropy, 'hhi': hhi,
            'rank1_conditional_share': top,
            'rank6_conditional_share': bottom,
            'rank1_to_rank6_share_ratio': float(top / bottom) if bottom > 0 else None,
            'winner_odds_rank': winner_rank,
            'rank1_exact_market_p': float(g.market_p.iloc[0]),
            'rank1_exact_hit': int(g.actual_hit.iloc[0]),
        })
    return pd.DataFrame(grows)


def flatness_agg(g: pd.DataFrame):
    if g.empty:
        return {'groups': 0}
    return {
        'groups': int(len(g)),
        'races': int(g.race_id.nunique()),
        'median_group_odds': float(g.effective_group_odds.median()),
        'mean_entropy_norm': float(g.entropy_norm.mean()),
        'median_entropy_norm': float(g.entropy_norm.median()),
        'mean_hhi': float(g.hhi.mean()),
        'median_rank1_conditional_share_pct': float(100 * g.rank1_conditional_share.median()),
        'median_rank1_to_rank6_share_ratio': float(g.rank1_to_rank6_share_ratio.median()),
    }


def conditional_rank1_calibration(g: pd.DataFrame):
    hit = g[g.group_hit == 1].copy()
    if hit.empty:
        return {'group_hits': 0, 'rank1_actual_wins': 0, 'rank1_conditional_expected_wins': 0.0, 'rank1_actual_over_conditional_market': None}
    actual = int((hit.winner_odds_rank == 1).sum())
    exp = float(hit.rank1_conditional_share.sum())
    return {
        'group_hits': int(len(hit)),
        'rank1_actual_wins': actual,
        'rank1_conditional_expected_wins': exp,
        'rank1_actual_over_conditional_market': float(actual / exp) if exp > 0 else None,
        'rank1_actual_share_pct': float(100 * actual / len(hit)),
        'rank1_market_expected_share_pct': float(100 * exp / len(hit)),
    }


def box_flattening_diag(groups: pd.DataFrame, cut: float):
    g = groups[groups.group_score_percentile <= cut].copy()
    g['log_group_odds'] = np.log(g.effective_group_odds)
    corr = {
        'spearman_log_group_odds_vs_entropy_norm': spearman(g.log_group_odds, g.entropy_norm),
        'spearman_log_group_odds_vs_hhi': spearman(g.log_group_odds, g.hhi),
        'spearman_log_group_odds_vs_rank1_share': spearman(g.log_group_odds, g.rank1_conditional_share),
        'spearman_log_group_odds_vs_rank1_to_rank6_ratio': spearman(g.log_group_odds, g.rank1_to_rank6_share_ratio),
    }
    by_odds = {}
    rank1_by_odds = {}
    for name, lo, hi in GROUP_ODDS_BINS:
        z = g[(g.effective_group_odds >= lo) & (g.effective_group_odds < hi)]
        by_odds[name] = flatness_agg(z)
        rank1_by_odds[name] = conditional_rank1_calibration(z)

    # Flatness quartiles are defined without using outcomes. Q4 = flattest / most BOX-like.
    try:
        g['flatness_quartile'] = pd.qcut(g.entropy_norm.rank(method='first'), 4, labels=['Q1_LEAST_FLAT', 'Q2', 'Q3', 'Q4_MOST_FLAT'])
    except Exception:
        g['flatness_quartile'] = None
    by_flat = {}
    for q, z in g.groupby('flatness_quartile', observed=True, sort=True):
        by_flat[str(q)] = {**flatness_agg(z), 'rank1_conditional_calibration': conditional_rank1_calibration(z)}

    role_sensitivity = {}
    for role, z in g.groupby('unordered_role_set', sort=True):
        if len(z) < 30:
            continue
        zz = z.copy(); zz['log_group_odds'] = np.log(zz.effective_group_odds)
        role_sensitivity[str(role)] = {
            'groups': int(len(zz)),
            'spearman_log_group_odds_vs_entropy_norm': spearman(zz.log_group_odds, zz.entropy_norm),
            'spearman_log_group_odds_vs_rank1_share': spearman(zz.log_group_odds, zz.rank1_conditional_share),
        }

    return {
        'groups': int(len(g)),
        'correlations': corr,
        'flatness_by_group_odds': by_odds,
        'rank1_conditional_calibration_by_group_odds': rank1_by_odds,
        'flatness_quartiles': by_flat,
        'role_set_sensitivity': role_sensitivity,
    }


def main():
    allg, skipped = build_all_group_market()
    if allg.empty:
        raise SystemExit('No all-group market rows')
    formation = formation_leakage_proxy(allg)

    groups = build_canonical_groups()
    groups.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    payload = {
        'status': 'exploratory_formation_to_box_chain_test',
        'hypothesis_chain': {
            'A_formation_omission_proxy': 'At similar pre-race group-score strength within the same race, three-line groups should carry longer effective group odds than two-line groups.',
            'B_box_flattening_proxy': 'As effective group odds rise, conditional market shares across the six exact orders should become flatter: entropy rises, HHI falls, rank1 share falls, rank1/rank6 ratio falls.',
            'C_practical_consequence': 'If BOX-like flattening overweights unlikely orderings, the lowest-odds exact order should become relatively underweighted, especially in flatter groups.',
        },
        'predeclared_support_pattern': {
            'A': 'median 3-line/2-line matched odds ratio > 1 and a majority of paired race-band cells > 1',
            'B': 'positive odds-vs-entropy correlation together with negative odds-vs-HHI/rank1-share/rank1-to-rank6 correlations, preferably repeated across role sets',
            'C': 'rank1 actual/conditional-market ratio higher in the flattest quartile than the least-flat quartile; group-odds-band version is secondary because hit counts become sparse in longshot tails',
        },
        'formation_omission_proxy': formation,
        'formation_omission_source_coverage': {'months': MONTHS, 'skipped': skipped},
        'box_flattening_primary_TOP50': box_flattening_diag(groups, 0.50),
        'box_flattening_sensitivity_STRUCTURAL': box_flattening_diag(groups, 1.00),
        'warning': 'We do not observe actual bettor betslips, so formation and BOX behavior remain mechanism interpretations. These are market-shape proxies on previously explored historical data, not fresh OOS evidence.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
