#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand the formation->BOX->cheapest-order mechanism sample.

Purpose: increase power for the conditional cheapest-order calibration test (C)
without changing the market-probability semantics. We do two things at once:
1) extend the observation window through 2026-08 when source files are present;
2) relax candidate filters stepwise, so we can see whether any apparent effect
   survives when sample size increases.

All variants keep: exactly three distinct true lines and each selected line must
contain at least two riders. Flatness quartiles are defined from pre-outcome market
shares only. Outcome evaluation is conditional on the unordered trio actually
finishing top 3.

Exploratory / same-data mechanism test, not fresh OOS validation.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
OUT = CTX / 'formation_box_chain_expanded_summary.json'
DETAIL = CTX / 'formation_box_chain_expanded_groups.csv'
MONTHS = [f'2024_{m:02d}' for m in range(1, 13)] + [f'2025_{m:02d}' for m in range(1, 13)] + [f'2026_{m:02d}' for m in range(1, 9)]
BOOTSTRAPS = 4000
SEED = 20260903

VARIANTS = {
    # Closest to the previous primary TOP50 definition.
    'CURRENT_TOP50': {'require_role': True, 'require_top3_spread': True, 'score_cut': 0.50},
    # Previous structural sensitivity, now with Jul-Aug added.
    'CURRENT_STRUCTURAL': {'require_role': True, 'require_top3_spread': True, 'score_cut': 1.00},
    # Cheapest relaxation: remove only the race-level top3-score dispersion gate.
    'NO_TOP3_SPREAD_TOP50': {'require_role': True, 'require_top3_spread': False, 'score_cut': 0.50},
    'NO_TOP3_SPREAD_STRUCTURAL': {'require_role': True, 'require_top3_spread': False, 'score_cut': 1.00},
    # Sensitivity ceiling: keep 3 distinct multi-rider lines but allow any selected role.
    'ANY_ROLE_TOP50': {'require_role': False, 'require_top3_spread': False, 'score_cut': 0.50},
    'ANY_ROLE_STRUCTURAL': {'require_role': False, 'require_top3_spread': False, 'score_cut': 1.00},
}


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


def build_groups():
    rows = []
    skipped = Counter()
    usable_by_month = {}

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
            if len(frames) < 4:
                skipped['too_few_riders'] += 1
                continue

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

            score_order = sorted(frames, key=lambda x: (-score[x], x))
            top3_spread = len({line_of[x] for x in score_order[:3]}) == 3

            all_groups = list(itertools.combinations(frames, 3))
            score_sums = {g: float(sum(score[x] for x in g)) for g in all_groups}
            ordered = sorted(all_groups, key=lambda g: (-score_sums[g], g))
            rank_of = {g: i + 1 for i, g in enumerate(ordered)}
            ng = len(ordered)

            for trio_u in all_groups:
                # Broadest retained structural boundary for this probe.
                if len({line_of[x] for x in trio_u}) != 3:
                    continue
                if any(size_of.get(x, 0) <= 1 for x in trio_u):
                    continue

                perms = list(itertools.permutations(trio_u))
                if any(p not in tri or tri[p] <= 0 for p in perms):
                    continue
                perms = sorted(perms, key=lambda p: (tri[p], p))
                exact_p = np.asarray([(1.0 / tri[p]) / z for p in perms], dtype=float)
                gp = float(exact_p.sum())
                if gp <= 0:
                    continue
                shares = exact_p / gp
                entropy = float(-np.sum(shares * np.log(shares)) / math.log(6.0))
                hhi = float(np.sum(shares ** 2))
                group_hit = int(frozenset(trio_u) == frozenset(actual))
                winner_rank = None
                if group_hit:
                    try:
                        winner_rank = perms.index(tuple(actual)) + 1
                    except ValueError:
                        winner_rank = None

                role_ok = all(style.get(x) == '両' or pos_of.get(x) == 2 for x in trio_u)
                role_set = '+'.join(sorted('RYO' if style.get(x) == '両' else ('BANTE' if pos_of.get(x) == 2 else 'OTHER') for x in trio_u))
                rows.append({
                    'month': month,
                    'race_id': rid,
                    'trio_key': '-'.join(map(str, sorted(trio_u))),
                    'group_score_percentile': float(rank_of[trio_u] / ng),
                    'role_ok': int(role_ok),
                    'top3_spread': int(top3_spread),
                    'unordered_role_set': role_set,
                    'group_market_p': gp,
                    'effective_group_odds': float(1.0 / gp),
                    'entropy_norm': entropy,
                    'hhi': hhi,
                    'rank1_conditional_share': float(shares[0]),
                    'rank6_conditional_share': float(shares[-1]),
                    'rank1_to_rank6_share_ratio': float(shares[0] / shares[-1]) if shares[-1] > 0 else None,
                    'group_hit': group_hit,
                    'winner_odds_rank': winner_rank,
                    'rank1_exact_hit': int(group_hit and winner_rank == 1),
                })

    return pd.DataFrame(rows), {'months_requested': MONTHS, 'usable_races_by_month': usable_by_month, 'skipped': dict(skipped)}


def conditional_calibration(g: pd.DataFrame):
    hit = g[g.group_hit == 1]
    if hit.empty:
        return {
            'group_hits': 0,
            'rank1_actual_wins': 0,
            'rank1_conditional_expected_wins': 0.0,
            'rank1_actual_over_conditional_market': None,
            'rank1_actual_share_pct': None,
            'rank1_market_expected_share_pct': None,
        }
    actual = int(hit.rank1_exact_hit.sum())
    exp = float(hit.rank1_conditional_share.sum())
    return {
        'group_hits': int(len(hit)),
        'rank1_actual_wins': actual,
        'rank1_conditional_expected_wins': exp,
        'rank1_actual_over_conditional_market': float(actual / exp) if exp > 0 else None,
        'rank1_actual_share_pct': float(100.0 * actual / len(hit)),
        'rank1_market_expected_share_pct': float(100.0 * exp / len(hit)),
    }


def bootstrap_quartile(g: pd.DataFrame, seed: int):
    # Resample races. Contributions are only from groups that actually hit,
    # matching the conditional calibration definition.
    x = g.copy()
    x['actual_contrib'] = x.rank1_exact_hit.astype(float)
    x['expected_contrib'] = x.group_hit.astype(float) * x.rank1_conditional_share.astype(float)
    rg = x.groupby('race_id', sort=False)[['actual_contrib', 'expected_contrib']].sum()
    if len(rg) < 2:
        return {'bootstrap_reps': 0, 'ratio_ci95': [None, None]}
    arr = rg.to_numpy(float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAPS):
        s = arr[rng.integers(0, n, n)].sum(axis=0)
        if s[1] > 0:
            vals.append(float(s[0] / s[1]))
    if not vals:
        return {'bootstrap_reps': 0, 'ratio_ci95': [None, None]}
    return {
        'bootstrap_reps': len(vals),
        'ratio_ci95': [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
    }


def q4_vs_q1_bootstrap(g: pd.DataFrame, seed: int):
    x = g.copy()
    x['actual_contrib'] = x.rank1_exact_hit.astype(float)
    x['expected_contrib'] = x.group_hit.astype(float) * x.rank1_conditional_share.astype(float)
    races = x.race_id.unique()
    if len(races) < 2:
        return {'bootstrap_reps': 0}
    by = {}
    for rid, rg in x.groupby('race_id', sort=False):
        d = {}
        for q in ['Q1_LEAST_FLAT', 'Q4_MOST_FLAT']:
            z = rg[rg.flatness_quartile.astype(str) == q]
            d[q] = (float(z.actual_contrib.sum()), float(z.expected_contrib.sum()))
        by[rid] = d
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(BOOTSTRAPS):
        samp = rng.choice(races, size=len(races), replace=True)
        a1 = e1 = a4 = e4 = 0.0
        for rid in samp:
            p1 = by[rid]['Q1_LEAST_FLAT']; p4 = by[rid]['Q4_MOST_FLAT']
            a1 += p1[0]; e1 += p1[1]; a4 += p4[0]; e4 += p4[1]
        if e1 > 0 and e4 > 0:
            diffs.append(float(a4 / e4 - a1 / e1))
    if not diffs:
        return {'bootstrap_reps': 0}
    arr = np.asarray(diffs, float)
    return {
        'bootstrap_reps': int(len(arr)),
        'q4_minus_q1_ratio_diff_ci95': [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        'prob_q4_ratio_gt_q1_pct': float(100.0 * np.mean(arr > 0)),
    }


def summarize_variant(allg: pd.DataFrame, name: str, cfg: dict):
    g = allg.copy()
    if cfg['require_role']:
        g = g[g.role_ok == 1]
    if cfg['require_top3_spread']:
        g = g[g.top3_spread == 1]
    g = g[g.group_score_percentile <= float(cfg['score_cut'])].copy()
    if g.empty:
        return {'groups': 0, 'config': cfg}

    g['log_group_odds'] = np.log(g.effective_group_odds)
    try:
        g['flatness_quartile'] = pd.qcut(
            g.entropy_norm.rank(method='first'), 4,
            labels=['Q1_LEAST_FLAT', 'Q2', 'Q3', 'Q4_MOST_FLAT']
        )
    except Exception:
        g['flatness_quartile'] = None

    by_q = {}
    for i, (q, z) in enumerate(g.groupby('flatness_quartile', observed=True, sort=True), 1):
        cal = conditional_calibration(z)
        cal['race_bootstrap'] = bootstrap_quartile(z, SEED + i + len(g))
        by_q[str(q)] = {
            'groups': int(len(z)),
            'races': int(z.race_id.nunique()),
            'median_group_odds': float(z.effective_group_odds.median()),
            'mean_entropy_norm': float(z.entropy_norm.mean()),
            'median_rank1_conditional_share_pct': float(100.0 * z.rank1_conditional_share.median()),
            'median_rank1_to_rank6_share_ratio': float(z.rank1_to_rank6_share_ratio.median()),
            'rank1_conditional_calibration': cal,
        }

    return {
        'config': cfg,
        'groups': int(len(g)),
        'races': int(g.race_id.nunique()),
        'group_hits_total': int(g.group_hit.sum()),
        'rank1_exact_hits_total': int(g.rank1_exact_hit.sum()),
        'spearman_log_group_odds_vs_entropy_norm': float(g.log_group_odds.corr(g.entropy_norm, method='spearman')) if len(g) >= 3 else None,
        'spearman_log_group_odds_vs_rank1_share': float(g.log_group_odds.corr(g.rank1_conditional_share, method='spearman')) if len(g) >= 3 else None,
        'flatness_quartiles': by_q,
        'q4_vs_q1_bootstrap': q4_vs_q1_bootstrap(g, SEED + 1000 + len(g)),
    }


def main():
    groups, coverage = build_groups()
    if groups.empty:
        raise SystemExit('No analyzable groups')
    groups.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    variants = {name: summarize_variant(groups, name, cfg) for name, cfg in VARIANTS.items()}
    payload = {
        'status': 'exploratory_formation_box_sample_expansion',
        'decision_question': 'Is the low Q4 point estimate mainly sparse-sample variance, and does cheapest-order relative value persist as filters are relaxed?',
        'sample_expansion': {
            'observation_window': '2024-01 through 2026-08 where all source files exist',
            'kept_in_all_variants': ['exactly 3 distinct true lines', 'selected lines have >=2 riders', 'full usable trifecta board'],
            'variants': VARIANTS,
        },
        'coverage': coverage,
        'broad_group_pool': {
            'groups': int(len(groups)),
            'races': int(groups.race_id.nunique()),
            'group_hits': int(groups.group_hit.sum()),
        },
        'variants': variants,
        'interpretation_rule': {
            'underpowered': 'quartile group-hit count is still small and bootstrap CI is wide / crosses 1 substantially',
            'supportive': 'Q4/Q3 ratios remain above market with materially larger hit counts and bootstrap evidence is directionally stable',
            'contradictory': 'with materially larger hit counts, Q4 remains below market and Q4-vs-Q1 bootstrap is consistently negative',
        },
        'warning': 'Exploratory same-data mechanism analysis. Actual bettor ticket types are not observed; BOX/formation remain market-shape interpretations.',
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
