#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Race-level diagnostic for conditions associated with 3-distinct-line top3 finishes.

Exploratory only; same discovery data, not OOS. Primary outcome is whether the
actual top3 riders come from three distinct true_line IDs. This script deliberately
does not require a complete odds board because the current question is incidence,
not profitability.

Predeclared structural features:
1) Within-line head/bante score separation.
2) Dispersion of the highest race_score riders across different lines.
3) Number of multi-rider lines where the bante out-scores the line head.

Secondary: the same summaries in races containing at least one multi-rider RYO
line head (running_style='両' and line_pos=1), plus simple two-way grids.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_three_line_finish_structural_conditions_summary.json'
DETAIL = CTX / 'h1_three_line_finish_structural_conditions_races.csv'


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


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def summarize(x: pd.DataFrame, baseline_rate: float):
    n = int(len(x))
    if n == 0:
        return {'races': 0, 'three_line_finishes': 0, 'rate': None, 'wilson95': [None, None], 'rate_vs_baseline': None}
    k = int(x.three_line_finish.sum())
    rate = k / n
    return {
        'races': n,
        'three_line_finishes': k,
        'rate': float(rate),
        'wilson95': wilson(k, n),
        'rate_vs_baseline': float(rate / baseline_rate) if baseline_rate > 0 else None,
    }


def gap_band(v: float):
    if v < 1:
        return 'LT_1'
    if v < 2:
        return '1_TO_2'
    if v < 4:
        return '2_TO_4'
    return 'GE_4'


def count_band(v: int):
    if v <= 0:
        return '0'
    if v == 1:
        return '1'
    return '2_PLUS'


def build_summary(df: pd.DataFrame):
    base_rate = float(df.three_line_finish.mean()) if len(df) else 0.0
    out = {
        'overall': summarize(df, base_rate),
        'feature_1_within_line_score_separation': {},
        'feature_2_top_score_line_dispersion': {},
        'feature_3_bante_stronger_lines': {},
        'combined_grids': {},
    }

    for col in ['mean_abs_head_bante_gap_band', 'max_abs_head_bante_gap_band', 'large_gap_count_band']:
        out['feature_1_within_line_score_separation'][col] = {
            str(key): summarize(g, base_rate) for key, g in df.groupby(col, dropna=False, sort=False)
        }

    for col in ['top3_score_distinct_lines', 'top4_score_distinct_lines']:
        out['feature_2_top_score_line_dispersion'][col] = {
            str(key): summarize(g, base_rate) for key, g in df.groupby(col, dropna=False, sort=True)
        }

    for col in ['bante_stronger_count_band', 'bante_stronger_by1_count_band']:
        out['feature_3_bante_stronger_lines'][col] = {
            str(key): summarize(g, base_rate) for key, g in df.groupby(col, dropna=False, sort=False)
        }

    # Two simple predeclared grids to see whether the mechanisms stack.
    grid1 = {}
    for d in sorted(df.top3_score_distinct_lines.dropna().unique()):
        row = {}
        for bc in ['0', '1', '2_PLUS']:
            x = df[(df.top3_score_distinct_lines == d) & (df.bante_stronger_count_band == bc)]
            row[bc] = summarize(x, base_rate)
        grid1[str(int(d))] = row
    out['combined_grids']['top3_score_distinct_lines_x_bante_stronger_count'] = grid1

    grid2 = {}
    for gc in ['0', '1', '2_PLUS']:
        row = {}
        for d in sorted(df.top3_score_distinct_lines.dropna().unique()):
            x = df[(df.large_gap_count_band == gc) & (df.top3_score_distinct_lines == d)]
            row[str(int(d))] = summarize(x, base_rate)
        grid2[gc] = row
    out['combined_grids']['large_gap_count_x_top3_score_distinct_lines'] = grid2

    return out


def main():
    rows = []
    skipped = Counter()
    context_rows = 0

    for month in h1.MONTHS:
        loaded = h1.load_month(month)
        if loaded is None:
            skipped['month_missing'] += 1
            continue
        b, c, _o = loaded
        use = c.copy()
        if 'context_quality' in use:
            use = use[use.context_quality.astype(str) == 'full']
        use = use.drop_duplicates('race_id', keep='last')
        context_rows += int(len(use))
        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}

        for cr in use.to_dict('records'):
            rid = str(cr['race_id'])
            pre = bby.get(rid)
            if pre is None:
                skipped['base_missing'] += 1
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

            line_of, pos_of, size_of, members_by_line = {}, {}, {}, {}
            for li, g in enumerate(lines, 1):
                gg = [int(x) for x in g]
                members_by_line[li] = gg
                for pos, fn in enumerate(gg, 1):
                    line_of[fn] = li
                    pos_of[fn] = pos
                    size_of[fn] = len(gg)

            if any(x not in line_of for x in actual):
                skipped['actual_line_missing'] += 1
                continue

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

            multi = []
            abs_gaps = []
            bante_stronger = 0
            bante_stronger_by1 = 0
            head_stronger_by2 = 0
            bante_stronger_by2 = 0
            for li, g in members_by_line.items():
                if len(g) < 2:
                    continue
                head, bante = g[0], g[1]
                delta = float(score[bante] - score[head])
                ad = abs(delta)
                multi.append((li, head, bante, delta, ad))
                abs_gaps.append(ad)
                if delta > 0:
                    bante_stronger += 1
                if delta >= 1:
                    bante_stronger_by1 += 1
                if delta <= -2:
                    head_stronger_by2 += 1
                if delta >= 2:
                    bante_stronger_by2 += 1
            if not multi:
                skipped['no_multi_rider_line'] += 1
                continue

            score_order = sorted(frames, key=lambda x: (-score[x], x))
            top3 = score_order[:min(3, len(score_order))]
            top4 = score_order[:min(4, len(score_order))]
            top3_distinct = len({line_of[x] for x in top3})
            top4_distinct = len({line_of[x] for x in top4})

            mean_abs_gap = float(np.mean(abs_gaps))
            max_abs_gap = float(np.max(abs_gaps))
            large_gap_count = int(sum(g >= 2 for g in abs_gaps))
            actual_lines = [line_of[x] for x in actual]
            three_line = int(len(set(actual_lines)) == 3)
            has_ryo_head = int(any(style.get(g[0]) == '両' for g in members_by_line.values() if len(g) > 1))

            rows.append({
                'month': month,
                'race_id': rid,
                'n_starters': len(frames),
                'n_lines': len(members_by_line),
                'n_multi_lines': len(multi),
                'three_line_finish': three_line,
                'actual_distinct_lines': len(set(actual_lines)),
                'has_ryo_head': has_ryo_head,
                'mean_abs_head_bante_gap': mean_abs_gap,
                'max_abs_head_bante_gap': max_abs_gap,
                'mean_abs_head_bante_gap_band': gap_band(mean_abs_gap),
                'max_abs_head_bante_gap_band': gap_band(max_abs_gap),
                'large_gap_count_ge2': large_gap_count,
                'large_gap_count_band': count_band(large_gap_count),
                'head_stronger_by2_count': head_stronger_by2,
                'bante_stronger_by2_count': bante_stronger_by2,
                'bante_stronger_count': bante_stronger,
                'bante_stronger_count_band': count_band(bante_stronger),
                'bante_stronger_by1_count': bante_stronger_by1,
                'bante_stronger_by1_count_band': count_band(bante_stronger_by1),
                'top3_score_distinct_lines': top3_distinct,
                'top4_score_distinct_lines': top4_distinct,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No usable races')
    df.to_csv(DETAIL, index=False)

    payload = {
        'status': 'exploratory_three_line_finish_structural_conditions',
        'warning': 'Same discovery data and exploratory feature search. Association only; do not freeze thresholds or infer causality from these cells.',
        'coverage': {
            'full_context_rows_seen': int(context_rows),
            'usable_races': int(len(df)),
            'skipped': dict(skipped),
        },
        'definitions': {
            'outcome': 'actual top3 riders occupy exactly three distinct true_line IDs',
            'within_line_gap': 'absolute race_score difference between line head and line_pos2 for each multi-rider line',
            'large_gap': 'absolute head-bante race_score gap >=2 points',
            'top3_score_distinct_lines': 'number of distinct true_line IDs represented among the three highest race_score riders',
            'bante_stronger_count': 'number of multi-rider lines with line_pos2 race_score > line head race_score',
            'bante_stronger_by1_count': 'same, requiring bante-head gap >=1 point',
            'ryo_subset': "race contains at least one multi-rider line with running_style='両' at line_pos1",
        },
        'all_usable_races': build_summary(df),
        'ryo_head_present_subset': build_summary(df[df.has_ryo_head == 1]),
        'selected_combined_signatures': {
            'top3_scores_on_3_lines_AND_2plus_bante_stronger': summarize(
                df[(df.top3_score_distinct_lines == 3) & (df.bante_stronger_count >= 2)],
                float(df.three_line_finish.mean()),
            ),
            'top3_scores_on_3_lines_AND_2plus_large_gaps': summarize(
                df[(df.top3_score_distinct_lines == 3) & (df.large_gap_count_ge2 >= 2)],
                float(df.three_line_finish.mean()),
            ),
            'top3_scores_on_3_lines_AND_2plus_bante_stronger_AND_2plus_large_gaps': summarize(
                df[(df.top3_score_distinct_lines == 3) & (df.bante_stronger_count >= 2) & (df.large_gap_count_ge2 >= 2)],
                float(df.three_line_finish.mean()),
            ),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
