#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Role-composition diagnostic for actual 3-distinct-line trifecta finishes.

Exploratory only; same discovery data, not OOS.

Question: once a race is structurally prone to a 3-line finish, which roles actually
occupy 1st/2nd/3rd?

Role taxonomy (pre-race only):
- RYO_HEAD: line_pos1, running_style='両', multi-rider line
- ESCAPE_HEAD: line_pos1, running_style='逃', multi-rider line
- OTHER_HEAD: line_pos1, other/unknown style, multi-rider line
- BANTE: line_pos2, multi-rider line
- THIRD_PLUS: line_pos>=3
- SOLO: single-rider line

Primary subsets:
- ALL_USABLE
- TOP3_SCORE_3LINES: top three race_score riders belong to three distinct lines
- TOP3_SCORE_3LINES_BANTE2PLUS: above + >=2 lines where bante score > head score
- TOP3_SCORE_3LINES_GAP2PLUS: above + >=2 multi-rider lines with |head-bante|>=2
- TOP3_SCORE_3LINES_BOTH: both structural add-ons
Each is also reported with RYO_HEAD_PRESENT.

Primary outputs are incidence among all races and composition share among actual
3-line finishes. No odds/ROI optimization in this diagnostic.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_three_line_role_composition_summary.json'
DETAIL = CTX / 'h1_three_line_role_composition_races.csv'

TARGET_PATTERNS = [
    'RYO_HEAD>BANTE>BANTE',
    'ESCAPE_HEAD>BANTE>BANTE',
    'OTHER_HEAD>BANTE>BANTE',
    'BANTE>BANTE>BANTE',
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


def role_of(fn, pos_of, size_of, style):
    if size_of.get(fn, 1) <= 1:
        return 'SOLO'
    pos = pos_of.get(fn)
    if pos == 1:
        st = style.get(fn, 'UNKNOWN')
        if st == '両':
            return 'RYO_HEAD'
        if st == '逃':
            return 'ESCAPE_HEAD'
        return 'OTHER_HEAD'
    if pos == 2:
        return 'BANTE'
    if pos is not None and pos >= 3:
        return 'THIRD_PLUS'
    return 'UNKNOWN'


def coarse_role(role):
    if role in {'RYO_HEAD', 'ESCAPE_HEAD', 'OTHER_HEAD'}:
        return 'HEAD'
    return role


def safe_rate(num, den):
    return float(num / den) if den else None


def top_counts(series: pd.Series, n=20):
    vc = series.value_counts(dropna=False)
    return [{'pattern': str(k), 'count': int(v)} for k, v in vc.head(n).items()]


def summarize_subset(x: pd.DataFrame):
    n = int(len(x))
    t = x[x.three_line_finish == 1]
    nt = int(len(t))
    out = {
        'races': n,
        'three_line_finishes': nt,
        'three_line_finish_rate': safe_rate(nt, n),
        'winner_role_distribution_within_three_line': {},
        'second_third_pair_distribution_within_three_line': {},
        'ordered_exact_role_patterns_top20': [],
        'ordered_coarse_role_patterns_top20': [],
        'target_patterns': {},
        'head_any_bante_bante': {},
    }
    if nt == 0:
        return out

    out['winner_role_distribution_within_three_line'] = {
        str(k): {
            'count': int(v),
            'share': float(v / nt),
            'incidence_all_races': float(v / n) if n else None,
        }
        for k, v in t.winner_role.value_counts().items()
    }
    out['second_third_pair_distribution_within_three_line'] = {
        str(k): {
            'count': int(v),
            'share': float(v / nt),
            'incidence_all_races': float(v / n) if n else None,
        }
        for k, v in t.second_third_pair.value_counts().items()
    }
    out['ordered_exact_role_patterns_top20'] = top_counts(t.ordered_roles, 20)
    out['ordered_coarse_role_patterns_top20'] = top_counts(t.ordered_roles_coarse, 20)

    for pat in TARGET_PATTERNS:
        c = int((t.ordered_roles == pat).sum())
        out['target_patterns'][pat] = {
            'count': c,
            'share_within_three_line': safe_rate(c, nt),
            'incidence_all_races': safe_rate(c, n),
        }

    headbb = t[
        t.winner_role.isin({'RYO_HEAD', 'ESCAPE_HEAD', 'OTHER_HEAD'}) &
        (t.second_role == 'BANTE') & (t.third_role == 'BANTE')
    ]
    c = int(len(headbb))
    out['head_any_bante_bante'] = {
        'count': c,
        'share_within_three_line': safe_rate(c, nt),
        'incidence_all_races': safe_rate(c, n),
        'winner_head_style_split': {
            str(k): int(v) for k, v in headbb.winner_role.value_counts().to_dict().items()
        },
    }
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
            bante_stronger_count = 0
            large_gap_count = 0
            has_ryo_head = 0
            for li, g in members_by_line.items():
                if len(g) < 2:
                    continue
                head, bante = g[0], g[1]
                delta = float(score[bante] - score[head])
                multi.append((li, head, bante, delta))
                if delta > 0:
                    bante_stronger_count += 1
                if abs(delta) >= 2:
                    large_gap_count += 1
                if style.get(head) == '両':
                    has_ryo_head = 1
            if not multi:
                skipped['no_multi_rider_line'] += 1
                continue

            score_order = sorted(frames, key=lambda x: (-score[x], x))
            top3_score = score_order[:min(3, len(score_order))]
            top3_score_distinct_lines = len({line_of[x] for x in top3_score})

            actual_lines = [line_of[x] for x in actual]
            three_line_finish = int(len(set(actual_lines)) == 3)
            roles = [role_of(x, pos_of, size_of, style) for x in actual]
            coarse = [coarse_role(r) for r in roles]

            rows.append({
                'month': month,
                'race_id': rid,
                'three_line_finish': three_line_finish,
                'actual_distinct_lines': len(set(actual_lines)),
                'top3_score_distinct_lines': top3_score_distinct_lines,
                'bante_stronger_count': bante_stronger_count,
                'large_gap_count_ge2': large_gap_count,
                'has_ryo_head': has_ryo_head,
                'winner_role': roles[0],
                'second_role': roles[1],
                'third_role': roles[2],
                'second_third_pair': f'{roles[1]}>{roles[2]}',
                'ordered_roles': '>'.join(roles),
                'ordered_roles_coarse': '>'.join(coarse),
                'winner_score': score[actual[0]],
                'second_score': score[actual[1]],
                'third_score': score[actual[2]],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No usable races')
    df.to_csv(DETAIL, index=False)

    subsets = {
        'ALL_USABLE': df,
        'TOP3_SCORE_3LINES': df[df.top3_score_distinct_lines == 3],
        'TOP3_SCORE_3LINES_BANTE2PLUS': df[(df.top3_score_distinct_lines == 3) & (df.bante_stronger_count >= 2)],
        'TOP3_SCORE_3LINES_GAP2PLUS': df[(df.top3_score_distinct_lines == 3) & (df.large_gap_count_ge2 >= 2)],
        'TOP3_SCORE_3LINES_BOTH': df[(df.top3_score_distinct_lines == 3) & (df.bante_stronger_count >= 2) & (df.large_gap_count_ge2 >= 2)],
    }
    # Add the same signatures only where a multi-rider RYO head exists pre-race.
    for key, x in list(subsets.items()):
        subsets[key + '_RYO_HEAD_PRESENT'] = x[x.has_ryo_head == 1]

    summaries = {key: summarize_subset(x) for key, x in subsets.items()}

    # Relative enrichment of named target patterns vs ALL_USABLE, using incidence
    # per race (not conditional share) so both finish frequency and role composition matter.
    baseline = summaries['ALL_USABLE']
    enrichment = {}
    for key, s in summaries.items():
        if key == 'ALL_USABLE':
            continue
        e = {}
        for pat in TARGET_PATTERNS:
            b = baseline['target_patterns'][pat]['incidence_all_races']
            v = s['target_patterns'][pat]['incidence_all_races']
            e[pat] = float(v / b) if b and v is not None else None
        b = baseline['head_any_bante_bante']['incidence_all_races']
        v = s['head_any_bante_bante']['incidence_all_races']
        e['HEAD_ANY>BANTE>BANTE'] = float(v / b) if b and v is not None else None
        enrichment[key] = e

    payload = {
        'status': 'exploratory_three_line_role_composition',
        'warning': 'Same discovery data. This is an association/composition diagnostic, not OOS and not a causal or betting rule.',
        'coverage': {
            'full_context_rows_seen': int(context_rows),
            'usable_races': int(len(df)),
            'skipped': dict(skipped),
        },
        'definitions': {
            'three_line_finish': 'actual top3 riders occupy exactly three distinct true_line IDs',
            'role_taxonomy': {
                'RYO_HEAD': "multi-rider line_pos1 with running_style='両'",
                'ESCAPE_HEAD': "multi-rider line_pos1 with running_style='逃'",
                'OTHER_HEAD': 'multi-rider line_pos1 with other/unknown style',
                'BANTE': 'multi-rider line_pos2',
                'THIRD_PLUS': 'line_pos>=3',
                'SOLO': 'single-rider line',
            },
            'top3_score_3lines': 'the three highest race_score riders are on three distinct true_line IDs',
            'bante2plus': 'at least two multi-rider lines have bante race_score > line-head race_score',
            'gap2plus': 'at least two multi-rider lines have absolute head-bante race_score gap >=2 points',
        },
        'subsets': summaries,
        'target_pattern_incidence_enrichment_vs_all_usable': enrichment,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
