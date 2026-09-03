#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-part diagnostic for the ryo-head / line-correlation hypothesis.

Exploratory only; same discovery data, not OOS.

Tests
1) Ryo line-head relative score band -> winning-method distribution when available.
   Winning method is treated only as an outcome diagnostic, not a cause.
2) Ryo line-head relative score band -> own-bante top3 accompaniment rate,
   conditional on the ryo head winning.
3) Ryo-head strength + own-bante weakness -> actual top3 line dispersion,
   conditional on the ryo head winning.
4) Market comparison with the same race and same ryo head fixed first:
   a) OTHER_LINE_PAIR: other line head + its bante occupy 2nd/3rd (either order)
      with a separate split where the other line head style is '逃'.
   b) CROSS_BANTES: bantes from two different other multi-rider lines occupy 2nd/3rd
      (either order).
   Compare normalized market probability, actual/market calibration, ROI, and a
   nearest-score matched price comparison within race/head.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_ryo_head_line_collapse_four_tests_summary.json'

HEAD_BANDS = [
    ('LE_0', float('-inf'), 0.0),
    ('0_TO_+2', 0.0, 2.0),
    ('+2_TO_+4', 2.0, 4.0),
    ('GE_+4', 4.0, float('inf')),
]
BANTE_BANDS = [
    ('LT_-3', float('-inf'), -3.0),
    ('-3_TO_-1', -3.0, -1.0),
    ('-1_TO_+1', -1.0, 1.0),
    ('GE_+1', 1.0, float('inf')),
]
MATCH_TOL = 0.50


def band_name(v, bands):
    for name, lo, hi in bands:
        if v >= lo and v < hi:
            return name
    return 'UNKNOWN'


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


def summarize_ticket_df(x):
    n = int(len(x))
    if not n:
        return {
            'tickets': 0, 'races': 0, 'actual_hits': 0,
            'normalized_market_expected_hits': 0.0,
            'actual_over_normalized_market': None,
            'gross_roi_pct': None, 'avg_odds': None, 'median_odds': None,
            'avg_market_p': None, 'median_market_p': None,
        }
    hits = int(x.actual_hit.sum())
    exp = float(x.market_p.sum())
    gross = float(x.loc[x.actual_hit == 1, 'odds'].sum())
    return {
        'tickets': n,
        'races': int(x.race_id.nunique()),
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'gross_roi_pct': float(100.0 * gross / n),
        'avg_odds': float(x.odds.mean()),
        'median_odds': float(x.odds.median()),
        'avg_market_p': float(x.market_p.mean()),
        'median_market_p': float(x.market_p.median()),
    }


def possible_kimarite_columns(cols):
    names = []
    for c in cols:
        s = str(c).lower()
        if ('kimarite' in s or 'winning_technique' in s or 'win_technique' in s or
                '決まり手' in str(c)):
            names.append(c)
    return names


def extract_kimarite(pre, winner_fn, candidates):
    if not candidates:
        return None
    row = pre[pd.to_numeric(pre['banum'], errors='coerce') == winner_fn]
    if row.empty:
        return None
    r = row.iloc[0]
    for c in candidates:
        val = str(r.get(c, '')).strip()
        if val and val.lower() != 'nan':
            return val
    return None


def main():
    skipped = Counter()
    winner_rows = []
    ticket_rows = []
    kimarite_cols_seen = set()
    available_context_rows = 0

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
        available_context_rows += int(len(use))

        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}
        oby = {str(k): g for k, g in o.groupby('race_id', sort=False)}
        month_kcols = possible_kimarite_columns(b.columns)
        kimarite_cols_seen.update(map(str, month_kcols))

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
                skipped['zero_mass'] += 1
                continue

            line_of, pos_of, size_of, members_by_line = {}, {}, {}, {}
            for li, g in enumerate(lines, 1):
                g2 = [int(x) for x in g]
                members_by_line[li] = g2
                for pos, fn in enumerate(g2, 1):
                    line_of[fn] = li
                    pos_of[fn] = pos
                    size_of[fn] = len(g2)

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

            race_mean = float(np.mean([score[x] for x in frames]))
            actual_lines = [line_of.get(x) for x in actual]
            actual_distinct_lines = len(set(actual_lines)) if None not in actual_lines else None

            ryo_heads = [x for x in frames if style.get(x) == '両' and pos_of.get(x) == 1 and size_of.get(x, 0) > 1]
            for a in ryo_heads:
                li_a = line_of[a]
                own_members = members_by_line[li_a]
                own_bante = own_members[1] if len(own_members) >= 2 else None
                if own_bante is None:
                    continue
                head_delta = float(score[a] - race_mean)
                bante_delta = float(score[own_bante] - race_mean)
                head_band = band_name(head_delta, HEAD_BANDS)
                bante_band = band_name(bante_delta, BANTE_BANDS)

                # Tests 1-3 are conditional on the ryo head actually winning.
                if actual[0] == a:
                    own_bante_top3 = int(own_bante in actual)
                    kimarite = extract_kimarite(pre, a, month_kcols)
                    winner_rows.append({
                        'month': month,
                        'race_id': rid,
                        'head': a,
                        'own_bante': own_bante,
                        'head_delta': head_delta,
                        'own_bante_delta': bante_delta,
                        'head_minus_bante': float(score[a] - score[own_bante]),
                        'head_band': head_band,
                        'bante_band': bante_band,
                        'own_bante_top3': own_bante_top3,
                        'actual_distinct_lines': actual_distinct_lines,
                        'three_distinct_lines': int(actual_distinct_lines == 3),
                        'kimarite': kimarite,
                    })

                # Test 4: same race/head, A fixed first in tickets.
                # 4a: other-line head + own bante, either 2nd/3rd order.
                for li_b, g in members_by_line.items():
                    if li_b == li_a or len(g) < 2:
                        continue
                    b_head, b_bante = g[0], g[1]
                    for p2, p3 in [(b_head, b_bante), (b_bante, b_head)]:
                        od = tri.get((a, p2, p3))
                        if od is None or od <= 0:
                            continue
                        trio_mean_delta = float(np.mean([score[a] - race_mean, score[p2] - race_mean, score[p3] - race_mean]))
                        ticket_rows.append({
                            'month': month, 'race_id': rid, 'head': a,
                            'type': 'OTHER_LINE_PAIR',
                            'subtype': 'OTHER_LINE_ESCAPE_PAIR' if style.get(b_head) == '逃' else 'OTHER_LINE_NON_ESCAPE_PAIR',
                            'ticket': f'{a}-{p2}-{p3}',
                            'odds': float(od), 'market_p': float((1.0 / od) / z),
                            'actual_hit': int((a, p2, p3) == actual),
                            'head_delta': head_delta,
                            'own_bante_delta': bante_delta,
                            'trio_mean_delta': trio_mean_delta,
                        })

                # 4b: bantes from two distinct other multi-rider lines, either order.
                other_bantes = []
                for li_b, g in members_by_line.items():
                    if li_b == li_a or len(g) < 2:
                        continue
                    other_bantes.append((li_b, g[1]))
                for (li1, b1), (li2, b2) in itertools.combinations(other_bantes, 2):
                    if li1 == li2:
                        continue
                    for p2, p3 in [(b1, b2), (b2, b1)]:
                        od = tri.get((a, p2, p3))
                        if od is None or od <= 0:
                            continue
                        trio_mean_delta = float(np.mean([score[a] - race_mean, score[p2] - race_mean, score[p3] - race_mean]))
                        ticket_rows.append({
                            'month': month, 'race_id': rid, 'head': a,
                            'type': 'CROSS_BANTES', 'subtype': 'CROSS_BANTES',
                            'ticket': f'{a}-{p2}-{p3}',
                            'odds': float(od), 'market_p': float((1.0 / od) / z),
                            'actual_hit': int((a, p2, p3) == actual),
                            'head_delta': head_delta,
                            'own_bante_delta': bante_delta,
                            'trio_mean_delta': trio_mean_delta,
                        })

    wdf = pd.DataFrame(winner_rows)
    tdf = pd.DataFrame(ticket_rows)
    if wdf.empty or tdf.empty:
        raise SystemExit('Insufficient rows')

    # Test 1: descriptive winning method distribution by head strength.
    test1 = {}
    for hb in [x[0] for x in HEAD_BANDS]:
        x = wdf[wdf.head_band == hb]
        known = x[x.kimarite.notna() & (x.kimarite.astype(str).str.len() > 0)]
        test1[hb] = {
            'winner_cases': int(len(x)),
            'kimarite_known': int(len(known)),
            'kimarite_counts': {str(k): int(v) for k, v in known.kimarite.value_counts().to_dict().items()},
        }

    # Test 2: own-bante accompaniment by head strength.
    test2 = {}
    for hb in [x[0] for x in HEAD_BANDS]:
        x = wdf[wdf.head_band == hb]
        n = len(x)
        test2[hb] = {
            'ryo_head_wins': int(n),
            'own_bante_top3': int(x.own_bante_top3.sum()) if n else 0,
            'own_bante_top3_rate': float(x.own_bante_top3.mean()) if n else None,
            'three_distinct_lines_rate': float(x.three_distinct_lines.mean()) if n else None,
            'avg_own_bante_delta': float(x.own_bante_delta.mean()) if n else None,
        }

    # Test 3: head strength x own-bante strength -> line dispersion.
    test3 = {}
    for hb in [x[0] for x in HEAD_BANDS]:
        row = {}
        for bb in [x[0] for x in BANTE_BANDS]:
            x = wdf[(wdf.head_band == hb) & (wdf.bante_band == bb)]
            n = len(x)
            row[bb] = {
                'ryo_head_wins': int(n),
                'own_bante_top3_rate': float(x.own_bante_top3.mean()) if n else None,
                'three_distinct_lines_rate': float(x.three_distinct_lines.mean()) if n else None,
                'avg_head_delta': float(x.head_delta.mean()) if n else None,
                'avg_own_bante_delta': float(x.own_bante_delta.mean()) if n else None,
            }
        test3[hb] = row

    # Test 4 aggregate ticket pricing/calibration.
    test4_agg = {
        'OTHER_LINE_PAIR': summarize_ticket_df(tdf[tdf.type == 'OTHER_LINE_PAIR']),
        'OTHER_LINE_ESCAPE_PAIR': summarize_ticket_df(tdf[tdf.subtype == 'OTHER_LINE_ESCAPE_PAIR']),
        'CROSS_BANTES': summarize_ticket_df(tdf[tdf.type == 'CROSS_BANTES']),
    }
    # Also the previously interesting head-strength region, explicitly labeled post-hoc.
    mid = tdf[(tdf.head_delta >= 0.0) & (tdf.head_delta < 4.0)]
    test4_mid = {
        'OTHER_LINE_PAIR': summarize_ticket_df(mid[mid.type == 'OTHER_LINE_PAIR']),
        'OTHER_LINE_ESCAPE_PAIR': summarize_ticket_df(mid[mid.subtype == 'OTHER_LINE_ESCAPE_PAIR']),
        'CROSS_BANTES': summarize_ticket_df(mid[mid.type == 'CROSS_BANTES']),
    }

    # Nearest trio-mean-score matched market-price comparison within same race/head.
    # For every CROSS_BANTES ticket, find the nearest OTHER_LINE_PAIR ticket in the same
    # race/head with abs(trio_mean_delta difference)<=0.5. Matching is descriptive;
    # reuse is allowed and counts are reported.
    pair_pool = defaultdict(list)
    for r in tdf[tdf.type == 'OTHER_LINE_PAIR'].itertuples(index=False):
        pair_pool[(r.race_id, r.head)].append(r)
    matched = []
    for r in tdf[tdf.type == 'CROSS_BANTES'].itertuples(index=False):
        pool = pair_pool.get((r.race_id, r.head), [])
        if not pool:
            continue
        best = min(pool, key=lambda q: abs(float(q.trio_mean_delta) - float(r.trio_mean_delta)))
        diff = abs(float(best.trio_mean_delta) - float(r.trio_mean_delta))
        if diff > MATCH_TOL:
            continue
        matched.append({
            'race_id': r.race_id,
            'head': int(r.head),
            'score_delta_abs_diff': float(diff),
            'cross_market_p': float(r.market_p),
            'pair_market_p': float(best.market_p),
            'cross_odds': float(r.odds),
            'pair_odds': float(best.odds),
            'cross_over_pair_market_p': float(r.market_p / best.market_p) if best.market_p > 0 else None,
            'cross_over_pair_odds': float(r.odds / best.odds) if best.odds > 0 else None,
        })
    mdf = pd.DataFrame(matched)
    if len(mdf):
        matched_summary = {
            'matched_cross_tickets': int(len(mdf)),
            'unique_races': int(mdf.race_id.nunique()),
            'median_score_delta_abs_diff': float(mdf.score_delta_abs_diff.median()),
            'median_cross_over_pair_market_p': float(mdf.cross_over_pair_market_p.median()),
            'mean_cross_over_pair_market_p': float(mdf.cross_over_pair_market_p.mean()),
            'median_cross_over_pair_odds': float(mdf.cross_over_pair_odds.median()),
            'mean_cross_over_pair_odds': float(mdf.cross_over_pair_odds.mean()),
            'share_cross_lower_market_p_than_pair': float((mdf.cross_market_p < mdf.pair_market_p).mean()),
        }
    else:
        matched_summary = {'matched_cross_tickets': 0}

    payload = {
        'status': 'exploratory_ryo_head_line_collapse_four_tests',
        'warning': 'Same discovery data. Tests are diagnostic and partially post-hoc; no causal claim and no production threshold should be frozen from these results.',
        'usable_context_rows': int(available_context_rows),
        'skipped': dict(skipped),
        'definitions': {
            'ryo_head': "running_style='両' AND line_pos=1 AND multi-rider line",
            'head_delta': 'ryo-head race_score minus race mean race_score',
            'own_bante_delta': 'own-bante race_score minus race mean race_score',
            'test_1_causal_note': 'winning method is an outcome diagnostic only; it is not treated as causing bante loss',
            'test_4_other_line_pair': 'A ryo head fixed 1st; another line head+bante occupy 2nd/3rd in either order',
            'test_4_cross_bantes': 'A ryo head fixed 1st; bantes from two different other lines occupy 2nd/3rd in either order',
            'test_4_match': 'same race/head; nearest trio_mean_delta within 0.5 points; descriptive reused nearest match',
        },
        'kimarite_columns_seen': sorted(kimarite_cols_seen),
        'test_1_head_strength_to_winning_method': test1,
        'test_2_head_strength_to_own_bante_accompaniment': test2,
        'test_3_head_and_bante_strength_to_three_line_finish': test3,
        'test_4_market_pair_vs_cross_bantes': {
            'all_head_strengths': test4_agg,
            'head_delta_0_to_4_posthoc_context': test4_mid,
            'matched_price_comparison': matched_summary,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
