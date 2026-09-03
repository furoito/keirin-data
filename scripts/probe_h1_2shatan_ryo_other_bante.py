#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic discovery-data Reality Probe for 2-sha-tan.

Hypothesis inspired by the 3-line trifecta exploration:
  first rider = running_style '両'
  second rider = line_pos 2 on a DIFFERENT line
  both riders belong to multi-rider lines (no solo)

No odds filter is used to select tickets. Pair race_score quality is reported at:
  - no score cut
  - top 50% of all unordered 2-rider score-sum groups in the race
  - top 40%

To bound web requests, candidate races are selected BEFORE reading outcomes/odds by a
fixed race_id SHA256 rule: hash modulo 10 == 0. This is an exploratory probe, not OOS.
Kdreams result pages provide confirmed/final 2-sha-tan odds, not stored pre-close odds.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import popular_head_skip_v01 as base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
OUT = CTX / 'h1_2shatan_ryo_other_bante_probe_summary.json'
DETAIL = CTX / 'h1_2shatan_ryo_other_bante_probe_tickets.csv'
MONTHS = [f'2025_{m:02d}' for m in range(1, 13)] + [f'2026_{m:02d}' for m in range(1, 7)]
KD = 'https://keirin.kdreams.jp'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36'
SAMPLE_MOD = 10
SAMPLE_KEEP = 0
ODDS_BINS = [
    ('<5', 0.0, 5.0),
    ('5-10', 5.0, 10.0),
    ('10-20', 10.0, 20.0),
    ('20-50', 20.0, 50.0),
    ('50-100', 50.0, 100.0),
    ('100+', 100.0, float('inf')),
]


def strict_banum(v, valid):
    vals = v if isinstance(v, tuple) else (v,)
    for x in reversed(vals):
        m = re.fullmatch(r'([1-9])(?:\.0)?', str(x).strip())
        if m:
            n = int(m.group(1))
            if n in valid:
                return n
    return None


def strict_odd(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace(',', '')
    if s in {'', '-', '--', 'nan', 'NaN'}:
        return None
    m = re.fullmatch(r'(\d{1,5}(?:\.\d+)?)', s)
    if not m:
        return None
    x = float(m.group(1))
    return x if x > 0 else None


def parse_2shatan(html, valid):
    """Fail-closed parser for the exact ordered 2-sha-tan matrix.

    Kdreams result pages expose a matrix where columns are 1st-place rider numbers and
    rows are 2nd-place rider numbers. We require the complete n*(n-1) board.
    """
    valid = set(int(x) for x in valid)
    n = len(valid)
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return {}, 'read_html_failed'

    for df in tables:
        if df.empty or not (n <= len(df) <= n + 2):
            continue

        rider_cols = {}
        for col in df.columns:
            b = strict_banum(col, valid)
            if b is not None:
                rider_cols.setdefault(b, col)
        if set(rider_cols) != valid:
            continue

        non_rider_cols = [c for c in df.columns if c not in set(rider_cols.values())]
        row_map = None
        for col in non_rider_cols:
            found = {}
            duplicate = False
            for ri, v in enumerate(df[col].tolist()):
                b = strict_banum(v, valid)
                if b is None:
                    continue
                if b in found:
                    duplicate = True
                    break
                found[b] = ri
            if not duplicate and set(found) == valid:
                row_map = found
                break
        if row_map is None:
            continue

        board = {}
        ok = True
        for b2, ri in row_map.items():
            for b1, col in rider_cols.items():
                od = strict_odd(df.iloc[ri][col])
                if b1 == b2:
                    # Diagonal must not contain an odds value.
                    if od is not None:
                        ok = False
                        break
                    continue
                if od is None:
                    ok = False
                    break
                board[(int(b1), int(b2))] = float(od)
            if not ok:
                break
        if ok and len(board) == n * (n - 1):
            return board, 'full'

    return {}, 'not_found'


def fetch_board(item):
    rid = item['race_id']
    url = f"{KD}/{item['venue_slug']}/racedetail/{rid}/?pageType=result"
    last = 'fetch_failed'
    for attempt in range(3):
        try:
            r = requests.get(url, headers={'User-Agent': UA}, timeout=25)
            if r.status_code == 404:
                return rid, {}, '404', url
            if r.status_code == 200 and r.text:
                board, q = parse_2shatan(r.text, item['frames'])
                if q == 'full':
                    return rid, board, q, url
                last = q
            else:
                last = f'http_{r.status_code}'
        except Exception as e:
            last = type(e).__name__
        time.sleep(1.0 + attempt)
    return rid, {}, last, url


def sampled(rid):
    h = hashlib.sha256(str(rid).encode('utf-8')).hexdigest()
    return int(h[:16], 16) % SAMPLE_MOD == SAMPLE_KEEP


def actual_ordered_top2(pre):
    vals = []
    for r in pre.itertuples(index=False):
        try:
            pos = int(str(r.rank).strip())
            fn = int(float(r.banum))
        except Exception:
            continue
        if pos in (1, 2):
            vals.append((pos, fn))
    vals.sort()
    if [p for p, _ in vals] != [1, 2]:
        return None
    return tuple(fn for _, fn in vals)


def agg(df):
    n = int(len(df))
    stake = float(n)
    gross = float(df.loc[df.actual_hit == 1, 'odds'].sum()) if n else 0.0
    hits = int(df.actual_hit.sum()) if n else 0
    exp = float(df.market_p.sum()) if n else 0.0
    return {
        'tickets': n,
        'races': int(df.race_id.nunique()) if n else 0,
        'stake_units': stake,
        'gross_return_units': gross,
        'gross_roi_pct': float(100.0 * gross / stake) if stake else None,
        'net_roi_pct': float(100.0 * (gross - stake) / stake) if stake else None,
        'actual_hits': hits,
        'normalized_market_expected_hits': exp,
        'actual_over_normalized_market': float(hits / exp) if exp > 0 else None,
        'avg_ticket_odds': float(df.odds.mean()) if n else None,
        'median_ticket_odds': float(df.odds.median()) if n else None,
    }


def odds_slices(df):
    return {name: agg(df[(df.odds >= lo) & (df.odds < hi)]) for name, lo, hi in ODDS_BINS}


def view(df):
    return {'all_odds': agg(df), 'ticket_odds_bins': odds_slices(df)}


def load_candidates():
    race_items = []
    candidate_rows = []
    skipped = {}

    def skip(k):
        skipped[k] = skipped.get(k, 0) + 1

    for month in MONTHS:
        bp = DATA / f'{month}_keirin.csv'
        cp = CTX / f'{month}_races.csv'
        if not (bp.exists() and cp.exists()):
            skip('month_file_missing')
            continue
        b = pd.read_csv(bp, encoding='utf-8-sig', dtype={'race_id': str})
        c = pd.read_csv(cp, encoding='utf-8-sig', dtype={'race_id': str}).drop_duplicates('race_id', keep='last')
        b['race_id'] = b.race_id.astype(str)
        c['race_id'] = c.race_id.astype(str)
        if 'line_quality' in c.columns:
            c = c[c.line_quality.astype(str) == 'full']
        else:
            c = c[c.context_quality.astype(str) == 'full']
        bby = {str(k): g for k, g in b.groupby('race_id', sort=False)}

        for cr in c.to_dict('records'):
            rid = str(cr['race_id'])
            pre = bby.get(rid)
            if pre is None or pre.empty:
                skip('base_missing')
                continue
            lines = base.parse_true_line(cr.get('true_line'))
            if not lines:
                skip('line_unresolved')
                continue
            frames = sorted({int(x) for g in lines for x in g})
            if len(frames) < 4:
                skip('too_few_riders')
                continue

            line_of = {}
            pos_of = {}
            line_size = {}
            for li, g in enumerate(lines, 1):
                gg = [int(x) for x in g]
                for pos, fn in enumerate(gg, 1):
                    line_of[fn] = li
                    pos_of[fn] = pos
                    line_size[fn] = len(gg)

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
            if set(frames) - set(score):
                skip('score_missing')
                continue

            pair_groups = list(itertools.combinations(frames, 2))
            sums = {g: float(score[g[0]] + score[g[1]]) for g in pair_groups}
            ordered = sorted(pair_groups, key=lambda g: (-sums[g], g))
            rank_of = {g: i + 1 for i, g in enumerate(ordered)}
            n_groups = len(ordered)
            actual = actual_ordered_top2(pre)
            if actual is None:
                skip('result_missing')
                continue

            candidates = []
            for b1 in frames:
                if style.get(b1) != '両' or line_size.get(b1, 0) <= 1:
                    continue
                for b2 in frames:
                    if b1 == b2:
                        continue
                    if pos_of.get(b2) != 2 or line_size.get(b2, 0) <= 1:
                        continue
                    if line_of.get(b1) == line_of.get(b2):
                        continue
                    key = tuple(sorted((b1, b2)))
                    pct = float(rank_of[key] / n_groups)
                    row = {
                        'month': month,
                        'race_id': rid,
                        'date': str(cr.get('date', '')),
                        'venue_slug': str(cr.get('venue_slug', '')),
                        'race_no': cr.get('race_no'),
                        'b1': int(b1),
                        'b2': int(b2),
                        'b1_line_pos': int(pos_of.get(b1, 0)),
                        'b2_line_pos': int(pos_of.get(b2, 0)),
                        'b1_score': float(score[b1]),
                        'b2_score': float(score[b2]),
                        'pair_score_sum': float(sums[key]),
                        'pair_score_percentile': pct,
                        'actual_hit': int(actual == (b1, b2)),
                    }
                    candidates.append(row)
                    candidate_rows.append(row)

            if candidates:
                race_items.append({
                    'race_id': rid,
                    'venue_slug': str(cr.get('venue_slug', '')),
                    'frames': frames,
                    'candidate_count': len(candidates),
                })

    return pd.DataFrame(candidate_rows), race_items, skipped


def main():
    cand, race_items, skipped = load_candidates()
    if cand.empty:
        raise SystemExit('No candidate 2-sha-tan tickets')

    # Race-level deterministic sample, fixed before any 2-sha-tan odds are read.
    race_by_id = {x['race_id']: x for x in race_items}
    sample_items = [x for rid, x in race_by_id.items() if sampled(rid)]

    fetched = {}
    status_counts = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_board, item): item['race_id'] for item in sample_items}
        for fut in as_completed(futs):
            rid, board, status, url = fut.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == 'full':
                fetched[rid] = (board, url)

    rows = []
    for r in cand.to_dict('records'):
        rid = str(r['race_id'])
        if rid not in fetched:
            continue
        board, url = fetched[rid]
        od = board.get((int(r['b1']), int(r['b2'])))
        if od is None or od <= 0:
            continue
        z = sum(1.0 / x for x in board.values() if x > 0)
        if z <= 0:
            continue
        x = dict(r)
        x['odds'] = float(od)
        x['market_p'] = float((1.0 / od) / z)
        x['odds_source_url'] = url
        x['snapshot_kind'] = 'confirmed_final'
        rows.append(x)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f'No fetched complete candidate tickets: {status_counts}')

    df = df.sort_values(['date', 'race_id', 'b1', 'b2']).reset_index(drop=True)
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAIL, index=False, encoding='utf-8-sig')

    views = {
        'NO_SCORE_CUT': view(df),
        'TOP50': view(df[df.pair_score_percentile <= 0.50]),
        'TOP40': view(df[df.pair_score_percentile <= 0.40]),
    }
    payload = {
        'status': 'exploratory_2shatan_ryo_other_line_bante_deterministic_probe',
        'hypothesis': "2-sha-tan first='両', second=other-line position-2 rider may be underpriced when both are members of multi-rider lines.",
        'fixed_structural_filters': [
            "1st candidate running_style='両'",
            '2nd candidate line_pos=2',
            'different true_line IDs',
            'both selected riders belong to multi-rider lines',
        ],
        'score_quality_views': ['NO_SCORE_CUT', 'TOP50 pair score-sum percentile', 'TOP40 pair score-sum percentile'],
        'explicitly_not_used': ['odds filter for candidate selection', 'individual score rank', 'score gap', 'same-line tickets'],
        'sample_rule': 'sha256(race_id) first64bits modulo 10 == 0; applied at race level before fetching 2-sha-tan odds',
        'discovery_period': '2025-01 through 2026-06',
        'price_semantics': 'Kdreams result-page confirmed/final 2-sha-tan odds; not a stored pre-close snapshot',
        'warning': 'Hypothesis was inspired by prior trifecta discovery and this uses a deterministic subset of the same discovery period. This is a Reality Probe, not OOS proof.',
        'candidate_races_before_sampling': int(len(race_by_id)),
        'candidate_tickets_before_sampling': int(len(cand)),
        'sampled_races_requested': int(len(sample_items)),
        'complete_2shatan_boards': int(len(fetched)),
        'fetch_status_counts': status_counts,
        'candidate_tickets_with_complete_board': int(len(df)),
        'views': views,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
