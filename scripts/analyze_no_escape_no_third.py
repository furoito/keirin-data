#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic: remove line position 3+ and optionally all running_style='逃'.

Frozen strategy is not changed. Uses only context_quality=full and price_usable races.
Compares:
- BASELINE: current popular_head_skip_v01 decision
- NO_THIRD: current gates/ranking, but line_pos >= 3 removed before ranking
- NO_ESCAPE_NO_THIRD: NO_THIRD plus every rider with running_style == '逃' removed
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base

DATA = Path('keirin_data')
CTX = DATA / 'strategy_context'
OUT_SUMMARY = CTX / 'no_escape_no_third_summary.json'
OUT_CASES = CTX / 'no_escape_no_third_cases.csv'
MONTHS = [f'2025_{m:02d}' for m in range(1,13)] + [f'2026_{m:02d}' for m in range(1,7)]


def style_map(pre_full: pd.DataFrame) -> dict[int, str]:
    out = {}
    for r in pre_full.itertuples(index=False):
        try:
            fn = int(float(r.banum))
        except Exception:
            continue
        out[fn] = str(getattr(r, 'running_style', '')).strip()
    return out


def filtered_decide(race_id: str, pre_full: pd.DataFrame, ctx_row: pd.Series,
                    og: pd.DataFrame, drop_third: bool, drop_escape: bool) -> base.Decision:
    lines = base.parse_true_line(ctx_row.get('true_line'))
    if not lines:
        return base.Decision(race_id, 'SKIP', 'line_unresolved')
    pre = pre_full[['race_id', 'banum', 'race_score']].copy()
    riders = base.make_riders(pre, lines)
    if len(riders) < 4:
        return base.Decision(race_id, 'SKIP', 'score_or_entry_missing')
    tri = base.odds_map(og)
    expected = len(riders) * (len(riders) - 1) * (len(riders) - 2)
    if len(tri) != expected:
        return base.Decision(race_id, 'SKIP', 'odds_board_incomplete')

    pop = base.detect_popular_group(riders, tri)
    if pop is None:
        return base.Decision(race_id, 'SKIP', 'popular_group_unresolved')
    pop_line, target, mass, is_solo = pop
    if is_solo:
        return base.Decision(race_id, 'SKIP', 'popular_group_is_solo', pop_line, target.frame_no, mass)

    outside = [r.race_score for r in riders if r.line_idx != pop_line]
    if not outside:
        return base.Decision(race_id, 'SKIP', 'no_rival_line', pop_line, target.frame_no, mass)
    if target.race_score >= max(outside) + base.SCORE_BOUNDARY:
        return base.Decision(race_id, 'SKIP', 'target_score_plus_3', pop_line, target.frame_no, mass)
    if base.popular_line_too_strong(riders, pop_line, target):
        return base.Decision(race_id, 'SKIP', 'popular_line_too_strong', pop_line, target.frame_no, mass)

    styles = style_map(pre_full)
    remaining = []
    for r in riders:
        if r.frame_no == target.frame_no:
            continue
        if drop_third and r.line_pos >= 3:
            continue
        if drop_escape and styles.get(r.frame_no, '') == '逃':
            continue
        remaining.append(r)
    if len(remaining) < 3:
        return base.Decision(race_id, 'SKIP', 'filter_left_lt3', pop_line, target.frame_no, mass)

    ranked, coherent = base.coherent_ranking(remaining, pop_line)
    if not coherent:
        return base.Decision(race_id, 'SKIP', 'ranking_cycle', pop_line, target.frame_no, mass)
    frames = [r.frame_no for r in ranked]
    orders, ambiguity = base.generate_orders(ranked, pop_line)
    if not orders:
        return base.Decision(race_id, 'SKIP', 'cannot_compress_to_two_bets', pop_line,
                             target.frame_no, mass, frames, ambiguity=ambiguity)

    eligible = [(o, tri[o]) for o in orders if o in tri and tri[o] >= base.TRIFECTA_MIN_ODDS]
    if ambiguity == 'third_slot':
        chosen = sorted(eligible, key=lambda x: x[1], reverse=True)[:2]
    elif len(eligible) > 1:
        chosen = [max(eligible, key=lambda x: x[1])]
    else:
        chosen = eligible[:1]
    if not chosen:
        return base.Decision(race_id, 'SKIP', 'odds_too_low', pop_line, target.frame_no, mass,
                             frames, orders, ambiguity=ambiguity)
    return base.Decision(race_id, 'BET', 'eligible', pop_line, target.frame_no, mass,
                         frames, orders, chosen, ambiguity)


def summarize(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {'bet_races': 0, 'tickets': 0, 'hits': 0, 'stake_yen': 0, 'return_yen': 0,
                'profit_yen': 0, 'roi_pct': None, 'race_hit_rate_pct': None,
                'head_bust_races': 0, 'head_bust_hit_races': 0, 'head_bust_set_cover_pct': None}
    z = rows[rows.action == 'BET'].copy()
    bet_races = len(z)
    tickets = int(z.tickets.sum()) if bet_races else 0
    hits = int(z.hit.sum()) if bet_races else 0
    stake = int(z.stake_yen.sum()) if bet_races else 0
    ret = int(z.return_yen.sum()) if bet_races else 0
    hb = z[z.head_bust == 1]
    hb_hits = int(hb.hit.sum()) if len(hb) else 0
    return {
        'bet_races': bet_races,
        'tickets': tickets,
        'hits': hits,
        'stake_yen': stake,
        'return_yen': ret,
        'profit_yen': ret - stake,
        'roi_pct': 100 * ret / stake if stake else None,
        'race_hit_rate_pct': 100 * hits / bet_races if bet_races else None,
        'avg_tickets_per_bet_race': tickets / bet_races if bet_races else None,
        'head_bust_races': len(hb),
        'head_bust_hit_races': hb_hits,
        'head_bust_set_cover_pct': 100 * hb_hits / len(hb) if len(hb) else None,
    }


def main():
    rows = []
    monthly_counts = Counter()
    for month in MONTHS:
        bp = DATA / f'{month}_keirin.csv'
        cp = CTX / f'{month}_races.csv'
        op = CTX / f'{month}_odds_3rentan.csv'
        if not (bp.exists() and cp.exists() and op.exists()):
            continue
        race = pd.read_csv(bp, encoding='utf-8-sig', dtype={'race_id': str})
        ctx = pd.read_csv(cp, encoding='utf-8-sig', dtype={'race_id': str})
        odds = pd.read_csv(op, encoding='utf-8-sig', dtype={'race_id': str})
        race['race_id'] = race.race_id.astype(str)
        ctx['race_id'] = ctx.race_id.astype(str)
        odds['race_id'] = odds.race_id.astype(str)
        use = ctx.copy()
        if 'context_quality' in use:
            use = use[use.context_quality.astype(str) == 'full']
        if 'price_usable' in use:
            use = use[use.price_usable.astype(str).str.lower().isin({'true', '1'})]
        use = use.drop_duplicates('race_id', keep='last')
        rb = {str(k): g for k, g in race.groupby('race_id', sort=False)}
        ob = {str(k): g for k, g in odds.groupby('race_id', sort=False)}
        monthly_counts[month] = len(use)

        for cr in use.itertuples(index=False):
            rid = str(cr.race_id)
            pre_full = rb.get(rid)
            og = ob.get(rid)
            if pre_full is None or og is None:
                continue
            ctxs = pd.Series(cr._asdict())
            pre = pre_full[['race_id', 'banum', 'race_score']].copy()
            variants = {
                'BASELINE': base.decide(rid, pre, ctxs, og),
                'NO_THIRD': filtered_decide(rid, pre_full, ctxs, og, True, False),
                'NO_ESCAPE_NO_THIRD': filtered_decide(rid, pre_full, ctxs, og, True, True),
            }
            act = base.actual_order(pre_full)
            for name, d in variants.items():
                bust = base.head_busted(pre_full, d.target)
                tickets = len(d.bets or []) if d.action == 'BET' else 0
                hit = 0
                payout = 0
                if d.action == 'BET' and act is not None:
                    for order, od in d.bets or []:
                        if tuple(order) == tuple(act):
                            hit = 1
                            payout += int(round(float(od) * base.STAKE_YEN))
                rows.append({
                    'month': month,
                    'race_id': rid,
                    'date': str(getattr(cr, 'date', '')),
                    'variant': name,
                    'action': d.action,
                    'reason': d.reason,
                    'target': d.target if d.target is not None else '',
                    'ranking': '-'.join(map(str, d.ranking or [])),
                    'bets': '|'.join('-'.join(map(str, o)) + f'@{od:.2f}' for o, od in (d.bets or [])),
                    'actual': '-'.join(map(str, act)) if act else '',
                    'head_bust': 1 if bust is True else (0 if bust is False else -1),
                    'tickets': tickets,
                    'hit': hit,
                    'stake_yen': tickets * base.STAKE_YEN,
                    'return_yen': payout,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CASES, index=False, encoding='utf-8-sig')
    payload = {
        'definition': {
            'baseline': 'current popular_head_skip_v01; popular line head already excluded',
            'no_third': 'baseline gates, then exclude all line_pos >= 3 before ranking',
            'no_escape_no_third': "NO_THIRD plus exclude every rider whose running_style is exactly '逃'",
            'quality_gate': 'context_quality=full AND price_usable=true',
            'odds_gate': f'3rentan >= {base.TRIFECTA_MIN_ODDS:.0f}x',
        },
        'months_used': sorted(monthly_counts.keys()),
        'usable_races_by_month': dict(sorted(monthly_counts.items())),
        'variants': {},
        'skip_reasons': {},
    }
    for name in ['BASELINE', 'NO_THIRD', 'NO_ESCAPE_NO_THIRD']:
        z = out[out.variant == name].copy()
        payload['variants'][name] = summarize(z)
        payload['skip_reasons'][name] = dict(Counter(z.loc[z.action != 'BET', 'reason'].astype(str)).most_common())
        by_month = {}
        for month, g in z.groupby('month', sort=True):
            by_month[month] = summarize(g)
        payload['variants'][name]['by_month'] = by_month

    OUT_SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
