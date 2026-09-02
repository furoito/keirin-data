#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a persistent performance ledger for the frozen popular-head-skip test.

The ledger records monthly and cumulative:
- scorable BET races
- head-bust races
- exact candidate-set races within head-busts
- trifecta hits
- stake / payout / profit
- ROI = payout / stake * 100

This is reporting only. It never changes selection or ordering rules.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

CTX = Path('keirin_data/strategy_context')
OUT_CSV = CTX / 'popular_head_skip_roi_ledger.csv'
OUT_JSON = CTX / 'popular_head_skip_roi_ledger.json'

SOURCES = [
    ('2026-07', CTX / 'single_bonus_order_july_summary.json'),
    ('2026-08', CTX / 'pop2_plus3_order_summary.json'),
]


def load_month(month: str, path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    all_bets = d.get('all_bets', {})
    hb = d.get('head_bust_only', {})
    exact = d.get('head_bust_and_set_exact', {})
    stake = int(all_bets.get('stake_yen') or 0)
    pay = int(all_bets.get('pay_yen') or 0)
    return {
        'month': month,
        'bet_races': int(all_bets.get('n') or 0),
        'head_bust_races': int(hb.get('n') or 0),
        'head_bust_rate_pct': (100.0 * int(hb.get('n') or 0) / int(all_bets.get('n') or 1)),
        'head_bust_set_exact_races': int(exact.get('set_match_n') or 0),
        'trifecta_hits': int(all_bets.get('bet_hit_n') or 0),
        'stake_yen': stake,
        'payout_yen': pay,
        'profit_yen': pay - stake,
        'roi_pct': (100.0 * pay / stake) if stake else None,
    }


def main():
    rows = [r for month, path in SOURCES if (r := load_month(month, path)) is not None]
    if not rows:
        raise SystemExit('no monthly summary files found')
    df = pd.DataFrame(rows).sort_values('month').reset_index(drop=True)
    df['cumulative_bet_races'] = df['bet_races'].cumsum()
    df['cumulative_head_bust_races'] = df['head_bust_races'].cumsum()
    df['cumulative_set_exact_races'] = df['head_bust_set_exact_races'].cumsum()
    df['cumulative_trifecta_hits'] = df['trifecta_hits'].cumsum()
    df['cumulative_stake_yen'] = df['stake_yen'].cumsum()
    df['cumulative_payout_yen'] = df['payout_yen'].cumsum()
    df['cumulative_profit_yen'] = df['profit_yen'].cumsum()
    df['cumulative_roi_pct'] = df.apply(
        lambda r: 100.0 * r.cumulative_payout_yen / r.cumulative_stake_yen
        if r.cumulative_stake_yen else None,
        axis=1,
    )
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    payload = {
        'definition': 'ROI = payout / stake * 100',
        'rule_version': 'candidate selection frozen v0.1b; trifecta order uses single-count +3 popular pos2, +1 popular pos3, then adjusted score only',
        'months': df.to_dict('records'),
        'latest_cumulative': df.iloc[-1].to_dict(),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
