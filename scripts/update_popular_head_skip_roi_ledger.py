#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a persistent performance ledger for the frozen popular-head-skip test.

Only summaries produced by validate_fixed_logic_month.py are included, so every
month uses the same current ordering and >=30x gate. Older July/August diagnostics
are intentionally excluded until they are recomputed with this exact evaluator.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

CTX = Path('keirin_data/strategy_context')
OUT_CSV = CTX / 'popular_head_skip_roi_ledger.csv'
OUT_JSON = CTX / 'popular_head_skip_roi_ledger.json'
PAT = re.compile(r'^fixed_logic_(\d{4})_(\d{2})_summary\.json$')


def discover_sources():
    out=[]
    for p in sorted(CTX.glob('fixed_logic_*_summary.json')):
        m=PAT.match(p.name)
        if m:
            out.append((f'{m.group(1)}-{m.group(2)}',p))
    return out


def load_month(month: str, path: Path):
    d = json.loads(path.read_text(encoding='utf-8'))
    all_bets = d.get('all_bets', {})
    hb = d.get('head_bust_only', {})
    exact = d.get('head_bust_and_set_exact', {})
    stake = int(all_bets.get('stake_yen') or 0)
    pay = int(all_bets.get('pay_yen') or 0)
    n = int(all_bets.get('n') or 0)
    hbn = int(hb.get('n') or 0)
    return {
        'month': month,
        'context_full_races': int(d.get('context_full_races') or 0),
        'bet_races': n,
        'head_bust_races': hbn,
        'head_bust_rate_pct': (100.0 * hbn / n) if n else None,
        'head_bust_set_exact_races': int(exact.get('set_match_n') or 0),
        'trifecta_order_matches': int(all_bets.get('order_match_n') or 0),
        'trifecta_hits': int(all_bets.get('bet_hit_n') or 0),
        'stake_yen': stake,
        'payout_yen': pay,
        'profit_yen': pay - stake,
        'roi_pct': (100.0 * pay / stake) if stake else None,
    }


def main():
    sources=discover_sources()
    if not sources:
        raise SystemExit('no fixed_logic monthly summary files found')
    rows=[load_month(month,path) for month,path in sources]
    df=pd.DataFrame(rows).sort_values('month').reset_index(drop=True)
    for col in ['context_full_races','bet_races','head_bust_races','head_bust_set_exact_races',
                'trifecta_order_matches','trifecta_hits','stake_yen','payout_yen','profit_yen']:
        df[f'cumulative_{col}']=df[col].cumsum()
    df['cumulative_head_bust_rate_pct']=df.apply(
        lambda r: 100.0*r.cumulative_head_bust_races/r.cumulative_bet_races if r.cumulative_bet_races else None,axis=1)
    df['cumulative_set_exact_given_bust_pct']=df.apply(
        lambda r: 100.0*r.cumulative_head_bust_set_exact_races/r.cumulative_head_bust_races if r.cumulative_head_bust_races else None,axis=1)
    df['cumulative_roi_pct']=df.apply(
        lambda r: 100.0*r.cumulative_payout_yen/r.cumulative_stake_yen if r.cumulative_stake_yen else None,axis=1)
    df.to_csv(OUT_CSV,index=False,encoding='utf-8-sig')
    payload={
        'definition':'ROI = payout / stake * 100',
        'scope':'fixed_logic_* monthly summaries only; legacy July/Aug diagnostics excluded until recomputed with the exact same evaluator',
        'rule_version':'v0.1b candidate sets; single-count +3 popular pos2, +1 popular pos3; adjusted-score order; current-order >=30x; max2',
        'months':df.to_dict('records'),
        'latest_cumulative':df.iloc[-1].to_dict(),
    }
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
