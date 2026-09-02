#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate v0.1b with the frozen priority logic used for trifecta order.

Important:
- popular-line detection = v0.1b joint market mass
- candidate ranking/order = existing 3-point boundary + positional priority
- no post-hoc order model
- same-set adjacent ambiguity keeps the existing canonical exception: among the two
  structurally indistinguishable orders, buy the higher-odds eligible one
- third-slot A-B-CD may buy both, max 2
- 30x minimum remains fixed
- missing monthly result ranks are filled from diagnostic_result_repairs.csv ONLY
  after the pre-race decision is frozen
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # noqa: F401; patches detector

DATA = Path('keirin_data')
CTX = DATA / 'strategy_context'
MONTH = '2026_08'
OUT = CTX / 'priority_order_v01b_results.csv'
SUMMARY = CTX / 'priority_order_v01b_summary.json'


def parse_rank(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(str(v).strip()); i = int(f)
        return i if abs(f-i) < 1e-9 else None
    except Exception:
        return None


def actual_order(g):
    vals=[]
    for r in g.itertuples(index=False):
        p=parse_rank(getattr(r,'rank',None))
        try: fn=int(float(r.banum))
        except Exception: continue
        if p is not None and 1 <= p <= 3:
            vals.append((p,fn))
    vals.sort()
    return tuple(fn for _,fn in vals) if [p for p,_ in vals] == [1,2,3] else None


def main():
    race=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{MONTH}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    repairs=pd.read_csv(CTX/'diagnostic_result_repairs.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds,repairs): d['race_id']=d.race_id.astype(str)
    use=ctx.copy()
    use=use[use.context_quality.astype(str)=='full']
    use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}
    ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    repb={k:g for k,g in repairs.groupby('race_id',sort=False)}

    rows=[]
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id)
        if rid not in rb or rid not in ob: continue
        full=rb[rid]
        pre=full[['race_id','banum','race_score']].copy()
        d=base.decide(rid,pre,pd.Series(cr._asdict()),ob[rid])
        if d.action!='BET': continue

        act=actual_order(full); source='monthly'
        if act is None and rid in repb:
            act=actual_order(repb[rid]); source='repair'
        if act is None: continue

        candidates=d.candidate_orders or []
        bets=d.bets or []
        actual_set=frozenset(act)
        candidate_sets={frozenset(o) for o in candidates}
        natural=candidates[0] if candidates else None
        chosen_orders=[o for o,_ in bets]
        hit=next(((o,od) for o,od in bets if o==act),None)

        # Was the actual set found by the reconstruction?
        set_match=int(actual_set in candidate_sets)
        # Did the strict first priority order (before any ambiguity exception) match?
        natural_order_match=int(natural==act) if natural else 0
        # Did any structurally plausible candidate order match?
        candidate_order_match=int(act in candidates)
        # Did the actually purchased order(s) match?
        bet_order_match=int(act in chosen_orders)

        target=d.target
        head_bust=int(target not in actual_set) if target is not None else 0
        stake=len(bets)*base.STAKE_YEN
        pay=int(round(hit[1]*base.STAKE_YEN)) if hit else 0

        rows.append({
            'race_id':rid,'date':cr.date,'venue':cr.venue_slug,'race_no':cr.race_no,
            'line':cr.true_line,'target':target,'result_source':source,'head_bust':head_bust,
            'ranking':'-'.join(map(str,d.ranking or [])),
            'ambiguity':d.ambiguity or '',
            'candidate_orders':'|'.join('-'.join(map(str,o)) for o in candidates),
            'natural_priority_order':'-'.join(map(str,natural)) if natural else '',
            'bets':'|'.join('-'.join(map(str,o)) for o,_ in bets),
            'bet_odds':'|'.join(f'{od:.2f}' for _,od in bets),
            'actual':'-'.join(map(str,act)),
            'set_match':set_match,
            'natural_order_match':natural_order_match,
            'candidate_order_match':candidate_order_match,
            'bet_order_match':bet_order_match,
            'stake':stake,'pay':pay,
        })

    out=pd.DataFrame(rows)
    out.to_csv(OUT,index=False,encoding='utf-8-sig')
    hb=out[out.head_bust==1]
    exact=hb[hb.set_match==1]
    def block(x):
        n=len(x); stake=int(x.stake.sum()) if n else 0; pay=int(x.pay.sum()) if n else 0
        return {
            'n':n,
            'set_match_n':int(x.set_match.sum()) if n else 0,
            'natural_priority_order_match_n':int(x.natural_order_match.sum()) if n else 0,
            'candidate_order_match_n':int(x.candidate_order_match.sum()) if n else 0,
            'actual_bet_hit_n':int(x.bet_order_match.sum()) if n else 0,
            'stake_yen':stake,'pay_yen':pay,'roi_pct':(pay/stake*100 if stake else None),
        }
    summary={
        'bet_races_scorable':len(out),
        'all_bets':block(out),
        'head_bust_only':block(hb),
        'head_bust_and_set_exact':block(exact),
        'head_bust_set_exact_rows':exact[['date','venue','race_no','line','target','ranking','ambiguity','candidate_orders','natural_priority_order','bets','bet_odds','actual','natural_order_match','candidate_order_match','bet_order_match','pay']].to_dict('records'),
        'rule':'3-point boundary + positional priority determines natural order; only frozen ambiguity exceptions use odds; >=30x; max2',
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
