#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate BASE and OH_HIGH side-by-side for one month.

Uses the same structural filters, ordering rule (+3 pop pos2, +1 pop pos3),
30x threshold, and max2 tickets. The only candidate-selection difference is:
BASE    : other-line head tier=1
OH_HIGH : other-line head tier=3 (same as popular-line pos3+ when score gap<3)
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # noqa: F401
import evaluate_other_head_promotion_oos as cmp

DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'; VARS=['BASE','OH_HIGH']


def summarize(x):
    sc=x[x.actual.astype(str)!=''] if not x.empty else x
    hb=sc[sc.head_bust==1] if not sc.empty else sc
    b=sc[sc.bet==1] if not sc.empty else sc
    bhb=b[b.head_bust==1] if not b.empty else b
    st=int(b.stake.sum()) if len(b) else 0; py=int(b.pay.sum()) if len(b) else 0
    return {'scorable_prefilter_races':len(sc),'head_bust_prefilter_races':len(hb),
            'candidate_set_exact_head_bust_n':int(hb.set_match.fillna(0).sum()) if len(hb) else 0,
            'candidate_set_exact_head_bust_pct':100*hb.set_match.fillna(0).mean() if len(hb) else None,
            'bet_races':len(b),'head_bust_bet_races':len(bhb),
            'set_exact_head_bust_bets':int(bhb.set_match.fillna(0).sum()) if len(bhb) else 0,
            'trifecta_hits':int(b.bet_hit.fillna(0).sum()) if len(b) else 0,
            'stake_yen':st,'pay_yen':py,'profit_yen':py-st,'roi_pct':100*py/st if st else None}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--month',required=True); a=ap.parse_args(); month=a.month
    race=pd.read_csv(DATA/f'{month}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{month}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{month}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds): d['race_id']=d.race_id.astype(str)
    use=ctx.copy()
    if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
    if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    repb=cmp.load_existing_repairs(month)
    rows=[]; new_repairs=[]
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id); full=rb.get(rid); og=ob.get(rid)
        if full is None or og is None: continue
        pre=full[['race_id','banum','race_score']].copy()
        state,why=cmp.prefilter(rid,pre,cr,og)
        if state is None: continue
        ds={v:cmp.variant_decision(state,v) for v in VARS}
        # Freeze both decisions before inspecting result.
        act=cmp.actual_order(full); source='monthly'
        if act is None and rid in repb:
            act=cmp.actual_order(repb[rid]); source='repair_existing'
        if act is None:
            got=cmp.fetch_result(rid,cr)
            if got:
                for x in got: x['month']=month
                new_repairs.extend(got); act=cmp.actual_order(pd.DataFrame(got)); source='repair_fetched'
        aset=frozenset(act) if act else None
        hb=int(state['target'].frame_no not in aset) if aset is not None else None
        for v,d in ds.items():
            sm=int(aset in d['sets']) if aset is not None else None
            hit=next(((o,od) for o,od in d['bets'] if act==o),None) if act else None
            st=len(d['bets'])*base.STAKE_YEN; py=int(round(hit[1]*base.STAKE_YEN)) if hit else 0
            rows.append({'month':month,'race_id':rid,'variant':v,'target':state['target'].frame_no,'head_bust':hb,
                         'result_source':source,'candidate_sets':'|'.join('-'.join(map(str,sorted(s))) for s in d['sets']),
                         'orders':'|'.join('-'.join(map(str,o)) for o in d['orders']),
                         'bets':'|'.join('-'.join(map(str,o)) for o,_ in d['bets']),
                         'actual':'-'.join(map(str,act)) if act else '','set_match':sm,'bet':int(bool(d['bets'])),
                         'bet_hit':int(hit is not None) if act else None,'stake':st,'pay':py if act else 0,'reason':d['reason']})
    out=pd.DataFrame(rows); out.to_csv(CTX/f'dual_logic_{month}_results.csv',index=False,encoding='utf-8-sig')
    if new_repairs:
        pd.DataFrame(new_repairs).drop_duplicates(['race_id','banum']).to_csv(CTX/f'dual_logic_{month}_result_repairs.csv',index=False,encoding='utf-8-sig')
    payload={'month':month,'context_full_races':len(use),'rules':{
        'BASE':'current candidate tiers','OH_HIGH':'other-line head promoted to tier3 when within 3 points',
        'shared':'v0.1b popular line; 3-point boundary; order +3 pop pos2/+1 pop pos3 once; >=30x; max2'},'variants':{}}
    for v in VARS: payload['variants'][v]=summarize(out[out.variant==v])
    # Direct gain/loss on head-bust candidate exactness.
    z=out[out.head_bust==1].pivot_table(index='race_id',columns='variant',values='set_match',aggfunc='first').dropna()
    if len(z):
        b=z.BASE==1; h=z.OH_HIGH==1
        payload['OH_HIGH_vs_BASE']={'gain_exact':int((~b & h).sum()),'loss_exact':int((b & ~h).sum()),'net_exact':int((~b & h).sum()-(b & ~h).sum()),'head_bust_compared':len(z)}
    (CTX/f'dual_logic_{month}_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
