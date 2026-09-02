#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen OH_HIGH + MKT1 for one month.

Prerequisite: validate_dual_logic_month.py has already produced
strategy_context/dual_logic_<month>_results.csv using frozen pre-race logic.

MKT1 is frozen as:
- take OH_HIGH candidate families exactly as produced pre-race
- enumerate every trifecta permutation for those 3-rider sets
- discard odds < 30x
- buy exactly one ticket: the lowest odds among remaining permutations
  (highest market support among eligible >=30x tickets)
- 100 yen stake

No result data is used to choose the ticket.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'keirin_data'/'strategy_context'
MIN_ODDS=30.0; STAKE=100


def parse_actual(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return None
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None


def parse_sets(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for part in str(v).split('|'):
        part=part.strip()
        if not part: continue
        try:s=tuple(sorted(int(float(x)) for x in part.split('-')))
        except Exception:continue
        if len(s)==3 and len(set(s))==3 and s not in out:out.append(s)
    return out


def odds_map(d):
    out={}
    for r in d.itertuples(index=False):
        try:k=(int(float(r.b1)),int(float(r.b2)),int(float(r.b3)));o=float(r.odds_decimal)
        except Exception:continue
        if o>0:out[k]=o
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--month',required=True);a=ap.parse_args();m=a.month
    rp=CTX/f'dual_logic_{m}_results.csv';op=CTX/f'{m}_odds_3rentan.csv'
    res=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(op,encoding='utf-8-sig',dtype={'race_id':str})
    res=res[res.variant.astype(str)=='OH_HIGH'].copy();res['race_id']=res.race_id.astype(str);odds['race_id']=odds.race_id.astype(str)
    ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    rows=[]
    for rr in res.itertuples(index=False):
        rid=str(rr.race_id);og=ob.get(rid)
        if og is None:continue
        om=odds_map(og);sets=parse_sets(getattr(rr,'candidate_sets',None));eligible=[]
        for s in sets:
            for p in itertools.permutations(s,3):
                o=om.get(p)
                if o is not None and o>=MIN_ODDS:eligible.append((o,p))
        eligible=sorted(set(eligible),key=lambda x:(x[0],x[1]))
        pick=eligible[0] if eligible else None
        act=parse_actual(getattr(rr,'actual',''))
        hit=bool(pick and act and pick[1]==act)
        stake=STAKE if pick else 0;pay=int(round(pick[0]*STAKE)) if hit else 0
        try:hb=int(float(rr.head_bust)) if not pd.isna(rr.head_bust) else None
        except Exception:hb=None
        try:sm=int(float(rr.set_match)) if not pd.isna(rr.set_match) else None
        except Exception:sm=None
        rows.append({'month':m,'race_id':rid,'target':getattr(rr,'target',''),'head_bust':hb,
                     'candidate_sets':getattr(rr,'candidate_sets',''),'actual':getattr(rr,'actual',''),'set_match':sm,
                     'eligible_count':len(eligible),'pick':'-'.join(map(str,pick[1])) if pick else '',
                     'pick_odds':pick[0] if pick else None,'bet':int(pick is not None),'hit':int(hit) if act else None,
                     'stake':stake,'pay':pay})
    out=pd.DataFrame(rows);out.to_csv(CTX/f'oh_high_mkt1_{m}_results.csv',index=False,encoding='utf-8-sig')
    sc=out[out.actual.astype(str)!=''] if len(out) else out
    hb=sc[pd.to_numeric(sc.head_bust,errors='coerce')==1] if len(sc) else sc
    st=int(sc.stake.sum()) if len(sc) else 0;py=int(sc.pay.sum()) if len(sc) else 0
    payload={'month':m,'rule':'frozen OH_HIGH candidate sets + MKT1 lowest odds among >=30x; 100 yen; 1 ticket',
             'scorable_races':len(sc),'bet_races':int(sc.bet.sum()) if len(sc) else 0,
             'head_bust_races':len(hb),'head_bust_bet_races':int(hb.bet.sum()) if len(hb) else 0,
             'candidate_set_exact_head_bust':int(pd.to_numeric(hb.set_match,errors='coerce').fillna(0).sum()) if len(hb) else 0,
             'hits':int(pd.to_numeric(sc.hit,errors='coerce').fillna(0).sum()) if len(sc) else 0,
             'stake_yen':st,'pay_yen':py,'profit_yen':py-st,'roi_pct':100*py/st if st else None}
    (CTX/f'oh_high_mkt1_{m}_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
