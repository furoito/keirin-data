#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate market-based ordering with frozen candidate-set logic.

Candidate-set construction is NOT changed. For each pre-race candidate family produced by
BASE / OH_HIGH, examine all 6 trifecta permutations using the stored pre-race odds.

Two ordering policies are compared:
- MKT1: among permutations priced >= 30x, buy the single lowest-odds permutation
        (highest market-implied probability among eligible tickets).
- MKT2: same, but buy the two lowest-odds eligible permutations.

This preserves the existing 30x minimum and max-2-ticket constraint.
No result information is used to choose tickets; result is read only for scoring.
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
OUT=CTX/'market_ordering_results.csv'; SUMMARY=CTX/'market_ordering_summary.json'
VARS=['BASE','OH_HIGH']; POLICIES={'MKT1':1,'MKT2':2}
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]
MIN_ODDS=30.0; STAKE=100


def flag(v):
    try:
        x=float(v)
        return None if math.isnan(x) else int(x)
    except Exception:return None


def parse_actual(v):
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None


def parse_candidate_sets(v):
    if v is None or (isinstance(v,float) and pd.isna(v)):return []
    out=[]
    for part in str(v).split('|'):
        part=part.strip()
        if not part:continue
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


def load_results():
    parts=[]
    big=CTX/'other_head_promotion_oos_results.csv'
    if big.exists():
        d=pd.read_csv(big,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.isin(VARS)].copy())
    for month in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{month}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.isin(VARS)].copy())
    x=pd.concat(parts,ignore_index=True);x['race_id']=x.race_id.astype(str)
    return x.drop_duplicates(['month','race_id','variant'],keep='last')


def summarize(df):
    st=int(df.stake.sum()) if len(df) else 0;py=int(df.pay.sum()) if len(df) else 0
    sc=df[df.actual.astype(str)!=''] if len(df) else df
    hb=sc[pd.to_numeric(sc.head_bust,errors='coerce')==1] if len(sc) else sc
    return {'scorable_races':len(sc),'bet_races':int(sc.bet.sum()) if len(sc) else 0,
            'tickets':int(sc.ticket_count.sum()) if len(sc) else 0,'head_bust_races':len(hb),
            'head_bust_bet_races':int(hb.bet.sum()) if len(hb) else 0,
            'candidate_set_exact_head_bust':int(pd.to_numeric(hb.set_match,errors='coerce').fillna(0).sum()) if len(hb) else 0,
            'trifecta_hits':int(sc.hit.sum()) if len(sc) else 0,'stake_yen':st,'pay_yen':py,'profit_yen':py-st,
            'roi_pct':100*py/st if st else None}


def main():
    res=load_results(); rows=[]
    for month in MONTHS:
        sub=res[res.month.astype(str)==month]
        if sub.empty:continue
        op=CTX/f'{month}_odds_3rentan.csv'
        if not op.exists():continue
        odds=pd.read_csv(op,encoding='utf-8-sig',dtype={'race_id':str});odds['race_id']=odds.race_id.astype(str)
        ob={k:g for k,g in odds.groupby('race_id',sort=False)}
        for rr in sub.itertuples(index=False):
            rid=str(rr.race_id);og=ob.get(rid)
            if og is None:continue
            sets=parse_candidate_sets(getattr(rr,'candidate_sets',None))
            act=parse_actual(getattr(rr,'actual',''))
            om=odds_map(og)
            eligible=[]
            for s in sets:
                for p in itertools.permutations(s,3):
                    o=om.get(p)
                    if o is not None and o>=MIN_ODDS:eligible.append((o,p))
            eligible=sorted(set(eligible),key=lambda x:(x[0],x[1]))
            for pol,n in POLICIES.items():
                picks=eligible[:n]
                hit=next(((o,p) for o,p in picks if act==p),None) if act else None
                st=len(picks)*STAKE;py=int(round(hit[0]*STAKE)) if hit else 0
                rows.append({'month':month,'race_id':rid,'variant':rr.variant,'policy':pol,
                             'target':getattr(rr,'target',''),'head_bust':getattr(rr,'head_bust',None),
                             'candidate_sets':getattr(rr,'candidate_sets',''),'actual':getattr(rr,'actual',''),
                             'set_match':getattr(rr,'set_match',None),'eligible_count':len(eligible),
                             'picks':'|'.join('-'.join(map(str,p)) for _,p in picks),
                             'pick_odds':'|'.join(f'{o:.2f}' for o,_ in picks),'ticket_count':len(picks),
                             'bet':int(bool(picks)),'hit':int(hit is not None) if act else 0,'stake':st,'pay':py})
    out=pd.DataFrame(rows);out.to_csv(OUT,index=False,encoding='utf-8-sig')
    payload={'rules':{'candidate_logic':'frozen BASE / OH_HIGH candidate families','MKT1':'lowest odds among >=30x, 1 ticket','MKT2':'two lowest odds among >=30x, max2','stake_yen':100},'variants':{}}
    for v in VARS:
        payload['variants'][v]={}
        for pol in POLICIES:
            payload['variants'][v][pol]=summarize(out[(out.variant==v)&(out.policy==pol)])
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
