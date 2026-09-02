#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze whether non-popular line heads place together with their followers.

Uses only already-scorable experiment rows. The popular line is inferred from the
stored pre-race target and true_line. No result information is used to define the
popular line.

Reports separately:
- all scorable races
- popular-head-bust races
and by non-popular line size (2 vs 3+).
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'keirin_data'/'strategy_context'
OUT=CTX/'other_head_line_cooccurrence_cases.csv'
SUMMARY=CTX/'other_head_line_cooccurrence_summary.json'
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]


def parse_order(v):
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None


def parse_lines(v):
    out=[]
    for part in str(v).split('/'):
        xs=[]
        for x in part.split('-'):
            try: xs.append(int(float(x)))
            except Exception: pass
        if xs: out.append(xs)
    return out


def load_rows():
    parts=[]
    big=CTX/'other_head_promotion_oos_results.csv'
    if big.exists():
        d=pd.read_csv(big,encoding='utf-8-sig',dtype={'race_id':str})
        d=d[d.variant.astype(str)=='OH_HIGH'].copy()
        parts.append(d)
    for m in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{m}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
            d=d[d.variant.astype(str)=='OH_HIGH'].copy()
            parts.append(d)
    x=pd.concat(parts,ignore_index=True)
    x['race_id']=x.race_id.astype(str)
    return x.drop_duplicates(['month','race_id'],keep='last')


def block(z):
    n=len(z)
    if not n:return {'other_head_top3_events':0}
    both=int(z.follower_top3.sum())
    return {
      'other_head_top3_events':n,
      'follower_also_top3_n':both,
      'follower_also_top3_pct':100*both/n,
      'head_without_follower_n':n-both,
      'head_without_follower_pct':100*(n-both)/n,
      'both_line_members_top3_and_pop_head_bust_n':int(((z.follower_top3==1)&(z.head_bust==1)).sum()),
    }


def main():
    exp=load_rows(); rows=[]
    for month in MONTHS:
        sub=exp[exp.month.astype(str)==month]
        if sub.empty: continue
        cp=CTX/f'{month}_races.csv'
        if not cp.exists(): continue
        ctx=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str});ctx['race_id']=ctx.race_id.astype(str)
        cm={str(r.race_id):r for r in ctx.itertuples(index=False)}
        for rr in sub.itertuples(index=False):
            act=parse_order(getattr(rr,'actual',''))
            if not act: continue
            cr=cm.get(str(rr.race_id))
            if cr is None: continue
            lines=parse_lines(getattr(cr,'true_line',''))
            try: target=int(float(rr.target))
            except Exception: continue
            pop_idx=next((i for i,g in enumerate(lines) if target in g),None)
            if pop_idx is None: continue
            aset=set(act)
            hb=int(target not in aset)
            for i,g in enumerate(lines):
                if i==pop_idx or len(g)<2: continue
                head=g[0]
                if head not in aset: continue
                followers=g[1:]
                follower_hits=[x for x in followers if x in aset]
                rows.append({
                  'month':month,'race_id':str(rr.race_id),'target':target,'head_bust':hb,
                  'true_line':getattr(cr,'true_line',''),'actual':'-'.join(map(str,act)),
                  'other_line':'-'.join(map(str,g)),'other_line_size':len(g),'other_head':head,
                  'followers':'-'.join(map(str,followers)),'follower_top3':int(bool(follower_hits)),
                  'follower_hits':'-'.join(map(str,follower_hits)),
                  'two_car_line':int(len(g)==2),
                })
    out=pd.DataFrame(rows);out.to_csv(OUT,index=False,encoding='utf-8-sig')
    payload={'scope':'2025-01..2026-06 scorable OH_HIGH experiment races; popular line inferred pre-race from stored target + true_line',
             'all':{},'popular_head_bust_only':{}}
    for name,cond in [('two_car',out.other_line_size==2),('three_plus',out.other_line_size>=3),('all_nonpopular_lines',out.other_line_size>=2)]:
        payload['all'][name]=block(out[cond])
        payload['popular_head_bust_only'][name]=block(out[cond & (out.head_bust==1)])
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
