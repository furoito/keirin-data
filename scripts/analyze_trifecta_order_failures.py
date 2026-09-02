#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why exact top-3 candidate sets rarely convert to trifecta hits.

Covers BASE and OH_HIGH across:
- untouched dual_logic_2025_01..2025_04
- other_head_promotion_oos_results.csv (2025_05..2026_06)

For head-bust BET races with an exact candidate set, classify:
- hit: actual order was bought
- price_gate: actual order was generated but not bought (e.g. under 30x)
- order_model: exact set found, but generated order differs from actual

For order_model misses, classify the permutation relative to the predicted order.
Reporting only; no strategy changes.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

CTX=Path('keirin_data/strategy_context')
OUT=CTX/'trifecta_order_failure_cases.csv'
SUMMARY=CTX/'trifecta_order_failure_summary.json'
VARS=['BASE','OH_HIGH']


def split_orders(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for part in str(v).split('|'):
        part=part.strip()
        if not part: continue
        try:
            q=tuple(int(float(x)) for x in part.split('-') if str(x).strip())
        except Exception:
            continue
        if len(q)==3: out.append(q)
    return out


def parse_order(v):
    x=split_orders(v)
    return x[0] if x else None


def perm_name(pred,act):
    if not pred or not act or set(pred)!=set(act): return 'not_same_set'
    if pred==act:return 'exact'
    mapping=tuple(pred.index(x)+1 for x in act)
    names={
        (2,1,3):'swap_1_2',
        (1,3,2):'swap_2_3',
        (3,2,1):'swap_1_3',
        (2,3,1):'rotate_231',
        (3,1,2):'rotate_312',
    }
    return names.get(mapping,'other_'+''.join(map(str,mapping)))


def load():
    parts=[]
    big=CTX/'other_head_promotion_oos_results.csv'
    if big.exists():
        d=pd.read_csv(big,encoding='utf-8-sig',dtype={'race_id':str})
        d=d[d.variant.isin(VARS)].copy(); parts.append(d)
    for month in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{month}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
            d=d[d.variant.isin(VARS)].copy(); parts.append(d)
    if not parts: raise SystemExit('no source result files')
    x=pd.concat(parts,ignore_index=True)
    x['race_id']=x.race_id.astype(str)
    x=x.drop_duplicates(['month','race_id','variant'],keep='last')
    return x


def main():
    x=load(); rows=[]; payload={'variants':{}}
    for v in VARS:
        z=x[x.variant==v].copy()
        hb=z[(pd.to_numeric(z.head_bust,errors='coerce')==1) & (pd.to_numeric(z.bet,errors='coerce')==1)].copy()
        exact=hb[pd.to_numeric(hb.set_match,errors='coerce')==1].copy()
        cause=Counter(); perm=Counter(); first_ok=second_ok=third_ok=0
        for r in exact.itertuples(index=False):
            act=parse_order(r.actual); orders=split_orders(r.orders); bets=split_orders(r.bets)
            same_set_orders=[o for o in orders if act and set(o)==set(act)]
            same_set_bets=[o for o in bets if act and set(o)==set(act)]
            if act and act in bets:
                c='hit'; p='exact'; pred=act
            elif act and act in orders:
                c='price_gate'; p='exact_generated_not_bought'; pred=act
            else:
                c='order_model'
                pred=same_set_orders[0] if same_set_orders else None
                p=perm_name(pred,act)
            cause[c]+=1; perm[p]+=1
            if pred and act:
                first_ok+=int(pred[0]==act[0]); second_ok+=int(pred[1]==act[1]); third_ok+=int(pred[2]==act[2])
            rows.append({'month':r.month,'race_id':r.race_id,'variant':v,'actual':r.actual,'orders':r.orders,'bets':r.bets,
                         'cause':c,'permutation':p,'predicted_same_set_order':'-'.join(map(str,pred)) if pred else ''})
        n=len(exact)
        payload['variants'][v]={
            'head_bust_bet_races':len(hb),
            'exact_candidate_set_bet_races':n,
            'exact_set_to_hit_pct':100*cause['hit']/n if n else None,
            'causes':cause.most_common(),
            'order_model_permutations':[(k,val) for k,val in perm.most_common() if k not in ('exact','exact_generated_not_bought')],
            'predicted_position_accuracy_within_exact_set_pct':{
                'first':100*first_ok/n if n else None,
                'second':100*second_ok/n if n else None,
                'third':100*third_ok/n if n else None,
            }
        }
    pd.DataFrame(rows).to_csv(OUT,index=False,encoding='utf-8-sig')
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
