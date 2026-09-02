#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare position-specific co-occurrence behind non-popular line heads.

Condition: non-popular line head finishes top3. Popular line is determined pre-race
from stored target + true_line. Report position-2 and position-3 top3 rates separately,
including popular-head-bust subset, to avoid inflating 3+ line rates by using
'any follower'.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'keirin_data'/'strategy_context'
OUT=CTX/'other_head_follower_positions_summary.json'
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
    p=CTX/'other_head_promotion_oos_results.csv'
    if p.exists():
        d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
        parts.append(d[d.variant.astype(str)=='OH_HIGH'].copy())
    for m in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{m}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
            parts.append(d[d.variant.astype(str)=='OH_HIGH'].copy())
    x=pd.concat(parts,ignore_index=True); x['race_id']=x.race_id.astype(str)
    return x.drop_duplicates(['month','race_id'],keep='last')


def stats(z):
    n=len(z)
    if not n:return {'events':0}
    out={'events':n,
         'pos2_top3_n':int(z.pos2_top3.sum()),'pos2_top3_pct':100*z.pos2_top3.mean(),
         'head_and_pos2_exact_pair_n':int(z.pos2_top3.sum())}
    if 'pos3_exists' in z:
        e=z[z.pos3_exists==1]
        out['pos3_eligible_events']=len(e)
        out['pos3_top3_n']=int(e.pos3_top3.sum()) if len(e) else 0
        out['pos3_top3_pct']=100*e.pos3_top3.mean() if len(e) else None
        out['pos2_or_pos3_top3_n']=int(((z.pos2_top3==1)|(z.pos3_top3==1)).sum())
        out['pos2_or_pos3_top3_pct']=100*((z.pos2_top3==1)|(z.pos3_top3==1)).mean()
        out['pos3_only_added_n']=int(((z.pos2_top3==0)&(z.pos3_top3==1)).sum())
        out['pos3_only_added_pct_of_events']=100*((z.pos2_top3==0)&(z.pos3_top3==1)).mean()
        out['both_pos2_pos3_top3_n']=int(((z.pos2_top3==1)&(z.pos3_top3==1)).sum())
    return out


def main():
    exp=load_rows(); rows=[]
    for month in MONTHS:
        sub=exp[exp.month.astype(str)==month]
        if sub.empty:continue
        cp=CTX/f'{month}_races.csv'
        if not cp.exists():continue
        ctx=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str});ctx['race_id']=ctx.race_id.astype(str)
        cm={str(r.race_id):r for r in ctx.itertuples(index=False)}
        for rr in sub.itertuples(index=False):
            act=parse_order(getattr(rr,'actual',''))
            if not act:continue
            cr=cm.get(str(rr.race_id))
            if cr is None:continue
            lines=parse_lines(getattr(cr,'true_line',''))
            try:target=int(float(rr.target))
            except Exception:continue
            pop_idx=next((i for i,g in enumerate(lines) if target in g),None)
            if pop_idx is None:continue
            aset=set(act); hb=int(target not in aset)
            for i,g in enumerate(lines):
                if i==pop_idx or len(g)<2 or g[0] not in aset:continue
                rows.append({'month':month,'race_id':str(rr.race_id),'head_bust':hb,
                             'line_size':len(g),'head':g[0],'pos2':g[1],
                             'pos2_top3':int(g[1] in aset),
                             'pos3_exists':int(len(g)>=3),'pos3':g[2] if len(g)>=3 else None,
                             'pos3_top3':int(len(g)>=3 and g[2] in aset)})
    d=pd.DataFrame(rows)
    payload={'scope':'2025-01..2026-06; condition = non-popular line head top3',
             'note':'Position-specific rates; no any-follower inflation.', 'all':{},'popular_head_bust_only':{}}
    for label,mask in [('two_car',d.line_size==2),('three_plus',d.line_size>=3)]:
        payload['all'][label]=stats(d[mask])
        payload['popular_head_bust_only'][label]=stats(d[mask & (d.head_bust==1)])
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
