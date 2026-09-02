#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]
OUT=CTX/'other_head_running_style_summary.json'

def parse_lines(v):
    out=[]
    for part in str(v).split('/'):
        xs=[]
        for x in part.split('-'):
            try: xs.append(int(float(x)))
            except: pass
        if xs: out.append(xs)
    return out

def parse_actual(v):
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return set(p) if len(p)==3 and len(set(p))==3 else None
    except:return None

def load_exp():
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

def block(z):
    n=len(z)
    return {'observations':n,'top3_n':int(z.top3.sum()) if n else 0,'top3_pct':100*z.top3.mean() if n else None}

def main():
    exp=load_exp(); rows=[]
    for month in MONTHS:
        sub=exp[exp.month.astype(str)==month]
        if sub.empty: continue
        rp=DATA/f'{month}_keirin.csv'; cp=CTX/f'{month}_races.csv'
        if not rp.exists() or not cp.exists(): continue
        race=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str}); race['race_id']=race.race_id.astype(str)
        ctx=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str}); ctx['race_id']=ctx.race_id.astype(str)
        rb={k:g for k,g in race.groupby('race_id',sort=False)}
        cm={str(r.race_id):r for r in ctx.itertuples(index=False)}
        for rr in sub.itertuples(index=False):
            rid=str(rr.race_id); full=rb.get(rid); cr=cm.get(rid)
            if full is None or cr is None: continue
            aset=parse_actual(getattr(rr,'actual',''))
            if not aset: continue
            try: target=int(float(rr.target))
            except: continue
            if target in aset: continue  # popular-head-bust only
            lines=parse_lines(getattr(cr,'true_line',''))
            pop_idx=next((i for i,g in enumerate(lines) if target in g),None)
            if pop_idx is None: continue
            style_map={int(float(r.banum)):str(getattr(r,'running_style','')).strip() for r in full.itertuples(index=False)}
            for i,g in enumerate(lines):
                if i==pop_idx or len(g)<2: continue
                head=g[0]
                rows.append({'month':month,'race_id':rid,'line_size':len(g),'head':head,
                             'style':style_map.get(head,''),'top3':int(head in aset)})
    d=pd.DataFrame(rows)
    payload={'scope':'2025-01..2026-06; popular-head-bust races only; all non-popular line heads', 'by_style':{},'by_style_and_line_size':{}}
    for style,z in d.groupby('style',dropna=False): payload['by_style'][str(style)]=block(z)
    for style,z in d.groupby('style',dropna=False):
        payload['by_style_and_line_size'][str(style)]={'two_car':block(z[z.line_size==2]),'three_plus':block(z[z.line_size>=3])}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
