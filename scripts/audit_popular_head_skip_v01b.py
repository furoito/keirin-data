#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

DATA=Path('keirin_data')
CTX=DATA/'strategy_context'
res=pd.read_csv(CTX/'popular_head_skip_v01b_results.csv',encoding='utf-8-sig',dtype={'race_id':str})
base=pd.read_csv(DATA/'2026_08_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
ctx=pd.read_csv(CTX/'2026_08_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
ctx=ctx.set_index('race_id')

def lines(s):
    out={}
    for li,g in enumerate(str(s).split('/'),1):
        for pos,x in enumerate(g.split('-'),1):
            if x.strip(): out[int(x)]=(li,pos)
    return out

def rank_value(v):
    s=str(v).strip()
    try:return int(float(s))
    except:return None

rows=[]
for r in res[res.action=='BET'].itertuples(index=False):
    rid=str(r.race_id); g=base[base.race_id==rid].copy()
    lm=lines(r.true_line)
    pop=int(float(r.popular_line)); target=int(float(r.target))
    pred=[int(x) for x in str(r.ranking).split('-') if x]
    predpos={x:i+1 for i,x in enumerate(pred)}
    desc=[]
    for x in sorted(pd.to_numeric(g.banum,errors='coerce').dropna().astype(int).unique()):
        q=g[pd.to_numeric(g.banum,errors='coerce')==x].iloc[0]
        li,lp=lm.get(x,(None,None)); ar=rank_value(q.get('rank'))
        tier=(4 if li==pop and lp==2 else 3 if li==pop and lp and lp>=3 else 2 if lp and lp>=2 else 1)
        desc.append(f"{x}:s={float(q.race_score):.2f},L={li},p={lp},t={tier},mr={predpos.get(x,'X')},fin={ar}")
    rows.append({
        'race_id':rid,'date':r.date,'venue':r.venue_slug,'race_no':r.race_no,'line':r.true_line,
        'target':target,'ranking':r.ranking,'candidates':r.candidate_orders,'actual':r.actual_order,
        'head_bust':r.head_bust,'top3_set_match':r.top3_set_match,'detail':' | '.join(desc)
    })
out=pd.DataFrame(rows)
out.to_csv(CTX/'popular_head_skip_v01b_audit.csv',index=False,encoding='utf-8-sig')
print(out.head(20).to_string(index=False))
