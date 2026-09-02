#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'
res=pd.read_csv(CTX/'popular_head_skip_v01b_results.csv',encoding='utf-8-sig',dtype={'race_id':str})
base=pd.read_csv(DATA/'2026_08_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})

def lm(s):
    d={}
    for li,g in enumerate(str(s).split('/'),1):
        for x in g.split('-'):
            try:d[int(x)]=li
            except:pass
    return d

rows=[]
for r in res.itertuples(index=False):
    g=base[base.race_id==str(r.race_id)]
    if g.empty or pd.isna(r.popular_line): continue
    line=lm(r.true_line)
    hon=g[pd.to_numeric(g.get('mark_num'),errors='coerce')==6]
    if hon.empty: continue
    h=int(pd.to_numeric(hon.iloc[0].banum))
    hline=line.get(h)
    pl=int(float(r.popular_line))
    rows.append({'race_id':r.race_id,'date':r.date,'venue':r.venue_slug,'race_no':r.race_no,'true_line':r.true_line,
                 'detected_line':pl,'detected_target':r.target,'honmei':h,'honmei_line':hline,'match':int(pl==hline)})
out=pd.DataFrame(rows)
print(f'n={len(out)} match={out.match.mean()*100:.1f}%')
print(out[out.match==0].head(25).to_string(index=False))
out.to_csv(CTX/'popular_line_detector_audit.csv',index=False,encoding='utf-8-sig')
