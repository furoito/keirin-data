#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ex-ante ROI decomposition of OH_HIGH + MKT1 by non-popular head running style."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
INP=CTX/'market_ordering_results.csv'
OUT_CASES=CTX/'oh_high_mkt1_style_roi_cases.csv'
OUT_SUMMARY=CTX/'oh_high_mkt1_style_roi_summary.json'
STYLE_ORDER={'逃':0,'両':1,'追':2,'':9}

def parse_lines(v):
    out=[]
    for part in str(v).split('/'):
        xs=[]
        for x in part.split('-'):
            try: xs.append(int(float(x)))
            except Exception: pass
        if xs: out.append(xs)
    return out

def parse_pick(v):
    s=str(v).split('|')[0].strip()
    try:
        p=tuple(int(float(x)) for x in s.split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None

def num(v):
    try:return float(str(v).split('|')[0])
    except Exception:return None

def max_drawdown_yen(z):
    if z.empty:return 0
    equity=peak=max_dd=0
    for r in z.sort_values(['date','race_id']).itertuples(index=False):
        equity+=int(r.pay)-int(r.stake); peak=max(peak,equity); max_dd=max(max_dd,peak-equity)
    return int(max_dd)

def max_losing_streak(z):
    best=cur=0
    for hit in z.sort_values(['date','race_id']).hit.astype(int).tolist():
        if hit: cur=0
        else: cur+=1; best=max(best,cur)
    return int(best)

def summarize(z):
    n=len(z); st=int(z.stake.sum()) if n else 0; py=int(z.pay.sum()) if n else 0; hits=int(z.hit.sum()) if n else 0
    hit_odds=z.loc[z.hit==1,'pick_odds_num'].dropna() if n else pd.Series(dtype=float)
    odds=z.pick_odds_num.dropna() if n else pd.Series(dtype=float)
    return {'bets':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,
            'avg_pick_odds':float(odds.mean()) if len(odds) else None,'median_pick_odds':float(odds.median()) if len(odds) else None,
            'avg_hit_odds':float(hit_odds.mean()) if len(hit_odds) else None,'stake_yen':st,'pay_yen':py,'profit_yen':py-st,
            'roi_pct':100*py/st if st else None,'max_losing_streak':max_losing_streak(z),'max_drawdown_yen':max_drawdown_yen(z)}

def main():
    mkt=pd.read_csv(INP,encoding='utf-8-sig',dtype={'race_id':str}); mkt['race_id']=mkt.race_id.astype(str)
    x=mkt[(mkt.variant.astype(str)=='OH_HIGH')&(mkt.policy.astype(str)=='MKT1')&(pd.to_numeric(mkt.bet,errors='coerce')==1)].copy()
    rows=[]
    for month,sub in x.groupby('month',sort=True):
        rp=DATA/f'{month}_keirin.csv'; cp=CTX/f'{month}_races.csv'
        if not rp.exists() or not cp.exists():continue
        race=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str}); ctx=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str})
        race['race_id']=race.race_id.astype(str); ctx['race_id']=ctx.race_id.astype(str)
        rb={k:g for k,g in race.groupby('race_id',sort=False)}; cm={str(r.race_id):r for r in ctx.itertuples(index=False)}
        for rr in sub.itertuples(index=False):
            rid=str(rr.race_id); full=rb.get(rid); cr=cm.get(rid); pick=parse_pick(getattr(rr,'picks',''))
            if full is None or cr is None or pick is None:continue
            lines=parse_lines(getattr(cr,'true_line',''))
            try:target=int(float(rr.target))
            except Exception:continue
            pop_idx=next((i for i,g in enumerate(lines) if target in g),None)
            if pop_idx is None:continue
            style_map={}
            for r in full.itertuples(index=False):
                try:fn=int(float(r.banum))
                except Exception:continue
                style_map[fn]=str(getattr(r,'running_style','')).strip()
            heads=[]
            for i,g in enumerate(lines):
                if i==pop_idx or len(g)<2:continue
                if g[0] in pick:heads.append({'frame':g[0],'style':style_map.get(g[0],''),'line_size':len(g)})
            styles=[h['style'] for h in heads]; sig='+'.join(sorted(styles,key=lambda s:(STYLE_ORDER.get(s,8),s))) if styles else 'NONE'
            contains_escape=int('逃' in styles); single=styles[0] if len(styles)==1 else ('NONE' if not styles else 'MULTI')
            es=[h['line_size'] for h in heads if h['style']=='逃']; estag='+'.join(sorted({'2' if n==2 else '3+' for n in es})) if es else 'NONE'
            rows.append({'month':month,'date':str(getattr(cr,'date','')),'race_id':rid,'target':target,'pick':'-'.join(map(str,pick)),
                         'pick_odds_num':num(getattr(rr,'pick_odds',None)),'hit':int(getattr(rr,'hit',0)),'stake':int(getattr(rr,'stake',0)),
                         'pay':int(getattr(rr,'pay',0)),'other_head_count':len(heads),'other_head_styles':sig,'single_other_head_style':single,
                         'contains_escape_head':contains_escape,'escape_line_size_tag':estag,
                         'other_heads':'|'.join(f"{h['frame']}:{h['style']}:{h['line_size']}" for h in heads)})
    d=pd.DataFrame(rows); d.to_csv(OUT_CASES,index=False,encoding='utf-8-sig')
    payload={'scope':'2025-01..2026-06 OH_HIGH + MKT1, ALL ex-ante bets','anti_leakage':'ROI groups use all bets; realized popular-head-bust is not used as a filter.',
             'overall':summarize(d),'core_contains_escape':{'contains_escape':summarize(d[d.contains_escape_head==1]),'no_escape':summarize(d[d.contains_escape_head==0])},
             'single_other_head_style':{},'exact_style_signature':{},'escape_line_size':{}}
    for key in ['逃','両','追','NONE','MULTI']:payload['single_other_head_style'][key]=summarize(d[d.single_other_head_style==key])
    for sig,z in d.groupby('other_head_styles',sort=True):payload['exact_style_signature'][str(sig)]=summarize(z)
    payload['escape_line_size']={'escape_2car_only':summarize(d[d.escape_line_size_tag=='2']),'escape_3plus_only':summarize(d[d.escape_line_size_tag=='3+']),
                                 'escape_mixed_sizes':summarize(d[d.escape_line_size_tag=='2+3+']),'no_escape':summarize(d[d.escape_line_size_tag=='NONE'])}
    OUT_SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
