#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal replication check for frozen H1 group-level hypothesis.

Frozen target view:
- unordered 3-rider groups
- exactly 3 distinct lines (line_span >= 3)
- group combined race_score in top 50% among all unordered 3-rider groups in that race
- no upper odds cap

We do NOT retune the hypothesis here. We only report whether the same market-vs-actual
sign appears across months and coarse time blocks for fixed minimum effective odds cuts.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import popular_head_skip_v01 as base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
OUT = CTX / 'h1_crossline_highscore_groups_replication.json'
MONTHS = [f'2025_{m:02d}' for m in range(1,13)] + [f'2026_{m:02d}' for m in range(1,7)]
MIN_ODDS = [10,20,30,50,100,200]


def load_month(month):
    bp=DATA/f'{month}_keirin.csv'; cp=CTX/f'{month}_races.csv'; op=CTX/f'{month}_odds_3rentan.csv'
    if not (bp.exists() and cp.exists() and op.exists()): return None
    b=pd.read_csv(bp,encoding='utf-8-sig',dtype={'race_id':str})
    c=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str}).drop_duplicates('race_id',keep='last')
    o=pd.read_csv(op,encoding='utf-8-sig',dtype={'race_id':str})
    for d in (b,c,o): d['race_id']=d.race_id.astype(str)
    return b,c,o


def actual_top3_set(g):
    vals=[]
    for r in g.itertuples(index=False):
        try: pos=int(str(r.rank).strip()); fn=int(r.banum)
        except Exception: continue
        if 1<=pos<=3: vals.append((pos,fn))
    vals.sort()
    if [x[0] for x in vals] != [1,2,3]: return None
    return frozenset(x[1] for x in vals)


def race_rows(month,rid,pre,cr,og):
    lines=base.parse_true_line(cr.get('true_line'))
    if not lines: return None,'line_unresolved'
    frames=sorted({int(x) for g in lines for x in g})
    tri=base.odds_map(og)
    expected=len(frames)*(len(frames)-1)*(len(frames)-2)
    if len(frames)<4 or len(tri)!=expected: return None,'board_or_entries_incomplete'
    z=sum(1.0/v for v in tri.values() if v>0)
    if z<=0: return None,'zero_mass'
    actual=actual_top3_set(pre)
    if actual is None: return None,'result_missing'

    score={}
    for r in pre.itertuples(index=False):
        try: fn=int(r.banum); sc=float(r.race_score)
        except Exception: continue
        if np.isfinite(sc): score[fn]=sc
    if set(frames)-set(score): return None,'score_missing'

    line_of={}
    for li,g in enumerate(lines,1):
        for fn in g: line_of[int(fn)]=li

    raw=Counter()
    for combo,od in tri.items():
        if od>0: raw[tuple(sorted(combo))]+=1.0/od
    keys=sorted(raw)
    sums={k:float(sum(score[x] for x in k)) for k in keys}
    ordered=sorted(keys,key=lambda k:(-sums[k],k))
    rank={k:i+1 for i,k in enumerate(ordered)}
    n=len(ordered)

    rows=[]
    for k in keys:
        p=float(raw[k]/z)
        rows.append({
            'month':month,'race_id':rid,'market_p':p,
            'effective_fair_odds':float(1.0/p),
            'actual_hit':int(frozenset(k)==actual),
            'line_span':len({line_of[x] for x in k}),
            'group_score_percentile':float(rank[k]/n),
        })
    return rows,None


def agg(x):
    e=float(x.market_p.sum()); h=int(x.actual_hit.sum())
    return {
        'groups':int(len(x)),
        'races':int(x.race_id.nunique()),
        'expected_hits':e,
        'actual_hits':h,
        'ratio_actual_over_market':float(h/e) if e>0 else None,
    }


def target(x,cut):
    return x[(x.line_span>=3)&(x.group_score_percentile<=0.50)&(x.effective_fair_odds>=cut)]


def main():
    rows=[]; skipped=Counter(); coverage=[]
    for month in MONTHS:
        loaded=load_month(month)
        if loaded is None: continue
        b,c,o=loaded
        use=c.copy()
        if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}
        oby={str(k):g for k,g in o.groupby('race_id',sort=False)}
        analyzed=0
        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None: skipped['base_or_odds_missing']+=1; continue
            rr,why=race_rows(month,rid,pre,cr,og)
            if rr is None: skipped[why]+=1
            else: rows.extend(rr); analyzed+=1
        coverage.append({'month':month,'context_rows':int(len(c)),'full_price_usable':int(len(use)),'analyzed_races':analyzed})

    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit('No rows')

    monthly=[]
    for m,x in df.groupby('month',sort=True):
        for cut in MIN_ODDS:
            z=target(x,cut)
            if not z.empty: monthly.append({'month':m,'min_effective_odds':cut,**agg(z)})

    blocks={
        '2025_H1':['2025_01','2025_02','2025_03','2025_04','2025_05','2025_06'],
        '2025_H2':['2025_07','2025_08','2025_09','2025_10','2025_11','2025_12'],
        '2026_H1':['2026_01','2026_02','2026_03','2026_04','2026_05','2026_06'],
        '2025_ALL':[f'2025_{m:02d}' for m in range(1,13)],
        '2026_H1_ALL':[f'2026_{m:02d}' for m in range(1,7)],
    }
    block_rows=[]
    for name,months in blocks.items():
        x=df[df.month.isin(months)]
        for cut in MIN_ODDS:
            z=target(x,cut)
            if not z.empty: block_rows.append({'block':name,'min_effective_odds':cut,**agg(z)})

    stability=[]
    for cut in MIN_ODDS:
        x=pd.DataFrame([r for r in monthly if r['min_effective_odds']==cut])
        if x.empty: continue
        valid=x[x.expected_hits>=1.0].copy()
        stability.append({
            'min_effective_odds':cut,
            'months_total':int(len(x)),
            'months_expected_hits_ge_1':int(len(valid)),
            'positive_months_ratio_gt_1':int((valid.ratio_actual_over_market>1).sum()) if len(valid) else 0,
            'negative_months_ratio_lt_1':int((valid.ratio_actual_over_market<1).sum()) if len(valid) else 0,
            'median_monthly_ratio':float(valid.ratio_actual_over_market.median()) if len(valid) else None,
        })

    payload={
        'frozen_hypothesis':'3-line unordered groups in top 50% by combined race_score are underpriced from medium odds upward; no upper cap',
        'purpose':'temporal replication only; no threshold tuning',
        'minimum_odds_cuts':MIN_ODDS,
        'coverage_by_month':coverage,
        'skipped':dict(skipped),
        'monthly':monthly,
        'time_blocks':block_rows,
        'stability':stability,
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
