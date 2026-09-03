#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct test: if the race-score top 3 are on 3 distinct lines, buy those 3 riders.

No role filter: escape heads, RYO, bante, third-plus, and solos are all allowed.
The selected unordered trio is exactly the three highest pre-race race_score riders.
We evaluate the unordered trio against normalized market probability, then all six
exact trifecta orders and cumulative lowest-odds exact-order rules.

Exploratory historical test, not fresh OOS.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'top3_score_three_lines_direct_summary.json'
DETAIL = CTX / 'top3_score_three_lines_direct_tickets.csv'
BOOTSTRAPS = 2000
SEED = 20260903


def period_of(month: str):
    y, m = map(int, month.split('_'))
    return f'{y}_H1' if m <= 6 else f'{y}_H2'


def actual_ordered_top3(pre):
    vals=[]
    for r in pre.itertuples(index=False):
        try:
            pos=int(str(r.rank).strip()); fn=int(float(r.banum))
        except Exception:
            continue
        if 1 <= pos <= 3:
            vals.append((pos,fn))
    vals.sort()
    return tuple(fn for _,fn in vals) if [p for p,_ in vals]==[1,2,3] else None


def ticket_agg(x: pd.DataFrame):
    n=int(len(x))
    if not n:
        return {'tickets':0,'races':0,'hits':0,'expected_hits':0.0,'market_ratio':None,'gross_roi_pct':None,'median_odds':None}
    hits=int(x.actual_hit.sum())
    exp=float(x.market_p.sum())
    gross=float(x.loc[x.actual_hit==1,'odds'].sum())
    return {
        'tickets':n,'races':int(x.race_id.nunique()),'hits':hits,'expected_hits':exp,
        'market_ratio':float(hits/exp) if exp>0 else None,
        'gross_roi_pct':float(100*gross/n),'median_odds':float(x.odds.median()),
    }


def bootstrap_ticket_ci(x: pd.DataFrame):
    if x.empty: return {'market_ratio_ci95':[None,None],'gross_roi_pct_ci95':[None,None]}
    y=x.copy(); y['gross']=np.where(y.actual_hit==1,y.odds,0.0)
    rg=y.groupby('race_id').agg(tickets=('actual_hit','size'),hits=('actual_hit','sum'),exp=('market_p','sum'),gross=('gross','sum')).reset_index(drop=True)
    if len(rg)<2: return {'market_ratio_ci95':[None,None],'gross_roi_pct_ci95':[None,None]}
    a=rg[['tickets','hits','exp','gross']].to_numpy(float); rng=np.random.default_rng(SEED+len(x)); ratios=[]; rois=[]
    for _ in range(BOOTSTRAPS):
        s=a[rng.integers(0,len(a),len(a))].sum(axis=0)
        if s[2]>0: ratios.append(s[1]/s[2])
        if s[0]>0: rois.append(100*s[3]/s[0])
    return {'market_ratio_ci95':[float(np.quantile(ratios,.025)),float(np.quantile(ratios,.975))],
            'gross_roi_pct_ci95':[float(np.quantile(rois,.025)),float(np.quantile(rois,.975))]}


def ticket_view(x):
    d=ticket_agg(x); d['race_bootstrap']=bootstrap_ticket_ci(x); return d


def group_agg(df: pd.DataFrame):
    g=df.groupby('race_id',sort=False).agg(group_market_p=('market_p','sum'),group_hit=('actual_hit','sum')).reset_index()
    # Exactly one of six exact orders can hit, so group_hit is 0/1.
    n=len(g); hits=int(g.group_hit.sum()); exp=float(g.group_market_p.sum())
    return {
        'races':int(n),'unordered_trio_hits':hits,
        'unordered_trio_hit_rate_pct':float(100*hits/n) if n else None,
        'normalized_market_expected_group_hits':exp,
        'actual_over_normalized_market':float(hits/exp) if exp>0 else None,
        'avg_normalized_market_group_probability_pct':float(100*g.group_market_p.mean()) if n else None,
    }


def main():
    rows=[]; skipped=Counter(); role_counts=Counter(); usable_by_month={}
    for month in h1.MONTHS:
        loaded=h1.load_month(month)
        if loaded is None:
            skipped['month_missing']+=1; continue
        b,c,o=loaded
        use=c.copy()
        if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        use=use.drop_duplicates('race_id',keep='last'); usable_by_month[month]=int(len(use))
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}; oby={str(k):g for k,g in o.groupby('race_id',sort=False)}

        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None: skipped['base_or_odds_missing']+=1; continue
            lines=base.parse_true_line(cr.get('true_line'))
            if not lines: skipped['line_unresolved']+=1; continue
            frames=sorted({int(x) for line in lines for x in line})
            line_of={}; pos_of={}; size_of={}
            for li,line in enumerate(lines,1):
                gg=[int(x) for x in line]
                for pos,fn in enumerate(gg,1): line_of[fn]=li; pos_of[fn]=pos; size_of[fn]=len(gg)

            actual=actual_ordered_top3(pre)
            if actual is None: skipped['ordered_result_missing']+=1; continue
            tri=base.odds_map(og); expected=len(frames)*(len(frames)-1)*(len(frames)-2)
            if len(tri)!=expected: skipped['odds_board_incomplete']+=1; continue
            z=sum(1.0/od for od in tri.values() if od>0)
            if z<=0: skipped['zero_market_mass']+=1; continue

            score={}; style={}
            for r in pre.itertuples(index=False):
                try: fn=int(float(r.banum)); sc=float(r.race_score)
                except Exception: continue
                if np.isfinite(sc): score[fn]=sc
                st=str(getattr(r,'running_style','')).strip()
                style[fn]=st if st and st.lower()!='nan' else 'UNKNOWN'
            if set(frames)-set(score): skipped['score_missing']+=1; continue

            selected=tuple(sorted(frames,key=lambda fn:(-score[fn],fn))[:3])
            if len(selected)<3 or len({line_of.get(fn) for fn in selected})!=3:
                skipped['top3_not_three_distinct_lines']+=1; continue

            def role(fn):
                if size_of.get(fn,0)==1: return 'SOLO'
                if pos_of.get(fn)==1:
                    return 'ESCAPE_HEAD' if style.get(fn)=='逃' else ('RYO_HEAD' if style.get(fn)=='両' else 'OTHER_HEAD')
                if pos_of.get(fn)==2: return 'BANTE'
                return 'THIRD_PLUS'
            roles=tuple(sorted(role(fn) for fn in selected)); role_key='+'.join(roles); role_counts[role_key]+=1

            for perm in itertools.permutations(selected):
                od=tri.get(tuple(perm))
                if od is None or od<=0: continue
                hit=int(tuple(perm)==actual)
                rows.append({'month':month,'period':period_of(month),'race_id':rid,
                             'selected_trio':'-'.join(map(str,sorted(selected))),
                             'ticket':'-'.join(map(str,perm)),'odds':float(od),
                             'market_p':float((1.0/od)/z),'actual_hit':hit,
                             'selected_role_set':role_key})

    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit('No qualifying tickets')
    sizes=df.groupby(['race_id','selected_trio']).size()
    if not (sizes==6).all(): raise SystemExit(f'Expected six permutations; bad groups={int((sizes!=6).sum())}')
    df=df.sort_values(['race_id','selected_trio','odds','ticket'],kind='mergesort')
    df['odds_rank_ascending']=df.groupby(['race_id','selected_trio']).cumcount()+1
    df.to_csv(DETAIL,index=False,encoding='utf-8-sig')

    selections={'ALL6_EQUAL':ticket_view(df)}
    for k in range(1,7):
        x=df[df.odds_rank_ascending<=k]
        selections[f'LOWEST_{k}']={'all':ticket_view(x),'periods':{p:ticket_view(g) for p,g in x.groupby('period',sort=True)}}

    payload={
        'status':'exploratory_direct_top3_score_three_lines',
        'rule':'If the three highest race_score riders are on three distinct true lines, select exactly those three riders. No role filter; escape heads are allowed.',
        'role_filter':None,
        'unordered_group':group_agg(df),
        'exact_order_bets':selections,
        'selected_role_sets_race_count':dict(role_counts),
        'coverage':{'usable_races_by_month':usable_by_month,'skipped':dict(skipped)},
        'warning':'Historical exploratory test on previously explored data; not fresh OOS.',
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
