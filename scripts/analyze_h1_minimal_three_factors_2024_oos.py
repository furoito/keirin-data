#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locked 2024 holdout test of the minimal three-factor ticket rule.

Rule is frozen from 2025-2026 discovery work:
1) exact trifecta 1st rider is running_style='両', line_pos=1, on a multi-rider line
2) that winner's own bante is excluded
3) 2nd and 3rd are bantes from two other distinct lines and each has race_score > own bante

No percentile, NO_THIRD, score-rank, score-order, or score-gap filter.
2024 is evaluated separately and must not be used to retune the rule.
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

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'keirin_data'/'strategy_context'
OUT=CTX/'h1_minimal_three_factors_2024_oos_summary.json'
MONTHS=[f'2024_{m:02d}' for m in range(1,13)]
ODDS_BINS=[
    ('<50',0.0,50.0),('50-100',50.0,100.0),('100-200',100.0,200.0),
    ('200-500',200.0,500.0),('500-1000',500.0,1000.0),('1000+',1000.0,float('inf')),
]


def actual_ordered_top3(pre):
    vals=[]
    for r in pre.itertuples(index=False):
        try: pos=int(str(r.rank).strip()); fn=int(float(r.banum))
        except Exception: continue
        if 1<=pos<=3: vals.append((pos,fn))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]: return None
    return tuple(fn for _,fn in vals)


def agg(x):
    n=int(len(x)); stake=float(n)
    gross=float(x.loc[x.actual_hit==1,'odds'].sum()) if n else 0.0
    hits=int(x.actual_hit.sum()) if n else 0
    exp=float(x.market_p.sum()) if n else 0.0
    return {
        'tickets':n,'races':int(x.race_id.nunique()) if n else 0,
        'stake_units':stake,'gross_return_units':gross,
        'gross_roi_pct':float(100*gross/stake) if stake else None,
        'net_roi_pct':float(100*(gross-stake)/stake) if stake else None,
        'actual_hits':hits,'normalized_market_expected_hits':exp,
        'actual_over_normalized_market':float(hits/exp) if exp>0 else None,
        'avg_ticket_odds':float(x.odds.mean()) if n else None,
        'median_ticket_odds':float(x.odds.median()) if n else None,
    }


def slices(df):
    return {name:agg(df[(df.odds>=lo)&(df.odds<hi)]) for name,lo,hi in ODDS_BINS}


def main():
    rows=[]; skipped=Counter(); usable_by_month={}
    for month in MONTHS:
        loaded=h1.load_month(month)
        if loaded is None:
            skipped[f'month_missing:{month}']+=1
            continue
        b,c,o=loaded
        use=c.copy()
        if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        use=use.drop_duplicates('race_id',keep='last')
        usable_by_month[month]=int(len(use))
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}
        oby={str(k):g for k,g in o.groupby('race_id',sort=False)}

        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing']+=1; continue
            lines=base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved']+=1; continue
            frames=sorted({int(x) for g in lines for x in g})
            tri=base.odds_map(og)
            expected=len(frames)*(len(frames)-1)*(len(frames)-2)
            if len(tri)!=expected:
                skipped['odds_board_incomplete']+=1; continue
            z=sum(1.0/od for od in tri.values() if od>0)
            if z<=0:
                skipped['zero_mass']+=1; continue
            actual=actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing']+=1; continue

            line_of={}; pos_of={}; members={}
            for li,g in enumerate(lines,1):
                members[li]=[int(x) for x in g]
                for pos,fn in enumerate(g,1):
                    line_of[int(fn)]=li; pos_of[int(fn)]=pos

            score={}; style={}
            for r in pre.itertuples(index=False):
                try: fn=int(float(r.banum))
                except Exception: continue
                try:
                    sc=float(r.race_score)
                    if np.isfinite(sc): score[fn]=sc
                except Exception: pass
                s=str(getattr(r,'running_style','')).strip()
                if not s or s.lower()=='nan': s='UNKNOWN'
                style[fn]=s

            for a in frames:
                if style.get(a)!='両' or pos_of.get(a)!=1: continue
                li=line_of.get(a); lm=members.get(li,[])
                if len(lm)<2: continue
                own=int(lm[1])
                if own not in score: continue

                stronger_other_bantes=[]
                for oli,ogroup in members.items():
                    if oli==li or len(ogroup)<2: continue
                    fn=int(ogroup[1])
                    if fn in score and score[fn] > score[own]:
                        stronger_other_bantes.append(fn)

                if len(stronger_other_bantes)<2: continue
                for b2,c3 in itertools.permutations(stronger_other_bantes,2):
                    if line_of.get(b2)==line_of.get(c3): continue
                    perm=(a,b2,c3)
                    od=tri.get(perm)
                    if od is None or od<=0: continue
                    p=(1.0/float(od))/z
                    rows.append({
                        'month':month,'race_id':rid,'ticket':'-'.join(map(str,perm)),
                        'odds':float(od),'market_p':float(p),'actual_hit':int(perm==actual),
                    })

    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit('No qualifying tickets')

    payload={
        'status':'locked_2024_holdout_minimal_three_factor_ticket_test',
        'period':'2024-01 through 2024-12',
        'fixed_factors':[
            "ticket 1st rider = running_style 両, line_pos=1, multi-rider line head",
            "winner own bante excluded",
            "ticket 2nd and 3rd riders = bantes from two other distinct lines, each with race_score > own bante",
        ],
        'explicitly_removed_filters':['group_score_percentile','NO_THIRD','ticket score rank','ticket score order','score gaps'],
        'holdout_note':'Rule frozen before looking at this 2024 result; do not retune on 2024 if preserving holdout value.',
        'usable_races_by_month':usable_by_month,'skipped':dict(skipped),
        'overall':agg(df),'ticket_odds_bins':slices(df),
        'min_ticket_odds':{str(c):agg(df[df.odds>=c]) for c in [50,100,200,500]},
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
