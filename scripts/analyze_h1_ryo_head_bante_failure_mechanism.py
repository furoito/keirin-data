#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mechanism diagnostic: when a multi-rider-line head with running_style='両'
wins, what pre-race features are associated with its own bante failing to make top3?

This is outcome-conditioned mechanism exploration, NOT a betting validation.
We keep only races passing context_quality=full and price_usable=true so the
sample matches the current research surface, but the labels use race results.

Primary pre-race features:
- head race_score rank in race
- own-bante race_score rank in race
- head minus own-bante score gap
- own-bante score versus the best bante from another line
- count of other-line bantes with higher race_score than own bante
- line size
- head score versus best other line-head score

The output reports own-bante failure rates by broad, pre-fixed bins. These bins
are descriptive and must not be canonized as strategy filters on the same data.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_ryo_head_bante_failure_mechanism_summary.json'
DETAIL = CTX / 'h1_ryo_head_bante_failure_mechanism_details.csv'


def actual_ordered_top3(pre: pd.DataFrame):
    vals=[]
    for r in pre.itertuples(index=False):
        try:
            pos=int(str(r.rank).strip()); fn=int(float(r.banum))
        except Exception:
            continue
        if 1 <= pos <= 3:
            vals.append((pos,fn))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]:
        return None
    return tuple(fn for _,fn in vals)


def rank_map(score: dict[int,float]) -> dict[int,int]:
    ordered=sorted(score, key=lambda fn:(-score[fn], fn))
    return {fn:i+1 for i,fn in enumerate(ordered)}


def fail_summary(x: pd.DataFrame) -> dict:
    n=int(len(x))
    if n == 0:
        return {'cases':0,'own_bante_failed':0,'own_bante_survived':0,'failure_rate_pct':None}
    failed=int(x.own_bante_failed.sum())
    return {
        'cases':n,
        'own_bante_failed':failed,
        'own_bante_survived':n-failed,
        'failure_rate_pct':float(100.0*failed/n),
    }


def by_category(df: pd.DataFrame, col: str) -> dict:
    out={}
    for v,g in df.groupby(col, dropna=False, sort=True):
        out[str(v)] = fail_summary(g)
    return out


def main():
    rows=[]; skipped=Counter(); usable_by_month={}

    for month in h1.MONTHS:
        loaded=h1.load_month(month)
        if loaded is None:
            continue
        b,c,o=loaded
        use=c.copy()
        if 'context_quality' in use:
            use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use:
            use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        use=use.drop_duplicates('race_id',keep='last')
        usable_by_month[month]=int(len(use))
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}

        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid)
            if pre is None:
                skipped['base_missing']+=1; continue
            actual=actual_ordered_top3(pre)
            if actual is None:
                skipped['result_missing']+=1; continue

            lines=base.parse_true_line(cr.get('true_line'))
            if not lines:
                skipped['line_unresolved']+=1; continue

            line_of={}; pos_of={}; line_members={}
            for li,g in enumerate(lines,1):
                line_members[li]=[int(x) for x in g]
                for pos,fn in enumerate(g,1):
                    line_of[int(fn)]=li; pos_of[int(fn)]=pos

            score={}; style={}
            for r in pre.itertuples(index=False):
                try: fn=int(float(r.banum))
                except Exception: continue
                try:
                    sc=float(r.race_score)
                    if np.isfinite(sc): score[fn]=sc
                except Exception:
                    pass
                s=str(getattr(r,'running_style','')).strip()
                if not s or s.lower()=='nan': s='UNKNOWN'
                style[fn]=s

            winner=int(actual[0])
            li=line_of.get(winner)
            if li is None or pos_of.get(winner)!=1 or style.get(winner)!='両':
                continue
            members=line_members.get(li,[])
            if len(members)<2:
                skipped['winner_solo']+=1; continue
            own_bante=int(members[1])
            if winner not in score or own_bante not in score:
                skipped['score_missing_head_or_bante']+=1; continue

            ranks=rank_map(score)
            own_score=score[own_bante]; head_score=score[winner]

            other_bantes=[]
            other_heads=[]
            for oli,og in line_members.items():
                if oli==li: continue
                if len(og)>=1 and og[0] in score:
                    other_heads.append(score[og[0]])
                if len(og)>=2 and og[1] in score:
                    other_bantes.append(score[og[1]])

            best_other_bante=max(other_bantes) if other_bantes else np.nan
            best_other_head=max(other_heads) if other_heads else np.nan
            higher_other_bantes=sum(1 for v in other_bantes if v>own_score)

            own_failed=int(own_bante not in actual)
            rows.append({
                'month':month,'race_id':rid,'winner':winner,'own_bante':own_bante,
                'own_bante_failed':own_failed,
                'head_score':head_score,'own_bante_score':own_score,
                'head_score_rank':int(ranks[winner]),
                'own_bante_score_rank':int(ranks[own_bante]),
                'head_minus_own_bante_score':float(head_score-own_score),
                'line_size':int(len(members)),
                'best_other_bante_score':float(best_other_bante) if np.isfinite(best_other_bante) else np.nan,
                'own_bante_minus_best_other_bante':float(own_score-best_other_bante) if np.isfinite(best_other_bante) else np.nan,
                'other_bantes_above_own':int(higher_other_bantes),
                'best_other_head_score':float(best_other_head) if np.isfinite(best_other_head) else np.nan,
                'head_minus_best_other_head':float(head_score-best_other_head) if np.isfinite(best_other_head) else np.nan,
            })

    df=pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No RYO-head winner cases')

    # Fixed broad bins for descriptive mechanism diagnosis.
    df['head_rank_bin']=pd.cut(df.head_score_rank,[0,2,4,99],labels=['1-2','3-4','5+'],include_lowest=True).astype(str)
    df['bante_rank_bin']=pd.cut(df.own_bante_score_rank,[0,2,4,99],labels=['1-2','3-4','5+'],include_lowest=True).astype(str)
    df['head_bante_gap_bin']=pd.cut(df.head_minus_own_bante_score,[-np.inf,0,2,4,np.inf],labels=['<=0','0-2','2-4','4+'],right=False).astype(str)
    df['own_vs_best_other_bante_bin']=pd.cut(df.own_bante_minus_best_other_bante,[-np.inf,-2,0,2,np.inf],labels=['<-2','-2-0','0-2','2+'],right=False).astype(str)
    df['other_bantes_above_bin']=df.other_bantes_above_own.map(lambda x:'0' if x==0 else ('1' if x==1 else '2+'))
    df['line_size_bin']=df.line_size.map(lambda x:'2' if x==2 else '3+')
    df['head_vs_best_other_head_bin']=pd.cut(df.head_minus_best_other_head,[-np.inf,0,2,4,np.inf],labels=['<0','0-2','2-4','4+'],right=False).astype(str)

    df=df.sort_values(['month','race_id'])
    df.to_csv(DETAIL,index=False,encoding='utf-8-sig')

    feature_views={
        'head_score_rank':by_category(df,'head_rank_bin'),
        'own_bante_score_rank':by_category(df,'bante_rank_bin'),
        'head_minus_own_bante_score':by_category(df,'head_bante_gap_bin'),
        'own_bante_vs_best_other_bante':by_category(df,'own_vs_best_other_bante_bin'),
        'other_bantes_above_own':by_category(df,'other_bantes_above_bin'),
        'line_size':by_category(df,'line_size_bin'),
        'head_vs_best_other_head':by_category(df,'head_vs_best_other_head_bin'),
    }

    numeric_cols=[
        'head_score','own_bante_score','head_score_rank','own_bante_score_rank',
        'head_minus_own_bante_score','line_size','own_bante_minus_best_other_bante',
        'other_bantes_above_own','head_minus_best_other_head'
    ]
    means={}
    for label,val in [('failed',1),('survived',0)]:
        z=df[df.own_bante_failed==val]
        means[label]={c:(float(z[c].mean()) if z[c].notna().any() else None) for c in numeric_cols}

    payload={
        'status':'exploratory_outcome_conditioned_mechanism_diagnostic',
        'cohort':'usable races where actual winner is running_style=両, line_pos=1, and belongs to a multi-rider line',
        'label':'own bante fails = winner own line_pos=2 rider is not in actual top3',
        'warning':'Outcome-conditioned descriptive analysis on the same discovery data. Do not use these bins as validated betting filters.',
        'usable_races_by_month':usable_by_month,
        'skipped':dict(skipped),
        'overall':fail_summary(df),
        'feature_views':feature_views,
        'numeric_means_by_outcome':means,
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print(f'detail={DETAIL}')


if __name__=='__main__':
    main()
