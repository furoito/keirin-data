#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ticket-level diagnostic for the RYO-head / own-bante-excluded hypothesis.

Fixed discovery surface:
- unordered trio group_score_percentile <= 0.45
- no rider at line_pos >= 3
- ordered ticket winner is running_style='両', line_pos=1, multi-rider line
- winner's own bante is excluded from trio

Compare two meanings of 'stronger other bantes':
A) race-level: number of all other-line bantes whose race_score > own bante: >=1 / >=2
B) in the bante-bante ticket structure: among actual 2nd/3rd ticket riders, count whose
   race_score > own bante: >=1 / both(2)

Evaluate quoted trifecta odds >=50 / >=100 / >=200, flat 1 unit per ticket.
Same-data discovery only.
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
OUT=CTX/'h1_other_bante_strength_filter_summary.json'
PCT=0.45
CUTS=[50,100,200]


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
    exp=float(x.market_p.sum()) if n else 0.0
    hits=int(x.actual_hit.sum()) if n else 0
    return {
        'tickets':n,'races':int(x.race_id.nunique()) if n else 0,'stake_units':stake,
        'gross_return_units':gross,'gross_roi_pct':float(100*gross/stake) if stake else None,
        'net_roi_pct':float(100*(gross-stake)/stake) if stake else None,
        'actual_hits':hits,'normalized_market_expected_hits':exp,
        'actual_over_normalized_market':float(hits/exp) if exp>0 else None,
        'avg_ticket_odds':float(x.odds.mean()) if n else None,
        'median_ticket_odds':float(x.odds.median()) if n else None,
    }


def summarize(z):
    return {
        'all_odds':agg(z),
        'min_ticket_odds':{str(c):agg(z[z.odds>=c]) for c in CUTS},
        'non_overlapping_bins':{
            '50-100':agg(z[(z.odds>=50)&(z.odds<100)]),
            '100-200':agg(z[(z.odds>=100)&(z.odds<200)]),
            '200-plus':agg(z[z.odds>=200]),
        },
    }


def main():
    rows=[]; skipped=Counter(); usable_by_month={}
    for month in h1.MONTHS:
        loaded=h1.load_month(month)
        if loaded is None: continue
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
            groups,why=h1.race_rows(month,rid,pre,cr,og)
            if groups is None:
                skipped[why]+=1; continue
            actual=actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing']+=1; continue
            tri=base.odds_map(og)
            z=sum(1.0/od for od in tri.values() if od>0)
            if z<=0:
                skipped['zero_mass']+=1; continue

            lines=base.parse_true_line(cr.get('true_line'))
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

            for q in groups:
                if float(q['group_score_percentile'])>PCT: continue
                trio=tuple(int(x) for x in q['trio'].split('-'))
                if any(pos_of.get(fn,99)>=3 for fn in trio): continue
                for perm in itertools.permutations(trio):
                    a,b2,c3=perm
                    if style.get(a)!='両' or pos_of.get(a)!=1: continue
                    li=line_of.get(a); lm=members.get(li,[])
                    if len(lm)<2: continue
                    own=int(lm[1])
                    if own in trio: continue
                    if own not in score: continue
                    od=tri.get(tuple(perm))
                    if od is None or od<=0: continue

                    other_bantes=[]
                    for oli,ogroup in members.items():
                        if oli==li or len(ogroup)<2: continue
                        fn=int(ogroup[1])
                        if fn in score: other_bantes.append(fn)
                    race_higher=sum(1 for fn in other_bantes if score[fn]>score[own])

                    bante_bante=int(pos_of.get(b2)==2 and pos_of.get(c3)==2)
                    selected_higher=None
                    if bante_bante and b2 in score and c3 in score:
                        selected_higher=int(score[b2]>score[own])+int(score[c3]>score[own])

                    p=(1.0/float(od))/z
                    rows.append({
                        'month':month,'race_id':rid,'ticket':'-'.join(map(str,perm)),
                        'odds':float(od),'market_p':float(p),'actual_hit':int(tuple(perm)==actual),
                        'race_higher_other_bantes':int(race_higher),
                        'bante_bante':bante_bante,
                        'selected_higher_bantes':selected_higher,
                        'line_span':int(q['line_span']),
                    })

    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit('No qualifying tickets')

    bb=df[df.bante_bante==1].copy()
    views={
        'OWN_BANTE_EXCLUDED_BASE':summarize(df),
        'RACE_OTHER_BANTE_HIGHER_GE1':summarize(df[df.race_higher_other_bantes>=1]),
        'RACE_OTHER_BANTE_HIGHER_GE2':summarize(df[df.race_higher_other_bantes>=2]),
        'BANTE_BANTE_BASE':summarize(bb),
        'BANTE_BANTE_SELECTED_HIGHER_GE1':summarize(bb[bb.selected_higher_bantes>=1]),
        'BANTE_BANTE_SELECTED_HIGHER_EQ2':summarize(bb[bb.selected_higher_bantes==2]),
        'BANTE_BANTE_RACE_HIGHER_GE1':summarize(bb[bb.race_higher_other_bantes>=1]),
        'BANTE_BANTE_RACE_HIGHER_GE2':summarize(bb[bb.race_higher_other_bantes>=2]),
    }
    payload={
        'status':'exploratory_same_data_other_bante_strength_ticket_diagnostic',
        'fixed_candidate':'group_score_top45pct AND NO_THIRD AND winner=multi-rider-line head with running_style=両 AND own_bante excluded',
        'definitions':{
            'race_ge1':'at least one other-line bante in the race has race_score greater than own bante',
            'race_ge2':'at least two other-line bantes in the race have race_score greater than own bante',
            'selected_ge1':'in bante-bante ticket, at least one of selected 2nd/3rd bantes has race_score greater than own bante',
            'selected_eq2':'in bante-bante ticket, both selected 2nd/3rd bantes have race_score greater than own bante',
        },
        'warning':'Same discovery data; these are candidate filters, not validation.',
        'usable_races_by_month':usable_by_month,'skipped':dict(skipped),'views':views,
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
