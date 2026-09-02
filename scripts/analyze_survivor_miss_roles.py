#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify structural roles of actual survivors missed by A on head-bust races."""
from __future__ import annotations
import ast, json
from collections import Counter
from pathlib import Path
import pandas as pd
import popular_head_skip_v01 as base

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'; MONTH='2026_08'
SRC=CTX/'reconstruction_abc_v4_full50.csv'
OUT=CTX/'survivor_miss_roles.csv'; SUMMARY=CTX/'survivor_miss_roles_summary.json'


def parse_order(s): return tuple(int(x) for x in str(s).split('-') if x!='')
def parse_candidates(s): return [parse_order(x) for x in str(s).split('|') if x]

def role(r,pop_line):
    if r.line_idx==pop_line:
        if r.line_pos==2:return 'popular_pos2'
        if r.line_pos>=3:return 'popular_pos3plus'
        return 'popular_head'
    if r.line_size==1:return 'solo'
    if r.line_pos==1:return 'other_head'
    if r.line_pos==2:return 'other_pos2'
    return 'other_pos3plus'


def main():
    x=pd.read_csv(SRC,encoding='utf-8-sig',dtype={'race_id':str})
    x=x[(x.head_bust==1)&(x.A_exact==0)].copy()
    race=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str}).set_index('race_id')
    for d in (race,):d['race_id']=d.race_id.astype(str)
    rb={k:g for k,g in race.groupby('race_id',sort=False)}
    rows=[]; miss_roles=Counter(); extra_roles=Counter(); miss_score_gap=[]
    for q in x.itertuples(index=False):
        rid=str(q.race_id); g=rb[rid]; cr=ctx.loc[rid]
        lines=base.parse_true_line(cr.true_line)
        pre=g[['race_id','banum','race_score']].copy(); riders=base.make_riders(pre,lines)
        by={r.frame_no:r for r in riders}
        target=int(q.target); pop_line=by[target].line_idx
        actual=set(parse_order(q.actual)); cands=parse_candidates(q.A_candidates)
        # Audit the candidate set with maximum actual overlap; ties preserve all variants.
        best=max((len(set(c)&actual) for c in cands),default=0)
        bests=[c for c in cands if len(set(c)&actual)==best]
        # union labels across tied best candidates; counts are per race-role presence to avoid double counting.
        race_miss=set(); race_extra=set(); detail=[]
        for c in bests:
            cs=set(c); missing=actual-cs; extra=cs-actual
            for fn in missing:
                rr=by[fn]; race_miss.add(role(rr,pop_line)); detail.append(f'miss{fn}:{role(rr,pop_line)}:s{rr.race_score:.2f}')
            for fn in extra:
                rr=by[fn]; race_extra.add(role(rr,pop_line)); detail.append(f'extra{fn}:{role(rr,pop_line)}:s{rr.race_score:.2f}')
        miss_roles.update(race_miss); extra_roles.update(race_extra)
        rows.append({'race_id':rid,'date':q.date,'venue':q.venue,'race_no':q.race_no,'line':q.line,'target':target,
                     'actual':q.actual,'A_candidates':q.A_candidates,'A_overlap':q.A_overlap,
                     'miss_roles':'|'.join(sorted(race_miss)),'extra_roles':'|'.join(sorted(race_extra)),
                     'detail':' ; '.join(detail)})
    out=pd.DataFrame(rows);out.to_csv(OUT,index=False,encoding='utf-8-sig')
    s={'head_bust_A_miss_races':len(out),'miss_role_race_counts':miss_roles.most_common(),
       'extra_role_race_counts':extra_roles.most_common(),
       'overlap_distribution':{str(k):int(v) for k,v in out.A_overlap.value_counts().sort_index().items()}}
    SUMMARY.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(s,ensure_ascii=False,indent=2));print(out.to_string(index=False))

if __name__=='__main__':main()
