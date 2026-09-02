#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full A/B/C reconstruction diagnostic using repaired result ranks when needed."""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import pandas as pd
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'; MONTH='2026_08'
OUT=CTX/'reconstruction_abc_v4_full50.csv'; SUMMARY=CTX/'reconstruction_abc_v4_full50_summary.json'


def parse_rank(v):
    try:
        if v is None or (isinstance(v,float) and math.isnan(v)): return None
        f=float(str(v).strip()); i=int(f)
        return i if abs(f-i)<1e-9 else None
    except Exception: return None


def actual_from_group(g):
    vals=[]
    for r in g.itertuples(index=False):
        p=parse_rank(getattr(r,'rank',None))
        try: fn=int(float(r.banum))
        except Exception: continue
        if p is not None and 1<=p<=3: vals.append((p,fn))
    vals.sort()
    return tuple(fn for _,fn in vals) if [p for p,_ in vals]==[1,2,3] else None


def score_set(riders,target):
    x=sorted([r for r in riders if r.frame_no!=target],key=lambda r:(-r.race_score,r.frame_no))[:3]
    return frozenset(r.frame_no for r in x)


def pos_family(riders,target,pop_line):
    by={}
    for r in riders:
        if r.frame_no==target: continue
        by.setdefault(base.position_tier(r,pop_line),[]).append(r.frame_no)
    chosen=[]; need=3; fam=set()
    for tier in sorted(by,reverse=True):
        ms=sorted(by[tier])
        if len(ms)<need:
            chosen+=ms; need-=len(ms); continue
        for pick in itertools.combinations(ms,need): fam.add(frozenset(chosen+list(pick)))
        need=0; break
    return fam if need==0 else set()


def best_overlap(fam,act): return max((len(set(x)&set(act)) for x in fam),default=0)
def candidate_sets(d): return {frozenset(o) for o in (d.candidate_orders or [])}


def metrics(x):
    n=len(x)
    if not n:return {'n':0}
    return {
      'n':n,
      'A_exact_n':int(x.A_exact.sum()),'A_exact_pct':float(x.A_exact.mean()*100),'A_avg_overlap':float(x.A_overlap.mean()),
      'B_exact_n':int(x.B_exact.sum()),'B_exact_pct':float(x.B_exact.mean()*100),'B_avg_overlap':float(x.B_overlap.mean()),
      'C_feasible_exact_n':int(x.C_exact.sum()),'C_feasible_exact_pct':float(x.C_exact.mean()*100),'C_avg_best_overlap':float(x.C_overlap.mean()),
      'C_unique_rate_pct':float(x.C_unique.mean()*100),
      'A_overlap_3_n':int((x.A_overlap==3).sum()),'A_overlap_2plus_n':int((x.A_overlap>=2).sum()),
      'B_overlap_3_n':int((x.B_overlap==3).sum()),'B_overlap_2plus_n':int((x.B_overlap>=2).sum()),
      'C_overlap_3_n':int((x.C_overlap==3).sum()),'C_overlap_2plus_n':int((x.C_overlap>=2).sum()),
    }


def main():
    race=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{MONTH}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    repairs=pd.read_csv(CTX/'diagnostic_result_repairs.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds,repairs): d['race_id']=d.race_id.astype(str)
    use=ctx[(ctx.context_quality.astype(str)=='full') & ctx.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    repb={k:g for k,g in repairs.groupby('race_id',sort=False)}
    rows=[]; bet_count=0; unresolved=[]
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id)
        if rid not in rb or rid not in ob: continue
        g=rb[rid]; pre=g[['race_id','banum','race_score']].copy()
        d=base.decide(rid,pre,pd.Series(cr._asdict()),ob[rid])
        if d.action!='BET': continue
        bet_count+=1
        act=actual_from_group(g)
        result_source='monthly'
        if not act and rid in repb:
            act=actual_from_group(repb[rid]); result_source='repair'
        if not act:
            unresolved.append(rid); continue
        actset=frozenset(act); riders=base.make_riders(pre,base.parse_true_line(cr.true_line))
        A=candidate_sets(d); B=score_set(riders,d.target); C=pos_family(riders,d.target,d.popular_line)
        rows.append({'race_id':rid,'date':cr.date,'venue':cr.venue_slug,'race_no':cr.race_no,'line':cr.true_line,
          'target':d.target,'result_source':result_source,'head_bust':int(d.target not in actset),'actual':'-'.join(map(str,act)),
          'A_candidates':'|'.join('-'.join(map(str,o)) for o in (d.candidate_orders or [])),
          'A_exact':int(actset in A),'A_overlap':best_overlap(A,actset),
          'B_set':'-'.join(map(str,sorted(B))),'B_exact':int(actset==B),'B_overlap':len(B&actset),
          'C_n':len(C),'C_exact':int(actset in C),'C_overlap':best_overlap(C,actset),'C_unique':int(len(C)==1)})
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False,encoding='utf-8-sig')
    hb=out[out.head_bust==1]
    s={'recomputed_bet_count':bet_count,'scorable_bet_count':len(out),'unresolved':unresolved,
       'repaired_result_races':int((out.result_source=='repair').sum()),
       'head_bust_count':len(hb),'head_bust_rate_pct':float(out.head_bust.mean()*100) if len(out) else None,
       'all_bets':metrics(out),'head_bust_only':metrics(hb)}
    SUMMARY.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(s,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
