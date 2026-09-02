#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools, json
from pathlib import Path
import pandas as pd
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # patches base popular-line detector

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'; MONTH='2026_08'
OUT=CTX/'reconstruction_abc_v2_diagnostic.csv'; SUMMARY=CTX/'reconstruction_abc_v2_summary.json'

def score_set(riders,target):
    x=[r for r in riders if r.frame_no!=target]
    x=sorted(x,key=lambda r:(-r.race_score,r.frame_no))[:3]
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

def cand_sets(orders): return {frozenset(o) for o in (orders or [])}
def best_overlap(fam,act): return max((len(set(x)&set(act)) for x in fam),default=0)

def metrics(x):
    n=len(x)
    if not n:return {'n':0}
    return {
      'n':n,
      'A_exact_pct':float(x.A_exact.mean()*100),'A_avg_overlap':float(x.A_overlap.mean()),
      'B_exact_pct':float(x.B_exact.mean()*100),'B_avg_overlap':float(x.B_overlap.mean()),
      'C_feasible_exact_pct':float(x.C_exact.mean()*100),'C_avg_best_overlap':float(x.C_overlap.mean()),
      'C_unique_rate_pct':float(x.C_unique.mean()*100),
    }

def main():
    race=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{MONTH}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds): d['race_id']=d.race_id.astype(str)
    use=ctx.copy()
    use=use[use.context_quality.astype(str)=='full']
    use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    rows=[]; bet_count=0
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id)
        if rid not in rb or rid not in ob: continue
        g=rb[rid]; pre=g[['race_id','banum','race_score']].copy(); og=ob[rid]
        d=base.decide(rid,pre,pd.Series(cr._asdict()),og)
        if d.action!='BET': continue
        bet_count+=1
        act=base.actual_order(g)
        if not act: continue
        actset=frozenset(act)
        lines=base.parse_true_line(cr.true_line); riders=base.make_riders(pre,lines)
        A=cand_sets(d.candidate_orders); B=score_set(riders,d.target); C=pos_family(riders,d.target,d.popular_line)
        rows.append({'race_id':rid,'date':cr.date,'venue':cr.venue_slug,'race_no':cr.race_no,'line':cr.true_line,
          'target':d.target,'head_bust':int(d.target not in actset),'actual':'-'.join(map(str,act)),
          'A_exact':int(actset in A),'A_overlap':best_overlap(A,actset),
          'B_exact':int(actset==B),'B_overlap':len(B&actset),
          'C_exact':int(actset in C),'C_overlap':best_overlap(C,actset),'C_unique':int(len(C)==1),
          'A_n':len(A),'C_n':len(C)})
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False,encoding='utf-8-sig')
    hb=out[out.head_bust==1]
    s={'recomputed_bet_count':bet_count,'scorable_bet_count':len(out),'head_bust_count':len(hb),
       'head_bust_rate_pct':float(out.head_bust.mean()*100) if len(out) else None,
       'all_bets':metrics(out),'head_bust_only':metrics(hb)}
    SUMMARY.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
