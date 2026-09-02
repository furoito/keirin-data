#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore pre-race signals for ordering an already-correct 3-rider candidate set.

Reporting only. No strategy changes.

Signals checked for BASE and OH_HIGH, on head-bust races where the candidate set was exact:
- current adjusted-score order (+3 popular pos2, +1 popular pos3)
- raw race_score descending
- KDreams mark_num descending, if available
- market favorite permutation among the six exact-set trifectas
- market implied first-place favorite within the exact set (sum inverse odds by first rider)
- actual finish-role distribution by line role
- winner score-rank and score gaps

Also reports the hypothetical performance of market-favorite exact-set ordering at the existing 30x minimum,
strictly as a diagnostic (not a proposed betting rule).
"""
from __future__ import annotations
import itertools, json, math
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
OUT=CTX/'ordering_signal_cases.csv'; SUMMARY=CTX/'ordering_signal_summary.json'
VARS=['BASE','OH_HIGH']
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]


def split_orders(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for p in str(v).split('|'):
        p=p.strip()
        if not p: continue
        try:q=tuple(int(float(x)) for x in p.split('-'))
        except Exception:continue
        if len(q)==3 and len(set(q))==3:out.append(q)
    return out


def parse_one(v):
    z=split_orders(v); return z[0] if z else None


def parse_line(s):
    groups=[]
    for g in str(s or '').split('/'):
        try:x=[int(float(v)) for v in g.split('-') if str(v).strip()]
        except Exception:return []
        if x:groups.append(x)
    return groups


def role_map(line,pop_line,target):
    d={}
    for li,g in enumerate(line,1):
        for pos,fn in enumerate(g,1):
            if fn==target:d[fn]='target'
            elif len(g)==1:d[fn]='solo'
            elif li==pop_line and pos==2:d[fn]='popular_pos2'
            elif li==pop_line and pos>=3:d[fn]='popular_pos3plus'
            elif li!=pop_line and pos==1:d[fn]='other_head'
            elif li!=pop_line and pos==2:d[fn]='other_pos2'
            else:d[fn]='other_pos3plus'
    return d


def odds_map(d):
    out={}
    for r in d.itertuples(index=False):
        try:k=(int(float(r.b1)),int(float(r.b2)),int(float(r.b3)));o=float(r.odds_decimal)
        except Exception:continue
        if o>0:out[k]=o
    return out


def load_results():
    parts=[]
    big=CTX/'other_head_promotion_oos_results.csv'
    if big.exists():
        d=pd.read_csv(big,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.isin(VARS)].copy())
    for month in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{month}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.isin(VARS)].copy())
    x=pd.concat(parts,ignore_index=True); x['race_id']=x.race_id.astype(str)
    return x.drop_duplicates(['month','race_id','variant'],keep='last')


def month_data(month):
    race=pd.read_csv(DATA/f'{month}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{month}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{month}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds):d['race_id']=d.race_id.astype(str)
    return ({k:g for k,g in race.groupby('race_id',sort=False)},
            {k:g.iloc[0] for k,g in ctx.groupby('race_id',sort=False)},
            {k:g for k,g in odds.groupby('race_id',sort=False)})


def pop_line_index(line,target):
    for li,g in enumerate(line,1):
        if target in g:return li
    return None


def ranked_order(frames,values):
    return tuple(sorted(frames,key=lambda f:(-values.get(f,float('-inf')),f)))


def main():
    res=load_results(); rows=[]
    ag={v:{'n':0,'current_exact':0,'raw_exact':0,'mark_exact':0,'mark_n':0,'market_perm_exact':0,
           'market_winner_exact':0,'market_winner_n':0,'raw_winner_exact':0,'winner_score_rank':Counter(),
           'role_by_finish':{1:Counter(),2:Counter(),3:Counter()},'market30_n':0,'market30_hits':0,'market30_stake':0,'market30_pay':0,
           'actual_order_odds':[],'favorite_order_odds':[],'score_gap_1_2':[],'score_gap_1_3':[]} for v in VARS}
    available_columns=set()
    for month in MONTHS:
        subset=res[res.month.astype(str)==month]
        if subset.empty:continue
        rb,cb,ob=month_data(month)
        for rr in subset.itertuples(index=False):
            if int(float(getattr(rr,'head_bust',0) or 0))!=1 or int(float(getattr(rr,'set_match',0) or 0))!=1:continue
            rid=str(rr.race_id); full=rb.get(rid); cr=cb.get(rid); og=ob.get(rid)
            if full is None or cr is None or og is None:continue
            available_columns.update(full.columns)
            act=parse_one(rr.actual)
            if not act:continue
            sets=split_orders(rr.candidate_sets)
            # candidate_sets are sorted set encodings; find actual set explicitly.
            cset=set(act)
            line=parse_line(cr.get('true_line',''))
            target=int(float(rr.target)); pl=pop_line_index(line,target)
            roles=role_map(line,pl,target)
            scores={int(float(r.banum)):float(r.race_score) for r in full.itertuples(index=False) if pd.notna(r.banum) and pd.notna(r.race_score)}
            marks={}
            if 'mark_num' in full.columns:
                for r in full.itertuples(index=False):
                    try:marks[int(float(r.banum))]=float(r.mark_num)
                    except Exception:pass
            om=odds_map(og); perms=list(itertools.permutations(act,3)); perm_odds={p:om.get(p) for p in perms if om.get(p) is not None}
            current=parse_one(rr.orders)
            raw=ranked_order(act,scores)
            mark=ranked_order(act,marks) if all(f in marks and not math.isnan(marks[f]) for f in act) else None
            fav=min(perm_odds,key=lambda p:perm_odds[p]) if perm_odds else None
            first_mass=defaultdict(float)
            for p,o in perm_odds.items():first_mass[p[0]]+=1.0/o
            market_first=max(first_mass,key=first_mass.get) if first_mass else None
            a=ag[rr.variant]; a['n']+=1
            a['current_exact']+=int(current==act); a['raw_exact']+=int(raw==act); a['raw_winner_exact']+=int(raw[0]==act[0])
            if mark:
                a['mark_n']+=1;a['mark_exact']+=int(mark==act)
            if fav:
                a['market_perm_exact']+=int(fav==act);a['favorite_order_odds'].append(perm_odds[fav]);a['actual_order_odds'].append(perm_odds.get(act,float('nan')))
                if perm_odds[fav]>=30:
                    a['market30_n']+=1;a['market30_stake']+=100
                    if fav==act:a['market30_hits']+=1;a['market30_pay']+=int(round(perm_odds[fav]*100))
            if market_first is not None:
                a['market_winner_n']+=1;a['market_winner_exact']+=int(market_first==act[0])
            sorted_scores=sorted(((scores.get(f,float('-inf')),f) for f in act),reverse=True)
            rank={f:i+1 for i,(_,f) in enumerate(sorted_scores)};a['winner_score_rank'][str(rank.get(act[0],0))]+=1
            if all(f in scores for f in act):
                a['score_gap_1_2'].append(scores[act[0]]-scores[act[1]]);a['score_gap_1_3'].append(scores[act[0]]-scores[act[2]])
            for pos,fn in enumerate(act,1):a['role_by_finish'][pos][roles.get(fn,'unknown')]+=1
            rows.append({'month':month,'race_id':rid,'variant':rr.variant,'actual':'-'.join(map(str,act)),
                         'current':'-'.join(map(str,current)) if current else '','raw':'-'.join(map(str,raw)),
                         'mark':'-'.join(map(str,mark)) if mark else '','market_fav':'-'.join(map(str,fav)) if fav else '',
                         'market_first':market_first if market_first is not None else '',
                         'actual_order_odds':perm_odds.get(act,''),'market_fav_odds':perm_odds.get(fav,'') if fav else '',
                         'winner_role':roles.get(act[0],''),'second_role':roles.get(act[1],''),'third_role':roles.get(act[2],''),
                         'winner_score':scores.get(act[0],''),'second_score':scores.get(act[1],''),'third_score':scores.get(act[2],''),
                         'winner_mark':marks.get(act[0],''),'second_mark':marks.get(act[1],''),'third_mark':marks.get(act[2],'')})
    payload={'available_monthly_columns':sorted(available_columns),'variants':{}}
    for v,a in ag.items():
        n=a['n']; mn=a['mark_n']; mwn=a['market_winner_n']; st=a['market30_stake'];py=a['market30_pay']
        payload['variants'][v]={
            'exact_set_cases':n,
            'exact_order_accuracy_pct':{
                'current_adjusted_score':100*a['current_exact']/n if n else None,
                'raw_race_score':100*a['raw_exact']/n if n else None,
                'mark_num':100*a['mark_exact']/mn if mn else None,
                'market_favorite_permutation':100*a['market_perm_exact']/n if n else None,
            },
            'winner_accuracy_pct':{
                'raw_score_highest':100*a['raw_winner_exact']/n if n else None,
                'market_first_mass':100*a['market_winner_exact']/mwn if mwn else None,
            },
            'winner_score_rank_distribution':a['winner_score_rank'].most_common(),
            'actual_role_by_finish':{str(p):c.most_common() for p,c in a['role_by_finish'].items()},
            'score_gap_actual_winner_minus_second':{'mean':sum(a['score_gap_1_2'])/len(a['score_gap_1_2']) if a['score_gap_1_2'] else None,'median':pd.Series(a['score_gap_1_2']).median() if a['score_gap_1_2'] else None},
            'score_gap_actual_winner_minus_third':{'mean':sum(a['score_gap_1_3'])/len(a['score_gap_1_3']) if a['score_gap_1_3'] else None,'median':pd.Series(a['score_gap_1_3']).median() if a['score_gap_1_3'] else None},
            'market_favorite_perm_at_30x_diagnostic':{'bets':a['market30_n'],'hits':a['market30_hits'],'stake_yen':st,'pay_yen':py,'roi_pct':100*py/st if st else None},
            'mark_cases':mn,
        }
    pd.DataFrame(rows).to_csv(OUT,index=False,encoding='utf-8-sig')
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
