#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate trifecta ordering with single-count line-position score bonuses.

Candidate SET selection stays frozen at v0.1b. Only order inside each selected
3-rider set changes.

Position advantage is encoded ONCE for trifecta ordering:
- popular-line position 2: race_score + 3.0
- popular-line position 3: race_score + 1.0
- everybody else:          race_score

After applying those bonuses, riders are ordered only by adjusted score.
The positional tier is NOT applied again, avoiding double-counting the same edge.
Raw race_score and frame number are deterministic tie-breakers only.

Candidate membership is never changed here. Third-slot A-B-CD can still yield two
sets, so max two bets. Trifecta minimum remains 30x.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # noqa: F401

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'; MONTH='2026_08'
OUT=CTX/'pop2_plus3_order_results.csv'; SUMMARY=CTX/'pop2_plus3_order_summary.json'
POS2_BONUS=3.0
POS3_BONUS=1.0


def parse_rank(v):
    if v is None or (isinstance(v,float) and math.isnan(v)): return None
    try:
        f=float(str(v).strip()); i=int(f)
        return i if abs(f-i)<1e-9 else None
    except Exception: return None


def actual_order(g):
    vals=[]
    for r in g.itertuples(index=False):
        p=parse_rank(getattr(r,'rank',None))
        try: fn=int(float(r.banum))
        except Exception: continue
        if p is not None and 1<=p<=3: vals.append((p,fn))
    vals.sort()
    return tuple(fn for _,fn in vals) if [p for p,_ in vals]==[1,2,3] else None


def unique_sets(orders):
    out=[]
    for o in orders or []:
        s=frozenset(o)
        if s not in out: out.append(s)
    return out


def effective_score(r, pop_line):
    if r.line_idx==pop_line:
        if r.line_pos==2: return r.race_score+POS2_BONUS
        if r.line_pos==3: return r.race_score+POS3_BONUS
    return r.race_score


def order_set(s, riders, pop_line):
    xs=[r for r in riders if r.frame_no in s]
    xs.sort(key=lambda r:(-effective_score(r,pop_line),-r.race_score,r.frame_no))
    return tuple(r.frame_no for r in xs)


def main():
    race=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{MONTH}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    repairs=pd.read_csv(CTX/'diagnostic_result_repairs.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds,repairs): d['race_id']=d.race_id.astype(str)
    use=ctx[(ctx.context_quality.astype(str)=='full') & ctx.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}
    repb={k:g for k,g in repairs.groupby('race_id',sort=False)}
    rows=[]
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id)
        if rid not in rb or rid not in ob: continue
        full=rb[rid]; pre=full[['race_id','banum','race_score']].copy(); og=ob[rid]
        d=base.decide(rid,pre,pd.Series(cr._asdict()),og)
        if d.action!='BET': continue
        act=actual_order(full); source='monthly'
        if act is None and rid in repb:
            act=actual_order(repb[rid]); source='repair'
        if act is None: continue

        lines=base.parse_true_line(cr.true_line); riders=base.make_riders(pre,lines); tri=base.odds_map(og)
        sets=unique_sets(d.candidate_orders)
        ordered=[order_set(s,riders,d.popular_line) for s in sets]
        eligible=[(o,tri[o]) for o in ordered if o in tri and tri[o]>=base.TRIFECTA_MIN_ODDS][:2]
        aset=frozenset(act)
        set_match=int(aset in sets)
        order_match=int(act in ordered)
        bet_hit=next(((o,od) for o,od in eligible if o==act),None)
        stake=len(eligible)*base.STAKE_YEN; pay=int(round(bet_hit[1]*base.STAKE_YEN)) if bet_hit else 0
        target=d.target; hb=int(target not in aset) if target is not None else 0

        eff=[]
        for fn in sorted(set().union(*sets)) if sets else []:
            r=next(x for x in riders if x.frame_no==fn)
            eff.append(f'{fn}:{effective_score(r,d.popular_line):.2f}')
        rows.append({'race_id':rid,'date':cr.date,'venue':cr.venue_slug,'race_no':cr.race_no,'line':cr.true_line,
          'target':target,'head_bust':hb,'result_source':source,'candidate_sets':'|'.join('-'.join(map(str,sorted(s))) for s in sets),
          'effective_scores':'|'.join(eff),'plus3_orders':'|'.join('-'.join(map(str,o)) for o in ordered),
          'bets':'|'.join('-'.join(map(str,o)) for o,_ in eligible),'bet_odds':'|'.join(f'{od:.2f}' for _,od in eligible),
          'actual':'-'.join(map(str,act)),'set_match':set_match,'order_match':order_match,'bet_hit':int(bet_hit is not None),'stake':stake,'pay':pay})

    out=pd.DataFrame(rows); out.to_csv(OUT,index=False,encoding='utf-8-sig')
    hb=out[out.head_bust==1]; exact=hb[hb.set_match==1]
    def m(x):
        n=len(x); st=int(x.stake.sum()) if n else 0; py=int(x.pay.sum()) if n else 0
        return {'n':n,'set_match_n':int(x.set_match.sum()) if n else 0,'order_match_n':int(x.order_match.sum()) if n else 0,
                'bet_hit_n':int(x.bet_hit.sum()) if n else 0,'stake_yen':st,'pay_yen':py,'roi_pct':py/st*100 if st else None}
    summary={'pos2_bonus':POS2_BONUS,'pos3_bonus':POS3_BONUS,
             'ordering_rule':'single-count position edge: +3 popular pos2, +1 popular pos3, then adjusted-score order only',
             'all_bets':m(out),'head_bust_only':m(hb),'head_bust_and_set_exact':m(exact),
             'exact_rows':exact[['date','venue','race_no','line','candidate_sets','effective_scores','plus3_orders','bets','bet_odds','actual','order_match','bet_hit','pay']].to_dict('records')}
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
