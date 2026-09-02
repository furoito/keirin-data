#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Untouched-July diagnostic for a one-rider correction to v0.1b.

The August miss audit suggested a very specific asymmetry:
- current candidate family sometimes keeps popular-line pos2;
- actual top3 often contains an omitted other-line head instead.

This script does NOT change the canonical strategy. It evaluates three minimal
pre-race correction candidates on July context that was not used to derive them:
  S0: replace popular pos2 with best omitted other-line head only if head_score >= pop2_score
  S1: same, allowing head within 1.0 point
  S3: same, allowing head within canonical 3.0-point boundary

Each correction preserves the number of candidate sets (max 2) and performs at
most one rider replacement per set. Results are used only after decisions are fixed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # patches popular-line detector
import keirin_scraper as ks

DATA = ROOT / 'keirin_data'
CTX = DATA / 'strategy_context'
MONTH = '2026_07'
OUT = CTX / 'pop2_swap_oos_july.csv'
SUMMARY = CTX / 'pop2_swap_oos_july_summary.json'
REPAIRS = CTX / 'pop2_swap_oos_july_result_repairs.csv'


def parse_rank(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(str(v).strip()); i = int(f)
        return i if abs(f-i) < 1e-9 else None
    except Exception:
        return None


def actual_from_rows(g):
    vals=[]
    for r in g.itertuples(index=False):
        p=parse_rank(getattr(r,'rank',None))
        try: fn=int(float(r.banum))
        except Exception: continue
        if p is not None and 1 <= p <= 3: vals.append((p,fn))
    vals.sort()
    return tuple(x for _,x in vals) if [p for p,_ in vals] == [1,2,3] else None


def family_from_orders(orders):
    fam=[]
    for o in orders or []:
        s=frozenset(o)
        if s not in fam: fam.append(s)
    return fam


def role(r, pop_line):
    if r.line_idx == pop_line:
        if r.line_pos == 2: return 'popular_pos2'
        if r.line_pos >= 3: return 'popular_pos3plus'
        return 'popular_head'
    if r.line_size == 1: return 'solo'
    if r.line_pos == 1: return 'other_head'
    if r.line_pos == 2: return 'other_pos2'
    return 'other_pos3plus'


def corrected_family(fam, riders, pop_line, margin):
    by={r.frame_no:r for r in riders}
    pop2=next((r for r in riders if r.line_idx==pop_line and r.line_pos==2),None)
    if pop2 is None: return fam
    out=[]
    for s in fam:
        ns=set(s)
        if pop2.frame_no in ns:
            heads=[r for r in riders if role(r,pop_line)=='other_head' and r.frame_no not in ns]
            if heads:
                alt=max(heads,key=lambda r:(r.race_score,-r.frame_no))
                if alt.race_score >= pop2.race_score - margin:
                    ns.remove(pop2.frame_no); ns.add(alt.frame_no)
        fs=frozenset(ns)
        if fs not in out: out.append(fs)
    return out[:2]


def best_overlap(fam, actual):
    aset=set(actual)
    return max((len(set(s)&aset) for s in fam),default=0)


def fetch_result(rid, venue, date, race_no):
    try: got=ks.parse_race(str(venue),str(rid))
    except Exception as e:
        print(f'RESULT ERROR {rid}: {type(e).__name__}: {e}')
        got=[]
    rows=[]
    for x in got:
        rows.append({'race_id':str(rid),'venue_slug':venue,'date':date,'race_no':race_no,
                     'banum':x.get('banum',''),'rank':x.get('rank','')})
    return rows


def metrics(df, prefix):
    n=len(df)
    if not n: return {'n':0}
    return {
        'n':n,
        'exact_n':int(df[f'{prefix}_exact'].sum()),
        'exact_pct':float(df[f'{prefix}_exact'].mean()*100),
        'avg_overlap':float(df[f'{prefix}_overlap'].mean()),
        'overlap_2plus_n':int((df[f'{prefix}_overlap']>=2).sum()),
    }


def main():
    base_df=pd.read_csv(DATA/f'{MONTH}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(CTX/f'{MONTH}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(CTX/f'{MONTH}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
    for d in (base_df,ctx,odds): d['race_id']=d.race_id.astype(str)
    use=ctx[(ctx.context_quality.astype(str)=='full') &
            (ctx.price_usable.astype(str).str.lower().isin({'true','1'}))].copy()
    rb={k:g for k,g in base_df.groupby('race_id',sort=False)}
    ob={k:g for k,g in odds.groupby('race_id',sort=False)}

    fixed=[]; pending=[]; decision_rows=[]
    for cr in use.itertuples(index=False):
        rid=str(cr.race_id)
        if rid not in rb or rid not in ob: continue
        g=rb[rid]; pre=g[['race_id','banum','race_score']].copy()
        d=base.decide(rid,pre,pd.Series(cr._asdict()),ob[rid])
        if d.action!='BET': continue
        act=actual_from_rows(g)
        if act is None:
            pending.append((rid,cr.venue_slug,cr.date,cr.race_no))
        decision_rows.append((cr,d,g,pre,act))

    # Result backfill only after all pre-race decisions are frozen.
    repairs=[]
    for i,(rid,venue,date,rno) in enumerate(pending,1):
        print(f'[{i}/{len(pending)}] result {date} {venue} {rno}R {rid}')
        repairs.extend(fetch_result(rid,venue,date,rno))
    rep=pd.DataFrame(repairs)
    rep.to_csv(REPAIRS,index=False,encoding='utf-8-sig')
    rep_by={k:g for k,g in rep.groupby('race_id',sort=False)} if not rep.empty else {}

    rows=[]
    for cr,d,g,pre,act in decision_rows:
        rid=str(cr.race_id)
        source='monthly'
        if act is None and rid in rep_by:
            act=actual_from_rows(rep_by[rid]); source='repair'
        if act is None: continue
        actual=frozenset(act)
        lines=base.parse_true_line(cr.true_line)
        riders=base.make_riders(pre,lines)
        fam=family_from_orders(d.candidate_orders)
        variants={'A':fam,'S0':corrected_family(fam,riders,d.popular_line,0.0),
                  'S1':corrected_family(fam,riders,d.popular_line,1.0),
                  'S3':corrected_family(fam,riders,d.popular_line,3.0)}
        rec={'race_id':rid,'date':cr.date,'venue':cr.venue_slug,'race_no':cr.race_no,
             'line':cr.true_line,'target':d.target,'result_source':source,
             'actual':'-'.join(map(str,act)),'head_bust':int(d.target not in actual)}
        for name,vfam in variants.items():
            rec[f'{name}_family']='|'.join('-'.join(map(str,sorted(s))) for s in vfam)
            rec[f'{name}_exact']=int(actual in vfam)
            rec[f'{name}_overlap']=best_overlap(vfam,act)
            rec[f'{name}_changed']=int(set(vfam)!=set(fam)) if name!='A' else 0
        rows.append(rec)

    out=pd.DataFrame(rows)
    out.to_csv(OUT,index=False,encoding='utf-8-sig')
    hb=out[out.head_bust==1]
    summary={
        'context_full_races':int(len(use)),
        'bet_races':int(len(decision_rows)),
        'scorable_bet_races':int(len(out)),
        'head_bust_races':int(len(hb)),
        'head_bust_rate_pct':float(out.head_bust.mean()*100) if len(out) else None,
        'all_bets':{k:metrics(out,k) for k in ['A','S0','S1','S3']},
        'head_bust_only':{k:metrics(hb,k) for k in ['A','S0','S1','S3']},
        'changed_races':{k:int(out[f'{k}_changed'].sum()) for k in ['S0','S1','S3']},
        'rule_note':'swap popular pos2 for best omitted other-line head only; max one rider per candidate set; no canonical rule change',
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
