#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the CURRENT frozen popular-head-skip logic for one month.

Selection:
- v0.1b popular-line detector
- existing candidate-set logic (3-point boundary + position structure)

Trifecta order:
- popular-line pos2: +3 score
- popular-line pos3: +1 score
- everyone else: raw score
- position edge counted ONCE; order by adjusted score only

Important: the 30x threshold is applied to the NEW adjusted-score order, not to the
legacy order emitted by v0.1/v0.1b. This fixes the prior evaluation mismatch without
changing strategy rules.

If monthly result ranks are missing, fetch showResult only AFTER the pre-race bet is
frozen and store repairs separately.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # noqa: F401; patches base detector
import keirin_scraper as ks

DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
POS2_BONUS=3.0; POS3_BONUS=1.0


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


def effective_score(r,pop_line):
    if r.line_idx==pop_line:
        if r.line_pos==2: return r.race_score+POS2_BONUS
        if r.line_pos==3: return r.race_score+POS3_BONUS
    return r.race_score


def order_set(s,riders,pop_line):
    xs=[r for r in riders if r.frame_no in s]
    xs.sort(key=lambda r:(-effective_score(r,pop_line),-r.race_score,r.frame_no))
    return tuple(r.frame_no for r in xs)


def candidate_state(rid,pre,cr,og):
    """Return structural candidate state BEFORE the legacy order/price gate."""
    d=base.decide(rid,pre,pd.Series(cr._asdict()),og)
    # BET and odds_too_low both already passed all structural filters and have
    # candidate_orders. Re-price them using the CURRENT order below.
    if d.action=='BET' or d.reason=='odds_too_low':
        return d
    return None


def fetch_result_rows(rid,venue,date,race_no):
    try: got=ks.parse_race(str(venue),str(rid))
    except Exception as e:
        print(f'RESULT ERROR {rid}: {type(e).__name__}: {e}')
        got=[]
    rows=[]
    for x in got:
        rows.append({'race_id':str(rid),'venue_slug':venue,'date':date,'race_no':race_no,
                     'banum':x.get('banum',''),'rank':x.get('rank','')})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--month',required=True,help='YYYY_MM')
    ap.add_argument('--no-fetch-results',action='store_true')
    a=ap.parse_args(); month=a.month
    base_path=DATA/f'{month}_keirin.csv'; race_path=CTX/f'{month}_races.csv'; odds_path=CTX/f'{month}_odds_3rentan.csv'
    for p in (base_path,race_path,odds_path):
        if not p.exists(): raise SystemExit(f'missing: {p}')
    race=pd.read_csv(base_path,encoding='utf-8-sig',dtype={'race_id':str})
    ctx=pd.read_csv(race_path,encoding='utf-8-sig',dtype={'race_id':str})
    odds=pd.read_csv(odds_path,encoding='utf-8-sig',dtype={'race_id':str})
    for d in (race,ctx,odds): d['race_id']=d.race_id.astype(str)
    use=ctx.copy()
    if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
    if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
    rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}

    # Load prior repairs for this exact month if present, then append only missing.
    repairs_path=CTX/f'fixed_logic_{month}_result_repairs.csv'
    prior=pd.read_csv(repairs_path,encoding='utf-8-sig',dtype={'race_id':str}) if repairs_path.exists() else pd.DataFrame()
    if not prior.empty: prior['race_id']=prior.race_id.astype(str)
    repb={k:g for k,g in prior.groupby('race_id',sort=False)} if not prior.empty else {}
    new_repairs=[]; rows=[]

    for cr in use.itertuples(index=False):
        rid=str(cr.race_id); full=rb.get(rid); og=ob.get(rid)
        if full is None or og is None: continue
        pre=full[['race_id','banum','race_score']].copy()
        d=candidate_state(rid,pre,cr,og)
        if d is None: continue
        lines=base.parse_true_line(cr.true_line); riders=base.make_riders(pre,lines); tri=base.odds_map(og)
        sets=unique_sets(d.candidate_orders)
        if not sets: continue
        ordered=[order_set(s,riders,d.popular_line) for s in sets]
        eligible=[(o,tri[o]) for o in ordered if o in tri and tri[o]>=base.TRIFECTA_MIN_ODDS][:2]
        if not eligible: continue  # current fixed logic = PASS

        # Decision is now frozen. Only now inspect/fetch result.
        act=actual_order(full); source='monthly'
        if act is None and rid in repb:
            act=actual_order(repb[rid]); source='repair_existing'
        if act is None and not a.no_fetch_results:
            got=fetch_result_rows(rid,getattr(cr,'venue_slug',''),getattr(cr,'date',''),getattr(cr,'race_no',''))
            if got:
                new_repairs.extend(got); g=pd.DataFrame(got); act=actual_order(g); source='repair_fetched'
        aset=frozenset(act) if act else None
        set_match=int(aset in sets) if aset is not None else None
        order_match=int(act in ordered) if act else None
        hit=next(((o,od) for o,od in eligible if act==o),None) if act else None
        stake=len(eligible)*base.STAKE_YEN; pay=int(round(hit[1]*base.STAKE_YEN)) if hit else 0
        hb=int(d.target not in aset) if aset is not None and d.target is not None else None
        rows.append({'race_id':rid,'date':getattr(cr,'date',''),'venue':getattr(cr,'venue_slug',''),'race_no':getattr(cr,'race_no',''),
          'line':getattr(cr,'true_line',''),'target':d.target,'result_source':source,'head_bust':hb,
          'candidate_sets':'|'.join('-'.join(map(str,sorted(s))) for s in sets),
          'orders':'|'.join('-'.join(map(str,o)) for o in ordered),
          'bets':'|'.join('-'.join(map(str,o)) for o,_ in eligible),'bet_odds':'|'.join(f'{od:.2f}' for _,od in eligible),
          'actual':'-'.join(map(str,act)) if act else '', 'set_match':set_match,'order_match':order_match,
          'bet_hit':int(hit is not None) if act else None,'stake':stake,'pay':pay if act else 0})

    if new_repairs:
        nr=pd.DataFrame(new_repairs)
        combined=nr if prior.empty else pd.concat([prior,nr],ignore_index=True)
        combined=combined.drop_duplicates(['race_id','banum'],keep='last')
        combined.to_csv(repairs_path,index=False,encoding='utf-8-sig')
    elif not repairs_path.exists():
        pd.DataFrame(columns=['race_id','venue_slug','date','race_no','banum','rank']).to_csv(repairs_path,index=False,encoding='utf-8-sig')

    out=pd.DataFrame(rows)
    results_path=CTX/f'fixed_logic_{month}_results.csv'; summary_path=CTX/f'fixed_logic_{month}_summary.json'
    out.to_csv(results_path,index=False,encoding='utf-8-sig')
    sc=out[out.actual.astype(str)!=''] if not out.empty else out
    hb=sc[sc.head_bust==1] if not sc.empty else sc
    exact=hb[hb.set_match==1] if not hb.empty else hb
    def m(x):
        n=len(x); st=int(x.stake.sum()) if n else 0; py=int(x.pay.sum()) if n else 0
        return {'n':n,'head_bust_n':int(x.head_bust.fillna(0).sum()) if n and 'head_bust' in x else 0,
                'set_match_n':int(x.set_match.fillna(0).sum()) if n and 'set_match' in x else 0,
                'order_match_n':int(x.order_match.fillna(0).sum()) if n and 'order_match' in x else 0,
                'bet_hit_n':int(x.bet_hit.fillna(0).sum()) if n and 'bet_hit' in x else 0,
                'stake_yen':st,'pay_yen':py,'profit_yen':py-st,'roi_pct':py/st*100 if st else None}
    payload={'month':month,'context_full_races':len(use),'current_bet_races':len(out),'scorable_bet_races':len(sc),
      'rule':'v0.1b candidate sets; single-count +3 popular pos2, +1 popular pos3; adjusted-score order; current-order >=30x; max2',
      'all_bets':m(sc),'head_bust_only':m(hb),'head_bust_and_set_exact':m(exact)}
    summary_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
