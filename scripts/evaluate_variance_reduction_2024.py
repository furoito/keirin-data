#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare frozen OH_HIGH candidate-set monetization on the 2024 holdout.

The candidate logic stays frozen. When OH_HIGH has more than one candidate set,
we use the set containing the already-frozen MKT1 pick; this makes all compared
products use the same selected 3-rider set.

Compared products:
- MKT1: existing one trifecta ticket, >=30x.
- BOX6: all six trifecta permutations of the MKT1-selected set, no odds filter.
- TRIO: one 3-ren-fuku ticket on the same selected set. Uses the official final
  payout stored in the monthly race CSV only for scoring. No pre-race 15x gate
  is applied because historical pre-race trio odds are not yet stored.

Result information is never used to select the set or tickets.
"""
from __future__ import annotations
import itertools, json, math, re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
MONTHS=[f'2024_{m:02d}' for m in range(1,13)]
STAKE=100
OUT=CTX/'variance_reduction_2024_cases.csv'
SUMMARY=CTX/'variance_reduction_2024_summary.json'


def parse_order(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return None
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None


def parse_pick(v):
    return parse_order(v)


def odds_map(d):
    out={}
    for r in d.itertuples(index=False):
        try:k=(int(float(r.b1)),int(float(r.b2)),int(float(r.b3)));o=float(r.odds_decimal)
        except Exception:continue
        if o>0:out[k]=o
    return out


def parse_payout_value(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return None
    s=str(v).replace(',','').replace('，','').strip()
    if not s or s.lower()=='nan': return None
    nums=[]
    for x in re.findall(r'\d+(?:\.\d+)?',s):
        try: nums.append(float(x))
        except Exception: pass
    if not nums:return None
    # Combination numbers are 1..9; the payout is normally the largest number.
    val=max(nums)
    return int(round(val)) if val>=100 else None


def race_trio_payout(full):
    if 'san_ren_fuku' not in full.columns:return None
    vals=[]
    for v in full['san_ren_fuku'].dropna().tolist():
        x=parse_payout_value(v)
        if x is not None: vals.append(x)
    if not vals:return None
    # The result payout is repeated across rider rows in the source; use mode.
    s=pd.Series(vals)
    mode=s.mode()
    return int(mode.iloc[0] if len(mode) else s.iloc[0])


def max_losing_streak(hit_flags):
    best=cur=0
    for h in hit_flags:
        if h:cur=0
        else:
            cur+=1;best=max(best,cur)
    return best


def max_drawdown(profits):
    equity=peak=0;maxdd=0
    for p in profits:
        equity+=p;peak=max(peak,equity);maxdd=max(maxdd,peak-equity)
    return int(maxdd)


def summarize(df, product):
    z=df[df['product']==product].copy()
    if z.empty:return {}
    st=int(z.stake.sum());py=int(z.pay.sum()); profits=(z.pay-z.stake).astype(float)
    returns=(z.pay/z.stake).where(z.stake>0).dropna()
    return {
        'races':len(z),'bets':int((z.stake>0).sum()),'hits':int(z.hit.sum()),
        'hit_rate_pct':100*z.hit.sum()/max(1,(z.stake>0).sum()),
        'stake_yen':st,'pay_yen':py,'profit_yen':py-st,'roi_pct':100*py/st if st else None,
        'max_losing_streak_bets':max_losing_streak(z.loc[z.stake>0,'hit'].astype(bool).tolist()),
        'max_drawdown_yen':max_drawdown(profits.tolist()),
        'per_bet_return_mean':float(returns.mean()) if len(returns) else None,
        'per_bet_return_std':float(returns.std(ddof=1)) if len(returns)>1 else None,
    }


def main():
    rows=[]; diagnostics={'trio_payout_missing_on_hit':0,'months':{}}
    for month in MONTHS:
        mp=CTX/f'oh_high_mkt1_{month}_results.csv';op=CTX/f'{month}_odds_3rentan.csv';rp=DATA/f'{month}_keirin.csv'
        if not (mp.exists() and op.exists() and rp.exists()):continue
        mkt=pd.read_csv(mp,encoding='utf-8-sig',dtype={'race_id':str});odds=pd.read_csv(op,encoding='utf-8-sig',dtype={'race_id':str});race=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str})
        for d in (mkt,odds,race):d['race_id']=d.race_id.astype(str)
        ob={k:g for k,g in odds.groupby('race_id',sort=False)};rb={k:g for k,g in race.groupby('race_id',sort=False)}
        dc={'cases':0,'trio_payout_parsed':0}
        for rr in mkt.itertuples(index=False):
            pick=parse_pick(getattr(rr,'pick',''))
            if not pick:continue
            rid=str(rr.race_id);og=ob.get(rid);full=rb.get(rid)
            if og is None or full is None:continue
            selected_set=frozenset(pick);act=parse_order(getattr(rr,'actual',''));actual_set=frozenset(act) if act else None
            om=odds_map(og);dc['cases']+=1
            # Existing MKT1.
            po=getattr(rr,'pick_odds',None)
            try:po=float(po)
            except Exception:po=om.get(pick)
            hit1=bool(act and act==pick);pay1=int(round(po*STAKE)) if hit1 and po else 0
            rows.append({'month':month,'race_id':rid,'product':'MKT1','selected_set':'-'.join(map(str,sorted(selected_set))),
                         'actual':getattr(rr,'actual',''),'stake':STAKE,'pay':pay1,'hit':int(hit1)})
            # Six-way trifecta BOX on exactly the same selected set.
            perms=list(itertools.permutations(sorted(selected_set),3));available=[(p,om.get(p)) for p in perms if om.get(p) is not None]
            stake6=len(available)*STAKE;hit6=bool(act and actual_set==selected_set and act in dict(available));pay6=int(round(om[act]*STAKE)) if hit6 else 0
            rows.append({'month':month,'race_id':rid,'product':'BOX6','selected_set':'-'.join(map(str,sorted(selected_set))),
                         'actual':getattr(rr,'actual',''),'stake':stake6,'pay':pay6,'hit':int(hit6)})
            # One 3-ren-fuku ticket on the same selected set.
            trio_pay=race_trio_payout(full);hit3=bool(act and actual_set==selected_set)
            if trio_pay is not None:dc['trio_payout_parsed']+=1
            if hit3 and trio_pay is None:diagnostics['trio_payout_missing_on_hit']+=1
            rows.append({'month':month,'race_id':rid,'product':'TRIO','selected_set':'-'.join(map(str,sorted(selected_set))),
                         'actual':getattr(rr,'actual',''),'stake':STAKE,'pay':int(trio_pay) if hit3 and trio_pay else 0,'hit':int(hit3 and trio_pay is not None)})
        diagnostics['months'][month]=dc
    out=pd.DataFrame(rows);out.to_csv(OUT,index=False,encoding='utf-8-sig')
    payload={'scope':'2024 holdout; frozen OH_HIGH; selected set = set containing frozen MKT1 pick',
             'important_note':'TRIO has no historical pre-race 15x filter because pre-race 3-ren-fuku odds are not stored; final payout is used only after the set is frozen for scoring.',
             'products':{p:summarize(out,p) for p in ['MKT1','BOX6','TRIO']},'diagnostics':diagnostics}
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
