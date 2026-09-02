#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen OH_HIGH selected set as one 3-ren-fuku ticket with >=15x gate.

Selection is frozen by the existing MKT1 pick: use exactly the 3-rider set
containing that pre-race MKT1 permutation. The only new decision is whether the
confirmed 3-ren-fuku odds for that set are >= 15.0. Results are read only after
that buy/skip decision is fixed.
"""
from __future__ import annotations
import argparse, json, random, re, time
from io import StringIO
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36'
MIN_ODDS=15.0; STAKE=100


def get(session,url,retries=4):
    for i in range(retries):
        try:
            r=session.get(url,headers={'User-Agent':UA},timeout=20)
            if r.status_code==200 and r.text:return r.text
            if r.status_code==404:return None
        except Exception:pass
        time.sleep(2**i+random.random())
    return None


def num(v):
    try:
        x=int(float(v)); return x if 1<=x<=9 else None
    except Exception:return None


def odd(v):
    try:
        x=float(str(v).replace(',','').strip()); return x if x>0 else None
    except Exception:return None


def parse_trio(html):
    try:tables=pd.read_html(StringIO(html))
    except Exception:return {}
    out={}
    for df in tables:
        if df.shape[1]!=3:continue
        m=re.match(r'\s*([1-9])',str(df.columns[0]))
        if not m:continue
        first=int(m.group(1))
        for _,r in df.iterrows():
            second=num(r.iloc[0]); third=num(r.iloc[1]); o=odd(r.iloc[2])
            if second is None or third is None or o is None:continue
            if len({first,second,third})<3:continue
            out[tuple(sorted((first,second,third)))]=o
    return out


def parse_pick(v):
    try:
        p=tuple(int(float(x)) for x in str(v).split('-'))
        return p if len(p)==3 and len(set(p))==3 else None
    except Exception:return None


def parse_actual(v): return parse_pick(v)


def parse_payout_value(v):
    if v is None or (isinstance(v,float) and pd.isna(v)):return None
    nums=[]
    for x in re.findall(r'\d+(?:\.\d+)?',str(v).replace(',','')):
        try:nums.append(float(x))
        except Exception:pass
    if not nums:return None
    z=max(nums); return int(round(z)) if z>=100 else None


def race_trio_payout(full):
    if 'san_ren_fuku' not in full.columns:return None
    vals=[parse_payout_value(v) for v in full.san_ren_fuku.dropna().tolist()]
    vals=[v for v in vals if v is not None]
    if not vals:return None
    mode=pd.Series(vals).mode(); return int(mode.iloc[0] if len(mode) else vals[0])


def max_losing_streak(hit_flags):
    best=cur=0
    for h in hit_flags:
        if h:cur=0
        else:cur+=1;best=max(best,cur)
    return best


def max_drawdown(profits):
    eq=peak=dd=0
    for p in profits:
        eq+=p;peak=max(peak,eq);dd=max(dd,peak-eq)
    return int(dd)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--month',required=True);a=ap.parse_args();m=a.month
    mp=CTX/f'oh_high_mkt1_{m}_results.csv';cp=CTX/f'{m}_races.csv';rp=DATA/f'{m}_keirin.csv'
    mkt=pd.read_csv(mp,encoding='utf-8-sig',dtype={'race_id':str});ctx=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str});race=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str})
    for d in (mkt,ctx,race):d['race_id']=d.race_id.astype(str)
    cm={str(r.race_id):r for r in ctx.itertuples(index=False)};rb={k:g for k,g in race.groupby('race_id',sort=False)}
    s=requests.Session();rows=[]
    for rr in mkt.itertuples(index=False):
        pick=parse_pick(getattr(rr,'pick',''))
        if not pick:continue
        rid=str(rr.race_id);cr=cm.get(rid);full=rb.get(rid)
        if cr is None or full is None:continue
        selected=tuple(sorted(pick)); venue=str(cr.venue_slug)
        url=f'https://keirin.kdreams.jp/{venue}/racedetail/{rid}/?pageType=odds&kakeshikiType=3renhuku'
        html=get(s,url); om=parse_trio(html) if html else {}; o=om.get(selected)
        buy=bool(o is not None and o>=MIN_ODDS)
        act=parse_actual(getattr(rr,'actual','')); aset=tuple(sorted(act)) if act else None
        set_hit=bool(act and aset==selected); payout=race_trio_payout(full)
        pay=int(payout) if buy and set_hit and payout is not None else 0
        rows.append({'month':m,'race_id':rid,'selected_set':'-'.join(map(str,selected)),'actual':getattr(rr,'actual',''),
                     'trio_odds':o,'parsed_trio_count':len(om),'bet':int(buy),'hit':int(buy and set_hit and payout is not None),
                     'stake':STAKE if buy else 0,'pay':pay,'source_url':url})
        time.sleep(random.uniform(.4,.9))
    out=pd.DataFrame(rows);out.to_csv(CTX/f'oh_high_trio15_{m}_results.csv',index=False,encoding='utf-8-sig')
    b=out[out.bet==1].copy() if len(out) else out;st=int(b.stake.sum()) if len(b) else 0;py=int(b.pay.sum()) if len(b) else 0
    payload={'month':m,'rule':'frozen OH_HIGH selected set; one 3-ren-fuku ticket only if confirmed odds >=15x',
             'candidate_cases':len(out),'trio_odds_found':int(out.trio_odds.notna().sum()) if len(out) else 0,
             'bets':len(b),'hits':int(b.hit.sum()) if len(b) else 0,'hit_rate_pct':100*b.hit.mean() if len(b) else None,
             'stake_yen':st,'pay_yen':py,'profit_yen':py-st,'roi_pct':100*py/st if st else None,
             'max_losing_streak_bets':max_losing_streak(b.hit.astype(bool).tolist()) if len(b) else 0,
             'max_drawdown_yen':max_drawdown((b.pay-b.stake).tolist()) if len(b) else 0}
    (CTX/f'oh_high_trio15_{m}_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
