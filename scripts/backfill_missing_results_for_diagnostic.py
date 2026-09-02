#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill result ranks only for unresolved diagnostic races.

Does not modify canonical monthly CSV. Reads the unresolved race IDs, fetches the
current K-Dreams showResult page through the existing scraper parser, and writes a
small repair table used only for validation scoring.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import keirin_scraper as ks

CTX=ROOT/'keirin_data'/'strategy_context'
UNRES=CTX/'reconstruction_abc_v3_unresolved.csv'
OUT=CTX/'diagnostic_result_repairs.csv'


def main():
    u=pd.read_csv(UNRES,encoding='utf-8-sig',dtype={'race_id':str})
    rows=[]
    for i,r in enumerate(u.itertuples(index=False),1):
        rid=str(r.race_id); venue=str(r.venue)
        print(f'[{i}/{len(u)}] {r.date} {venue} {r.race_no}R {rid}')
        try:
            got=ks.parse_race(venue,rid)
        except Exception as e:
            print(f'  ERROR {type(e).__name__}: {e}')
            got=[]
        for x in got:
            rows.append({
                'race_id':rid,'venue_slug':venue,'date':getattr(r,'date',''),'race_no':getattr(r,'race_no',''),
                'banum':x.get('banum',''),'rank':x.get('rank',''),
                'san_ren_tan':x.get('san_ren_tan',''),'san_ren_fuku':x.get('san_ren_fuku','')
            })
        if got:
            ranks=[str(x.get('rank','')) for x in got]
            print('  ranks='+'|'.join(ranks))
    out=pd.DataFrame(rows)
    out.to_csv(OUT,index=False,encoding='utf-8-sig')
    if out.empty:
        print('No repairs fetched')
        return
    def scorable(g):
        vals=[]
        for v in g['rank']:
            try:
                f=float(str(v).strip()); i=int(f)
                if abs(f-i)<1e-9 and 1<=i<=3: vals.append(i)
            except Exception: pass
        return sorted(vals)==[1,2,3]
    ok=out.groupby('race_id').apply(scorable,include_groups=False)
    print(f'fetched_races={out.race_id.nunique()} scorable={int(ok.sum())}/{len(ok)} rows={len(out)}')

if __name__=='__main__': main()
