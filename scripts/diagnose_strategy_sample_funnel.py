#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]
OUT=CTX/'strategy_sample_funnel_summary.json'

def load_eval():
    parts=[]
    big=CTX/'other_head_promotion_oos_results.csv'
    if big.exists():
        d=pd.read_csv(big,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.astype(str)=='OH_HIGH'].copy())
    for m in ['2025_01','2025_02','2025_03','2025_04']:
        p=CTX/f'dual_logic_{m}_results.csv'
        if p.exists():
            d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str}); parts.append(d[d.variant.astype(str)=='OH_HIGH'].copy())
    if not parts:return pd.DataFrame()
    x=pd.concat(parts,ignore_index=True); x['race_id']=x.race_id.astype(str)
    return x.drop_duplicates(['month','race_id'],keep='last')

def load_mkt():
    p=CTX/'market_ordering_results.csv'
    if not p.exists():return pd.DataFrame()
    d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str}); d['race_id']=d.race_id.astype(str)
    return d[(d.variant.astype(str)=='OH_HIGH')&(d.policy.astype(str)=='MKT1')].copy()

def only_classes(g, allowed):
    vals=set(g.dropna().astype(str).str.strip())
    return bool(vals) and vals.issubset(allowed)

def main():
    ev=load_eval(); mk=load_mkt(); rows=[]; reason_total=Counter()
    for month in MONTHS:
        rp=DATA/f'{month}_keirin.csv'; cp=CTX/f'{month}_races.csv'; op=CTX/f'{month}_odds_3rentan.csv'
        if not rp.exists():continue
        r=pd.read_csv(rp,encoding='utf-8-sig',dtype={'race_id':str}); r['race_id']=r.race_id.astype(str)
        raw_races=r.race_id.nunique()
        cls=r.groupby('race_id').player_class.apply(lambda s: set(s.dropna().astype(str).str.strip()))
        aa=cls[cls.apply(lambda s: bool(s) and s.issubset({'A1','A2'}))].index
        aa_count=len(aa)
        mixed_a=cls[cls.apply(lambda s: bool(s) and s.issubset({'A1','A2','A3'}) and not s.issubset({'A1','A2'}))].index
        class_counts=Counter()
        for s in cls:
            class_counts['/'.join(sorted(s)) if s else 'UNKNOWN']+=1
        context_rows=context_full=price_usable=line_full=odds_full=0
        context_aa=0
        if cp.exists():
            c=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str}); c['race_id']=c.race_id.astype(str)
            context_rows=c.race_id.nunique(); context_aa=c[c.race_id.isin(set(aa))].race_id.nunique()
            if 'context_quality' in c: context_full=c[c.context_quality.astype(str)=='full'].race_id.nunique()
            if 'price_usable' in c: price_usable=c[c.price_usable.astype(str).str.lower().isin({'true','1'})].race_id.nunique()
            if 'line_quality' in c: line_full=c[c.line_quality.astype(str)=='full'].race_id.nunique()
            if 'odds_quality' in c: odds_full=c[c.odds_quality.astype(str)=='full'].race_id.nunique()
        es=ev[ev.month.astype(str)==month] if not ev.empty else pd.DataFrame()
        eval_rows=len(es)
        scorable=int(es.actual.astype(str).ne('').sum()) if len(es) and 'actual' in es else 0
        eval_bets=int(pd.to_numeric(es.bet,errors='coerce').fillna(0).sum()) if len(es) and 'bet' in es else 0
        reasons=Counter(es.reason.astype(str)) if len(es) and 'reason' in es else Counter()
        reason_total.update(reasons)
        ms=mk[mk.month.astype(str)==month] if not mk.empty else pd.DataFrame()
        mkt_rows=len(ms); mkt_bets=int(pd.to_numeric(ms.bet,errors='coerce').fillna(0).sum()) if len(ms) else 0
        rows.append({
            'month':month,'raw_unique_races':raw_races,'a1_a2_only_races':aa_count,'a1_a2_a3_mixed_excluded':len(mixed_a),
            'context_rows':context_rows,'context_rows_matching_a1a2':context_aa,'line_full':line_full,'odds_full':odds_full,
            'price_usable':price_usable,'context_full':context_full,'eval_rows':eval_rows,'scorable':scorable,
            'old_order_eval_bets':eval_bets,'mkt1_rows':mkt_rows,'mkt1_bets':mkt_bets,
            'class_mix_top':dict(class_counts.most_common(8)),'eval_reason_counts':dict(reasons)
        })
    d=pd.DataFrame(rows)
    totals={k:int(d[k].sum()) for k in ['raw_unique_races','a1_a2_only_races','context_rows','context_rows_matching_a1a2','line_full','odds_full','price_usable','context_full','eval_rows','scorable','old_order_eval_bets','mkt1_rows','mkt1_bets']}
    avgs={k:float(d[k].mean()) for k in ['raw_unique_races','a1_a2_only_races','context_full','scorable','mkt1_bets']}
    rates={
        'a1a2_share_of_raw_pct':100*totals['a1_a2_only_races']/totals['raw_unique_races'] if totals['raw_unique_races'] else None,
        'context_coverage_of_a1a2_pct':100*totals['context_rows_matching_a1a2']/totals['a1_a2_only_races'] if totals['a1_a2_only_races'] else None,
        'full_context_of_a1a2_pct':100*totals['context_full']/totals['a1_a2_only_races'] if totals['a1_a2_only_races'] else None,
        'mkt1_bets_of_a1a2_pct':100*totals['mkt1_bets']/totals['a1_a2_only_races'] if totals['a1_a2_only_races'] else None,
        'mkt1_bets_of_raw_pct':100*totals['mkt1_bets']/totals['raw_unique_races'] if totals['raw_unique_races'] else None,
    }
    payload={'scope':'2025-01..2026-06','monthly':rows,'totals':totals,'monthly_average':avgs,'funnel_rates':rates,'eval_reason_totals':dict(reason_total)}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
