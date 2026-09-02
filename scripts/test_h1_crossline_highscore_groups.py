#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 v2: cross-line HIGH-SCORE GROUPS may be underpriced from medium odds upward.

Correction from v1: "high score" is NOT defined by how many individually top-3 riders
are included. The unit is the 3-rider GROUP itself.

For each race:
- enumerate every unordered 3-rider group;
- sum the three pre-race race_score values;
- rank that group among every possible 3-rider group in the same race by score sum;
- compare market-implied group probability vs realized top3-set frequency;
- stratify by line span and fixed effective-odds bins, with NO upper odds cap.

Primary high-score-group views are pre-fixed at top 10%, top 25%, and top 50% of
all 3-rider groups in the same race (by score sum). We also preserve continuous
score-group percentile and exact score-sum rank in the detail output.

No previous strategy rules are used: no popular-line fade, no 3-point boundary,
no OH_HIGH, no running style, no MKT1, and no 30x ticket threshold.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import popular_head_skip_v01 as base

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
OUT=CTX/'h1_crossline_highscore_groups_summary.json'
DETAIL=CTX/'h1_crossline_highscore_groups_details.csv'
MONTHS=[f'2025_{m:02d}' for m in range(1,13)]+[f'2026_{m:02d}' for m in range(1,7)]
ODDS_BINS=[0,10,20,30,50,100,200,500,float('inf')]
ODDS_LABELS=['<10','10-20','20-30','30-50','50-100','100-200','200-500','500+']
MIN_ODDS_CUTS=[10,20,30,50,100,200,500]
GROUP_CUTS=[('top10pct',0.10),('top25pct',0.25),('top50pct',0.50)]


def load_month(month):
    bp=DATA/f'{month}_keirin.csv'; cp=CTX/f'{month}_races.csv'; op=CTX/f'{month}_odds_3rentan.csv'
    if not (bp.exists() and cp.exists() and op.exists()): return None
    b=pd.read_csv(bp,encoding='utf-8-sig',dtype={'race_id':str})
    c=pd.read_csv(cp,encoding='utf-8-sig',dtype={'race_id':str}).drop_duplicates('race_id',keep='last')
    o=pd.read_csv(op,encoding='utf-8-sig',dtype={'race_id':str})
    for d in (b,c,o): d['race_id']=d.race_id.astype(str)
    return b,c,o


def actual_top3_set(g):
    vals=[]
    for r in g.itertuples(index=False):
        try: pos=int(str(r.rank).strip()); fn=int(r.banum)
        except Exception: continue
        if 1<=pos<=3: vals.append((pos,fn))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]: return None
    return frozenset(fn for _,fn in vals)


def odds_bin(v):
    for i in range(len(ODDS_BINS)-1):
        if ODDS_BINS[i] <= v < ODDS_BINS[i+1]: return ODDS_LABELS[i]
    return ODDS_LABELS[-1]


def race_rows(month,rid,pre,cr,og):
    lines=base.parse_true_line(cr.get('true_line'))
    if not lines: return None,'line_unresolved'
    frames=sorted({int(x) for g in lines for x in g})
    if len(frames)<4: return None,'too_few_riders'
    tri=base.odds_map(og)
    expected=len(frames)*(len(frames)-1)*(len(frames)-2)
    if len(tri)!=expected: return None,'odds_board_incomplete'
    z=sum(1.0/v for v in tri.values() if v>0)
    if z<=0: return None,'zero_mass'
    actual=actual_top3_set(pre)
    if actual is None: return None,'result_missing'

    score={}
    for r in pre.itertuples(index=False):
        try: fn=int(r.banum); sc=float(r.race_score)
        except Exception: continue
        if np.isfinite(sc): score[fn]=sc
    if set(frames)-set(score): return None,'score_missing'

    line_of={}
    for li,g in enumerate(lines,1):
        for fn in g: line_of[int(fn)]=li

    # Collapse the ordered trifecta board to unordered-group market probability.
    raw=Counter()
    for combo,od in tri.items():
        if od>0: raw[tuple(sorted(combo))]+=1.0/od
    keys=sorted(raw)
    score_sums={k:float(sum(score[x] for x in k)) for k in keys}

    # Rank GROUPS by their three-rider score sum. Higher sum = better group.
    ordered_groups=sorted(keys,key=lambda k:(-score_sums[k],k))
    rank_of={k:i+1 for i,k in enumerate(ordered_groups)}
    n_groups=len(ordered_groups)

    rows=[]
    for key in keys:
        p=float(raw[key]/z); rank=rank_of[key]
        # Percentile is rank / N: lower is a stronger score group.
        pct=float(rank/n_groups)
        line_span=len({line_of[x] for x in key})
        eff=float(1.0/p) if p>0 else float('inf')
        rows.append({
            'month':month,'race_id':rid,'trio':'-'.join(map(str,key)),
            'market_p':p,'effective_fair_odds':eff,'odds_bin':odds_bin(eff),
            'actual_hit':int(frozenset(key)==actual),
            'line_span':int(line_span),'cross_line':int(line_span>=2),'three_lines':int(line_span>=3),
            'group_score_sum':score_sums[key],
            'group_score_rank':int(rank),'group_count_in_race':int(n_groups),
            'group_score_percentile':pct,
        })
    return rows,None


def agg(x):
    expected=float(x.market_p.sum()); hits=int(x.actual_hit.sum()); races=int(x.race_id.nunique())
    ratio=float(hits/expected) if expected>0 else None
    return {'groups':int(len(x)),'races':races,'expected_hits':expected,'actual_hits':hits,
            'calibration_ratio_actual_over_market':ratio,
            'market_share_pct':float(100*expected/races) if races else None,
            'actual_hit_rate_per_race_pct':float(100*hits/races) if races else None}


def block_bootstrap(df, fn, draws=4000, seed=20260902):
    rids=df.race_id.unique(); pairs=[]
    for rid in rids:
        x=fn(df[df.race_id==rid]); pairs.append((float(x.market_p.sum()),int(x.actual_hit.sum())))
    if not pairs: return [None,None]
    a=np.asarray(pairs,float); rng=np.random.default_rng(seed); vals=[]
    for _ in range(draws):
        s=a[rng.integers(0,len(a),size=len(a))]; e=s[:,0].sum(); h=s[:,1].sum()
        if e>0: vals.append(h/e)
    if not vals: return [None,None]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def summarize(df,skipped,context):
    by_odds=[]; cumulative=[]
    defs=[]
    for label,cut in GROUP_CUTS:
        defs.extend([
            (f'crossline_{label}',lambda x,c=cut:x[(x.line_span>=2)&(x.group_score_percentile<=c)]),
            (f'threelines_{label}',lambda x,c=cut:x[(x.line_span>=3)&(x.group_score_percentile<=c)]),
            (f'sameline_{label}_control',lambda x,c=cut:x[(x.line_span==1)&(x.group_score_percentile<=c)]),
        ])

    for name,fn in defs:
        for ob in ODDS_LABELS:
            x=fn(df[df.odds_bin==ob])
            if not x.empty: by_odds.append({'view':name,'odds_bin':ob,**agg(x)})
        for cut in MIN_ODDS_CUTS:
            base_df=df[df.effective_fair_odds>=cut]; x=fn(base_df)
            if x.empty: continue
            d={'view':name,'min_effective_odds':cut,'no_upper_cap':True,**agg(x)}
            if name in {'crossline_top10pct','crossline_top25pct','crossline_top50pct'}:
                d['race_block_bootstrap_95pct_ci']=block_bootstrap(base_df,fn,seed=20260902+cut)
            cumulative.append(d)

    # Exact score-group rank diagnostics: 1st, top3 groups, top5 groups, etc.
    exact=[]
    for max_rank in [1,2,3,5,10]:
        for span_label,mask in [('crossline',lambda x:x.line_span>=2),('three_lines',lambda x:x.line_span>=3)]:
            for ob in ODDS_LABELS:
                x=df[(df.group_score_rank<=max_rank)&mask(df)&(df.odds_bin==ob)]
                if not x.empty: exact.append({'line_view':span_label,'max_group_score_rank':max_rank,'odds_bin':ob,**agg(x)})

    return {
        'hypothesis':'Cross-line 3-rider GROUPS that rank highly by combined race_score are underpriced from medium odds upward; no upper odds cap.',
        'status':'exploratory_reset_test_v2_group_level',
        'correction':'High score is defined at the GROUP level, not by count of individually top-3 riders.',
        'unit':'unordered 3-rider group collapsed from six trifecta permutations',
        'group_score_definition':'sum of the three riders pre-race race_score, ranked among all unordered 3-rider groups in that race',
        'fixed_group_views':['top10pct','top25pct','top50pct'],
        'fixed_odds_bins':ODDS_LABELS,'no_upper_cap':True,
        'context':context,'skipped':skipped,
        'races_analyzed':int(df.race_id.nunique()),'groups_analyzed':int(len(df)),
        'views_by_odds':by_odds,'cumulative_min_odds_views':cumulative,'exact_group_rank_views':exact,
    }


def main():
    rows=[]; skipped=Counter(); context={'context_rows':0,'full_price_usable_rows':0}
    for month in MONTHS:
        loaded=load_month(month)
        if loaded is None: continue
        b,c,o=loaded; context['context_rows']+=len(c)
        use=c.copy()
        if 'context_quality' in use: use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use: use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        context['full_price_usable_rows']+=len(use)
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}; oby={str(k):g for k,g in o.groupby('race_id',sort=False)}
        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None: skipped['base_or_odds_missing']+=1; continue
            rr,why=race_rows(month,rid,pre,cr,og)
            if rr is None: skipped[why]+=1
            else: rows.extend(rr)
    if not rows: raise SystemExit('No analyzable rows')
    df=pd.DataFrame(rows).sort_values(['month','race_id','group_score_rank','trio'])
    df.to_csv(DETAIL,index=False,encoding='utf-8-sig')
    payload=summarize(df,dict(skipped),context)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print(f'detail={DETAIL}')
    print(f'summary={OUT}')

if __name__=='__main__': main()
