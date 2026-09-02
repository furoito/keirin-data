#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reset H1: cross-line high-score unordered trios may be underpriced from medium odds upward.

Hypothesis under test:
- low odds need not have edge;
- unordered 3-rider sets spanning multiple lines and containing high race_score riders
  may be underpriced by the trifecta market once their effective market odds rise;
- no upper odds cap is imposed.

The unit is an UNORDERED 3-rider set. For every complete trifecta board, the six
permutations belonging to each set are collapsed into one normalized market probability.
Because normalized ordered probabilities sum to 1 within a race, unordered-set market
probabilities also sum to 1. Thus for any pre-defined stratum:
    expected_hits = sum(market_probability)
    actual_hits   = number of races whose realized top3 set falls in the stratum
    calibration_ratio = actual_hits / expected_hits
A ratio > 1 means the stratum realized more often than the market priced; < 1 means less.

No previous strategy rules are used: no popular-line fade, no 3-point gap, no OH_HIGH,
no running style, no MKT1, and no 30x ticket threshold.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import popular_head_skip_v01 as base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "keirin_data"
CTX = DATA / "strategy_context"
OUT = CTX / "h1_crossline_highscore_trios_summary.json"
DETAIL = CTX / "h1_crossline_highscore_trios_details.csv"
MONTHS = [f"2025_{m:02d}" for m in range(1,13)] + [f"2026_{m:02d}" for m in range(1,7)]

# Fixed BEFORE looking at results. Last bin has no upper cap.
ODDS_BINS = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, float("inf")]
ODDS_LABELS = ["<10","10-20","20-30","30-50","50-100","100-200","200-500","500+"]
# Exploratory cumulative cuts; again, no upper cap.
CUMULATIVE_MIN_ODDS = [10,20,30,50,100,200,500]


def actual_top3_set(g: pd.DataFrame):
    vals=[]
    for r in g.itertuples(index=False):
        try:
            rank=int(str(r.rank).strip()); fn=int(r.banum)
        except Exception:
            continue
        if 1 <= rank <= 3:
            vals.append((rank,fn))
    vals.sort()
    if [x[0] for x in vals] != [1,2,3]:
        return None
    return frozenset(x[1] for x in vals)


def load_month(month):
    bp=DATA/f"{month}_keirin.csv"; cp=CTX/f"{month}_races.csv"; op=CTX/f"{month}_odds_3rentan.csv"
    if not (bp.exists() and cp.exists() and op.exists()): return None
    b=pd.read_csv(bp,encoding="utf-8-sig",dtype={"race_id":str})
    c=pd.read_csv(cp,encoding="utf-8-sig",dtype={"race_id":str}).drop_duplicates("race_id",keep="last")
    o=pd.read_csv(op,encoding="utf-8-sig",dtype={"race_id":str})
    for d in (b,c,o): d["race_id"]=d.race_id.astype(str)
    return b,c,o


def odds_bin(v):
    for i in range(len(ODDS_BINS)-1):
        if ODDS_BINS[i] <= v < ODDS_BINS[i+1]: return ODDS_LABELS[i]
    return ODDS_LABELS[-1]


def make_line_lookup(lines):
    out={}
    for li,g in enumerate(lines,1):
        for fn in g: out[int(fn)]=li
    return out


def score_ranks(pre, frames):
    score={}
    for r in pre.itertuples(index=False):
        try: fn=int(r.banum); sc=float(r.race_score)
        except Exception: continue
        if np.isfinite(sc): score[fn]=sc
    if set(frames) - set(score): return None,None
    ordered=sorted(frames,key=lambda x:(-score[x],x))
    rank={fn:i+1 for i,fn in enumerate(ordered)}
    return score,rank


def race_rows(month,rid,pre,cr,og):
    lines=base.parse_true_line(cr.get("true_line"))
    if not lines: return None,"line_unresolved"
    frames=sorted({int(x) for g in lines for x in g})
    if len(frames) < 4: return None,"too_few_riders"
    tri=base.odds_map(og)
    expected=len(frames)*(len(frames)-1)*(len(frames)-2)
    if len(tri) != expected: return None,"odds_board_incomplete"
    z=sum(1.0/v for v in tri.values() if v>0)
    if z<=0: return None,"zero_mass"
    actual=actual_top3_set(pre)
    if actual is None: return None,"result_missing"
    score,rank=score_ranks(pre,frames)
    if score is None: return None,"score_missing"
    line_of=make_line_lookup(lines)

    # Collapse ordered board into unordered-set probability.
    pset=Counter()
    raw_inverse=Counter()
    for combo,od in tri.items():
        if od<=0: continue
        key=tuple(sorted(combo)); w=1.0/od
        raw_inverse[key]+=w
    for key,w in raw_inverse.items(): pset[key]=w/z

    rows=[]
    for key,p in pset.items():
        s=frozenset(key)
        line_span=len({line_of[x] for x in key})
        ranks=sorted(rank[x] for x in key)
        top3_count=sum(r<=3 for r in ranks)
        top4_count=sum(r<=4 for r in ranks)
        mean_rank=float(np.mean(ranks))
        score_sum=float(sum(score[x] for x in key))
        # Fair effective odds from normalized market probability; no upper cap.
        eff=float(1.0/p) if p>0 else float("inf")
        rows.append({
            "month":month,"race_id":rid,"trio":"-".join(map(str,key)),
            "market_p":float(p),"effective_fair_odds":eff,"odds_bin":odds_bin(eff),
            "actual_hit":int(s==actual),"line_span":int(line_span),
            "cross_line":int(line_span>=2),"three_lines":int(line_span>=3),
            "score_rank_1":int(ranks[0]),"score_rank_2":int(ranks[1]),"score_rank_3":int(ranks[2]),
            "top3_score_count":int(top3_count),"top4_score_count":int(top4_count),
            "mean_score_rank":mean_rank,"score_sum":score_sum,
        })
    return rows,None


def agg(x):
    expected=float(x.market_p.sum()); hits=int(x.actual_hit.sum()); n=int(len(x)); races=int(x.race_id.nunique())
    ratio=float(hits/expected) if expected>0 else None
    return {"triples":n,"races":races,"expected_hits":expected,"actual_hits":hits,
            "calibration_ratio_actual_over_market":ratio,
            "market_share_pct":float(100*expected/races) if races else None,
            "actual_hit_rate_per_race_pct":float(100*hits/races) if races else None}


def block_bootstrap_ratio(df, filter_fn, draws=4000, seed=20260902):
    # Bootstrap races as blocks to preserve dependence among triples from the same race.
    rids=df.race_id.unique()
    if len(rids)==0: return [None,None]
    by={rid:filter_fn(df[df.race_id==rid]) for rid in rids}
    pairs=[]
    for rid,x in by.items():
        pairs.append((float(x.market_p.sum()),int(x.actual_hit.sum())))
    arr=np.asarray(pairs,float)
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(draws):
        idx=rng.integers(0,len(arr),size=len(arr)); samp=arr[idx]
        e=samp[:,0].sum(); h=samp[:,1].sum()
        if e>0: vals.append(h/e)
    if not vals: return [None,None]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def summarize(df, skipped, context):
    # Core grid: odds x line span x number of race top3 score riders in trio.
    grid=[]
    for ob in ODDS_LABELS:
        for ls in [1,2,3]:
            for t3 in [0,1,2,3]:
                x=df[(df.odds_bin==ob)&(df.line_span==ls)&(df.top3_score_count==t3)]
                if x.empty: continue
                row={"odds_bin":ob,"line_span":ls,"top3_score_count":t3,**agg(x)}
                grid.append(row)

    # Most literal hypothesis views.
    views=[]
    definitions=[
        ("crossline_top3_all3",lambda x:x[(x.line_span>=2)&(x.top3_score_count==3)]),
        ("crossline_top3_atleast2",lambda x:x[(x.line_span>=2)&(x.top3_score_count>=2)]),
        ("three_lines_top3_all3",lambda x:x[(x.line_span>=3)&(x.top3_score_count==3)]),
        ("three_lines_top3_atleast2",lambda x:x[(x.line_span>=3)&(x.top3_score_count>=2)]),
        ("same_line_top3_atleast2_control",lambda x:x[(x.line_span==1)&(x.top3_score_count>=2)]),
    ]
    for name,fn in definitions:
        for ob in ODDS_LABELS:
            x=fn(df[df.odds_bin==ob])
            if x.empty: continue
            views.append({"view":name,"odds_bin":ob,**agg(x)})

    cumulative=[]
    for name,fn in definitions:
        for cut in CUMULATIVE_MIN_ODDS:
            x=fn(df[df.effective_fair_odds>=cut])
            if x.empty: continue
            d={"view":name,"min_effective_odds":cut,"no_upper_cap":True,**agg(x)}
            # CI only for the two main hypothesis views to keep runtime reasonable.
            if name in {"crossline_top3_all3","crossline_top3_atleast2"}:
                base_df=df[df.effective_fair_odds>=cut]
                d["race_block_bootstrap_95pct_ci"] = block_bootstrap_ratio(base_df,fn,seed=20260902+cut)
            cumulative.append(d)

    by_month=[]
    for m,xm in df.groupby("month",sort=True):
        x=xm[(xm.line_span>=2)&(xm.top3_score_count>=2)&(xm.effective_fair_odds>=20)]
        if not x.empty: by_month.append({"month":m,"view":"crossline_top3_atleast2_odds20plus",**agg(x)})

    return {
        "hypothesis":"Cross-line unordered trios containing high race_score riders become underpriced from medium odds upward; no upper odds cap.",
        "status":"exploratory_reset_test",
        "scope":"2025-01..2026-06 available full context at run time",
        "unit":"unordered 3-rider set collapsed from six trifecta permutations",
        "market_probability":"inverse trifecta odds normalized over complete ordered outcome board, then summed by unordered set",
        "fixed_odds_bins":ODDS_LABELS,
        "no_upper_cap":True,
        "high_score_operationalization":"report exact count of riders among race top3 race_score; primary views use all3 or at least2",
        "context":context,"skipped":skipped,"races_analyzed":int(df.race_id.nunique()),"triples_analyzed":int(len(df)),
        "core_grid":grid,"hypothesis_views_by_odds":views,"cumulative_min_odds_views":cumulative,"monthly_diagnostic":by_month,
    }


def main():
    allrows=[]; skipped=Counter(); context={"context_rows":0,"full_price_usable_rows":0}
    for month in MONTHS:
        loaded=load_month(month)
        if loaded is None: continue
        b,c,o=loaded; context["context_rows"]+=len(c)
        use=c.copy()
        if "context_quality" in use: use=use[use.context_quality.astype(str)=="full"]
        if "price_usable" in use: use=use[use.price_usable.astype(str).str.lower().isin({"true","1"})]
        context["full_price_usable_rows"]+=len(use)
        bby={str(k):g for k,g in b.groupby("race_id",sort=False)}; oby={str(k):g for k,g in o.groupby("race_id",sort=False)}
        for cr in use.to_dict("records"):
            rid=str(cr["race_id"]); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None: skipped["base_or_odds_missing"]+=1; continue
            rows,why=race_rows(month,rid,pre,cr,og)
            if rows is None: skipped[why]+=1
            else: allrows.extend(rows)
    if not allrows: raise SystemExit("No analyzable rows")
    df=pd.DataFrame(allrows).sort_values(["month","race_id","trio"])
    df.to_csv(DETAIL,index=False,encoding="utf-8-sig")
    payload=summarize(df,dict(skipped),context)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print(f"detail={DETAIL}")
    print(f"summary={OUT}")

if __name__=="__main__": main()
