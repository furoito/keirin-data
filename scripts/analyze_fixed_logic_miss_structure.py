#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural diagnosis of candidate-set misses across all fixed-logic months.

Reporting only; DOES NOT change the frozen strategy.

For every scorable BET where the popular head actually misses top3:
- compare the actual top3 set to the candidate family;
- choose candidate variant(s) with maximum overlap;
- classify missing/extra riders by structural role;
- measure whether misses are one-rider replacements;
- measure score differences for one-rider replacements;
- aggregate replacement role pairs and score-gap buckets.
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base

DATA=Path('keirin_data'); CTX=DATA/'strategy_context'
OUT=CTX/'fixed_logic_miss_structure.csv'
SUMMARY=CTX/'fixed_logic_miss_structure_summary.json'
PAT=re.compile(r'^fixed_logic_(\d{4})_(\d{2})_results\.csv$')


def parse_order(s):
    if s is None or (isinstance(s,float) and pd.isna(s)): return tuple()
    out=[]
    for x in str(s).split('-'):
        x=x.strip()
        if not x: continue
        try: out.append(int(float(x)))
        except Exception: pass
    return tuple(out)


def parse_sets(s):
    if s is None or (isinstance(s,float) and pd.isna(s)): return []
    out=[]
    for part in str(s).split('|'):
        q=parse_order(part)
        if q: out.append(frozenset(q))
    # de-duplicate while preserving order
    seen=[]
    for q in out:
        if q not in seen: seen.append(q)
    return seen


def role(r,pop_line):
    if r.line_idx==pop_line:
        if r.line_pos==2: return 'popular_pos2'
        if r.line_pos>=3: return 'popular_pos3plus'
        return 'popular_head'
    if r.line_size==1: return 'solo'
    if r.line_pos==1: return 'other_head'
    if r.line_pos==2: return 'other_pos2'
    return 'other_pos3plus'


def gap_bucket(g):
    # g = missing actual score - extra predicted score
    if g >= 3: return 'missing_+3_or_more'
    if g >= 1: return 'missing_+1_to_3'
    if g > -1: return 'within_1'
    if g > -3: return 'extra_+1_to_3'
    return 'extra_+3_or_more'


def month_sources():
    xs=[]
    for p in sorted(CTX.glob('fixed_logic_*_results.csv')):
        m=PAT.match(p.name)
        if m: xs.append((f'{m.group(1)}_{m.group(2)}',p))
    return xs


def main():
    sources=month_sources()
    if not sources: raise SystemExit('no fixed_logic monthly result files')

    miss_roles=Counter(); extra_roles=Counter(); replacement_pairs=Counter(); overlap_dist=Counter()
    gap_buckets=Counter(); miss_months=Counter(); all_bust=0; exact=0; miss=0
    one_replace=0; one_replace_gaps=[]; one_replace_rows=[]; rows=[]
    actual_role_presence=Counter(); predicted_role_presence=Counter()

    # score-gap directional counts for common hypotheses
    hyp=Counter()

    for month,p in sources:
        result=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
        result['race_id']=result.race_id.astype(str)
        base_path=DATA/f'{month}_keirin.csv'; ctx_path=CTX/f'{month}_races.csv'
        if not base_path.exists() or not ctx_path.exists(): continue
        race=pd.read_csv(base_path,encoding='utf-8-sig',dtype={'race_id':str}); race['race_id']=race.race_id.astype(str)
        ctx=pd.read_csv(ctx_path,encoding='utf-8-sig',dtype={'race_id':str}); ctx['race_id']=ctx.race_id.astype(str)
        rb={k:g for k,g in race.groupby('race_id',sort=False)}
        cb={k:g.iloc[0] for k,g in ctx.groupby('race_id',sort=False)}

        bust=result[(pd.to_numeric(result.head_bust,errors='coerce')==1) & (result.actual.astype(str)!='')].copy()
        all_bust += len(bust)
        exact += int((pd.to_numeric(bust.set_match,errors='coerce')==1).sum())
        bad=bust[pd.to_numeric(bust.set_match,errors='coerce')!=1].copy()
        miss += len(bad); miss_months[month]+=len(bad)

        for q in bad.itertuples(index=False):
            rid=str(q.race_id); g=rb.get(rid); cr=cb.get(rid)
            if g is None or cr is None: continue
            lines=base.parse_true_line(str(cr.true_line))
            pre=g[['race_id','banum','race_score']].copy()
            riders=base.make_riders(pre,lines); by={r.frame_no:r for r in riders}
            try: target=int(float(q.target))
            except Exception: continue
            if target not in by: continue
            pop_line=by[target].line_idx
            actual=set(parse_order(q.actual)); cands=parse_sets(q.candidate_sets)
            if len(actual)!=3 or not cands: continue
            best_overlap=max(len(c & actual) for c in cands)
            overlap_dist[best_overlap]+=1
            bests=[c for c in cands if len(c & actual)==best_overlap]

            race_miss_roles=set(); race_extra_roles=set(); race_pairs=set(); details=[]
            # Use all tied best candidate variants for role-presence counts, but de-dupe per race.
            for c in bests:
                missing=actual-set(c); extra=set(c)-actual
                for fn in actual:
                    if fn in by: actual_role_presence[role(by[fn],pop_line)]+=1
                for fn in c:
                    if fn in by: predicted_role_presence[role(by[fn],pop_line)]+=1
                for fn in missing:
                    if fn in by: race_miss_roles.add(role(by[fn],pop_line))
                for fn in extra:
                    if fn in by: race_extra_roles.add(role(by[fn],pop_line))
                if len(missing)==1 and len(extra)==1:
                    mf=next(iter(missing)); ef=next(iter(extra))
                    if mf in by and ef in by:
                        mr=role(by[mf],pop_line); er=role(by[ef],pop_line)
                        gap=float(by[mf].race_score-by[ef].race_score)
                        race_pairs.add((er,mr))
                        details.append(f'extra{ef}:{er}:s{by[ef].race_score:.2f}->miss{mf}:{mr}:s{by[mf].race_score:.2f}:gap{gap:+.2f}')

            miss_roles.update(race_miss_roles); extra_roles.update(race_extra_roles)
            for pair in race_pairs: replacement_pairs[pair]+=1

            # one-replacement metrics based on first best variant only; candidate variants tied by overlap
            c=set(bests[0]); missing=actual-c; extra=c-actual
            gap=None; mr=er=''; mf=ef=None
            if len(missing)==1 and len(extra)==1:
                one_replace+=1; mf=next(iter(missing)); ef=next(iter(extra))
                if mf in by and ef in by:
                    mr=role(by[mf],pop_line); er=role(by[ef],pop_line)
                    gap=float(by[mf].race_score-by[ef].race_score); one_replace_gaps.append(gap); gap_buckets[gap_bucket(gap)]+=1
                    if er=='popular_pos2' and mr=='other_head':
                        hyp['popular_pos2_to_other_head']+=1
                        if gap>=0: hyp['popular_pos2_to_other_head_missing_score_ge_extra']+=1
                        if gap>=-1: hyp['popular_pos2_to_other_head_missing_within1']+=1
                        if gap>=-3: hyp['popular_pos2_to_other_head_missing_within3']+=1
                    if er=='popular_pos2': hyp['extra_popular_pos2_one_replace']+=1
                    if mr=='other_head': hyp['miss_other_head_one_replace']+=1
                    one_replace_rows.append({'month':month,'race_id':rid,'extra_role':er,'missing_role':mr,'score_gap_missing_minus_extra':gap})

            rows.append({
                'month':month,'race_id':rid,'date':getattr(q,'date',''),'venue':getattr(q,'venue',''),'race_no':getattr(q,'race_no',''),
                'line':getattr(q,'line',''),'target':target,'actual':q.actual,'candidate_sets':q.candidate_sets,
                'best_overlap':best_overlap,'n_best_variants':len(bests),
                'miss_roles':'|'.join(sorted(race_miss_roles)),'extra_roles':'|'.join(sorted(race_extra_roles)),
                'one_replacement':int(len(missing)==1 and len(extra)==1),
                'extra_frame':ef or '','extra_role':er,'missing_frame':mf or '','missing_role':mr,
                'score_gap_missing_minus_extra':gap if gap is not None else '',
                'detail':' ; '.join(details)
            })

    out=pd.DataFrame(rows); out.to_csv(OUT,index=False,encoding='utf-8-sig')
    gaps=pd.Series(one_replace_gaps,dtype=float)
    summary={
        'scope_months':[m for m,_ in sources],
        'head_bust_races':all_bust,
        'candidate_set_exact_races':exact,
        'candidate_set_miss_races':miss,
        'candidate_set_exact_given_bust_pct':100*exact/all_bust if all_bust else None,
        'miss_overlap_distribution':dict(sorted((str(k),v) for k,v in overlap_dist.items())),
        'one_rider_replacement_misses':one_replace,
        'one_rider_replacement_share_of_misses_pct':100*one_replace/miss if miss else None,
        'missing_role_race_counts':miss_roles.most_common(),
        'extra_role_race_counts':extra_roles.most_common(),
        'replacement_role_pair_counts':[[f'{a}->{b}',n] for (a,b),n in replacement_pairs.most_common()],
        'one_replacement_score_gap_missing_minus_extra':{
            'n':len(gaps),'mean':float(gaps.mean()) if len(gaps) else None,'median':float(gaps.median()) if len(gaps) else None,
            'missing_score_ge_extra_n':int((gaps>=0).sum()) if len(gaps) else 0,
            'missing_within_1_below_or_better_n':int((gaps>=-1).sum()) if len(gaps) else 0,
            'missing_within_3_below_or_better_n':int((gaps>=-3).sum()) if len(gaps) else 0,
            'gap_buckets':gap_buckets.most_common(),
        },
        'hypothesis_counts':dict(hyp),
        'misses_by_month':dict(sorted(miss_months.items())),
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
