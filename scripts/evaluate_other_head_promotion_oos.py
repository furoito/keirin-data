#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test whether promoting OTHER-LINE HEADS improves candidate reconstruction.

This is an experiment only; it does NOT alter the frozen production/backtest logic.

Predeclared variants (only candidate position tier changes):
- BASE: current tiers: popular pos2=4, popular pos3+=3, other follower=2, other head/solo=1
- OH_EQ: other-line head (non-solo) tier=2, equal to other followers
- OH_HIGH: other-line head tier=3, equal to popular pos3+
- OH_TOP: other-line head tier=4, equal to popular pos2

Everything else stays fixed:
- popular-line detector v0.1b
- 3-point ability/position boundary
- target +3 skip
- provisional strong-line skip
- max two candidate sets / compression rules
- trifecta ORDER uses +3 popular pos2, +1 popular pos3 exactly once
- current ordered ticket must be >=30x

Anti-overfit protocol:
- TRAIN = 2025_05..2025_12
- OOS   = 2026_01..2026_06
- selected_variant is chosen using TRAIN candidate-set exact count ONLY.
  If no alternative beats BASE, BASE is retained.
- OOS never participates in selection.

We report both gains and losses versus BASE so promotion is penalized for races it breaks.
"""
from __future__ import annotations

import json, math, sys
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import popular_head_skip_v01 as base
import popular_head_skip_v01b as v01b  # noqa: F401; patches detector
import keirin_scraper as ks

DATA=ROOT/'keirin_data'; CTX=DATA/'strategy_context'
TRAIN=['2025_05','2025_06','2025_07','2025_08','2025_09','2025_10','2025_11','2025_12']
OOS=['2026_01','2026_02','2026_03','2026_04','2026_05','2026_06']
VARIANTS=['BASE','OH_EQ','OH_HIGH','OH_TOP']
OUT=CTX/'other_head_promotion_oos_results.csv'
SUMMARY=CTX/'other_head_promotion_oos_summary.json'
REPAIRS=CTX/'other_head_promotion_oos_result_repairs.csv'


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


def tier(r,pop_line,variant):
    if r.line_idx==pop_line:
        if r.line_pos==2: return 4
        if r.line_pos>=3: return 3
    if r.line_pos>=2: return 2
    if r.line_size>1 and r.line_pos==1 and r.line_idx!=pop_line:
        return {'BASE':1,'OH_EQ':2,'OH_HIGH':3,'OH_TOP':4}[variant]
    return 1  # solo


def pair_winner(a,b,pop_line,variant):
    gap=abs(a.race_score-b.race_score)
    if gap>=base.SCORE_BOUNDARY and a.race_score!=b.race_score:
        return a if a.race_score>b.race_score else b
    if gap<base.SCORE_BOUNDARY:
        ta,tb=tier(a,pop_line,variant),tier(b,pop_line,variant)
        if ta!=tb: return a if ta>tb else b
    if a.race_score!=b.race_score: return a if a.race_score>b.race_score else b
    return a if a.frame_no<b.frame_no else b


def coherent_ranking(riders,pop_line,variant):
    by={r.frame_no:r for r in riders}; adj={r.frame_no:set() for r in riders}; indeg={r.frame_no:0 for r in riders}
    for i,a in enumerate(riders):
        for b in riders[i+1:]:
            w=pair_winner(a,b,pop_line,variant); lo=b if w.frame_no==a.frame_no else a
            if lo.frame_no not in adj[w.frame_no]: adj[w.frame_no].add(lo.frame_no); indeg[lo.frame_no]+=1
    order=[]; zeros=[k for k,v in indeg.items() if v==0]
    while zeros:
        if len(zeros)!=1: return [],False
        u=zeros.pop(); order.append(by[u])
        for v in adj[u]:
            indeg[v]-=1
            if indeg[v]==0: zeros.append(v)
    return order,len(order)==len(riders)


def indist(a,b,pop_line,variant):
    return abs(a.race_score-b.race_score)<base.SCORE_BOUNDARY and tier(a,pop_line,variant)==tier(b,pop_line,variant)


def generate_orders(ranked,pop_line,variant):
    if len(ranked)<3:return [],'insufficient'
    a,b,c=ranked[:3]
    ab=indist(a,b,pop_line,variant); bc=indist(b,c,pop_line,variant)
    cd=len(ranked)>=4 and indist(ranked[2],ranked[3],pop_line,variant)
    if int(ab)+int(bc)+int(cd)>1:return [],'multiple_ambiguities'
    main=(a.frame_no,b.frame_no,c.frame_no)
    if ab:return [main,(b.frame_no,a.frame_no,c.frame_no)],'order_ab'
    if bc:return [main,(a.frame_no,c.frame_no,b.frame_no)],'order_bc'
    if cd:
        if len(ranked)>=5 and indist(ranked[3],ranked[4],pop_line,variant):return [],'third_slot_too_wide'
        d=ranked[3]; return [main,(a.frame_no,b.frame_no,d.frame_no)],'third_slot'
    return [main],'fixed'


def unique_sets(orders):
    out=[]
    for o in orders:
        s=frozenset(o)
        if s not in out: out.append(s)
    return out


def effective_score(r,pop_line):
    if r.line_idx==pop_line:
        if r.line_pos==2:return r.race_score+3.0
        if r.line_pos==3:return r.race_score+1.0
    return r.race_score


def order_set(s,riders,pop_line):
    xs=[r for r in riders if r.frame_no in s]
    xs.sort(key=lambda r:(-effective_score(r,pop_line),-r.race_score,r.frame_no))
    return tuple(r.frame_no for r in xs)


def prefilter(rid,pre,cr,og):
    lines=base.parse_true_line(cr.true_line)
    if not lines:return None,'line_unresolved'
    riders=base.make_riders(pre,lines)
    if len(riders)<4:return None,'score_or_entry_missing'
    tri=base.odds_map(og); n=len(riders); expected=n*(n-1)*(n-2)
    if len(tri)!=expected:return None,'odds_board_incomplete'
    pop=base.detect_popular_group(riders,tri)
    if pop is None:return None,'popular_group_unresolved'
    pop_line,target,mass,is_solo=pop
    if is_solo:return None,'popular_group_is_solo'
    outside=[r.race_score for r in riders if r.line_idx!=pop_line]
    if not outside:return None,'no_rival_line'
    if target.race_score>=max(outside)+base.SCORE_BOUNDARY:return None,'target_score_plus_3'
    if base.popular_line_too_strong(riders,pop_line,target):return None,'popular_line_too_strong'
    return {'riders':riders,'tri':tri,'pop_line':pop_line,'target':target},'eligible'


def variant_decision(state,variant):
    rem=[r for r in state['riders'] if r.frame_no!=state['target'].frame_no]
    ranked,ok=coherent_ranking(rem,state['pop_line'],variant)
    if not ok:return {'sets':[],'orders':[],'bets':[],'reason':'ranking_cycle'}
    legacy_orders,amb=generate_orders(ranked,state['pop_line'],variant)
    if not legacy_orders:return {'sets':[],'orders':[],'bets':[],'reason':'cannot_compress_'+amb}
    sets=unique_sets(legacy_orders)
    ordered=[order_set(s,state['riders'],state['pop_line']) for s in sets]
    eligible=[(o,state['tri'][o]) for o in ordered if o in state['tri'] and state['tri'][o]>=base.TRIFECTA_MIN_ODDS][:2]
    return {'sets':sets,'orders':ordered,'bets':eligible,'reason':'BET' if eligible else 'odds_too_low'}


def load_existing_repairs(month):
    p=CTX/f'fixed_logic_{month}_result_repairs.csv'
    if not p.exists():return {}
    d=pd.read_csv(p,encoding='utf-8-sig',dtype={'race_id':str})
    if d.empty:return {}
    d['race_id']=d.race_id.astype(str)
    return {k:g for k,g in d.groupby('race_id',sort=False)}


def fetch_result(rid,cr):
    try: got=ks.parse_race(str(getattr(cr,'venue_slug','')),str(rid))
    except Exception: got=[]
    rows=[]
    for x in got:
        rows.append({'race_id':str(rid),'month':'','venue_slug':getattr(cr,'venue_slug',''),'date':getattr(cr,'date',''),
                     'race_no':getattr(cr,'race_no',''),'banum':x.get('banum',''),'rank':x.get('rank','')})
    return rows


def run():
    rows=[]; new_repairs=[]
    months=TRAIN+OOS
    for month in months:
        race=pd.read_csv(DATA/f'{month}_keirin.csv',encoding='utf-8-sig',dtype={'race_id':str})
        ctx=pd.read_csv(CTX/f'{month}_races.csv',encoding='utf-8-sig',dtype={'race_id':str})
        odds=pd.read_csv(CTX/f'{month}_odds_3rentan.csv',encoding='utf-8-sig',dtype={'race_id':str})
        for d in (race,ctx,odds):d['race_id']=d.race_id.astype(str)
        use=ctx.copy()
        if 'context_quality' in use:use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use:use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        rb={k:g for k,g in race.groupby('race_id',sort=False)}; ob={k:g for k,g in odds.groupby('race_id',sort=False)}
        repb=load_existing_repairs(month)
        for cr in use.itertuples(index=False):
            rid=str(cr.race_id); full=rb.get(rid); og=ob.get(rid)
            if full is None or og is None:continue
            pre=full[['race_id','banum','race_score']].copy()
            state,why=prefilter(rid,pre,cr,og)
            if state is None:continue
            decisions={v:variant_decision(state,v) for v in VARIANTS}
            # Freeze every variant before seeing results.
            act=actual_order(full); source='monthly'
            if act is None and rid in repb:
                act=actual_order(repb[rid]); source='repair_existing'
            if act is None:
                got=fetch_result(rid,cr)
                if got:
                    for x in got:x['month']=month
                    new_repairs.extend(got); act=actual_order(pd.DataFrame(got)); source='repair_fetched'
            aset=frozenset(act) if act else None
            hb=int(state['target'].frame_no not in aset) if aset is not None else None
            split='TRAIN' if month in TRAIN else 'OOS'
            for v,d in decisions.items():
                sm=int(aset in d['sets']) if aset is not None else None
                hit=next(((o,od) for o,od in d['bets'] if act==o),None) if act else None
                stake=len(d['bets'])*base.STAKE_YEN; pay=int(round(hit[1]*base.STAKE_YEN)) if hit else 0
                rows.append({'split':split,'month':month,'race_id':rid,'variant':v,'target':state['target'].frame_no,
                             'head_bust':hb,'result_source':source,'candidate_sets':'|'.join('-'.join(map(str,sorted(s))) for s in d['sets']),
                             'orders':'|'.join('-'.join(map(str,o)) for o in d['orders']),'bets':'|'.join('-'.join(map(str,o)) for o,_ in d['bets']),
                             'actual':'-'.join(map(str,act)) if act else '','set_match':sm,'bet':int(bool(d['bets'])),
                             'bet_hit':int(hit is not None) if act else None,'stake':stake,'pay':pay if act else 0,'reason':d['reason']})
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False,encoding='utf-8-sig')
    if new_repairs:pd.DataFrame(new_repairs).drop_duplicates(['race_id','banum']).to_csv(REPAIRS,index=False,encoding='utf-8-sig')
    return out


def summarize_variant(x):
    sc=x[x.actual.astype(str)!='']; hb=sc[sc.head_bust==1]; b=sc[sc.bet==1]; bhb=b[b.head_bust==1]
    st=int(b.stake.sum()); py=int(b.pay.sum())
    return {'scorable_prefilter_races':len(sc),'head_bust_prefilter_races':len(hb),
            'candidate_set_exact_head_bust_n':int(hb.set_match.fillna(0).sum()),
            'candidate_set_exact_head_bust_pct':100*hb.set_match.fillna(0).mean() if len(hb) else None,
            'bet_races':len(b),'head_bust_bet_races':len(bhb),'set_exact_head_bust_bets':int(bhb.set_match.fillna(0).sum()) if len(bhb) else 0,
            'trifecta_hits':int(b.bet_hit.fillna(0).sum()) if len(b) else 0,'stake_yen':st,'pay_yen':py,
            'profit_yen':py-st,'roi_pct':100*py/st if st else None,
            'skip_reasons':Counter(sc.loc[sc.bet==0,'reason']).most_common()}


def compare_to_base(x,variant):
    piv=x[x.head_bust==1].pivot_table(index='race_id',columns='variant',values='set_match',aggfunc='first')
    piv=piv.dropna(subset=['BASE',variant])
    baseok=piv.BASE==1; vok=piv[variant]==1
    gain=int((~baseok & vok).sum()); loss=int((baseok & ~vok).sum())
    changed=0
    # candidate-family changed count on head-bust races
    z=x[x.head_bust==1].pivot_table(index='race_id',columns='variant',values='candidate_sets',aggfunc='first')
    if 'BASE' in z and variant in z:changed=int((z.BASE.fillna('')!=z[variant].fillna('')).sum())
    return {'gain_exact_vs_base':gain,'loss_exact_vs_base':loss,'net_exact_vs_base':gain-loss,'changed_candidate_family_races':changed}


def main():
    out=run(); payload={'protocol':{'train_months':TRAIN,'oos_months':OOS,'variants':VARIANTS,
      'selection_rule':'Choose highest TRAIN candidate-set exact count among variants; retain BASE unless an alternative strictly beats BASE. OOS not used for selection.'},'splits':{}}
    train_stats={}
    for split in ['TRAIN','OOS','ALL']:
        sx=out if split=='ALL' else out[out.split==split]
        block={}
        for v in VARIANTS:
            stats=summarize_variant(sx[sx.variant==v])
            if v!='BASE':stats['vs_base']=compare_to_base(sx,v)
            block[v]=stats
        payload['splits'][split]=block
        if split=='TRAIN':train_stats=block
    base_exact=train_stats['BASE']['candidate_set_exact_head_bust_n']
    alternatives=[v for v in VARIANTS if v!='BASE' and train_stats[v]['candidate_set_exact_head_bust_n']>base_exact]
    if alternatives:
        # deterministic: exact count desc, loss asc, declaration order asc
        selected=sorted(alternatives,key=lambda v:(-train_stats[v]['candidate_set_exact_head_bust_n'],train_stats[v]['vs_base']['loss_exact_vs_base'],VARIANTS.index(v)))[0]
    else:selected='BASE'
    payload['selected_variant_from_train_only']=selected
    payload['selected_oos']=payload['splits']['OOS'][selected]
    if selected!='BASE':payload['selected_oos_vs_base']=payload['splits']['OOS'][selected]['vs_base']
    SUMMARY.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
