#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ticket-level diagnostic for a requested two-line structure.

Keep discovery filters:
- unordered trio group-score top 45% within race
- no rider at line_pos >= 3

Requested ordered trifecta structure (2-line span):
1st: running_style == '両' and line_pos == 1
2nd: line_pos == 2 of the SAME line as 1st
3rd: line_pos == 2 of a DIFFERENT line

Comparison control (current 3-line structure):
1st: running_style == '両' and line_pos == 1
2nd: line_pos == 2 of another line
3rd: line_pos == 2 of a third line

Evaluate quoted trifecta odds >=50 / >=100 / >=200 with flat 1 unit per ticket.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
OUT = CTX / 'h1_ryo_head_same_line_mate_other_bante_summary.json'
CUTS = [50, 100, 200]
PCT = 0.45


def actual_ordered_top3(pre: pd.DataFrame):
    vals=[]
    for r in pre.itertuples(index=False):
        try:
            pos=int(str(r.rank).strip()); fn=int(r.banum)
        except Exception:
            continue
        if 1 <= pos <= 3:
            vals.append((pos,fn))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]:
        return None
    return tuple(fn for _,fn in vals)


def agg(x: pd.DataFrame) -> dict:
    n=int(len(x)); stake=float(n)
    gross=float(x.loc[x.actual_hit==1,'odds'].sum()) if n else 0.0
    exp=float(x.market_p.sum()) if n else 0.0
    hits=int(x.actual_hit.sum()) if n else 0
    return {
        'tickets':n,
        'races':int(x.race_id.nunique()) if n else 0,
        'stake_units':stake,
        'gross_return_units':gross,
        'gross_roi_pct':float(100*gross/stake) if stake else None,
        'net_roi_pct':float(100*(gross-stake)/stake) if stake else None,
        'actual_hits':hits,
        'normalized_market_expected_hits':exp,
        'actual_over_normalized_market':float(hits/exp) if exp>0 else None,
        'avg_ticket_odds':float(x.odds.mean()) if n else None,
        'median_ticket_odds':float(x.odds.median()) if n else None,
    }


def summarize(z: pd.DataFrame) -> dict:
    return {
        'all_odds':agg(z),
        'min_ticket_odds':{str(c):agg(z[z.odds>=c]) for c in CUTS},
        'non_overlapping_bins':{
            '50-100':agg(z[(z.odds>=50)&(z.odds<100)]),
            '100-200':agg(z[(z.odds>=100)&(z.odds<200)]),
            '200-plus':agg(z[z.odds>=200]),
        },
    }


def main():
    rows=[]; skipped=Counter(); usable_by_month={}
    for month in h1.MONTHS:
        loaded=h1.load_month(month)
        if loaded is None:
            continue
        b,c,o=loaded
        use=c.copy()
        if 'context_quality' in use:
            use=use[use.context_quality.astype(str)=='full']
        if 'price_usable' in use:
            use=use[use.price_usable.astype(str).str.lower().isin({'true','1'})]
        use=use.drop_duplicates('race_id',keep='last')
        usable_by_month[month]=int(len(use))
        bby={str(k):g for k,g in b.groupby('race_id',sort=False)}
        oby={str(k):g for k,g in o.groupby('race_id',sort=False)}

        for cr in use.to_dict('records'):
            rid=str(cr['race_id']); pre=bby.get(rid); og=oby.get(rid)
            if pre is None or og is None:
                skipped['base_or_odds_missing']+=1; continue
            groups,why=h1.race_rows(month,rid,pre,cr,og)
            if groups is None:
                skipped[why]+=1; continue
            actual=actual_ordered_top3(pre)
            if actual is None:
                skipped['ordered_result_missing']+=1; continue

            tri=base.odds_map(og)
            z=sum(1.0/od for od in tri.values() if od>0)
            if z<=0:
                skipped['zero_mass']+=1; continue

            lines=base.parse_true_line(cr.get('true_line'))
            line_of={}; line_pos={}; line_size={}
            for li,g in enumerate(lines,1):
                for pos,fn in enumerate(g,1):
                    line_of[int(fn)]=li; line_pos[int(fn)]=pos; line_size[int(fn)]=len(g)

            style={}
            for r in pre.itertuples(index=False):
                try: fn=int(r.banum)
                except Exception: continue
                s=str(getattr(r,'running_style','')).strip()
                style[fn]=s

            for q in groups:
                if float(q['group_score_percentile']) > PCT:
                    continue
                trio=tuple(int(x) for x in q['trio'].split('-'))
                if any(line_pos.get(fn,99)>=3 for fn in trio):
                    continue
                for perm in itertools.permutations(trio):
                    od=tri.get(tuple(perm))
                    if od is None or od<=0:
                        continue
                    a,b2,c3=perm
                    pattern=None
                    # Requested 2-line form: RYO head -> own mate -> other-line mate.
                    if (
                        q['line_span']==2 and
                        style.get(a)=='両' and line_pos.get(a)==1 and line_size.get(a,0)>=2 and
                        line_pos.get(b2)==2 and line_of.get(b2)==line_of.get(a) and
                        line_pos.get(c3)==2 and line_of.get(c3)!=line_of.get(a)
                    ):
                        pattern='RYO_HEAD_OWN_BANTE_OTHER_BANTE_2LINES'
                    # Current 3-line control: RYO head -> other-line bante -> third-line bante.
                    elif (
                        q['line_span']==3 and
                        style.get(a)=='両' and line_pos.get(a)==1 and
                        line_pos.get(b2)==2 and line_of.get(b2)!=line_of.get(a) and
                        line_pos.get(c3)==2 and len({line_of.get(a),line_of.get(b2),line_of.get(c3)})==3
                    ):
                        pattern='RYO_HEAD_OTHER_BANTE_OTHER_BANTE_3LINES'
                    if pattern is None:
                        continue
                    p=(1.0/float(od))/z
                    rows.append({
                        'month':month,'race_id':rid,'pattern':pattern,
                        'ticket':'-'.join(map(str,perm)),'odds':float(od),'market_p':float(p),
                        'actual_hit':int(tuple(perm)==actual),
                        'group_score_percentile':float(q['group_score_percentile']),
                        'line_span':int(q['line_span']),
                    })

    df=pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No matching tickets')
    views={name:summarize(g) for name,g in df.groupby('pattern',sort=True)}
    payload={
        'status':'exploratory_same_data_requested_structure_diagnostic',
        'group_filter':'group_score_top45pct AND NO_THIRD',
        'requested_structure':'RYO line head -> own line bante -> other-line bante (exactly 2 lines)',
        'control_structure':'RYO line head -> bante from second line -> bante from third line (exactly 3 lines)',
        'stake_model':'flat 1 unit per ordered trifecta ticket',
        'warning':'Same discovery data; do not canonize until re-tested after backfill.',
        'usable_races_by_month':usable_by_month,
        'skipped':dict(skipped),
        'views':views,
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
