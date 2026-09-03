#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
SRC = CTX / 'core_top40_rank1_tickets.csv'
OUT = CTX / 'core_20_40_top3spread_group_hits.json'


def main():
    df = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'month': str, 'race_id': str, 'trio_key': str})
    x = df[(df.role_ok == 1) & (df.top3_spread == 1) &
           (df.group_score_percentile > 0.20) & (df.group_score_percentile <= 0.40) &
           (df.group_hit == 1)].copy()
    cols = ['month','race_id','trio_key','rank1_order','rank1_posted_odds','rank1_exact_hit','winner_odds_rank','effective_group_odds']
    x = x[cols].sort_values(['rank1_exact_hit','rank1_posted_odds'], ascending=[False, True])
    wins = x[x.rank1_exact_hit == 1].to_dict('records')
    misses = x[x.rank1_exact_hit == 0].to_dict('records')
    payload = {
        'count': int(len(x)),
        'wins_count': int(len(wins)),
        'misses_count': int(len(misses)),
        'wins': wins,
        'misses': misses,
        'winning_cheapest_odds': [float(v) for v in x.loc[x.rank1_exact_hit == 1, 'rank1_posted_odds'].tolist()],
        'miss_cheapest_odds': [float(v) for v in x.loc[x.rank1_exact_hit == 0, 'rank1_posted_odds'].tolist()],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
