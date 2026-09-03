#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refine the 25-45 percentile region into 5-point bins while preserving
non-overlapping quoted trifecta ticket-odds bins and the current structural views.
Same-data discovery only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'analyze_h1_percentile_x_ticket_odds.py'
OUT = ROOT / 'keirin_data' / 'strategy_context' / 'h1_percentile_25_45_fivepct_summary.json'

spec = importlib.util.spec_from_file_location('pctbase', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.PCT_BINS = [
    ('25-30', 0.25, 0.30, False),
    ('30-35', 0.30, 0.35, False),
    ('35-40', 0.35, 0.40, False),
    ('40-45', 0.40, 0.45, False),
]
mod.PCT_CUMULATIVE = [
    ('top30', 0.30),
    ('top35', 0.35),
    ('top40', 0.40),
    ('top45', 0.45),
]
mod.OUT = OUT

# Reuse the exact candidate generation, market normalization and odds bins.
mod.main()

# Patch metadata to make the narrower question explicit.
payload = json.loads(OUT.read_text(encoding='utf-8'))
payload['status'] = 'exploratory_25_45_percentile_five_point_ticket_odds_diagnostic'
payload['question'] = 'Within the previously strong 25-45 group-score percentile region, which 5-point sub-bands carry the signal by quoted trifecta ticket-odds bin?'
payload['percentile_nonoverlap_bins'] = ['25-30','30-35','35-40','40-45']
payload['percentile_cumulative_cuts'] = ['top30','top35','top40','top45']
payload['warning'] = 'Same discovery data and a refinement chosen after observing 25-45 strength. Treat as localization only, not validation.'
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False, indent=2))
