#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the fixed canonical H1 + cheapest-order test on all backfilled history.

Observation window: 2022-01..2026-06.
Only the observation window changes. Candidate rules, score-cut ladder,
normalized market probability, and LOWEST_1..LOWEST_6 betting rules remain
identical to the earlier canonical experiments.

This is a historical sample-size/stability expansion, not fresh OOS validation.
"""
from pathlib import Path

import analyze_h1_canonical_three_line_ryo_bante_strong as canonical
import analyze_h1_cheapest_order_cumulative as cheapest

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / 'keirin_data' / 'strategy_context'
MONTHS = (
    [f'2022_{m:02d}' for m in range(1, 13)]
    + [f'2023_{m:02d}' for m in range(1, 13)]
    + [f'2024_{m:02d}' for m in range(1, 13)]
    + [f'2025_{m:02d}' for m in range(1, 13)]
    + [f'2026_{m:02d}' for m in range(1, 7)]
)

CANONICAL_OUT = CTX / 'h1_canonical_three_line_ryo_bante_strong_full_2022_2026_summary.json'
CANONICAL_DETAIL = CTX / 'h1_canonical_three_line_ryo_bante_strong_full_2022_2026_tickets.csv'
CHEAPEST_OUT = CTX / 'h1_cheapest_order_cumulative_full_2022_2026_summary.json'
CHEAPEST_DETAIL = CTX / 'h1_cheapest_order_cumulative_full_2022_2026_tickets.csv'


def period_of(month: str):
    y, m = month.split('_')
    y = int(y)
    m = int(m)
    return f'{y}_H1' if m <= 6 else f'{y}_H2'


def main():
    canonical.h1.MONTHS = MONTHS
    canonical.period_of = period_of
    canonical.OUT = CANONICAL_OUT
    canonical.DETAIL = CANONICAL_DETAIL
    canonical.main()

    cheapest.SRC = CANONICAL_DETAIL
    cheapest.OUT = CHEAPEST_OUT
    cheapest.DETAIL = CHEAPEST_DETAIL
    cheapest.main()


if __name__ == '__main__':
    main()
