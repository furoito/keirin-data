#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose survivor reconstruction without changing the betting strategy.

Universe: the exact races that v0.1b classified BET in 2026-08.
A = current v0.1b reconstruction (3-point boundary + position; includes its 1-2 candidate sets)
B = score only: top 3 remaining riders by race_score
C = position only: use only canonical positional tiers. Because position alone often ties,
    evaluate the complete family of top-3 sets allowed by the cutoff tier rather than
    breaking ties with score or frame number.

Report both:
- all BET races
- head-bust subset only (target actually outside top 3), which is the conditional world
  the strategy is intended to exploit.

No result data influences A/B/C selection. Results are read only for scoring.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

import popular_head_skip_v01 as base

DATA = Path("keirin_data")
CTX = DATA / "strategy_context"
MONTH = "2026_08"
RESULTS = CTX / "popular_head_skip_v01b_results.csv"
OUT = CTX / "reconstruction_abc_diagnostic.csv"
SUMMARY = CTX / "reconstruction_abc_summary.json"


def parse_order(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    vals = []
    for x in str(s).split("-"):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(int(float(x)))
        except ValueError:
            return None
    return tuple(vals) if len(vals) == 3 else None


def parse_candidate_sets(s):
    sets = set()
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return sets
    for part in str(s).split("|"):
        o = parse_order(part)
        if o:
            sets.add(frozenset(o))
    return sets


def overlap(candidate, actual):
    return len(set(candidate) & set(actual)) if candidate and actual else 0


def score_only_set(riders, target):
    remain = [r for r in riders if r.frame_no != target]
    remain.sort(key=lambda r: (-r.race_score, r.frame_no))
    return frozenset(r.frame_no for r in remain[:3])


def position_only_family(riders, target, popular_line):
    """All top-3 sets consistent with position tier alone.

    Higher tiers are mandatory. At the cutoff tier, choose any required riders.
    This avoids sneaking score or arbitrary frame-number priority into C.
    """
    remain = [r for r in riders if r.frame_no != target]
    by_tier = {}
    for r in remain:
        by_tier.setdefault(base.position_tier(r, popular_line), []).append(r.frame_no)
    chosen = []
    family = set()
    need = 3
    for tier in sorted(by_tier, reverse=True):
        members = sorted(by_tier[tier])
        if len(members) < need:
            chosen.extend(members)
            need -= len(members)
            continue
        if len(members) == need:
            family.add(frozenset(chosen + members))
            need = 0
            break
        for pick in itertools.combinations(members, need):
            family.add(frozenset(chosen + list(pick)))
        need = 0
        break
    if need > 0:
        return set()
    return family


def best_overlap(family, actual):
    if not family or not actual:
        return 0
    return max(len(set(x) & set(actual)) for x in family)


def main():
    decisions = pd.read_csv(RESULTS, encoding="utf-8-sig", dtype={"race_id": str})
    decisions = decisions[decisions.action.astype(str) == "BET"].copy()
    race = pd.read_csv(DATA / f"{MONTH}_keirin.csv", encoding="utf-8-sig", dtype={"race_id": str})
    ctx = pd.read_csv(CTX / f"{MONTH}_races.csv", encoding="utf-8-sig", dtype={"race_id": str})
    race["race_id"] = race.race_id.astype(str)
    ctx["race_id"] = ctx.race_id.astype(str)
    cidx = ctx.set_index("race_id")
    gidx = {str(k): g for k, g in race.groupby("race_id", sort=False)}

    rows = []
    for d in decisions.itertuples(index=False):
        rid = str(d.race_id)
        if rid not in gidx or rid not in cidx.index:
            continue
        g = gidx[rid]
        cr = cidx.loc[rid]
        lines = base.parse_true_line(cr.true_line)
        pre = g[["race_id", "banum", "race_score"]].copy()
        riders = base.make_riders(pre, lines)
        if not riders:
            continue
        target = int(float(d.target))
        pop_line = int(float(d.popular_line))
        actual = parse_order(d.actual_order)
        if not actual:
            actual = base.actual_order(g)
        if not actual:
            continue
        actual_set = frozenset(actual)

        a_family = parse_candidate_sets(d.candidate_orders)
        a_rank = []
        for x in str(d.ranking).split("-")[:3]:
            try:
                a_rank.append(int(float(x)))
            except ValueError:
                pass
        a_rank_set = frozenset(a_rank) if len(a_rank) == 3 else frozenset()
        b_set = score_only_set(riders, target)
        c_family = position_only_family(riders, target, pop_line)

        head_bust = int(target not in actual_set)
        rows.append({
            "race_id": rid,
            "date": d.date,
            "venue": d.venue_slug,
            "race_no": d.race_no,
            "line": cr.true_line,
            "target": target,
            "head_bust": head_bust,
            "actual": "-".join(map(str, actual)),
            "A_rank_set": "-".join(map(str, sorted(a_rank_set))),
            "A_family_n": len(a_family),
            "A_exact": int(actual_set in a_family),
            "A_rank_exact": int(actual_set == a_rank_set),
            "A_best_overlap": best_overlap(a_family, actual_set),
            "B_set": "-".join(map(str, sorted(b_set))),
            "B_exact": int(actual_set == b_set),
            "B_overlap": overlap(b_set, actual_set),
            "C_family_n": len(c_family),
            "C_exact_feasible": int(actual_set in c_family),
            "C_best_overlap": best_overlap(c_family, actual_set),
            "C_unique": int(len(c_family) == 1),
            "C_unique_exact": int(len(c_family) == 1 and actual_set in c_family),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    def metrics(x):
        n = len(x)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "A_exact_pct": float(x.A_exact.mean() * 100),
            "A_rank_exact_pct": float(x.A_rank_exact.mean() * 100),
            "A_avg_overlap_of_3": float(x.A_best_overlap.mean()),
            "B_exact_pct": float(x.B_exact.mean() * 100),
            "B_avg_overlap_of_3": float(x.B_overlap.mean()),
            "C_feasible_exact_pct": float(x.C_exact_feasible.mean() * 100),
            "C_avg_best_overlap_of_3": float(x.C_best_overlap.mean()),
            "C_unique_rate_pct": float(x.C_unique.mean() * 100),
            "C_unique_exact_pct_all": float(x.C_unique_exact.mean() * 100),
            "C_unique_exact_pct_when_unique": float(
                x.loc[x.C_unique == 1, "C_unique_exact"].mean() * 100
            ) if (x.C_unique == 1).any() else None,
        }

    summary = {
        "universe": "v0.1b BET races only",
        "all_bet_races": metrics(out),
        "head_bust_only": metrics(out[out.head_bust == 1]),
        "head_bust_rate_pct": float(out.head_bust.mean() * 100) if len(out) else None,
        "definitions": {
            "A": "v0.1b 3-point boundary + position, candidate-family scoring",
            "B": "score-only top3 after target removal",
            "C": "position-tier-only; all sets allowed by cutoff tie are scored",
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(out):
        hb = out[out.head_bust == 1]
        print("\nHEAD-BUST diagnostic counts")
        print(f"A exact {hb.A_exact.sum()}/{len(hb)}")
        print(f"B exact {hb.B_exact.sum()}/{len(hb)}")
        print(f"C feasible exact {hb.C_exact_feasible.sum()}/{len(hb)}")
        print("\nCases where B beats A on exact-set scoring:")
        cols = ["date","venue","race_no","line","target","actual","A_rank_set","B_set","A_exact","B_exact","A_best_overlap","B_overlap"]
        print(hb[(hb.B_exact == 1) & (hb.A_exact == 0)][cols].to_string(index=False))
        print("\nCases where A beats B on exact-set scoring:")
        print(hb[(hb.A_exact == 1) & (hb.B_exact == 0)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
