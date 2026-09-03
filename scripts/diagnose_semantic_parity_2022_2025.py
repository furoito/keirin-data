#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose whether 2022 backfill and 2025 canonical data have comparable semantics.

This is a pipeline-parity probe, not a strategy search. It compares:
- base schema and race_score scale;
- running-style / role semantics used by the current selector;
- true-line structure and source domains;
- 3rentan board completeness and overround semantics.

We intentionally use 2022-01..08 (bulk-complete at probe design time) and
2025-01..12. Missing months are reported rather than treated as zero.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

import popular_head_skip_v01 as base
import test_h1_crossline_highscore_groups as h1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "keirin_data"
CTX = DATA / "strategy_context"
OUT = CTX / "semantic_parity_2022_2025_summary.json"

MONTHS = {
    "2022": [f"2022_{m:02d}" for m in range(1, 9)],
    "2025": [f"2025_{m:02d}" for m in range(1, 13)],
}


def finite_series(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return s[np.isfinite(s)]


def dist(values):
    s = finite_series(values)
    if s.empty:
        return {"n": 0, "missing_or_nonfinite": int(len(pd.Series(values))), "mean": None, "std": None,
                "min": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "max": None}
    q = s.quantile([.05, .25, .50, .75, .95])
    return {
        "n": int(len(s)),
        "missing_or_nonfinite": int(len(pd.Series(values)) - len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p05": float(q.loc[.05]),
        "p25": float(q.loc[.25]),
        "p50": float(q.loc[.50]),
        "p75": float(q.loc[.75]),
        "p95": float(q.loc[.95]),
        "max": float(s.max()),
    }


def proportions(counter: Counter):
    total = sum(counter.values())
    return {
        str(k): {"n": int(v), "pct": float(100.0 * v / total) if total else None}
        for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))
    }


def domain(url):
    try:
        return urlparse(str(url)).netloc.lower() or "EMPTY"
    except Exception:
        return "INVALID"


def normalize_style(v):
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return "UNKNOWN"
    # Preserve the exact semantics used by the current analysis code: it expects '両'.
    # Prefix normalization is only for descriptive auditing of source variants.
    if s.startswith("両"):
        return "両"
    if s.startswith("逃"):
        return "逃"
    if s.startswith("ま") or s.startswith("捲"):
        return "捲"
    if s.startswith("差"):
        return "差"
    if s.startswith("追"):
        return "追"
    return s


def column_signature(df: pd.DataFrame):
    return [str(x) for x in df.columns]


def audit_year(year: str):
    schema = {"base": Counter(), "context": Counter(), "odds": Counter()}
    score_all = []
    score_usable = []
    styles_all = Counter()
    styles_usable = Counter()
    line_sources = Counter()
    odds_sources = Counter()
    n_lines = Counter()
    line_sizes = Counter()
    role_counts = Counter()
    overround = []
    parsed_ratio = []
    month_rows = {}
    skipped = Counter()

    usable_races = 0
    board_complete_races = 0
    raw_context_rows = 0
    context_full_rows = 0
    price_usable_rows = 0

    for month in MONTHS[year]:
        loaded = h1.load_month(month)
        if loaded is None:
            skipped["month_missing"] += 1
            month_rows[month] = {"status": "missing"}
            continue
        b, c, o = loaded
        schema["base"][tuple(column_signature(b))] += 1
        schema["context"][tuple(column_signature(c))] += 1
        schema["odds"][tuple(column_signature(o))] += 1

        raw_context_rows += len(c)
        bscore = pd.to_numeric(b.get("race_score"), errors="coerce") if "race_score" in b else pd.Series(dtype=float)
        score_all.extend(bscore.tolist())
        if "running_style" in b:
            styles_all.update(normalize_style(x) for x in b.running_style)

        use = c.copy()
        if "context_quality" in use:
            mask = use.context_quality.astype(str) == "full"
            context_full_rows += int(mask.sum())
            use = use[mask]
        else:
            context_full_rows += len(use)
        if "price_usable" in use:
            pmask = use.price_usable.astype(str).str.lower().isin({"true", "1"})
            price_usable_rows += int(pmask.sum())
            use = use[pmask]
        else:
            price_usable_rows += len(use)
        use = use.drop_duplicates("race_id", keep="last")
        usable_races += len(use)

        bby = {str(k): g for k, g in b.groupby("race_id", sort=False)}
        oby = {str(k): g for k, g in o.groupby("race_id", sort=False)}

        m_complete = 0
        for cr in use.to_dict("records"):
            rid = str(cr.get("race_id"))
            line_sources[domain(cr.get("line_source_url", ""))] += 1
            odds_sources[domain(cr.get("odds_source_url", ""))] += 1
            try:
                nl = int(float(cr.get("n_lines")))
                n_lines[nl] += 1
            except Exception:
                pass
            try:
                parsed = float(cr.get("parsed_3rentan"))
                expected_ctx = float(cr.get("expected_3rentan"))
                if expected_ctx > 0:
                    parsed_ratio.append(parsed / expected_ctx)
            except Exception:
                pass

            lines = base.parse_true_line(cr.get("true_line"))
            if not lines:
                skipped["line_unresolved"] += 1
                continue
            pre = bby.get(rid)
            og = oby.get(rid)
            if pre is None or og is None:
                skipped["base_or_odds_missing"] += 1
                continue

            frames = sorted({int(x) for g in lines for x in g})
            for g in lines:
                line_sizes[len(g)] += 1
            tri = base.odds_map(og)
            expected = len(frames) * (len(frames) - 1) * (len(frames) - 2)
            if len(tri) != expected:
                skipped["odds_board_incomplete"] += 1
                continue
            z = sum(1.0 / od for od in tri.values() if od > 0)
            if not np.isfinite(z) or z <= 0:
                skipped["zero_or_invalid_market_mass"] += 1
                continue
            overround.append(z)
            board_complete_races += 1
            m_complete += 1

            score_map = {}
            style_map = {}
            for r in pre.itertuples(index=False):
                try:
                    fn = int(float(r.banum))
                except Exception:
                    continue
                try:
                    sc = float(r.race_score)
                    if np.isfinite(sc):
                        score_map[fn] = sc
                        score_usable.append(sc)
                except Exception:
                    pass
                st = normalize_style(getattr(r, "running_style", "UNKNOWN"))
                style_map[fn] = st
                styles_usable[st] += 1

            for g in lines:
                gg = [int(x) for x in g]
                if len(gg) < 2:
                    continue
                for pos, fn in enumerate(gg, 1):
                    st = style_map.get(fn, "UNKNOWN")
                    role_counts["eligible_multirider_riders"] += 1
                    if st == "両":
                        role_counts["style_ryo"] += 1
                    if pos == 2:
                        role_counts["line_pos2"] += 1
                    if st == "両" or pos == 2:
                        role_counts["role_ok"] += 1
                    if st == "両" and pos == 1:
                        role_counts["ryo_line_head"] += 1
                    if st == "両" and pos == 2:
                        role_counts["ryo_bante"] += 1

        month_rows[month] = {
            "status": "loaded",
            "base_rows": int(len(b)),
            "context_rows": int(len(c)),
            "usable_context_races": int(len(use)),
            "complete_board_races": int(m_complete),
        }

    schema_out = {}
    for k, v in schema.items():
        schema_out[k] = [
            {"months": int(n), "columns": list(cols)}
            for cols, n in v.items()
        ]

    rc = dict(role_counts)
    elig = role_counts.get("eligible_multirider_riders", 0)
    rc["style_ryo_pct"] = float(100 * role_counts.get("style_ryo", 0) / elig) if elig else None
    rc["line_pos2_pct"] = float(100 * role_counts.get("line_pos2", 0) / elig) if elig else None
    rc["role_ok_pct"] = float(100 * role_counts.get("role_ok", 0) / elig) if elig else None

    return {
        "months_requested": MONTHS[year],
        "month_rows": month_rows,
        "schema_signatures": schema_out,
        "base_race_score_all_rows": dist(score_all),
        "race_score_on_usable_full_board_races": dist(score_usable),
        "running_style_all_rows": proportions(styles_all),
        "running_style_on_usable_full_board_races": proportions(styles_usable),
        "context": {
            "raw_rows": int(raw_context_rows),
            "context_full_rows": int(context_full_rows),
            "price_usable_rows_after_full_filter": int(price_usable_rows),
            "usable_races": int(usable_races),
            "line_source_domains": proportions(line_sources),
            "odds_source_domains": proportions(odds_sources),
            "n_lines": proportions(n_lines),
            "line_sizes": proportions(line_sizes),
            "role_semantics_multirider_lines": rc,
        },
        "odds": {
            "complete_board_races": int(board_complete_races),
            "overround_sum_inverse_odds": dist(overround),
            "parsed_over_expected_ratio": dist(parsed_ratio),
        },
        "skipped": dict(skipped),
    }


def compare(a, b):
    flags = []
    notes = []

    # Schema parity: one stable signature per source type and exact equality across years.
    for kind in ["base", "context", "odds"]:
        sa = [x["columns"] for x in a["schema_signatures"][kind]]
        sb = [x["columns"] for x in b["schema_signatures"][kind]]
        if len(sa) != 1 or len(sb) != 1 or sa[0] != sb[0]:
            flags.append(f"SCHEMA_MISMATCH_{kind.upper()}")

    for key in ["line_source_domains", "odds_source_domains"]:
        da = set(a["context"][key])
        db = set(b["context"][key])
        if da != db:
            flags.append(f"SOURCE_DOMAIN_MISMATCH_{key.upper()}")

    s22 = a["race_score_on_usable_full_board_races"]
    s25 = b["race_score_on_usable_full_board_races"]
    if s22["p50"] is not None and s25["p50"] is not None:
        med_delta = float(s25["p50"] - s22["p50"])
        notes.append({"metric": "race_score_median_2025_minus_2022", "value": med_delta})
        # This threshold is a diagnostic tripwire, not a claim that distributions must match.
        if abs(med_delta) > 5.0:
            flags.append("RACE_SCORE_SCALE_SHIFT_GT5")

    o22 = a["odds"]["overround_sum_inverse_odds"]["p50"]
    o25 = b["odds"]["overround_sum_inverse_odds"]["p50"]
    if o22 is not None and o25 is not None and o22 > 0:
        rel = float(o25 / o22)
        notes.append({"metric": "overround_median_ratio_2025_over_2022", "value": rel})
        if rel < .90 or rel > 1.10:
            flags.append("OVERROUND_MEDIAN_SHIFT_GT10PCT")

    r22 = a["context"]["role_semantics_multirider_lines"].get("role_ok_pct")
    r25 = b["context"]["role_semantics_multirider_lines"].get("role_ok_pct")
    if r22 is not None and r25 is not None:
        delta = float(r25 - r22)
        notes.append({"metric": "role_ok_pct_2025_minus_2022", "value": delta})
        if abs(delta) > 10.0:
            flags.append("ROLE_MIX_SHIFT_GT10PP")

    verdict = "PARITY_SUPPORTED" if not flags else "PARITY_NEEDS_REVIEW"
    return {"verdict": verdict, "red_flags": flags, "diagnostic_notes": notes,
            "interpretation_guardrail": "Different natural race mixes are allowed. Red flags identify possible semantic/pipeline shifts, not proof of a bug."}


def main():
    by_year = {year: audit_year(year) for year in ["2022", "2025"]}
    payload = {
        "status": "semantic_parity_probe_2022_vs_2025",
        "scope": {
            "2022": "2022-01..08 only; chosen because these were bulk-complete before the active September backfill",
            "2025": "2025-01..12",
        },
        "definition_invariants": [
            "race_score is consumed directly from the monthly base CSV, not recomputed by year",
            "role_ok is running_style == '両' OR true-line position == 2",
            "single-rider lines are excluded from the current 3-line selector",
            "3rentan raw market weight is 1/posted_odds and normalized by the full race board",
            "unordered group market probability sums the six exact-order normalized probabilities",
        ],
        "by_year": by_year,
        "comparison": compare(by_year["2022"], by_year["2025"]),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
