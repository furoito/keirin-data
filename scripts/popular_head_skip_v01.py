#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Popular-line lead fade v0.1 backtest on furoito/keirin-data.

Selection uses only pre-race fields:
- monthly CSV: race_score (result columns are not passed to selector)
- strategy_context: true_line + full 3-ren-tan board

Result columns (rank / winning payout) are read only after a Decision is fixed.

v0.1 fixed rules
- ability/position boundary: 3.0 points
- target score >= every outside-line rider +3.0 => skip
- popular line too strong (provisional): after target removal, pos2+pos3 are raw-score top2 => skip
- under 3 points, positional priority:
    popular line pos2 > popular line pos3+ > other-line pos2+ > other head/solo
- >=3 points, higher race_score wins
- max 2 trifecta bets, no box
- trifecta minimum 30x
- A-B-CD third-slot form is allowed (2 bets max)
- same 3 riders but order ambiguous => choose the higher-odds plausible order

Important limitations
- Popular line is operationalized as inverse-odds mass of head-first trifectas that include
  at least one line mate. This is v0.1 and should be validated against human labels.
- 3-ren-puku candidate odds are not available in strategy_context, so trio ROI is not scored here.
- The subjective EV gate P(hit) > 1/odds is not automated in v0.1.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path("keirin_data")
CTX = DATA / "strategy_context"
SCORE_BOUNDARY = 3.0
TRIFECTA_MIN_ODDS = 30.0
STAKE_YEN = 100


@dataclass(frozen=True)
class Rider:
    frame_no: int
    race_score: float
    line_idx: int
    line_pos: int
    line_size: int


@dataclass
class Decision:
    race_id: str
    action: str
    reason: str
    popular_line: int | None = None
    target: int | None = None
    market_mass: float | None = None
    ranking: list[int] | None = None
    candidate_orders: list[tuple[int, int, int]] | None = None
    bets: list[tuple[tuple[int, int, int], float]] | None = None
    ambiguity: str | None = None


def parse_true_line(s: object) -> list[list[int]]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    groups: list[list[int]] = []
    for g in str(s).strip().split("/"):
        g = g.strip()
        if not g:
            continue
        try:
            nums = [int(x) for x in g.split("-") if x.strip()]
        except ValueError:
            return []
        if nums:
            groups.append(nums)
    flat = [x for g in groups for x in g]
    if not flat or len(flat) != len(set(flat)):
        return []
    return groups


def make_riders(pre: pd.DataFrame, lines: list[list[int]]) -> list[Rider]:
    score = {
        int(r.banum): float(r.race_score)
        for r in pre.itertuples(index=False)
        if pd.notna(r.banum) and pd.notna(r.race_score)
    }
    out: list[Rider] = []
    for li, group in enumerate(lines, 1):
        for pos, fn in enumerate(group, 1):
            if fn not in score:
                return []
            out.append(Rider(fn, score[fn], li, pos, len(group)))
    return out


def odds_map(og: pd.DataFrame) -> dict[tuple[int, int, int], float]:
    out = {}
    for r in og.itertuples(index=False):
        try:
            k = (int(r.b1), int(r.b2), int(r.b3))
            v = float(r.odds_decimal)
        except (TypeError, ValueError):
            continue
        if v > 0 and len(set(k)) == 3:
            out[k] = v
    return out


def line_members(riders: list[Rider]) -> dict[int, list[Rider]]:
    d: dict[int, list[Rider]] = defaultdict(list)
    for r in riders:
        d[r.line_idx].append(r)
    for x in d.values():
        x.sort(key=lambda r: r.line_pos)
    return dict(d)


def detect_popular_group(riders: list[Rider], tri: dict[tuple[int, int, int], float]):
    """Return (line_idx, leader, mass, is_solo) for market-heaviest group."""
    best = None
    for li, members in line_members(riders).items():
        leader = members[0]
        mates = {r.frame_no for r in members[1:]}
        mass = 0.0
        for (a, b, c), od in tri.items():
            if a != leader.frame_no or od <= 0:
                continue
            if mates and b not in mates and c not in mates:
                continue
            mass += 1.0 / od
        if mass <= 0:
            continue
        item = (li, leader, mass, len(members) == 1)
        if best is None or mass > best[2]:
            best = item
    return best


def position_tier(r: Rider, popular_line: int) -> int:
    if r.line_idx == popular_line:
        if r.line_pos == 2:
            return 4
        if r.line_pos >= 3:
            return 3
    if r.line_pos >= 2:
        return 2
    return 1


def pair_winner(a: Rider, b: Rider, popular_line: int) -> Rider:
    gap = abs(a.race_score - b.race_score)
    if gap >= SCORE_BOUNDARY:
        if a.race_score != b.race_score:
            return a if a.race_score > b.race_score else b
    else:
        ta, tb = position_tier(a, popular_line), position_tier(b, popular_line)
        if ta != tb:
            return a if ta > tb else b
    if a.race_score != b.race_score:
        return a if a.race_score > b.race_score else b
    return a if a.frame_no < b.frame_no else b


def coherent_ranking(riders: list[Rider], popular_line: int) -> tuple[list[Rider], bool]:
    """Topological-sort all pairwise preferences; fail closed on a cycle."""
    by = {r.frame_no: r for r in riders}
    adj = {r.frame_no: set() for r in riders}
    indeg = {r.frame_no: 0 for r in riders}
    for i, a in enumerate(riders):
        for b in riders[i + 1:]:
            w = pair_winner(a, b, popular_line)
            lo = b if w.frame_no == a.frame_no else a
            if lo.frame_no not in adj[w.frame_no]:
                adj[w.frame_no].add(lo.frame_no)
                indeg[lo.frame_no] += 1
    order: list[Rider] = []
    zeros = [k for k, v in indeg.items() if v == 0]
    while zeros:
        if len(zeros) != 1:
            return [], False
        u = zeros.pop()
        order.append(by[u])
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                zeros.append(v)
    return order, len(order) == len(riders)


def indistinguishable(a: Rider, b: Rider, popular_line: int) -> bool:
    return (
        abs(a.race_score - b.race_score) < SCORE_BOUNDARY
        and position_tier(a, popular_line) == position_tier(b, popular_line)
    )


def popular_line_too_strong(riders: list[Rider], popular_line: int, target: Rider) -> bool:
    followers = sorted(
        [r for r in riders if r.line_idx == popular_line and r.frame_no != target.frame_no],
        key=lambda r: r.line_pos,
    )
    p2 = next((r for r in followers if r.line_pos == 2), None)
    p3 = next((r for r in followers if r.line_pos == 3), None)
    if p2 is None or p3 is None:
        return False
    remain = [r for r in riders if r.frame_no != target.frame_no]
    raw_top2 = sorted(remain, key=lambda r: (-r.race_score, r.frame_no))[:2]
    return {r.frame_no for r in raw_top2} == {p2.frame_no, p3.frame_no}


def generate_orders(ranked: list[Rider], popular_line: int):
    if len(ranked) < 3:
        return [], "insufficient"
    a, b, c = ranked[:3]
    ab = indistinguishable(a, b, popular_line)
    bc = indistinguishable(b, c, popular_line)
    cd = len(ranked) >= 4 and indistinguishable(ranked[2], ranked[3], popular_line)
    if int(ab) + int(bc) + int(cd) > 1:
        return [], "multiple_ambiguities"
    base = (a.frame_no, b.frame_no, c.frame_no)
    if ab:
        return [base, (b.frame_no, a.frame_no, c.frame_no)], "order_ab"
    if bc:
        return [base, (a.frame_no, c.frame_no, b.frame_no)], "order_bc"
    if cd:
        if len(ranked) >= 5 and indistinguishable(ranked[3], ranked[4], popular_line):
            return [], "third_slot_too_wide"
        d = ranked[3]
        return [base, (a.frame_no, b.frame_no, d.frame_no)], "third_slot"
    return [base], "fixed"


def decide(race_id: str, pre: pd.DataFrame, ctx_row: pd.Series, og: pd.DataFrame) -> Decision:
    lines = parse_true_line(ctx_row.get("true_line"))
    if not lines:
        return Decision(race_id, "SKIP", "line_unresolved")
    riders = make_riders(pre, lines)
    if len(riders) < 4:
        return Decision(race_id, "SKIP", "score_or_entry_missing")
    tri = odds_map(og)
    expected = len(riders) * (len(riders) - 1) * (len(riders) - 2)
    if len(tri) != expected:
        return Decision(race_id, "SKIP", "odds_board_incomplete")

    pop = detect_popular_group(riders, tri)
    if pop is None:
        return Decision(race_id, "SKIP", "popular_group_unresolved")
    pop_line, target, mass, is_solo = pop
    if is_solo:
        return Decision(race_id, "SKIP", "popular_group_is_solo", pop_line, target.frame_no, mass)

    outside = [r.race_score for r in riders if r.line_idx != pop_line]
    if not outside:
        return Decision(race_id, "SKIP", "no_rival_line", pop_line, target.frame_no, mass)
    if target.race_score >= max(outside) + SCORE_BOUNDARY:
        return Decision(race_id, "SKIP", "target_score_plus_3", pop_line, target.frame_no, mass)
    if popular_line_too_strong(riders, pop_line, target):
        return Decision(race_id, "SKIP", "popular_line_too_strong", pop_line, target.frame_no, mass)

    remaining = [r for r in riders if r.frame_no != target.frame_no]
    ranked, coherent = coherent_ranking(remaining, pop_line)
    if not coherent:
        return Decision(race_id, "SKIP", "ranking_cycle", pop_line, target.frame_no, mass)
    frames = [r.frame_no for r in ranked]
    orders, ambiguity = generate_orders(ranked, pop_line)
    if not orders:
        return Decision(race_id, "SKIP", "cannot_compress_to_two_bets", pop_line,
                        target.frame_no, mass, frames, ambiguity=ambiguity)

    eligible = [(o, tri[o]) for o in orders if o in tri and tri[o] >= TRIFECTA_MIN_ODDS]
    if ambiguity == "third_slot":
        chosen = sorted(eligible, key=lambda x: x[1], reverse=True)[:2]
    elif len(eligible) > 1:
        chosen = [max(eligible, key=lambda x: x[1])]
    else:
        chosen = eligible[:1]
    if not chosen:
        return Decision(race_id, "SKIP", "odds_too_low", pop_line, target.frame_no, mass,
                        frames, orders, ambiguity=ambiguity)
    return Decision(race_id, "BET", "eligible", pop_line, target.frame_no, mass,
                    frames, orders, chosen, ambiguity)


def actual_order(result_rows: pd.DataFrame) -> tuple[int, int, int] | None:
    vals = []
    for r in result_rows.itertuples(index=False):
        try:
            pos = int(str(r.rank).strip())
            fn = int(r.banum)
        except (TypeError, ValueError):
            continue
        if 1 <= pos <= 3:
            vals.append((pos, fn))
    vals.sort()
    if [p for p, _ in vals] != [1, 2, 3]:
        return None
    return tuple(fn for _, fn in vals)


def head_busted(result_rows: pd.DataFrame, target: int | None):
    if target is None:
        return None
    r = result_rows[pd.to_numeric(result_rows.banum, errors="coerce") == int(target)]
    if r.empty:
        return None
    raw = str(r.iloc[0].get("rank", "")).strip()
    try:
        p = int(raw)
        return p >= 4
    except ValueError:
        return True if raw else None


def load_month(month: str):
    base_path = DATA / f"{month}_keirin.csv"
    race_path = CTX / f"{month}_races.csv"
    odds_path = CTX / f"{month}_odds_3rentan.csv"
    for p in (base_path, race_path, odds_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    base = pd.read_csv(base_path, encoding="utf-8-sig", dtype={"race_id": str})
    ctx = pd.read_csv(race_path, encoding="utf-8-sig", dtype={"race_id": str})
    odds = pd.read_csv(odds_path, encoding="utf-8-sig", dtype={"race_id": str})
    base["race_id"] = base["race_id"].astype(str)
    ctx["race_id"] = ctx["race_id"].astype(str)
    odds["race_id"] = odds["race_id"].astype(str)
    return base, ctx, odds


def run(month: str) -> pd.DataFrame:
    base, ctx, odds = load_month(month)
    use = ctx.copy()
    if "context_quality" in use:
        use = use[use.context_quality.astype(str) == "full"]
    if "price_usable" in use:
        use = use[use.price_usable.astype(str).str.lower().isin({"true", "1"})]
    base_by = {str(k): g for k, g in base.groupby("race_id", sort=False)}
    odds_by = {str(k): g for k, g in odds.groupby("race_id", sort=False)}

    rows = []
    for cr in use.itertuples(index=False):
        rid = str(cr.race_id)
        pre_full = base_by.get(rid)
        og = odds_by.get(rid)
        if pre_full is None or og is None:
            continue
        pre = pre_full[["race_id", "banum", "race_score"]].copy()
        d = decide(rid, pre, pd.Series(cr._asdict()), og)

        act = actual_order(pre_full)
        aset = frozenset(act) if act else None
        bust = head_busted(pre_full, d.target)
        cand = d.candidate_orders or []
        cand_sets = {frozenset(o) for o in cand}
        bets = d.bets or []
        hit = next(((o, od) for o, od in bets if act == o), None)
        cost = len(bets) * STAKE_YEN
        pay = int(round(hit[1] * STAKE_YEN)) if hit else 0
        rows.append({
            "race_id": rid,
            "date": getattr(cr, "date", ""),
            "venue_slug": getattr(cr, "venue_slug", ""),
            "race_no": getattr(cr, "race_no", ""),
            "true_line": getattr(cr, "true_line", ""),
            "action": d.action,
            "reason": d.reason,
            "popular_line": d.popular_line,
            "target": d.target,
            "market_mass": d.market_mass,
            "ranking": "-".join(map(str, d.ranking or [])),
            "ambiguity": d.ambiguity or "",
            "candidate_orders": "|".join("-".join(map(str, o)) for o in cand),
            "bets": "|".join("-".join(map(str, o)) for o, _ in bets),
            "bet_odds": "|".join(f"{od:.2f}" for _, od in bets),
            "bet_count": len(bets),
            "head_bust": None if bust is None else int(bust),
            "top3_set_match": int(aset is not None and aset in cand_sets),
            "candidate_order_match": int(act is not None and act in cand),
            "trifecta_hit": int(hit is not None),
            "cost": cost,
            "pay": pay,
            "actual_order": "-".join(map(str, act or [])),
        })
    return pd.DataFrame(rows)


def pct(x, n):
    return 100.0 * x / n if n else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    bet = df[df.action == "BET"]
    resolved = df[df.target.notna()]
    cost = int(bet.cost.sum()) if len(bet) else 0
    pay = int(bet.pay.sum()) if len(bet) else 0
    hb = resolved.head_bust.dropna()
    bhb = bet.head_bust.dropna()
    return {
        "races": len(df),
        "bets_races": len(bet),
        "bet_rate_pct": pct(len(bet), len(df)),
        "popular_head_bust_pct": float(hb.mean() * 100) if len(hb) else None,
        "bet_head_bust_pct": float(bhb.mean() * 100) if len(bhb) else None,
        "top3_set_match_pct": float(bet.top3_set_match.mean() * 100) if len(bet) else None,
        "candidate_order_match_pct": float(bet.candidate_order_match.mean() * 100) if len(bet) else None,
        "trifecta_hit_races": int(bet.trifecta_hit.sum()) if len(bet) else 0,
        "trifecta_hit_pct": float(bet.trifecta_hit.mean() * 100) if len(bet) else None,
        "stake_yen": cost,
        "pay_yen": pay,
        "roi_pct": pct(pay, cost) if cost else None,
        "skip_reasons": Counter(df.loc[df.action == "SKIP", "reason"]).most_common(),
    }


def print_report(df: pd.DataFrame, s: dict):
    print("=" * 88)
    print("POPULAR LINE HEAD SKIP v0.1 / structural + trifecta test")
    print("=" * 88)
    print(f"races={s['races']:,} bet_races={s['bets_races']:,} bet_rate={s['bet_rate_pct']:.1f}%")
    print(f"popular_head_bust={s['popular_head_bust_pct']:.2f}%" if s['popular_head_bust_pct'] is not None else "popular_head_bust=n/a")
    print(f"BET head_bust={s['bet_head_bust_pct']:.2f}%" if s['bet_head_bust_pct'] is not None else "BET head_bust=n/a")
    print(f"BET top3_set_match={s['top3_set_match_pct']:.2f}%" if s['top3_set_match_pct'] is not None else "BET top3_set_match=n/a")
    print(f"BET candidate_order_match={s['candidate_order_match_pct']:.2f}%" if s['candidate_order_match_pct'] is not None else "BET candidate_order_match=n/a")
    if s['roi_pct'] is not None:
        print(f"trifecta hits={s['trifecta_hit_races']}/{s['bets_races']} ({s['trifecta_hit_pct']:.2f}%) stake={s['stake_yen']:,} pay={s['pay_yen']:,} ROI={s['roi_pct']:.1f}%")
    print("skip reasons:")
    for reason, n in s["skip_reasons"]:
        sub = df[(df.action == "SKIP") & (df.reason == reason)]
        hb = sub.head_bust.dropna()
        extra = f" head_bust={hb.mean()*100:.1f}%" if len(hb) else ""
        print(f"  {reason:<30} {n:>5}{extra}")
    print("\nNotes: trio ROI and subjective EV gate are intentionally not scored in v0.1.")
    print("Popular-line identification is market_mass_v0.1 and must be checked against human labels.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026_08")
    ap.add_argument("--out", default="keirin_data/strategy_context/popular_head_skip_v01_results.csv")
    ap.add_argument("--summary", default="keirin_data/strategy_context/popular_head_skip_v01_summary.json")
    a = ap.parse_args()
    df = run(a.month)
    s = summarize(df)
    print_report(df, s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    Path(a.summary).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
