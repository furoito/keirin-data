#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.1b: implementation-alignment patch for popular-line detection only.

The canonical rule is "identify the most market-supported LINE from odds + line
composition". v0.1 accidentally scored solo riders and only leader-first tickets,
which measured a different object. This patch changes only that identification:

- solo groups are not eligible to be the popular LINE;
- for every line with >=2 riders, sum 1/odds over every trifecta outcome whose
  top-3 contains at least two riders from that line;
- the highest such joint market mass is the popular line; its pos1 rider is target.

No result fields are used. All downstream 3-point/position/compression/odds rules
remain exactly v0.1.
"""
from __future__ import annotations

import popular_head_skip_v01 as base


def detect_popular_line_joint_mass(riders, tri):
    best = None
    for li, members in base.line_members(riders).items():
        if len(members) < 2:
            continue
        member_frames = {r.frame_no for r in members}
        mass = 0.0
        for combo, od in tri.items():
            if od <= 0:
                continue
            if len(member_frames.intersection(combo)) >= 2:
                mass += 1.0 / od
        if mass <= 0:
            continue
        leader = members[0]
        item = (li, leader, mass, False)
        if best is None or mass > best[2]:
            best = item
    return best


base.detect_popular_group = detect_popular_line_joint_mass

if __name__ == "__main__":
    base.main()
