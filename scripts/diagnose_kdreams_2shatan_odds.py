#!/usr/bin/env python3
# trigger workflow after workflow file creation
from __future__ import annotations

from io import StringIO
import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = 'https://keirin.kdreams.jp/hiratsuka/racedetail/3520260728030010/?pageType=result'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36'
VALID = set(range(1, 8))


def num(v):
    s = str(v).strip()
    m = re.fullmatch(r'([1-9])(?:\.0)?', s)
    if not m:
        return None
    x = int(m.group(1))
    return x if x in VALID else None


def odd(v):
    s = str(v).strip().replace(',', '')
    if s in {'', '-', '--', 'nan', 'NaN'}:
        return None
    m = re.fullmatch(r'(\d{1,5}(?:\.\d+)?)', s)
    if not m:
        return None
    x = float(m.group(1))
    return x if x > 0 else None


def inspect_table(df, idx):
    out = {
        'index': idx,
        'shape': list(df.shape),
        'columns': [str(x) for x in df.columns],
        'head': df.head(10).astype(str).values.tolist(),
    }
    colmap = {}
    for c in df.columns:
        vals = c if isinstance(c, tuple) else (c,)
        for v in vals:
            x = num(v)
            if x is not None:
                colmap.setdefault(x, c)
    rowmap = {}
    for ri in range(len(df)):
        for c in df.columns:
            x = num(df.iloc[ri][c])
            if x is not None:
                rowmap.setdefault(x, ri)
    pairs = []
    if set(colmap) == VALID and set(rowmap) == VALID:
        for b1 in sorted(VALID):
            for b2 in sorted(VALID):
                if b1 == b2:
                    continue
                val = odd(df.iloc[rowmap[b2]][colmap[b1]])
                if val is not None:
                    pairs.append((b1, b2, val))
    out['candidate_pair_count'] = len(pairs)
    out['candidate_pairs_head'] = pairs[:12]
    return out


def main():
    r = requests.get(URL, headers={'User-Agent': UA}, timeout=30)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(' ', strip=True)
    print('URL', URL)
    print('contains_2shatan', '2車単' in text)
    tables = pd.read_html(StringIO(html))
    print('table_count', len(tables))
    payload = [inspect_table(df, i) for i, df in enumerate(tables)]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
