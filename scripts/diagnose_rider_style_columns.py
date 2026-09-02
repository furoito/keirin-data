#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
p=Path('keirin_data/2025_01_keirin.csv')
d=pd.read_csv(p,encoding='utf-8-sig',nrows=50)
print('COLUMNS')
for c in d.columns: print(repr(c))
print('\nCANDIDATE STYLE COLUMNS')
for c in d.columns:
    lc=str(c).lower()
    if any(k in lc for k in ['style','type','tactic','leg','foot','kyaku','senko','nige','maku','jizai','脚','逃','捲']):
        print(c, d[c].dropna().astype(str).value_counts().head(20).to_dict())
print('\nFIRST ROWS')
print(d.head(10).to_string(index=False))
