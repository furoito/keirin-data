#!/usr/bin/env python3
from io import StringIO
import re
import pandas as pd, requests
from bs4 import BeautifulSoup
URL='https://keirin.kdreams.jp/hiratsuka/racedetail/3520240101030001/?kakeshikiType=3rentan&pageType=odds'
h=requests.get(URL,headers={'User-Agent':'Mozilla/5.0'},timeout=20).text
s=BeautifulSoup(h,'html.parser')
print('links')
for a in s.find_all('a',href=True):
    href=a['href']; txt=a.get_text(' ',strip=True)
    if 'kakeshikiType' in href or '3連複' in txt:
        print(txt, href)
print('tables')
for i,df in enumerate(pd.read_html(StringIO(h))):
    print('TABLE',i,'shape',df.shape,'cols',list(map(str,df.columns))[:10])
    print(df.head(8).to_string(index=False)[:3000])
    print('---')
