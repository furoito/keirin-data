# -*- coding: utf-8 -*-
"""Backfill real keirin lines and full 3-ren-tan odds for strategy testing."""
from __future__ import annotations

import argparse, glob, itertools, random, re, sys, time
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA = Path("keirin_data")
OUT = DATA / "strategy_context"
KD = "https://keirin.kdreams.jp"
OP = "https://sp.oddspark.com/keirin/yosou"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"
SLUG = {"kouchi": "kochi", "kochi": "kochi"}


def get(session, url, retries=4):
    for i in range(retries):
        try:
            r = session.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 200 and r.text:
                return r.text
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** i + random.random())
    return None


def wait():
    time.sleep(random.uniform(.8, 1.8))


def banum(v, valid):
    vals = v if isinstance(v, tuple) else [v]
    for x in vals:
        m = re.fullmatch(r"([1-9])(?:\.0)?", str(x).strip())
        if m and int(m.group(1)) in valid:
            return int(m.group(1))
    return None


def parse_lines(html, valid):
    valid = set(valid)
    found = []
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        toks = [banum(c.get_text(" ", strip=True), valid) for c in tr.find_all(["td", "th"])]
        nums = [x for x in toks if x is not None]
        if len(nums) != len(valid) or set(nums) != valid or len(set(nums)) != len(nums):
            continue
        groups, cur = [], []
        for x in toks:
            if x is None:
                if cur: groups.append(cur); cur = []
            else:
                cur.append(x)
        if cur: groups.append(cur)
        if set(itertools.chain.from_iterable(groups)) == valid:
            found.append(groups)
    # A row containing all rider numbers with no separators can be a generic
    # race-card row rather than a line prediction. Fail closed unless at least
    # two distinct lines are visible.
    found = [groups for groups in found if len(groups) >= 2]
    if not found:
        return [], "not_found"
    found.sort(key=len, reverse=True)
    return found[0], "full"


def oddspark_url(venue, date, race_no):
    d = datetime.strptime(date, "%Y-%m-%d")
    slug = SLUG.get(venue, venue)
    fn = f"{d:%m%d}.html" if int(race_no) == 1 else f"{d:%m%d}_{int(race_no)}.html"
    return f"{OP}/{slug}/{d:%Y}/{fn}"


def decimal_odd(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "nan", "NaN"}: return None
    m = re.search(r"(?<!\d)(\d{1,5}(?:\.\d+)?)(?!\d)", s)
    return float(m.group(1)) if m else None


def matrix_info(df, valid):
    valid = set(valid)
    cmap = {}
    for col in df.columns:
        b = banum(col, valid)
        if b is not None: cmap.setdefault(b, col)
    if len(cmap) != len(valid)-1:
        for ri in range(min(3, len(df))):
            trial = {}
            for col in df.columns:
                b = banum(df.iloc[ri][col], valid)
                if b is not None: trial.setdefault(b, col)
            if len(trial) == len(valid)-1:
                cmap = trial; break
    if len(cmap) != len(valid)-1: return None
    cset = set(cmap)
    missing = valid-cset
    if len(missing) != 1: return None
    first = next(iter(missing))
    labels = {}
    for col in df.columns:
        x = {}
        for ri, v in enumerate(df[col].tolist()):
            b = banum(v, valid)
            if b is not None: x.setdefault(b, ri)
        if set(x) == cset:
            labels = x; break
    if not labels:
        x = {}
        for ri, v in enumerate(df.index.tolist()):
            b = banum(v, valid)
            if b is not None: x.setdefault(b, ri)
        if set(x) == cset: labels = x
    return (first, labels, cmap) if labels else None


def parse_odds(html, valid):
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    snap = close = None
    m = re.search(r"(20\d{2})/(\d{2})/(\d{2})\s+(\d{1,2}:\d{2})現在", text)
    if m: snap = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}"
    m = re.search(r"投票締切\s*(\d{1,2}:\d{2})", text)
    if m: close = m.group(1)
    try: tables = pd.read_html(StringIO(html))
    except Exception: return pd.DataFrame(), "read_html_failed", snap, close
    rows, seen = [], set()
    for df in tables:
        if df.empty: continue
        info = matrix_info(df, valid)
        if not info: continue
        first, labels, cmap = info
        for third, ri in labels.items():
            for second, col in cmap.items():
                if len({first, second, third}) < 3: continue
                odd = decimal_odd(df.iloc[ri][col])
                if odd is None: continue
                key = (first, second, third)
                if key in seen: continue
                seen.add(key)
                rows.append({"b1": first, "b2": second, "b3": third, "odds_decimal": odd})
    out = pd.DataFrame(rows)
    exp = len(valid)*(len(valid)-1)*(len(valid)-2)
    q = "full" if len(out)==exp else ("partial" if len(out) else "not_found")
    return out, q, snap, close


def load_data():
    fps = [x for x in sorted(glob.glob(str(DATA / "20??_??_keirin.csv"))) if "bak" not in x and "sample" not in x]
    if not fps: raise SystemExit("No monthly keirin CSV files found")
    frames=[]
    for fp in fps:
        try: frames.append(pd.read_csv(fp, encoding="utf-8-sig", dtype={"race_id":str}))
        except Exception as e: print(f"WARN skip {fp}: {e}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True); df["race_id"] = df["race_id"].astype(str)
    return df


def eligible(df, start, end, classes, debug):
    x = df[df.race_id==str(debug)].copy() if debug else df.copy()
    if not debug:
        if start: x=x[x.date.astype(str)>=start]
        if end: x=x[x.date.astype(str)<=end]
    ok=x.groupby("race_id").player_class.apply(lambda s: bool(len(s)) and set(s.dropna().astype(str)).issubset(classes))
    x=x[x.race_id.isin(ok[ok].index)]
    return x.groupby("race_id", as_index=False).agg(
        venue_slug=("venue_slug","first"), date=("date","first"), race_no=("race_no","first"),
        classes=("player_class",lambda s:",".join(sorted(set(s.dropna().astype(str)))))
    ).sort_values(["date","venue_slug","race_no"])


def paths(date):
    ym=str(date)[:7].replace("-","_")
    return OUT/f"{ym}_races.csv", OUT/f"{ym}_odds_3rentan.csv"


def read_csv(path):
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id":str})
    except Exception: return pd.DataFrame()


def upsert(path, new, keys):
    if new is None or new.empty: return
    path.parent.mkdir(parents=True, exist_ok=True)
    old=read_csv(path)
    out=new if old.empty else pd.concat([old,new],ignore_index=True).drop_duplicates(keys,keep="last")
    out.to_csv(path,index=False,encoding="utf-8-sig")


def price_quality(kind, snap, close, date):
    if kind=="confirmed": return "confirmed"
    if not snap: return "unknown"
    if not close: return "snapshot"
    try:
        s=datetime.strptime(snap,"%Y-%m-%d %H:%M")
        c=datetime.strptime(f"{date} {close}","%Y-%m-%d %H:%M")
        mins=(c-s).total_seconds()/60
        if 0<=mins<=10: return "preclose_10m"
        if mins<0: return "after_close_snapshot"
        return "early_snapshot"
    except Exception: return "snapshot"


def process(session, race, base):
    rid=str(race.race_id); venue=str(race.venue_slug); date=str(race.date); rno=int(race.race_no)
    g=base[base.race_id==rid]
    valid=sorted(pd.to_numeric(g.banum,errors="coerce").dropna().astype(int).unique())
    exp=len(valid)*(len(valid)-1)*(len(valid)-2)

    lurl=oddspark_url(venue,date,rno); lhtml=get(session,lurl); lines=[]; lq="fetch_failed"
    if lhtml: lines,lq=parse_lines(lhtml,valid)
    wait()

    best=pd.DataFrame(); oq="fetch_failed"; snap=close=None; ourl=""; kind=""
    urls=[f"{KD}/{venue}/racedetail/{rid}/?kakeshikiType=3rentan&pageType=odds",
          f"{KD}/{venue}/racedetail/{rid}/?pageType=showResult"]
    for url in urls:
        html=get(session,url)
        if not html: continue
        p,q,s,c=parse_odds(html,valid)
        if len(p)>len(best):
            best,oq,snap,close,ourl=p,q,s,c,url
            txt=BeautifulSoup(html,"html.parser").get_text(" ",strip=True)
            kind="confirmed" if "確定オッズ" in txt else ("snapshot" if s else "unknown")
        if q=="full": break
        wait()
    wait()

    if not best.empty:
        best=best.copy(); best.insert(0,"race_id",rid); best["date"]=date; best["venue_slug"]=venue; best["race_no"]=rno
        best["snapshot_at"]=snap or ""; best["snapshot_kind"]=kind; best["source_url"]=ourl
        best=best[["race_id","date","venue_slug","race_no","b1","b2","b3","odds_decimal","snapshot_at","snapshot_kind","source_url"]]

    flat=[b for line in lines for b in line]
    line_full=len(flat)==len(valid) and set(flat)==set(valid) and len(set(flat))==len(flat)
    odds_full=len(best)==exp
    pq=price_quality(kind,snap,close,date)
    usable=odds_full and pq in {"confirmed","preclose_10m","after_close_snapshot"}
    meta={
        "race_id":rid,"date":date,"venue_slug":venue,"race_no":rno,"classes":race.classes,"n_players":len(valid),
        "expected_3rentan":exp,"true_line":"/".join("-".join(map(str,x)) for x in lines),"n_lines":len(lines),
        "line_quality":"full" if line_full else lq,"line_source_url":lurl,
        "parsed_3rentan":len(best),"odds_quality":"full" if odds_full else oq,"odds_snapshot_at":snap or "",
        "odds_snapshot_kind":kind,"bet_close_time":close or "","price_quality":pq,"price_usable":bool(usable),
        "odds_source_url":ourl,"context_quality":"full" if line_full and usable else "partial",
        "enriched_at":datetime.now().isoformat(timespec="seconds")}
    return meta,best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start-date"); ap.add_argument("--end-date"); ap.add_argument("--classes",default="A1,A2")
    ap.add_argument("--limit",type=int); ap.add_argument("--resume",action="store_true"); ap.add_argument("--debug-race")
    a=ap.parse_args(); classes={x.strip() for x in a.classes.split(",") if x.strip()}
    base=load_data(); races=eligible(base,a.start_date,a.end_date,classes,a.debug_race)
    if a.resume and not races.empty:
        keep=[]
        for _,r in races.iterrows():
            fp,_=paths(r.date); old=read_csv(fp)
            done=set(old.loc[old.context_quality=="full","race_id"].astype(str)) if not old.empty and "context_quality" in old else set()
            if str(r.race_id) not in done: keep.append(r)
        races=pd.DataFrame(keep,columns=races.columns)
    if a.limit: races=races.head(a.limit)
    print(f"eligible={len(races)} classes={sorted(classes)}")
    s=requests.Session(); full=partial=0
    for i,r in races.reset_index(drop=True).iterrows():
        try:
            meta,odds=process(s,r,base); rfp,ofp=paths(r.date); upsert(rfp,pd.DataFrame([meta]),["race_id"])
            if not odds.empty: upsert(ofp,odds,["race_id","b1","b2","b3"])
            full += meta["context_quality"]=="full"; partial += meta["context_quality"]!="full"
            print(f"[{i+1}/{len(races)}] {r.date} {r.venue_slug} {int(r.race_no)}R line={meta['true_line'] or '-'} {meta['line_quality']} odds={meta['parsed_3rentan']}/{meta['expected_3rentan']} {meta['price_quality']} => {meta['context_quality']}")
        except KeyboardInterrupt: return 130
        except Exception as e: partial+=1; print(f"ERROR race_id={r.race_id}: {type(e).__name__}: {e}",file=sys.stderr)
    print(f"done full={full} partial={partial} out={OUT}")
    return 0

if __name__=="__main__": raise SystemExit(main())
