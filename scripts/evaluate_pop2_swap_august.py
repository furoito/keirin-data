#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'keirin_data'/'strategy_context'
BASE=ROOT/'keirin_data'/'2026_08_keirin.csv'
IN=CTX/'reconstruction_abc_v4_full50.csv'
OUT=CTX/'pop2_swap_august_diagnostic.csv'
SUM=CTX/'pop2_swap_august_summary.json'

def parse_line(s):
    out={}
    for li,g in enumerate(str(s).split('/'),1):
        xs=[int(x) for x in g.split('-')]
        for pos,x in enumerate(xs,1): out[x]=(li,pos,len(xs))
    return out

def parse_fam(s):
    fam=[]
    for p in str(s).split('|'):
        try: fs=frozenset(int(x) for x in p.split('-'))
        except: continue
        if len(fs)==3 and fs not in fam:fam.append(fs)
    return fam

def role(fn,lmap,popline):
    li,pos,sz=lmap[fn]
    if li==popline:
        return 'popular_pos2' if pos==2 else ('popular_pos3plus' if pos>=3 else 'popular_head')
    if sz==1:return 'solo'
    if pos==1:return 'other_head'
    return 'other_pos2plus'

def correct(fam,lmap,scores,popline,margin):
    pop2=next((fn for fn,(li,pos,sz) in lmap.items() if li==popline and pos==2),None)
    if pop2 is None:return fam
    out=[]
    for s in fam:
        ns=set(s)
        if pop2 in ns:
            heads=[fn for fn in lmap if role(fn,lmap,popline)=='other_head' and fn not in ns]
            if heads:
                alt=max(heads,key=lambda fn:(scores.get(fn,-999),-fn))
                if scores.get(alt,-999)>=scores.get(pop2,-999)-margin:
                    ns.remove(pop2);ns.add(alt)
        fs=frozenset(ns)
        if fs not in out:out.append(fs)
    return out[:2]
def overlap(fam,actual):return max((len(set(x)&set(actual)) for x in fam),default=0)
def main():
    d=pd.read_csv(IN,encoding='utf-8-sig',dtype={'race_id':str})
    b=pd.read_csv(BASE,encoding='utf-8-sig',dtype={'race_id':str})
    sb={k:{int(float(r.banum)):float(r.race_score) for r in g.itertuples()} for k,g in b.groupby('race_id')}
    rows=[]
    for r in d.itertuples(index=False):
        if int(r.head_bust)!=1:continue
        lmap=parse_line(r.line)
        target=int(r.target)
        popline=lmap[target][0]
        actual=frozenset(int(x) for x in str(r.actual).split('-'))
        A=parse_fam(r.A_candidates)
        rec={'race_id':r.race_id,'date':r.date,'actual':r.actual,'A_family':r.A_candidates}
        for name,m in [('A',None),('S0',0.0),('S1',1.0),('S3',3.0)]:
            fam=A if m is None else correct(A,lmap,sb.get(r.race_id,{}),popline,m)
            rec[name+'_family']='|'.join('-'.join(map(str,sorted(x))) for x in fam)
            rec[name+'_exact']=int(actual in fam)
            rec[name+'_overlap']=overlap(fam,actual)
            rec[name+'_changed']=int(set(fam)!=set(A)) if m is not None else 0
        rows.append(rec)
    o=pd.DataFrame(rows);o.to_csv(OUT,index=False,encoding='utf-8-sig')
    s={'n':len(o)}
    for name in ['A','S0','S1','S3']:
        s[name]={'exact_n':int(o[name+'_exact'].sum()),'exact_pct':float(o[name+'_exact'].mean()*100),'avg_overlap':float(o[name+'_overlap'].mean()),'overlap_2plus_n':int((o[name+'_overlap']>=2).sum()),'changed_n':int(o[name+'_changed'].sum()) if name!='A' else 0}
    SUM.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
