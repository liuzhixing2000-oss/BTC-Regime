from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import normalize
from free_event_reliability_v2 import fetch_json, alpha_time


CATEGORIES = {
    "ETF_FLOWS": [r"\betf\b", r"exchange.traded fund", r"blackrock", r"fidelity", r"grayscale", r"ibit\b"],
    "REGULATION_LEGAL": [r"regulat", r"\bsec\b", r"cftc", r"lawsuit", r"court", r"legislation", r"crypto bill", r"ban\b", r"crackdown"],
    "SECURITY_HACK": [r"hack", r"exploit", r"breach", r"stolen", r"cyber", r"vulnerab"],
    "EXCHANGE_CREDIT": [r"exchange", r"binance", r"coinbase", r"kraken", r"bybit", r"bankrupt", r"insolven", r"withdrawal", r"liquidity crisis"],
    "INSTITUTIONAL_SOVEREIGN": [r"institution", r"treasury", r"sovereign", r"government.*bitcoin", r"company.*bitcoin", r"strategy.*bitcoin", r"reserve.*bitcoin"],
    "STABLECOIN": [r"stablecoin", r"\busdt\b", r"tether", r"\busdc\b", r"depeg"],
    "FED_MACRO": [r"federal reserve", r"\bfed\b", r"interest rate", r"inflation", r"\bcpi\b", r"payroll", r"jobs report", r"fomc", r"powell"],
    "GEOPOLITICAL": [r"war\b", r"missile", r"invasion", r"sanction", r"geopolit", r"conflict", r"tariff"],
    "NETWORK_PROTOCOL": [r"halving", r"bitcoin network", r"upgrade", r"fork\b", r"lightning network", r"protocol"],
    "MINING": [r"bitcoin min", r"miner", r"hashrate", r"hash rate"],
}

POSITIVE = [r"approv", r"inflow", r"buy|buys|bought", r"purchase", r"adopt", r"launch", r"legaliz", r"reserve", r"record demand", r"rate cut", r"easing", r"dismiss.*lawsuit"]
NEGATIVE = [r"reject", r"outflow", r"sell|sells|sold", r"ban\b", r"crackdown", r"lawsuit", r"hack", r"exploit", r"stolen", r"bankrupt", r"insolven", r"depeg", r"liquidat", r"rate hike", r"hawkish", r"war\b", r"attack"]
REACTIVE = [r"bitcoin (?:price )?(?:rises|falls|jumps|drops|surges|slides|rallies|plunges)", r"why bitcoin", r"price prediction", r"technical analysis", r"bitcoin today"]


def classify(text: str) -> str:
    t=text.lower()
    scores={k:sum(bool(re.search(p,t)) for p in pats) for k,pats in CATEGORIES.items()}
    best=max(scores,key=scores.get)
    return best if scores[best] else "OTHER"


def semantic_sign(text: str) -> int:
    t=text.lower(); pos=sum(bool(re.search(p,t)) for p in POSITIVE); neg=sum(bool(re.search(p,t)) for p in NEGATIVE)
    return 1 if pos>neg else (-1 if neg>pos else 0)


def fetch_news(start: pd.Timestamp,end: pd.Timestamp,key: str,cache: Path) -> pd.DataFrame:
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        x=pd.read_csv(cache); x["timestamp"]=pd.to_datetime(x.timestamp,utc=True); return x
    rows=[]; cursor=start.floor("D")
    while cursor<end:
        stop=min(cursor+pd.Timedelta(days=31),end)
        params={"function":"NEWS_SENTIMENT","tickers":"CRYPTO:BTC","time_from":alpha_time(cursor),
                "time_to":alpha_time(stop),"sort":"EARLIEST","limit":1000,"apikey":key}
        payload=fetch_json("https://www.alphavantage.co/query?"+urllib.parse.urlencode(params))
        if any(k in payload for k in ["Information","Note","Error Message"]):
            raise RuntimeError(payload.get("Information") or payload.get("Note") or payload.get("Error Message"))
        for item in payload.get("feed",[]):
            try: ts=pd.to_datetime(item.get("time_published"),format="%Y%m%dT%H%M%S",utc=True)
            except Exception: continue
            tick=[z for z in item.get("ticker_sentiment",[]) if z.get("ticker")=="CRYPTO:BTC"]
            rel=max([float(z.get("relevance_score",0)) for z in tick] or [0.0])
            rows.append({"timestamp":ts,"title":item.get("title","").strip(),"summary":item.get("summary","").strip(),
                         "url":item.get("url",""),"source":item.get("source",""),"relevance":rel})
        cursor=stop; time.sleep(1.05)
    x=pd.DataFrame(rows).drop_duplicates("url").sort_values("timestamp")
    x.to_csv(cache,index=False); return x


def prepare_events(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    x=news.loc[news.relevance.ge(.35)].copy(); text=(x.title.fillna("")+" "+x.summary.fillna(""))
    x["category"]=[classify(t) for t in text]; x["semantic_sign"]=[semantic_sign(t) for t in text]
    x["reactive_title"]=[any(re.search(p,t.lower()) for p in REACTIVE) for t in text]
    # Collapse repeated coverage of the same category into one 2-hour event cluster.
    x["cluster_time"]=x.timestamp.dt.floor("2h")
    x=x.sort_values("relevance",ascending=False).drop_duplicates(["category","cluster_time"]).sort_values("timestamp")
    rows=[]
    for _,e in x.iterrows():
        pos=prices.index.searchsorted(e.timestamp)
        if pos<4 or pos+96>=len(prices): continue
        p=prices.close.iloc[pos]; pre=prices.close.iloc[pos-4]/p-1
        hist=prices.close.pct_change(4).abs().iloc[max(0,pos-672):pos]
        reactive_move=abs(pre)>hist.quantile(.80) if len(hist.dropna())>=100 else False
        row=e.to_dict(); row["pre_return_1h"]=-pre; row["reactive_move"]=bool(reactive_move)
        for h in [1,4,8,24]:
            q=pos+h*4; ret=prices.close.iloc[q]/p-1
            row[f"return_{h}h"]=ret; row[f"abs_return_{h}h"]=abs(ret)
            row[f"aligned_return_{h}h"]=e.semantic_sign*ret if e.semantic_sign else np.nan
        rows.append(row)
    out=pd.DataFrame(rows)
    return out.loc[~out.reactive_title & ~out.reactive_move].copy()


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p=pvalues.fillna(1).to_numpy(); order=np.argsort(p); ranked=p[order]; n=len(p)
    q=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
    out=np.empty(n); out[order]=q; return pd.Series(out,index=pvalues.index)


def category_tests(events: pd.DataFrame,prices: pd.DataFrame,seed=73):
    rng=np.random.default_rng(seed); split=events.timestamp.min()+(events.timestamp.max()-events.timestamp.min())/2
    # Random non-event 15m bars, excluding first/last day.
    valid=np.arange(96,len(prices)-96); event_pos=prices.index.searchsorted(events.timestamp)
    blocked=set(j for p in event_pos for j in range(max(0,p-16),min(len(prices),p+17)))
    controls=np.array([j for j in valid if j not in blocked]); control_abs={}
    for h in [1,4,8,24]:
        pick=rng.choice(controls,min(10000,len(controls)),replace=False)
        control_abs[h]=np.abs(prices.close.iloc[pick+h*4].to_numpy()/prices.close.iloc[pick].to_numpy()-1)
    rows=[]
    for cat,g in events.groupby("category"):
        row={"category":cat,"events":len(g),"train_events":int((g.timestamp<=split).sum()),"test_events":int((g.timestamp>split).sum())}
        for h in [1,4,8,24]:
            a=g[f"abs_return_{h}h"].dropna().to_numpy(); base=control_abs[h]
            row[f"abs_median_{h}h"]=np.median(a); row[f"baseline_abs_median_{h}h"]=np.median(base)
            row[f"vol_uplift_{h}h"]=np.median(a)/np.median(base)-1 if len(a) else np.nan
            row[f"vol_p_{h}h"]=(np.sum(base>=np.median(a))+1)/(len(base)+1) if len(a) else np.nan
            row[f"train_uplift_{h}h"]=g.loc[g.timestamp<=split,f"abs_return_{h}h"].median()/np.median(base)-1
            row[f"test_uplift_{h}h"]=g.loc[g.timestamp>split,f"abs_return_{h}h"].median()/np.median(base)-1
        d=g.loc[g.semantic_sign.ne(0)]
        row["directional_events"]=len(d)
        for h in [1,4,8,24]:
            z=d[f"aligned_return_{h}h"].dropna(); row[f"direction_hit_{h}h"]=(z>0).mean() if len(z) else np.nan
            row[f"direction_mean_{h}h"]=z.mean() if len(z) else np.nan
            # Sign permutation: under null, hit rate is 0.5.
            if len(z):
                wins=int((z>0).sum()); sims=rng.binomial(len(z),.5,5000); row[f"direction_p_{h}h"]=(np.sum(sims>=wins)+1)/5001
            else: row[f"direction_p_{h}h"]=np.nan
        rows.append(row)
    result=pd.DataFrame(rows)
    result["vol_q_4h"]=bh_fdr(result.vol_p_4h); result["direction_q_4h"]=bh_fdr(result.direction_p_4h)
    result["volatility_pass"]=(result.events>=40)&(result.train_events>=15)&(result.test_events>=15)&(result.vol_uplift_4h>=.10)&(result.train_uplift_4h>0)&(result.test_uplift_4h>0)&(result.vol_q_4h<.10)
    result["direction_pass"]=(result.directional_events>=40)&(result.direction_hit_4h>=.55)&(result.direction_mean_4h>0)&(result.direction_q_4h<.10)
    return result.sort_values(["volatility_pass","direction_pass","events"],ascending=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/BTCUSDT_15m.csv"); ap.add_argument("--output",default="output_news_types_v3"); args=ap.parse_args()
    key=os.environ.get("ALPHA_VANTAGE_API_KEY","").strip()
    if not key: raise SystemExit("Missing ALPHA_VANTAGE_API_KEY")
    prices=normalize(pd.read_csv(args.input)); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    news=fetch_news(prices.index.min(),prices.index.max(),key,out/"alpha_news_detailed.csv")
    events=prepare_events(news,prices); stats=category_tests(events,prices)
    whitelist=stats.loc[stats.volatility_pass|stats.direction_pass,["category","events","volatility_pass","direction_pass","vol_uplift_4h","direction_hit_4h","vol_q_4h","direction_q_4h"]]
    events.to_csv(out/"classified_deduplicated_events.csv",index=False); stats.to_csv(out/"news_category_reliability.csv",index=False); whitelist.to_csv(out/"news_whitelist.csv",index=False)
    report=["BTC NEWS TYPE RESEARCH V3","="*42,f"Raw news: {len(news)}",f"Non-reactive deduplicated events: {len(events)}",f"Whitelisted categories: {len(whitelist)}","",
            "CATEGORY RELIABILITY",stats.to_string(index=False),"","WHITELIST",whitelist.to_string(index=False) if len(whitelist) else "EMPTY — no category passed reliability rules.","",
            "Passing rules: >=40 events, both halves represented, >=10% 4h volatility uplift with FDR q<0.10; direction requires >=55% hit and FDR q<0.10."]
    text="\n".join(report); (out/"validation_report.txt").write_text(text,encoding="utf-8"); print(text)


if __name__=="__main__": main()
