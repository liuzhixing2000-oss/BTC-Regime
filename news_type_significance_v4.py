from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import normalize
from news_type_research_v3 import fetch_news, prepare_events, bh_fdr


HORIZONS = [1, 4, 8, 24]


def median_bootstrap_p(control: np.ndarray, observed: float, sample_n: int, rng, reps=4000) -> float:
    """Null distribution of a sample median with the same N as the news category."""
    if sample_n < 1 or not np.isfinite(observed): return np.nan
    exceed=0; done=0
    while done < reps:
        batch=min(200,reps-done)
        draws=rng.choice(control,size=(batch,sample_n),replace=True)
        exceed += int((np.median(draws,axis=1) >= observed).sum()); done += batch
    return (exceed+1)/(reps+1)


def exact_direction_p(wins: int,n: int,rng,reps=20000) -> float:
    if n<1: return np.nan
    sims=rng.binomial(n,.5,reps)
    return (int((sims>=wins).sum())+1)/(reps+1)


def corrected_tests(events: pd.DataFrame, prices: pd.DataFrame, seed=74) -> pd.DataFrame:
    rng=np.random.default_rng(seed); split=events.timestamp.min()+(events.timestamp.max()-events.timestamp.min())/2
    valid=np.arange(96,len(prices)-96); positions=prices.index.searchsorted(events.timestamp)
    blocked=set(j for p in positions for j in range(max(0,p-16),min(len(prices),p+17)))
    controls=np.array([j for j in valid if j not in blocked])
    control_abs={h:np.abs(prices.close.iloc[controls+h*4].to_numpy()/prices.close.iloc[controls].to_numpy()-1) for h in HORIZONS}
    rows=[]
    for cat,g in events.groupby("category"):
        train=g.timestamp<=split; row={"category":cat,"events":len(g),"train_events":int(train.sum()),"test_events":int((~train).sum())}
        d=g.loc[g.semantic_sign.ne(0)].copy(); dtrain=d.timestamp<=split
        row["directional_events"]=len(d); row["direction_train_events"]=int(dtrain.sum()); row["direction_test_events"]=int((~dtrain).sum())
        for h in HORIZONS:
            a=g[f"abs_return_{h}h"].dropna(); base=control_abs[h]; observed=a.median()
            row[f"vol_uplift_{h}h"]=observed/np.median(base)-1
            row[f"train_uplift_{h}h"]=g.loc[train,f"abs_return_{h}h"].median()/np.median(base)-1
            row[f"test_uplift_{h}h"]=g.loc[~train,f"abs_return_{h}h"].median()/np.median(base)-1
            row[f"vol_p_{h}h"]=median_bootstrap_p(base,observed,len(a),rng)
            z=d[f"aligned_return_{h}h"].dropna(); ztrain=d.loc[dtrain,f"aligned_return_{h}h"].dropna(); ztest=d.loc[~dtrain,f"aligned_return_{h}h"].dropna()
            row[f"direction_hit_{h}h"]=(z>0).mean() if len(z) else np.nan
            row[f"direction_mean_{h}h"]=z.mean() if len(z) else np.nan
            row[f"direction_train_hit_{h}h"]=(ztrain>0).mean() if len(ztrain) else np.nan
            row[f"direction_test_hit_{h}h"]=(ztest>0).mean() if len(ztest) else np.nan
            row[f"direction_p_{h}h"]=exact_direction_p(int((z>0).sum()),len(z),rng)
        rows.append(row)
    r=pd.DataFrame(rows)
    for h in HORIZONS:
        r[f"vol_q_{h}h"]=bh_fdr(r[f"vol_p_{h}h"])
        r[f"direction_q_{h}h"]=bh_fdr(r[f"direction_p_{h}h"])
        r[f"vol_pass_{h}h"]=(r.events>=40)&(r.train_events>=15)&(r.test_events>=15)&(r[f"vol_uplift_{h}h"]>=.10)&(r[f"train_uplift_{h}h"]>0)&(r[f"test_uplift_{h}h"]>0)&(r[f"vol_q_{h}h"]<.10)
        r[f"direction_pass_{h}h"]=(r.directional_events>=40)&(r.direction_train_events>=15)&(r.direction_test_events>=15)&(r[f"direction_hit_{h}h"]>=.55)&(r[f"direction_train_hit_{h}h"]>.50)&(r[f"direction_test_hit_{h}h"]>.50)&(r[f"direction_mean_{h}h"]>0)&(r[f"direction_q_{h}h"]<.10)
    r["volatility_pass"]=r[[f"vol_pass_{h}h" for h in HORIZONS]].any(axis=1)
    r["direction_pass"]=r[[f"direction_pass_{h}h" for h in HORIZONS]].any(axis=1)
    return r.sort_values(["volatility_pass","direction_pass","events"],ascending=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/BTCUSDT_15m.csv"); ap.add_argument("--output",default="output_news_types_v4"); args=ap.parse_args()
    key=os.environ.get("ALPHA_VANTAGE_API_KEY","").strip()
    if not key: raise SystemExit("Missing ALPHA_VANTAGE_API_KEY")
    prices=normalize(pd.read_csv(args.input)); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    news=fetch_news(prices.index.min(),prices.index.max(),key,out/"alpha_news_detailed.csv")
    events=prepare_events(news,prices); stats=corrected_tests(events,prices)
    keepcols=["category","events","directional_events","volatility_pass","direction_pass"]
    for h in HORIZONS: keepcols += [f"vol_uplift_{h}h",f"vol_q_{h}h",f"vol_pass_{h}h",f"direction_hit_{h}h",f"direction_q_{h}h",f"direction_pass_{h}h"]
    whitelist=stats.loc[stats.volatility_pass|stats.direction_pass,keepcols]
    stats.to_csv(out/"corrected_category_reliability.csv",index=False); whitelist.to_csv(out/"corrected_news_whitelist.csv",index=False)
    text="\n".join(["BTC NEWS SIGNIFICANCE V4","="*40,f"Events tested: {len(events)}",f"Whitelisted categories: {len(whitelist)}","",
                     "CORRECTED RELIABILITY",stats[keepcols].to_string(index=False),"","CORRECTED WHITELIST",whitelist.to_string(index=False) if len(whitelist) else "EMPTY — no category passed corrected rules."])
    (out/"validation_report.txt").write_text(text,encoding="utf-8"); print(text)


if __name__=="__main__": main()
