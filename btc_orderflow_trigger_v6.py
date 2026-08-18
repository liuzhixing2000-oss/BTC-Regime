from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import add_indicators, build_scores, load_config, normalize


def get_json(url,params):
    req=urllib.request.Request(url+"?"+urllib.parse.urlencode(params),headers={"User-Agent":"btc-regime-v6/1.0"})
    with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read().decode())


def download_binance_flow(start,end):
    rows=[]; cursor=int(start.timestamp()*1000); finish=int(end.timestamp()*1000)
    while cursor<finish:
        batch=get_json("https://fapi.binance.com/fapi/v1/klines",{"symbol":"BTCUSDT","interval":"15m","startTime":cursor,"endTime":finish,"limit":1000})
        if not batch:break
        rows.extend(batch); nxt=int(batch[-1][0])+900000
        if nxt<=cursor:break
        cursor=nxt;time.sleep(.08)
    cols=["timestamp","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_volume","taker_buy_quote","ignore"]
    x=pd.DataFrame(rows,columns=cols);x["timestamp"]=pd.to_datetime(pd.to_numeric(x.timestamp),unit="ms",utc=True)
    for c in ["volume","taker_buy_volume","trades"]:x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def build_conditions(prices,cfg,flow):
    x=build_scores(prices,cfg).join(flow[["taker_buy_volume","trades"]],how="left")
    x["taker_imbalance"]=2*x.taker_buy_volume/x.volume.replace(0,np.nan)-1
    x["bar_vol_ratio"]=x.volume/x.volume.rolling(20).mean()
    x["prior_high20"]=x.high.rolling(20).max().shift(1);x["prior_low20"]=x.low.rolling(20).min().shift(1)
    width_cut=x.h4_bb_width.rolling(2880,min_periods=960).quantile(.35)
    compressed=x.h4_bb_width.le(width_cut)
    dist_hi=(x.close-x.h4_last_swing_high).abs();dist_lo=(x.close-x.h4_last_swing_low).abs()
    near_level=pd.concat([dist_hi,dist_lo],axis=1).min(axis=1).le(.75*x.h4_atr14)
    pre_state=x.market_state.isin(["RANGE","BREAKOUT_PENDING","BULLISH_DEVELOPING","BEARISH_DEVELOPING"])
    x["loaded_spring"]=compressed&near_level&pre_state&x.h1_vol_ratio.lt(1.05)&x.confidence_score.ge(60)
    bull_break=x.close.gt(x.prior_high20);bear_break=x.close.lt(x.prior_low20)
    x["break_side"]=np.select([bull_break,bear_break],[1,-1],default=0)
    x["price_break_trigger"]=x.loaded_spring.shift(1,fill_value=False)&x.break_side.ne(0)
    x["volume_trigger"]=x.price_break_trigger&x.bar_vol_ratio.ge(1.30)
    candle_side=np.sign(x.close-x.open)
    x["orderflow_trigger"]=x.volume_trigger&x.taker_imbalance.abs().ge(.08)&candle_side.eq(x.break_side)&np.sign(x.taker_imbalance).eq(x.break_side)
    oi_change=x.open_interest.pct_change(4)
    x["new_position_trigger"]=x.orderflow_trigger&oi_change.gt(0)
    x["liquidation_trigger"]=x.orderflow_trigger&oi_change.lt(0)
    return x


def sample_loaded(x,cooldown=16):
    idx=[];last=-999999
    for i,v in enumerate(x.loaded_spring.to_numpy()):
        if v and i-last>=cooldown:idx.append(i);last=i
    return idx


def evaluate(x):
    groups={"LOADED_ONLY":sample_loaded(x),"PRICE_BREAK":list(np.flatnonzero(x.price_break_trigger)),"PLUS_VOLUME":list(np.flatnonzero(x.volume_trigger)),
            "PLUS_ORDERFLOW":list(np.flatnonzero(x.orderflow_trigger)),"NEW_POSITION":list(np.flatnonzero(x.new_position_trigger)),"LIQUIDATION":list(np.flatnonzero(x.liquidation_trigger))}
    rows=[];events=[]
    for name,indices in groups.items():
        vals={h:[] for h in [1,4,8,24]}
        for i in indices:
            if i+96>=len(x):continue
            side=int(x.break_side.iloc[i]) if name!="LOADED_ONLY" else int(np.sign(x.direction_score.iloc[i]))
            if side==0:continue
            r={"policy":name,"timestamp":x.index[i],"side":side,"price":x.close.iloc[i],"imbalance":x.taker_imbalance.iloc[i],"volume_ratio":x.bar_vol_ratio.iloc[i]}
            for h in [1,4,8,24]:
                future=x.iloc[i+1:i+1+h*4];ret=side*(future.close.iloc[-1]/x.close.iloc[i]-1);vals[h].append(ret);r[f"return_{h}h"]=ret
                r[f"mfe_{h}h"]=(future.high.max()/x.close.iloc[i]-1) if side==1 else (1-future.low.min()/x.close.iloc[i])
                r[f"mae_{h}h"]=(future.low.min()/x.close.iloc[i]-1) if side==1 else (1-future.high.max()/x.close.iloc[i])
            events.append(r)
        row={"policy":name,"signals":len(vals[1])}
        for h in [1,4,8,24]:
            a=pd.Series(vals[h],dtype=float);row[f"avg_{h}h"]=a.mean();row[f"hit_{h}h"]=(a>0).mean() if len(a) else np.nan
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(events)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="config.yaml");ap.add_argument("--input",default="data/BTCUSDT_15m.csv");ap.add_argument("--output",default="output_orderflow_v6");args=ap.parse_args()
    cfg=load_config(args.config);prices=normalize(pd.read_csv(args.input));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    flow=download_binance_flow(prices.index.min(),prices.index.max());x=build_conditions(prices,cfg,flow);summary,events=evaluate(x)
    summary.to_csv(out/"trigger_ladder_summary.csv",index=False);events.to_csv(out/"trigger_events.csv",index=False)
    report="BTC ORDERFLOW TRIGGER V6\n"+"="*40+"\nFlow coverage: "+f"{flow.index.min()} to {flow.index.max()} ({len(flow)} bars)\n\n"+summary.to_string(index=False)
    (out/"validation_report.txt").write_text(report,encoding="utf-8");print(report)


if __name__=="__main__":main()
