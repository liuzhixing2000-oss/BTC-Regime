from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import build_scores, load_config, normalize
from news_type_research_v3 import fetch_news, prepare_events, bh_fdr


def technical_ready(scores: pd.DataFrame) -> pd.Series:
    state=scores.market_state.isin(["BULLISH_DEVELOPING","BEARISH_DEVELOPING","BREAKOUT_PENDING","BULLISH_CONFIRMED","BEARISH_CONFIRMED"])
    structure=scores.h4_structure_event.str.contains("BREAK_CONFIRMED|RETESTING|RETEST_HELD|TREND_CONTINUATION",regex=True,na=False)
    return scores.opportunity_score.ge(65)&scores.confidence_score.ge(60)&(state|structure)


def attach_pre_news(events: pd.DataFrame,scores: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    ready=technical_ready(scores)
    for _,e in events.iterrows():
        pos=scores.index.searchsorted(e.timestamp,side="left")-1
        if pos<0 or pos+16>=len(scores): continue
        r=scores.iloc[pos]; side=int(np.sign(r.direction_score))
        pre_atr=float(r.atr14/r.close) if np.isfinite(r.atr14) else np.nan
        future=scores.iloc[pos+1:pos+17]; ret=float(future.close.iloc[-1]/r.close-1)
        row=e.to_dict(); row.update({"technical_timestamp":scores.index[pos],"technical_ready":bool(ready.iloc[pos]),
            "market_state":r.market_state,"direction_score":r.direction_score,"opportunity_score":r.opportunity_score,
            "tech_side":side,"atr_pct":pre_atr,"return_4h":ret,"tech_aligned_return_4h":side*ret if side else np.nan,
            "abs_return_4h":abs(ret),"max_up_4h":future.high.max()/r.close-1,"max_down_4h":future.low.min()/r.close-1})
        rows.append(row)
    return pd.DataFrame(rows)


def controls(scores: pd.DataFrame, attached: pd.DataFrame) -> pd.DataFrame:
    ready=technical_ready(scores); blocked=np.zeros(len(scores),dtype=bool)
    for ts in attached.timestamp:
        p=scores.index.searchsorted(ts)
        blocked[max(0,p-16):min(len(scores),p+17)]=True
    rows=[]
    for pos in np.where(ready.to_numpy()&~blocked)[0]:
        if pos+16>=len(scores): continue
        r=scores.iloc[pos]; side=int(np.sign(r.direction_score)); future=scores.iloc[pos+1:pos+17]
        ret=float(future.close.iloc[-1]/r.close-1)
        rows.append({"timestamp":scores.index[pos],"market_state":r.market_state,"tech_side":side,
                     "direction_score":r.direction_score,"opportunity_score":r.opportunity_score,
                     "abs_return_4h":abs(ret),"tech_aligned_return_4h":side*ret if side else np.nan})
    return pd.DataFrame(rows)


def permutation_p(news_values,control_values,rng,reps=10000):
    a=np.asarray(news_values,float); b=np.asarray(control_values,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<5 or len(b)<5:return np.nan
    observed=np.mean(a); exceed=0
    for _ in range(reps):
        sample=rng.choice(b,len(a),replace=True)
        exceed += (sample.mean()>=observed)
    return (exceed+1)/(reps+1)


def analyse(attached: pd.DataFrame, control: pd.DataFrame, seed=75):
    rng=np.random.default_rng(seed); ready=attached.loc[attached.technical_ready].copy()
    # Thresholds are determined only by comparable technical-ready no-news bars.
    vol_threshold=control.abs_return_4h.quantile(.75)
    ready["large_release"]=ready.abs_return_4h.ge(vol_threshold)
    ready["directional_support"]=ready.tech_side.ne(0)&ready.semantic_sign.ne(0)&ready.tech_side.eq(ready.semantic_sign)
    ready["effective_catalyst"]=ready.large_release & (ready.tech_side.eq(0)|ready.tech_aligned_return_4h.gt(0))
    control_effective=control.abs_return_4h.ge(vol_threshold)&(control.tech_side.eq(0)|control.tech_aligned_return_4h.gt(0))
    summary=pd.DataFrame([{
        "all_nonreactive_news":len(attached),"news_with_technical_support":len(ready),
        "support_share":len(ready)/len(attached) if len(attached) else np.nan,
        "effective_news":int(ready.effective_catalyst.sum()),"effective_news_rate":ready.effective_catalyst.mean(),
        "matched_tech_no_news_rate":control_effective.mean(),
        "incremental_effective_rate":ready.effective_catalyst.mean()-control_effective.mean(),
        "large_move_threshold_4h":vol_threshold,
        "mean_abs_move_news":ready.abs_return_4h.mean(),"mean_abs_move_no_news":control.abs_return_4h.mean(),
        "abs_move_permutation_p":permutation_p(ready.abs_return_4h,control.abs_return_4h,rng,3000)
    }])
    rows=[]
    for cat,g in ready.groupby("category"):
        c=control
        # Prefer same technical direction when enough controls exist.
        sides=set(g.tech_side.unique()); matched=c.loc[c.tech_side.isin(sides)] if sides else c
        base_rate=((matched.abs_return_4h>=vol_threshold)&(matched.tech_side.eq(0)|matched.tech_aligned_return_4h.gt(0))).mean()
        split=g.timestamp.min()+(g.timestamp.max()-g.timestamp.min())/2
        rows.append({"category":cat,"supported_news":len(g),"effective_news":int(g.effective_catalyst.sum()),
                     "effective_rate":g.effective_catalyst.mean(),"matched_no_news_rate":base_rate,
                     "incremental_rate":g.effective_catalyst.mean()-base_rate,"mean_abs_move_4h":g.abs_return_4h.mean(),
                     "matched_abs_move_4h":matched.abs_return_4h.mean(),"move_p":permutation_p(g.abs_return_4h,matched.abs_return_4h,rng,3000),
                     "train_effective_rate":g.loc[g.timestamp<=split,"effective_catalyst"].mean(),
                     "test_effective_rate":g.loc[g.timestamp>split,"effective_catalyst"].mean(),
                     "direction_supported_news":int(g.directional_support.sum()),
                     "direction_supported_hit":g.loc[g.directional_support,"tech_aligned_return_4h"].gt(0).mean() if g.directional_support.any() else np.nan})
    cats=pd.DataFrame(rows)
    if not cats.empty:
        cats["move_q"]=bh_fdr(cats.move_p)
        cats["catalyst_pass"]=(cats.supported_news>=30)&(cats.incremental_rate>=.05)&(cats.train_effective_rate>cats.matched_no_news_rate)&(cats.test_effective_rate>cats.matched_no_news_rate)&(cats.move_q<.10)
        cats=cats.sort_values(["catalyst_pass","incremental_rate","supported_news"],ascending=False)
    whitelist=cats.loc[cats.catalyst_pass].copy() if not cats.empty else cats.copy()
    return ready,summary,cats,whitelist


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="config.yaml");ap.add_argument("--input",default="data/BTCUSDT_15m.csv");ap.add_argument("--output",default="output_catalyst_v5");args=ap.parse_args()
    key=os.environ.get("ALPHA_VANTAGE_API_KEY","").strip()
    if not key:raise SystemExit("Missing ALPHA_VANTAGE_API_KEY")
    cfg=load_config(args.config);prices=normalize(pd.read_csv(args.input));scores=build_scores(prices,cfg).dropna(subset=["h4_bb_mid","h1_ema50"])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    news=fetch_news(prices.index.min(),prices.index.max(),key,out/"alpha_news_detailed.csv");events=prepare_events(news,prices)
    attached=attach_pre_news(events,scores);control=controls(scores,attached);ready,summary,cats,whitelist=analyse(attached,control)
    ready.to_csv(out/"supported_news_events.csv",index=False);summary.to_csv(out/"catalyst_summary.csv",index=False);cats.to_csv(out/"category_conditional_effect.csv",index=False);whitelist.to_csv(out/"conditional_news_whitelist.csv",index=False)
    text="\n".join(["BTC NEWS × TECHNICAL CATALYST V5","="*46,"","OVERALL",summary.to_string(index=False),"","CONDITIONAL CATEGORY EFFECT",cats.to_string(index=False),"","CONDITIONAL WHITELIST",whitelist.to_string(index=False) if len(whitelist) else "EMPTY — no category passed conditional catalyst rules."])
    (out/"validation_report.txt").write_text(text,encoding="utf-8");print(text)


if __name__=="__main__":main()
