from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import build_scores, load_config, normalize
from news_type_research_v3 import REACTIVE, classify, fetch_news, semantic_sign


HORIZONS = (4, 8, 12, 24)
STOP_ATR = (0.75, 1.25, 1.75)


def build_loaded(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    x = build_scores(prices, cfg)
    width_cut = x.h4_bb_width.rolling(2880, min_periods=960).quantile(.35)
    compressed = x.h4_bb_width.le(width_cut)
    dist_hi = (x.close - x.h4_last_swing_high).abs()
    dist_lo = (x.close - x.h4_last_swing_low).abs()
    near_level = pd.concat([dist_hi, dist_lo], axis=1).min(axis=1).le(.75 * x.h4_atr14)
    state = x.market_state.isin([
        "RANGE", "BREAKOUT_PENDING", "BULLISH_DEVELOPING", "BEARISH_DEVELOPING"
    ])
    x["loaded"] = compressed & near_level & state & x.h1_vol_ratio.lt(1.05) & x.confidence_score.ge(60)
    x["side"] = np.sign(x.direction_score).astype(int)
    x.loc[x.direction_score.abs().lt(15), "side"] = 0
    x["bar_body"] = (x.close - x.open) / x.atr14.replace(0, np.nan)
    x["prior_high4"] = x.high.rolling(4).max().shift(1)
    x["prior_low4"] = x.low.rolling(4).min().shift(1)
    return x


def loaded_onsets(x: pd.DataFrame, cooldown_bars: int = 16) -> list[int]:
    candidates = np.flatnonzero((x.loaded & ~x.loaded.shift(1, fill_value=False) & x.side.ne(0)).to_numpy())
    out, last = [], -10**9
    for i in candidates:
        if i - last >= cooldown_bars:
            out.append(int(i)); last = int(i)
    return out


def news_map(prices: pd.DataFrame, output: Path) -> pd.DataFrame:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        return pd.DataFrame(columns=["timestamp", "semantic_sign", "category", "title"])
    raw = fetch_news(prices.index.min(), prices.index.max(), key, output / "alpha_news_detailed.csv")
    # Do not use post-news price movement to select events: that would leak the outcome.
    x = raw.loc[raw.relevance.ge(.35)].copy()
    text = x.title.fillna("") + " " + x.summary.fillna("")
    x["category"] = [classify(t) for t in text]
    x["semantic_sign"] = [semantic_sign(t) for t in text]
    x["reactive_title"] = [any(re.search(p, t.lower()) for p in REACTIVE) for t in text]
    x = x.loc[~x.reactive_title].copy()
    x["cluster_time"] = x.timestamp.dt.floor("2h")
    return (x.sort_values("relevance", ascending=False)
             .drop_duplicates(["category", "cluster_time"])
             .sort_values("timestamp"))


def first_news(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    if events.empty:
        return None
    a = events.timestamp.searchsorted(start, side="left")
    if a >= len(events) or events.iloc[a].timestamp > end:
        return None
    return events.iloc[a]


def find_entries(x: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin in loaded_onsets(x):
        if origin + 96 >= len(x):
            continue
        side = int(x.side.iloc[origin]); atr = float(x.atr14.iloc[origin])
        if not np.isfinite(atr) or atr <= 0:
            continue
        base = {"origin": origin, "loaded_time": x.index[origin], "side": side}
        rows.append({**base, "route": "EARLY", "entry_i": origin})

        # A shallow counter-trend excursion followed by an aligned close.
        for j in range(origin + 1, min(origin + 17, len(x))):
            excursion = side * (x.close.iloc[j] / x.close.iloc[origin] - 1)
            aligned_candle = side * (x.close.iloc[j] - x.open.iloc[j]) > 0
            not_invalid = excursion > -1.0 * atr / x.close.iloc[origin]
            touched = excursion <= -.20 * atr / x.close.iloc[origin]
            if touched and aligned_candle and not_invalid:
                rows.append({**base, "route": "PULLBACK", "entry_i": j}); break

        # First genuine directional impulse, deliberately earlier than a 20-bar breakout.
        for j in range(origin + 1, min(origin + 9, len(x))):
            level = x.prior_high4.iloc[j] if side == 1 else x.prior_low4.iloc[j]
            crossed = x.close.iloc[j] > level if side == 1 else x.close.iloc[j] < level
            if crossed and side * x.bar_body.iloc[j] >= .20:
                rows.append({**base, "route": "FIRST_IMPULSE", "entry_i": j}); break

        event = first_news(events, x.index[origin], x.index[min(origin + 32, len(x) - 1)])
        if event is not None:
            j = int(x.index.searchsorted(event.timestamp, side="left"))
            relation = "NEUTRAL" if int(event.semantic_sign) == 0 else (
                "ALIGNED" if int(event.semantic_sign) == side else "CONTRARY"
            )
            rows.append({**base, "route": "NEWS_" + relation, "entry_i": j,
                         "news_time": event.timestamp, "news_category": event.category,
                         "news_title": event.title})
    return pd.DataFrame(rows)


def one_trade(x: pd.DataFrame, row: pd.Series, hours: int, stop_atr: float, fee: float) -> dict:
    i, side = int(row.entry_i), int(row.side)
    entry = float(x.close.iloc[i]); atr = float(x.atr14.iloc[i])
    stop = entry - side * stop_atr * atr
    end = min(i + hours * 4, len(x) - 1); exit_i = end; stopped = False
    for j in range(i + 1, end + 1):
        hit = x.low.iloc[j] <= stop if side == 1 else x.high.iloc[j] >= stop
        if hit:
            exit_i, stopped = j, True; break
    exit_price = stop if stopped else float(x.close.iloc[exit_i])
    gross = side * (exit_price / entry - 1); net = gross - fee
    path = x.iloc[i + 1:exit_i + 1]
    if len(path):
        mfe = (path.high.max() / entry - 1) if side == 1 else (1 - path.low.min() / entry)
        mae = (path.low.min() / entry - 1) if side == 1 else (1 - path.high.max() / entry)
    else:
        mfe = mae = 0.0
    return {"exit_time": x.index[exit_i], "exit_reason": "STOP" if stopped else f"TIME_{hours}H",
            "gross_return": gross, "net_return": net, "mfe": mfe, "mae": mae}


def backtest(x: pd.DataFrame, entries: pd.DataFrame, fee: float) -> pd.DataFrame:
    trades = []
    for _, e in entries.iterrows():
        for h in HORIZONS:
            for stop in STOP_ATR:
                z = e.to_dict(); z.update({"hold_hours": h, "stop_atr": stop,
                    "entry_time": x.index[int(e.entry_i)], "entry_price": x.close.iloc[int(e.entry_i)]})
                z.update(one_trade(x, e, h, stop, fee)); trades.append(z)
    return pd.DataFrame(trades)


def summarise(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in trades.groupby(["route", "hold_hours", "stop_atr"]):
        r = g.net_return
        wins = r[r > 0].sum(); losses = -r[r < 0].sum()
        rows.append({"route": keys[0], "hold_hours": keys[1], "stop_atr": keys[2],
            "trades": len(g), "avg_net": r.mean(), "median_net": r.median(),
            "win_rate": (r > 0).mean(), "profit_factor": wins / losses if losses else np.nan,
            "stop_rate": g.exit_reason.eq("STOP").mean(), "avg_mae": g.mae.mean(), "avg_mfe": g.mfe.mean()})
    return pd.DataFrame(rows).sort_values(["avg_net", "profit_factor"], ascending=False)


def chronological_holdout(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = pd.Series(sorted(trades.loaded_time.unique()))
    split = origins.iloc[int(len(origins) * .60)]
    train, test = trades[trades.loaded_time < split], trades[trades.loaded_time >= split]
    train_summary = summarise(train)
    eligible = train_summary[train_summary.trades >= 20].copy()
    selected = eligible.sort_values(["avg_net", "profit_factor"], ascending=False).groupby("route").head(1)
    rows = []
    for _, s in selected.iterrows():
        g = test[(test.route == s.route) & (test.hold_hours == s.hold_hours) & (test.stop_atr == s.stop_atr)]
        if g.empty:
            continue
        rows.append({"route": s.route, "selected_hold_hours": s.hold_hours, "selected_stop_atr": s.stop_atr,
            "train_trades": int(s.trades), "train_avg_net": s.avg_net, "test_trades": len(g),
            "test_avg_net": g.net_return.mean(), "test_win_rate": (g.net_return > 0).mean(),
            "test_stop_rate": g.exit_reason.eq("STOP").mean()})
    result = pd.DataFrame(rows, columns=["route", "selected_hold_hours", "selected_stop_atr",
        "train_trades", "train_avg_net", "test_trades", "test_avg_net", "test_win_rate", "test_stop_rate"])
    if len(result):
        result = result.sort_values("test_avg_net", ascending=False)
    return result, pd.DataFrame([{"split_time": split}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--input", default="data/BTCUSDT_15m.csv")
    ap.add_argument("--output", default="output_loaded_entry_v7"); ap.add_argument("--round-trip-fee", type=float, default=.0012)
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    prices = normalize(pd.read_csv(args.input))
    # Exchange timestamps mark candle OPEN. Shift to candle CLOSE before building any signal.
    prices.index = prices.index + pd.Timedelta(minutes=15)
    x = build_loaded(prices, load_config(args.config))
    events = news_map(prices, out); entries = find_entries(x, events); trades = backtest(x, entries, args.round_trip_fee)
    summary = summarise(trades); holdout, split = chronological_holdout(trades)
    entries.to_csv(out / "candidate_entries.csv", index=False); trades.to_csv(out / "trade_events.csv", index=False)
    summary.to_csv(out / "all_route_results.csv", index=False); holdout.to_csv(out / "chronological_holdout.csv", index=False)
    report = "\n".join(["BTC LOADED-STATE ENTRY V7", "=" * 44,
        f"Loaded events: {entries.loaded_time.nunique()} | Candidate entries: {len(entries)} | News events available: {len(events)}",
        f"Round-trip fee: {args.round_trip_fee:.4%} | Holdout split: {split.split_time.iloc[0]}", "",
        "TOP 15 FULL-SAMPLE COMBINATIONS", summary.head(15).to_string(index=False), "",
        "CHRONOLOGICAL HOLDOUT — PARAMETERS SELECTED ON TRAIN ONLY",
        holdout.to_string(index=False) if len(holdout) else "No eligible holdout result.", "",
        "Decision rule: only consider a route if test_avg_net > 0 after fees, with adequate trades and acceptable stop rate."])
    (out / "validation_report.txt").write_text(report, encoding="utf-8"); print(report)


if __name__ == "__main__":
    main()
