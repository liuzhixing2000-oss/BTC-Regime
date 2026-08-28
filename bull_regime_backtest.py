from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BINANCE = "https://api.binance.com"


@dataclass(frozen=True)
class Config:
    fee_rate: float = 0.00055
    slippage_rate: float = 0.00020
    atr_stop: float = 1.5
    max_stop_pct: float = 0.03
    chase_atr: float = 1.5
    retest_atr: float = 0.30
    breakout_lookback: int = 120  # 20 days of 4H bars
    retest_window: int = 12
    swing_left: int = 2
    swing_right: int = 2


def get_json(path: str, params: dict, retries: int = 5):
    url = BINANCE + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btc-bull-regime/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def download_4h(start: str, end: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step = 1000 * 4 * 60 * 60 * 1000
    starts = list(range(start_ms, end_ms, step))

    def fetch(cursor):
        return get_json("/api/v3/klines", {
            "symbol": "BTCUSDT", "interval": "4h", "startTime": cursor,
            "endTime": min(end_ms, cursor + step - 1), "limit": 1000,
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(fetch, starts))
    rows = [row for batch in batches for row in batch]
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    out = pd.DataFrame(rows, columns=cols)
    out.index = pd.to_datetime(out.open_time.astype("int64") + 4 * 60 * 60 * 1000,
                               unit="ms", utc=True)
    out.index.name = "timestamp"
    out = out[["open", "high", "low", "close", "volume"]].astype(float)
    return out[~out.index.duplicated()].sort_index()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ohlcv_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()


def confirmed_swings(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    x = df.copy()
    width = left + right + 1
    raw_hi = x.high.eq(x.high.rolling(width, center=True).max())
    raw_lo = x.low.eq(x.low.rolling(width, center=True).min())
    confirmed_hi = x.high.where(raw_hi).shift(right)
    confirmed_lo = x.low.where(raw_lo).shift(right)

    def last_two(events: pd.Series):
        last = np.full(len(events), np.nan)
        previous = np.full(len(events), np.nan)
        a = b = np.nan
        for i, value in enumerate(events.to_numpy()):
            if np.isfinite(value):
                b, a = a, value
            last[i], previous[i] = a, b
        return last, previous

    x["swing_high"], x["prev_swing_high"] = last_two(confirmed_hi)
    x["swing_low"], x["prev_swing_low"] = last_two(confirmed_lo)
    x["hh_hl"] = (x.swing_high > x.prev_swing_high) & (x.swing_low > x.prev_swing_low)
    return x


def prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = confirmed_swings(df, cfg.swing_left, cfg.swing_right)
    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)
    x["atr"] = atr(x)
    x["daily_mid"] = ohlcv_resample(df, "1D").close.rolling(20).mean().reindex(x.index, method="ffill")

    daily = ohlcv_resample(df, "1D")
    daily["sma200"] = daily.close.rolling(200).mean()
    daily["sma200_slope"] = daily.sma200.diff(20)
    daily["ema20"] = ema(daily.close, 20)
    daily["ema50"] = ema(daily.close, 50)
    weekly = ohlcv_resample(df, "W-MON")
    weekly["ema20"] = ema(weekly.close, 20)
    weekly["weekly_ok"] = weekly.close > weekly.ema20
    daily["weekly_ok"] = weekly.weekly_ok.reindex(daily.index, method="ffill")
    daily["above_sma200"] = daily.close > daily.sma200
    daily["regime_score"] = (
        daily.above_sma200.astype(int)
        + (daily.sma200_slope > 0).astype(int)
        + (daily.ema20 > daily.ema50).astype(int)
        + daily.weekly_ok.fillna(False).astype(int)
    )
    x["regime_score"] = daily.regime_score.reindex(x.index, method="ffill")
    x["above_sma200"] = daily.above_sma200.reindex(x.index, method="ffill").fillna(False)
    x["regime"] = pd.cut(x.regime_score, [-1, 1, 2, 3, 4], labels=["OFF", "EARLY_BULL", "CONFIRMED_BULL", "STRONG_BULL"])

    x["prior_20d_high"] = x.high.rolling(cfg.breakout_lookback).max().shift(1)
    x["breakout"] = x.close > x.prior_20d_high
    anchors = np.full(len(x), np.nan)
    ages = np.full(len(x), np.nan)
    anchor = np.nan
    age = 10_000
    for i, row in enumerate(x.itertuples()):
        if row.breakout:
            anchor, age = row.prior_20d_high, 0
        elif np.isfinite(anchor):
            age += 1
        anchors[i], ages[i] = anchor, age
    x["breakout_anchor"] = anchors
    x["breakout_age"] = ages

    trend = (x.ema20 > x.ema50) & x.hh_hl
    not_chasing = (x.close - x.ema20) <= cfg.chase_atr * x.atr
    reclaim = (x.close > x.open) & (x.close > x.high.shift(1))
    touched = (
        (x.low <= x.ema20 + cfg.retest_atr * x.atr)
        | (x.low <= x.ema50 + cfg.retest_atr * x.atr)
        | (x.low <= x.daily_mid + cfg.retest_atr * x.atr)
    )
    x["sig_pullback"] = trend & not_chasing & touched & reclaim
    x["sig_retest"] = (
        trend & not_chasing & x.breakout_age.between(1, cfg.retest_window)
        & (x.low <= x.breakout_anchor + cfg.retest_atr * x.atr)
        & (x.close >= x.breakout_anchor) & (x.close > x.open)
    )
    return x


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else np.nan


def simulate(x: pd.DataFrame, family: str, exit_name: str, min_score: int, cfg: Config):
    signal_col = "sig_pullback" if family == "PULLBACK_RECLAIM" else "sig_retest"
    trades = []
    i = 0
    equity = 1.0
    while i < len(x) - 1:
        sig = x.iloc[i]
        if not bool(sig[signal_col]) or sig.regime_score < min_score:
            i += 1
            continue
        entry_bar = x.iloc[i + 1]
        entry = entry_bar.open * (1 + cfg.slippage_rate)
        stop = min(sig.swing_low, entry - cfg.atr_stop * sig.atr)
        stop_pct = (entry - stop) / entry
        if not np.isfinite(stop) or stop >= entry or stop_pct > cfg.max_stop_pct:
            i += 1
            continue
        risk = entry - stop
        peak = entry
        partial = False
        realized_r = 0.0
        remaining = 1.0
        exit_price = np.nan
        reason = "DATA_END"
        j = i + 1
        for j in range(i + 1, len(x)):
            bar = x.iloc[j]
            peak = max(peak, bar.high)
            target1 = entry + risk
            target2 = entry + 2 * risk
            trail = peak - 3 * bar.atr if np.isfinite(bar.atr) else stop
            active_stop = stop
            if exit_name == "D_CHANDELIER":
                active_stop = max(stop, trail)
            elif exit_name == "B_PARTIAL_EMA" and partial and np.isfinite(bar.ema20):
                active_stop = max(stop, bar.ema20)
            stop_hit = bar.low <= active_stop
            if stop_hit:
                exit_price = active_stop * (1 - cfg.slippage_rate)
                realized_r += remaining * (exit_price - entry) / risk
                reason = "STOP_OR_TRAIL"
                break
            if exit_name == "A_FIXED_2R" and bar.high >= target2:
                exit_price = target2 * (1 - cfg.slippage_rate)
                realized_r = (exit_price - entry) / risk
                reason = "2R"
                break
            if exit_name == "B_PARTIAL_EMA" and not partial and bar.high >= target1:
                px = target1 * (1 - cfg.slippage_rate)
                realized_r += 0.25 * (px - entry) / risk
                remaining, partial = 0.75, True
            if exit_name == "E_HALF_2R_REGIME" and not partial and bar.high >= target2:
                px = target2 * (1 - cfg.slippage_rate)
                realized_r += 0.50 * (px - entry) / risk
                remaining, partial = 0.50, True
            structure_break = np.isfinite(bar.swing_low) and bar.close < bar.swing_low
            if exit_name == "C_STRUCTURE" and structure_break:
                exit_price = bar.close * (1 - cfg.slippage_rate)
                realized_r = (exit_price - entry) / risk
                reason = "STRUCTURE_BREAK"
                break
            if exit_name == "E_HALF_2R_REGIME" and bar.regime_score < min_score:
                exit_price = bar.close * (1 - cfg.slippage_rate)
                realized_r += remaining * (exit_price - entry) / risk
                reason = "REGIME_FAIL"
                break
        if not np.isfinite(exit_price):
            # An open trade at the end of the dataset has no knowable exit and
            # must not enter performance statistics.
            break
        fee_r = cfg.fee_rate * (entry + exit_price) / risk
        net_r = realized_r - fee_r
        risk_fraction = 0.005 if sig.regime_score == 2 else 0.01
        equity *= max(0.01, 1 + risk_fraction * net_r)
        trades.append({
            "family": family, "exit": exit_name, "min_score": min_score,
            "signal_time": x.index[i], "entry_time": x.index[i + 1], "exit_time": x.index[j],
            "entry": entry, "stop": stop, "exit_price": exit_price, "stop_pct": stop_pct,
            "gross_r": realized_r, "fee_r": fee_r, "net_r": net_r, "reason": reason,
            "regime_score": int(sig.regime_score), "risk_fraction": risk_fraction, "equity": equity,
        })
        i = max(j + 1, i + 1)
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    wins = t.net_r[t.net_r > 0].sum()
    losses = -t.net_r[t.net_r < 0].sum()
    equity = (1 + t.risk_fraction * t.net_r).cumprod() if "risk_fraction" in t else t.equity
    return {
        "trades": len(t), "avg_net_r": t.net_r.mean(), "median_net_r": t.net_r.median(),
        "win_rate": (t.net_r > 0).mean(), "profit_factor": wins / losses if losses else np.inf,
        "total_return": equity.iloc[-1] - 1, "max_drawdown": max_drawdown(equity),
    }


def benchmarks(x: pd.DataFrame) -> pd.DataFrame:
    ret = x.close.pct_change().fillna(0)
    ema_signal = (x.ema20 > x.ema50).shift(1).fillna(False)
    sma_signal = x.above_sma200.shift(1).fillna(False)
    rows = []
    for name, signal in {
        "BUY_HOLD": pd.Series(True, index=x.index),
        "CLOSE_ABOVE_SMA200_PROXY": sma_signal,
        "EMA20_ABOVE_EMA50_4H": ema_signal,
    }.items():
        strategy_ret = ret * signal.astype(float)
        equity = (1 + strategy_ret).cumprod()
        for split, mask in {
            "FULL": pd.Series(True, index=x.index),
            "TRAIN_2017_2021": x.index < pd.Timestamp("2022-01-01", tz="UTC"),
            "VALID_2022_2024": (x.index >= pd.Timestamp("2022-01-01", tz="UTC")) & (x.index < pd.Timestamp("2025-01-01", tz="UTC")),
            "HOLDOUT_2025_NOW": x.index >= pd.Timestamp("2025-01-01", tz="UTC"),
        }.items():
            rr = strategy_ret[mask]
            eq = (1 + rr).cumprod()
            rows.append({"benchmark": name, "split": split, "bars": len(rr),
                         "total_return": eq.iloc[-1] - 1 if len(eq) else np.nan,
                         "max_drawdown": max_drawdown(eq), "exposure": signal[mask].mean()})
    return pd.DataFrame(rows)


def split_name(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2022-01-01", tz="UTC"):
        return "TRAIN_2017_2021"
    if ts < pd.Timestamp("2025-01-01", tz="UTC"):
        return "VALID_2022_2024"
    return "HOLDOUT_2025_NOW"


def variant_signal(x: pd.DataFrame, location: str, confirmation: str,
                   trend_atr: float, tolerance_atr: float) -> pd.Series:
    """Predeclared local sensitivity variants; all terms use current/past closed bars."""
    tol = tolerance_atr * x.atr
    shallow = (x.low <= x.ema20 + tol) & (x.low > x.ema50 - tol)
    deep = (x.low <= x.ema50 + tol) | (x.low <= x.daily_mid + tol)
    if location == "SHALLOW":
        touched = shallow
    elif location == "DEEP":
        touched = deep
    else:
        touched = shallow | deep

    one_bar = touched & (x.close > x.open) & (x.close > x.high.shift(1))
    two_bar = (
        touched.rolling(2).max().fillna(0).astype(bool)
        & (x.close > x.ema20)
        & (x.close.shift(1) > x.ema20.shift(1))
        & (x.low > x.low.shift(1))
    )
    confirm = one_bar if confirmation == "ONE_BAR_RECLAIM" else two_bar
    strength = (x.ema20 - x.ema50) / x.atr
    trend = (x.ema20 > x.ema50) & x.hh_hl & (strength >= trend_atr)
    not_chasing = (x.close - x.ema20) <= 1.5 * x.atr
    return trend & not_chasing & confirm


def run_sensitivity(x: pd.DataFrame, cfg: Config, out: Path):
    rows, ledgers = [], []
    for location in ["SHALLOW", "DEEP", "ANY"]:
        for confirmation in ["ONE_BAR_RECLAIM", "TWO_BAR_STABLE"]:
            for trend_atr in [0.0, 0.5, 1.0]:
                for tolerance_atr in [0.15, 0.30, 0.50]:
                    for min_score in [2, 3, 4]:
                        work = x.copy()
                        work["sig_pullback"] = variant_signal(
                            work, location, confirmation, trend_atr, tolerance_atr)
                        trades = simulate(work, "PULLBACK_RECLAIM", "A_FIXED_2R", min_score, cfg)
                        variant = f"{location}__{confirmation}__T{trend_atr:.1f}__R{tolerance_atr:.2f}__S{min_score}"
                        if not trades.empty:
                            trades["split"] = trades.entry_time.map(split_name)
                            trades["variant"] = variant
                            ledgers.append(trades)
                        for split in ["TRAIN_2017_2021", "VALID_2022_2024", "HOLDOUT_2025_NOW"]:
                            group = trades[trades.entry_time.map(split_name) == split] if not trades.empty else trades
                            rows.append({"variant": variant, "location": location,
                                         "confirmation": confirmation, "trend_atr": trend_atr,
                                         "tolerance_atr": tolerance_atr, "min_score": min_score,
                                         "split": split, **summarize(group.reset_index(drop=True))})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "sensitivity_grid.csv", index=False)
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()

    train = grid[(grid.split == "TRAIN_2017_2021") & (grid.trades >= 12)].copy()
    train["train_rank"] = train.avg_net_r.rank(method="first", ascending=False)
    selected = train.sort_values(["avg_net_r", "profit_factor"], ascending=False).head(10)[["variant", "train_rank"]]
    reveal = grid.merge(selected, on="variant", how="inner").sort_values(["train_rank", "split"])
    reveal.to_csv(out / "sensitivity_train_top10_reveal.csv", index=False)
    if not ledger.empty and not selected.empty:
        ledger[ledger.variant.isin(selected.variant)].to_csv(out / "sensitivity_selected_ledger.csv", index=False)
    return reveal


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    if cache.exists() and not args.refresh:
        df = pd.read_csv(cache, parse_dates=["timestamp"]).set_index("timestamp")
    else:
        df = download_4h(args.start, args.end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache, index=False)
    cfg = Config()
    x = prepare(df, cfg)
    x.reset_index().to_csv(out / "regime_history.csv", index=False)
    benchmarks(x).to_csv(out / "benchmark_summary.csv", index=False)
    all_trades, rows = [], []
    exits = ["A_FIXED_2R", "B_PARTIAL_EMA", "C_STRUCTURE", "D_CHANDELIER", "E_HALF_2R_REGIME"]
    for family in ["PULLBACK_RECLAIM", "BREAKOUT_RETEST"]:
        for exit_name in exits:
            for min_score in [2, 3, 4]:
                trades = simulate(x, family, exit_name, min_score, cfg)
                if not trades.empty:
                    trades["split"] = trades.entry_time.map(split_name)
                    trades["year"] = trades.entry_time.dt.year
                    all_trades.append(trades)
                rows.append({"family": family, "exit": exit_name, "min_score": min_score,
                             "split": "FULL", **summarize(trades)})
                if not trades.empty:
                    for split, group in trades.groupby("split"):
                        rows.append({"family": family, "exit": exit_name, "min_score": min_score,
                                     "split": split, **summarize(group.reset_index(drop=True))})
    ledger = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    ledger.to_csv(out / "trade_ledger.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "strategy_summary.csv", index=False)
    if not ledger.empty:
        yearly = ledger.groupby(["family", "exit", "min_score", "year"], observed=True).apply(
            lambda g: pd.Series(summarize(g.reset_index(drop=True))), include_groups=False).reset_index()
        yearly.to_csv(out / "yearly_summary.csv", index=False)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "start": args.start, "end": args.end,
                   "data": "Binance BTCUSDT spot 4H proxy", "funding_included": False}, f, indent=2)
    ranked = summary[(summary.split == "FULL") & (summary.trades >= 10)].sort_values(
        ["avg_net_r", "profit_factor"], ascending=False)
    print(ranked.head(15).to_string(index=False))
    if args.sensitivity:
        print("\nTRAIN-SELECTED SENSITIVITY VARIANTS (validation and holdout revealed after selection)")
        print(run_sensitivity(x, cfg, out).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2017-08-17")
    p.add_argument("--end", default=str(pd.Timestamp.now(tz="UTC").date()))
    p.add_argument("--cache", default="data/BTCUSDT_4h_spot.csv")
    p.add_argument("--output", default="output/bull_regime_v1")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--sensitivity", action="store_true")
    run(p.parse_args())
