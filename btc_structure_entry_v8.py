from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btc_regime import add_indicators, normalize


TARGET_R = (1.5, 2.0, 3.0)


def features(prices: pd.DataFrame) -> pd.DataFrame:
    x = add_indicators(prices)
    x["atr_median_96"] = x.atr14.rolling(96, min_periods=48).median()
    x["body_atr"] = (x.close - x.open) / x.atr14.replace(0, np.nan)
    return x


def range_at(x: pd.DataFrame, i: int, lookback: int = 32):
    if i < max(200, lookback + 96):
        return None
    w = x.iloc[i - lookback:i]
    atr = float(x.atr14.iloc[i - 1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    hi, lo = float(w.high.max()), float(w.low.min())
    width = hi - lo
    # A usable box: compressed, but not so narrow that fees dominate it.
    if not (1.5 * atr <= width <= 5.0 * atr):
        return None
    if x.atr14.iloc[i - 1] > 0.90 * x.atr_median_96.iloc[i - 1]:
        return None
    tol = .30 * atr
    upper_tests = int((w.high >= hi - tol).sum())
    lower_tests = int((w.low <= lo + tol).sum())
    if upper_tests < 2 or lower_tests < 2:
        return None
    return {"range_high": hi, "range_low": lo, "range_width": width, "range_atr": atr,
            "upper_tests": upper_tests, "lower_tests": lower_tests}


def breakouts(x: pd.DataFrame) -> list[dict]:
    out, blocked_until = [], -1
    for i in range(len(x) - 12):
        if i <= blocked_until:
            continue
        box = range_at(x, i)
        if box is None:
            continue
        atr = box["range_atr"]; close = float(x.close.iloc[i])
        if close >= box["range_high"] + .15 * atr:
            side, excess = 1, close - box["range_high"]
        elif close <= box["range_low"] - .15 * atr:
            side, excess = -1, box["range_low"] - close
        else:
            continue
        # Avoid entering the study after an already-exhausted candle.
        if excess > 1.50 * atr:
            continue
        level = box["range_high"] if side == 1 else box["range_low"]
        out.append({"break_i": i, "break_time": x.index[i], "break_side": side,
                    "break_level": level, "break_excess_atr": excess / atr,
                    "break_volume_ratio": float(x.vol_ratio.iloc[i]), **box})
        blocked_until = i + 12
    return out


def setups(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in breakouts(x):
        i, side, level, atr = b["break_i"], b["break_side"], b["break_level"], b["range_atr"]

        # True breakout: price revisits the level, closes outside, then has an aligned body.
        for j in range(i + 1, min(i + 9, len(x) - 1)):
            touched = x.low.iloc[j] <= level + .25 * atr if side == 1 else x.high.iloc[j] >= level - .25 * atr
            held = x.close.iloc[j] >= level - .05 * atr if side == 1 else x.close.iloc[j] <= level + .05 * atr
            aligned = side * (x.close.iloc[j] - x.open.iloc[j]) >= .10 * atr
            if touched and held and aligned:
                entry_i = j + 1
                extreme = x.low.iloc[i:j + 1].min() if side == 1 else x.high.iloc[i:j + 1].max()
                stop = min(extreme, level - .20 * atr) if side == 1 else max(extreme, level + .20 * atr)
                rows.append({**b, "route": "BREAK_RETEST_HOLD", "signal_i": j,
                             "signal_time": x.index[j], "entry_i": entry_i, "trade_side": side,
                             "structure_stop": float(stop), "confirmation_bars": j - i})
                break

        # False breakout: a close returns materially inside the old box within four bars.
        for j in range(i + 1, min(i + 5, len(x) - 1)):
            reclaimed = x.close.iloc[j] <= level - .10 * atr if side == 1 else x.close.iloc[j] >= level + .10 * atr
            reverse_body = side * (x.close.iloc[j] - x.open.iloc[j]) <= -.10 * atr
            if reclaimed and reverse_body:
                trade_side, entry_i = -side, j + 1
                extreme = x.high.iloc[i:j + 1].max() if trade_side == -1 else x.low.iloc[i:j + 1].min()
                stop = extreme + .15 * atr if trade_side == -1 else extreme - .15 * atr
                rows.append({**b, "route": "FALSE_BREAK_RECLAIM", "signal_i": j,
                             "signal_time": x.index[j], "entry_i": entry_i, "trade_side": trade_side,
                             "structure_stop": float(stop), "confirmation_bars": j - i})
                break
    return pd.DataFrame(rows)


def simulate(x: pd.DataFrame, setup: pd.Series, target_r: float, costs: float,
             max_hold_bars: int = 96) -> dict | None:
    i, side = int(setup.entry_i), int(setup.trade_side)
    if i >= len(x):
        return None
    entry = float(x.open.iloc[i]); stop = float(setup.structure_stop)
    risk = side * (entry - stop)
    risk_pct = risk / entry
    if not (.002 <= risk_pct <= .020):
        return None
    target = entry + side * target_r * risk
    end = min(i + max_hold_bars, len(x) - 1)
    exit_i, exit_price, reason = end, float(x.close.iloc[end]), "TIME_24H"
    for j in range(i, end + 1):
        stop_hit = x.low.iloc[j] <= stop if side == 1 else x.high.iloc[j] >= stop
        target_hit = x.high.iloc[j] >= target if side == 1 else x.low.iloc[j] <= target
        # If both occur in one 15m candle, take the conservative ordering.
        if stop_hit:
            exit_i, exit_price, reason = j, stop, "STOP"; break
        if target_hit:
            exit_i, exit_price, reason = j, target, "TARGET"; break
    gross = side * (exit_price / entry - 1)
    net = gross - costs
    return {"entry_time": x.index[i], "entry_price": entry, "stop_price": stop,
            "target_price": target, "target_r": target_r, "risk_pct": risk_pct,
            "exit_time": x.index[exit_i], "exit_price": exit_price, "exit_reason": reason,
            "gross_return": gross, "net_return": net, "net_r": net / risk_pct}


def backtest(x: pd.DataFrame, candidates: pd.DataFrame, costs: float) -> pd.DataFrame:
    rows = []
    for _, s in candidates.iterrows():
        for target_r in TARGET_R:
            result = simulate(x, s, target_r, costs)
            if result is not None:
                row = s.to_dict(); row.update(result); rows.append(row)
    return pd.DataFrame(rows)


def stats(g: pd.DataFrame) -> dict:
    r = g.net_return
    win = r[r > 0].sum(); loss = -r[r < 0].sum()
    return {"trades": len(g), "avg_net": r.mean(), "median_net": r.median(),
            "win_rate": (r > 0).mean(), "profit_factor": win / loss if loss else np.nan,
            "avg_net_r": g.net_r.mean(), "stop_rate": g.exit_reason.eq("STOP").mean(),
            "target_rate": g.exit_reason.eq("TARGET").mean()}


def summaries(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (route, target), g in trades.groupby(["route", "target_r"]):
        rows.append({"route": route, "target_r": target, **stats(g)})
    return pd.DataFrame(rows).sort_values(["avg_net", "profit_factor"], ascending=False)


def holdout(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    times = pd.Series(sorted(trades.break_time.unique()))
    split = pd.Timestamp(times.iloc[int(len(times) * .60)])
    train, test = trades[trades.break_time < split], trades[trades.break_time >= split]
    rows = []
    for route, g in train.groupby("route"):
        options = summaries(g)
        eligible = options[options.trades >= 20]
        if eligible.empty:
            continue
        selected = eligible.iloc[0]
        z = test[(test.route == route) & (test.target_r == selected.target_r)]
        if len(z):
            rows.append({"route": route, "selected_target_r": selected.target_r,
                         **{"train_" + k: v for k, v in stats(g[g.target_r == selected.target_r]).items()},
                         **{"test_" + k: v for k, v in stats(z).items()}})
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("test_avg_net", ascending=False)
    return result, split


def quarterly(trades: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in selected.iterrows():
        g = trades[(trades.route == s.route) & (trades.target_r == s.selected_target_r)].copy()
        g["quarter"] = g.entry_time.dt.to_period("Q").astype(str)
        for q, z in g.groupby("quarter"):
            rows.append({"quarter": q, "route": s.route, "target_r": s.selected_target_r, **stats(z)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/BTCUSDT_15m.csv")
    ap.add_argument("--output", default="output_structure_entry_v8")
    ap.add_argument("--costs", type=float, default=.0014,
                    help="Round-trip fees plus slippage; default 0.14%%")
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    prices = normalize(pd.read_csv(args.input))
    # Bybit timestamps are candle opens; all decisions below use known candle closes.
    prices.index = prices.index + pd.Timedelta(minutes=15)
    x = features(prices); candidates = setups(x); trades = backtest(x, candidates, args.costs)
    if trades.empty:
        raise SystemExit("No eligible V8 trades found.")
    overall = summaries(trades); test, split = holdout(trades); quarters = quarterly(trades, test)
    candidates.to_csv(out / "candidate_setups.csv", index=False)
    trades.to_csv(out / "trade_events.csv", index=False)
    overall.to_csv(out / "full_sample_results.csv", index=False)
    test.to_csv(out / "chronological_holdout.csv", index=False)
    quarters.to_csv(out / "quarter_stability.csv", index=False)
    report = "\n".join(["BTC STRUCTURE ENTRY V8", "=" * 42,
        f"Breakout structures: {candidates.break_time.nunique()} | Candidate routes: {len(candidates)} | Eligible trades: {len(trades) // len(TARGET_R)}",
        f"Costs: {args.costs:.4%} | Next-bar-open execution | Holdout split: {split}", "",
        "FULL SAMPLE", overall.to_string(index=False), "",
        "CHRONOLOGICAL HOLDOUT — TARGET SELECTED ON TRAIN ONLY",
        test.to_string(index=False) if len(test) else "No route had enough training observations.", "",
        "QUARTER STABILITY", quarters.to_string(index=False) if len(quarters) else "No selected route.", "",
        "Pass rule: positive test expectancy after costs, PF > 1, adequate observations, and no dependence on one quarter."])
    (out / "validation_report.txt").write_text(report, encoding="utf-8"); print(report)


if __name__ == "__main__":
    main()
