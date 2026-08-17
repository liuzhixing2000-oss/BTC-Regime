from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


BYBIT = "https://api.bybit.com"


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_series(values) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unit = "ms" if numeric.dropna().median() > 10_000_000_000 else "s"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "timestamp" not in df and "time" in df:
        df = df.rename(columns={"time": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        parsed = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if parsed.isna().mean() > 0.5:
            parsed = utc_series(df["timestamp"])
        df["timestamp"] = parsed
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=list(required)).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    for col in ["buy_volume", "open_interest", "funding_rate", "event_risk"]:
        if col not in df:
            df[col] = np.nan
    return df


def get_json(path: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            url = BYBIT + path + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "btc-regime-v1/1.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("retCode") != 0:
                raise RuntimeError(data.get("retMsg", "Bybit API error"))
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def download_klines(symbol: str, category: str, interval: str, days: int) -> pd.DataFrame:
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows: list[list[str]] = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        payload = get_json("/v5/market/kline", {
            "category": category, "symbol": symbol, "interval": interval,
            "start": start_ms, "end": cursor_end, "limit": 1000,
        })
        batch = payload["result"]["list"]
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x[0]) for x in batch)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.05)
    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    return normalize(pd.DataFrame(rows, columns=cols))


def download_open_interest(symbol: str, category: str, days: int) -> pd.Series:
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows = []
    cursor = None
    while True:
        params = {"category": category, "symbol": symbol, "intervalTime": "15min",
                  "startTime": start_ms, "endTime": end_ms, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        payload = get_json("/v5/market/open-interest", params)
        result = payload["result"]
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
        time.sleep(0.05)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(int(x["timestamp"]), unit="ms", utc=True): float(x["openInterest"]) for x in rows})
    return s.sort_index()


def download_funding(symbol: str, category: str, days: int) -> pd.Series:
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        payload = get_json("/v5/market/funding/history", {
            "category": category, "symbol": symbol, "startTime": start_ms,
            "endTime": cursor_end, "limit": 200,
        })
        batch = payload["result"].get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x["fundingRateTimestamp"]) for x in batch)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.05)
    return pd.Series({pd.to_datetime(int(x["fundingRateTimestamp"]), unit="ms", utc=True): float(x["fundingRate"]) for x in rows}).sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in ["turnover", "buy_volume"]:
        if c in df:
            agg[c] = "sum"
    for c in ["open_interest", "funding_rate", "event_risk"]:
        if c in df:
            agg[c] = "last" if c != "event_risk" else "max"
    return df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open", "high", "low", "close"])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for n in [20, 50, 200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False, min_periods=n).mean()
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["bb_mid"] = x.close.rolling(20).mean()
    std = x.close.rolling(20).std(ddof=0)
    x["bb_upper"], x["bb_lower"] = x.bb_mid + 2*std, x.bb_mid - 2*std
    x["bb_width"] = (x.bb_upper-x.bb_lower)/x.bb_mid
    x["vol_ma20"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume/x.vol_ma20.replace(0, np.nan)
    x["atr_pct"] = x.atr14/x.close
    return x


def confirmed_swings(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Swing becomes visible only `right` bars later; values are shifted to confirmation time."""
    x = df.copy()
    w = left + right + 1
    raw_hi = x.high.eq(x.high.rolling(w, center=True).max())
    raw_lo = x.low.eq(x.low.rolling(w, center=True).min())
    x["confirmed_swing_high"] = x.high.where(raw_hi).shift(right)
    x["confirmed_swing_low"] = x.low.where(raw_lo).shift(right)
    x["last_swing_high"] = x.confirmed_swing_high.ffill()
    x["last_swing_low"] = x.confirmed_swing_low.ffill()
    x["prev_swing_high"] = x.confirmed_swing_high.ffill().where(x.confirmed_swing_high.notna()).ffill().shift(1)
    x["prev_swing_low"] = x.confirmed_swing_low.ffill().where(x.confirmed_swing_low.notna()).ffill().shift(1)
    return x


def structure_state(df4: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    x = confirmed_swings(add_indicators(df4), cfg["swing_left"], cfg["swing_right"])
    states, anchors = [], []
    state, anchor, age = "NOT_TESTED", np.nan, 0
    for _, r in x.iterrows():
        atr = r.atr14
        if not np.isfinite(atr) or not np.isfinite(r.bb_mid):
            states.append(state); anchors.append(anchor); continue
        tol = cfg["retest_tolerance_atr"] * atr
        margin = cfg["breakout_atr"] * atr
        bull_break = r.close > r.bb_mid + margin and r.close > r.ema20 and r.vol_ratio >= 1.0
        bear_break = r.close < r.bb_mid - margin and r.close < r.ema20 and r.vol_ratio >= 1.0
        if bull_break and state not in {"BULL_BREAK_CONFIRMED", "BULL_RETESTING", "BULL_RETEST_HELD"}:
            state, anchor, age = "BULL_BREAK_CONFIRMED", r.bb_mid, 0
        elif bear_break and state not in {"BEAR_BREAK_CONFIRMED", "BEAR_RETESTING", "BEAR_RETEST_HELD"}:
            state, anchor, age = "BEAR_BREAK_CONFIRMED", r.bb_mid, 0
        elif state.startswith("BULL_"):
            age += 1
            if r.close < anchor - tol:
                state = "BULL_RETEST_FAILED"
            elif r.low <= anchor + tol:
                state = "BULL_RETEST_HELD" if r.close >= anchor and r.close >= r.open else "BULL_RETESTING"
            elif age > cfg["retest_window_4h_bars"]:
                state = "BULL_TREND_CONTINUATION"
        elif state.startswith("BEAR_"):
            age += 1
            if r.close > anchor + tol:
                state = "BEAR_RETEST_FAILED"
            elif r.high >= anchor - tol:
                state = "BEAR_RETEST_HELD" if r.close <= anchor and r.close <= r.open else "BEAR_RETESTING"
            elif age > cfg["retest_window_4h_bars"]:
                state = "BEAR_TREND_CONTINUATION"
        states.append(state); anchors.append(anchor)
    x["structure_event"] = states
    x["structure_anchor"] = anchors
    return x


def clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def build_scores(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    one = add_indicators(resample_ohlcv(base, "1h")).add_prefix("h1_")
    four = structure_state(resample_ohlcv(base, "4h"), cfg["structure"]).add_prefix("h4_")
    x = add_indicators(base).join(one.reindex(base.index, method="ffill")).join(four.reindex(base.index, method="ffill"))

    direction = pd.Series(0.0, index=x.index)
    direction += np.where(x.h4_close > x.h4_bb_mid, 18, -18)
    direction += np.where(x.h4_ema20 > x.h4_ema50, 15, -15)
    direction += np.where(x.h1_close > x.h1_ema20, 10, -10)
    direction += np.where(x.h1_ema20 > x.h1_ema50, 10, -10)
    direction += np.where(x.h4_bb_mid.diff().fillna(0) > 0, 7, -7)
    bull_state = x.h4_structure_event.str.contains("BULL_RETEST_HELD|BULL_TREND_CONTINUATION", regex=True, na=False)
    bear_state = x.h4_structure_event.str.contains("BEAR_RETEST_HELD|BEAR_TREND_CONTINUATION", regex=True, na=False)
    direction += np.where(bull_state, 20, 0)
    direction -= np.where(bear_state, 20, 0)
    oi_change = x.open_interest.pct_change(16).replace([np.inf, -np.inf], np.nan)
    direction += np.where((x.close.pct_change(16)>0) & (oi_change>0), 6, 0)
    direction -= np.where((x.close.pct_change(16)<0) & (oi_change>0), 6, 0)
    if x.buy_volume.notna().any():
        imbalance = (2*x.buy_volume/x.volume.replace(0, np.nan)-1).rolling(4).mean()
        direction += np.where(imbalance > .08, 6, np.where(imbalance < -.08, -6, 0))

    compression = (x.h4_bb_width / x.h4_bb_width.rolling(30).median()).clip(0, 2)
    expansion = (x.h4_atr_pct / x.h4_atr_pct.rolling(30).median()).clip(0, 2)
    opportunity = 28 + 18*(1-compression.clip(0,1)) + 16*expansion.clip(0,1.5)
    opportunity += 15*np.minimum(x.h4_vol_ratio.fillna(0), 2)/2
    opportunity += np.where(x.h4_structure_event.str.contains("BREAK_CONFIRMED|RETEST", regex=True, na=False), 18, 0)

    components = pd.DataFrame({
        "h4_mid": x.h4_bb_mid.notna(), "h4_ema": x.h4_ema50.notna(),
        "h1_ema": x.h1_ema50.notna(), "volume": x.h4_vol_ratio.notna(),
        "oi": x.open_interest.notna(), "funding": x.funding_rate.notna(),
    })
    weights = pd.Series({"h4_mid": 25, "h4_ema": 20, "h1_ema": 20, "volume": 20, "oi": 10, "funding": 5})
    confidence = components.mul(weights).sum(axis=1)
    event_risk = x.event_risk.fillna(0).clip(0, 100)
    funding_heat = (x.funding_rate.abs()/0.001).clip(0, 1).fillna(0)*20
    event_risk = np.maximum(event_risk, funding_heat)

    x["direction_score"] = direction.clip(-100, 100)
    x["opportunity_score"] = pd.Series(opportunity, index=x.index).clip(0, 100)
    x["confidence_score"] = confidence.clip(0, 100)
    x["event_risk_score"] = pd.Series(event_risk, index=x.index).clip(0, 100)
    x["market_state"] = np.select([
        x.h4_structure_event.str.contains("BULL_RETEST_HELD|BULL_TREND", regex=True, na=False),
        x.h4_structure_event.str.contains("BEAR_RETEST_HELD|BEAR_TREND", regex=True, na=False),
        x.h4_structure_event.str.contains("BULL_BREAK", regex=True, na=False),
        x.h4_structure_event.str.contains("BEAR_BREAK", regex=True, na=False),
        x.h4_bb_width < x.h4_bb_width.rolling(30).quantile(.3),
    ], ["BULLISH_CONFIRMED", "BEARISH_CONFIRMED", "BULLISH_DEVELOPING", "BEARISH_DEVELOPING", "BREAKOUT_PENDING"], default="RANGE")
    t = cfg["thresholds"]
    x["permission"] = "NO_TRADE"
    x.loc[x.event_risk_score >= t["max_event_risk"], "permission"] = "EVENT_LOCKOUT"
    safe = x.event_risk_score < t["max_event_risk"]
    ready = safe & (x.opportunity_score >= t["min_opportunity"]) & (x.confidence_score >= t["min_confidence"])
    x.loc[ready & (x.direction_score >= t["long_direction"]), "permission"] = "LONG_ONLY"
    x.loc[ready & (x.direction_score <= t["short_direction"]), "permission"] = "SHORT_ONLY"
    x.loc[ready & x.direction_score.abs().lt(25), "permission"] = "BOTH_ALLOWED"
    pending = safe & x.market_state.eq("BREAKOUT_PENDING") & x.direction_score.abs().lt(t["long_direction"])
    x.loc[pending, "permission"] = "WAIT_FOR_BREAKOUT"
    return x


def validate_forward(x: pd.DataFrame, horizons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = x[["close", "permission", "market_state", "direction_score", "opportunity_score", "confidence_score"]].copy()
    bars_per_hour = 4
    for h in horizons:
        n = h * bars_per_hour
        future_high = x.high.shift(-1).rolling(n).max().shift(-(n-1))
        future_low = x.low.shift(-1).rolling(n).min().shift(-(n-1))
        future_close = x.close.shift(-n)
        out[f"mfe_{h}h"] = future_high/out.close-1
        out[f"mae_{h}h"] = future_low/out.close-1
        out[f"return_{h}h"] = future_close/out.close-1
        out[f"aligned_{h}h"] = np.where(out.permission.eq("SHORT_ONLY"), -out[f"return_{h}h"], out[f"return_{h}h"])
    rows = []
    for permission, g in out.groupby("permission"):
        row = {"permission": permission, "observations": len(g)}
        for h in horizons:
            aligned = g[f"aligned_{h}h"].dropna()
            row[f"avg_aligned_{h}h"] = aligned.mean()
            row[f"median_aligned_{h}h"] = aligned.median()
            row[f"direction_hit_{h}h"] = (aligned > 0).mean()
        rows.append(row)
    return out, pd.DataFrame(rows)


def latest_payload(x: pd.DataFrame) -> dict:
    r = x.iloc[-1]
    return {
        "timestamp_utc": x.index[-1].isoformat(), "symbol": "BTCUSDT",
        "market_state": r.market_state, "permission": r.permission,
        "direction_score": round(float(r.direction_score), 1),
        "opportunity_score": round(float(r.opportunity_score), 1),
        "confidence_score": round(float(r.confidence_score), 1),
        "event_risk_score": round(float(r.event_risk_score), 1),
        "price": round(float(r.close), 2),
        "h4_structure_event": str(r.h4_structure_event),
        "h4_structure_anchor": None if pd.isna(r.h4_structure_anchor) else round(float(r.h4_structure_anchor), 2),
        "note": "Environment filter only; not an entry signal.",
    }


def run_pipeline(df: pd.DataFrame, cfg: dict) -> dict:
    outdir = Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    scores = build_scores(df, cfg)
    usable = scores.dropna(subset=["h4_bb_mid", "h1_ema50"]).copy()
    forward, summary = validate_forward(usable, cfg["backtest"]["horizons_hours"])
    event_mask = usable.permission.ne(usable.permission.shift()) | usable.market_state.ne(usable.market_state.shift())
    save_cols = ["open", "high", "low", "close", "volume", "open_interest", "funding_rate",
                 "h4_structure_event", "h4_structure_anchor", "market_state", "permission",
                 "direction_score", "opportunity_score", "confidence_score", "event_risk_score"]
    usable[save_cols].to_csv(outdir/"regime_history.csv")
    usable.loc[event_mask, save_cols].to_csv(outdir/"regime_events.csv")
    forward.to_csv(outdir/"forward_validation.csv")
    summary.to_csv(outdir/"validation_summary.csv", index=False)
    latest = latest_payload(usable)
    with open(outdir/"latest_regime.json", "w", encoding="utf-8") as f:
        json.dump(latest, f, ensure_ascii=False, indent=2)
    return latest


def synthetic_data(n=3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    regime = np.repeat([.00015, -.00012, 0, .00025], math.ceil(n/4))[:n]
    ret = regime + rng.normal(0, .0022, n)
    close = 60000*np.exp(np.cumsum(ret)); open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(.0005, .0035, n)*close
    volume = rng.lognormal(7, .55, n)*(1+np.abs(ret)*80)
    return normalize(pd.DataFrame({"timestamp": idx, "open": open_, "high": np.maximum(open_, close)+spread,
        "low": np.minimum(open_, close)-spread, "close": close, "volume": volume,
        "buy_volume": volume*np.clip(.5+ret*25+rng.normal(0,.06,n), .05,.95),
        "open_interest": 1e9*np.exp(np.cumsum(rng.normal(0,.0008,n))), "funding_rate": rng.normal(0,.00008,n)}))


def main():
    ap = argparse.ArgumentParser(description="BTC market-regime radar V1")
    ap.add_argument("command", choices=["download", "run", "self-test"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/BTCUSDT_15m.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = Path(args.input); path.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "download":
        df = download_klines(cfg["symbol"], cfg["category"], cfg["base_interval"], cfg["history_days"])
        try:
            oi = download_open_interest(cfg["symbol"], cfg["category"], cfg["history_days"])
            df["open_interest"] = oi.reindex(df.index, method="ffill")
        except Exception as e:
            print(f"OI download warning: {e}")
        try:
            funding = download_funding(cfg["symbol"], cfg["category"], cfg["history_days"])
            df["funding_rate"] = funding.reindex(df.index, method="ffill")
        except Exception as e:
            print(f"Funding download warning: {e}")
        df.reset_index().to_csv(path, index=False)
        print(f"Saved {len(df)} bars to {path}")
        return
    df = synthetic_data() if args.command == "self-test" else normalize(pd.read_csv(path))
    latest = run_pipeline(df, cfg)
    print(json.dumps(latest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
