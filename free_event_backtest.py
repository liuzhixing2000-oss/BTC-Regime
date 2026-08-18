from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from btc_regime import build_scores, load_config, normalize


UA = {"User-Agent": "btc-regime-research/1.1 contact=personal-research"}
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict:
    return json.loads(fetch_text(url))


def clean_html(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def collect_bls_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Parse official annual BLS list pages; only CPI and Employment Situation."""
    rows = []
    months = "January February March April May June July August September October November December"
    date_re = re.compile(
        rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s+({months.replace(' ', '|')})\s+(\d{{1,2}}),?\s+(\d{{4}})\s+"
        r"(\d{1,2}:\d{2})\s+(AM|PM)\s+(Consumer Price Index|Employment Situation)", re.I
    )
    for year in range(start.year, end.year + 1):
        urls = [f"https://www.bls.gov/schedule/{year}/home.htm"]
        text = ""
        for url in urls:
            try:
                text = clean_html(fetch_text(url))
                break
            except Exception:
                continue
        for month, day, yr, clock, ampm, name in date_re.findall(text):
            local = pd.Timestamp(f"{month} {day} {yr} {clock} {ampm}", tz=NY)
            ts = local.tz_convert("UTC")
            if start <= ts <= end:
                rows.append({"timestamp": ts, "event_type": "CPI" if "Consumer" in name else "NFP",
                             "importance": 3, "source": "BLS", "source_quality": 1.0})
    return pd.DataFrame(rows)


def collect_fomc_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Use official statement links; scheduled decisions are normally released 14:00 ET."""
    pages = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
             "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"]
    dates = set()
    link_re = re.compile(r"monetary(20\d{6})a\.htm", re.I)
    for page in pages:
        try:
            html = fetch_text(page)
        except Exception:
            continue
        for ymd in link_re.findall(html):
            d = datetime.strptime(ymd, "%Y%m%d")
            local = pd.Timestamp(d.date(), tz=NY) + pd.Timedelta(hours=14)
            ts = local.tz_convert("UTC")
            if start <= ts <= end:
                dates.add(ts)
    return pd.DataFrame([{"timestamp": ts, "event_type": "FOMC", "importance": 3,
                          "source": "Federal Reserve", "source_quality": 0.95} for ts in sorted(dates)])


def collect_macro_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = [collect_bls_events(start, end), collect_fomc_events(start, end)]
    frames = [x for x in parts if not x.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "event_type", "importance", "source", "source_quality"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "event_type"]).sort_values("timestamp")


def alpha_time(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y%m%dT%H%M")


def collect_alpha_news(start: pd.Timestamp, end: pd.Timestamp, api_key: str, cache: Path) -> pd.DataFrame:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        old = pd.read_csv(cache)
        if not old.empty:
            old["timestamp"] = pd.to_datetime(old.timestamp, utc=True)
            if old.timestamp.min() <= start + pd.Timedelta(days=2) and old.timestamp.max() >= end - pd.Timedelta(days=2):
                return old
    rows, cursor = [], start.floor("D")
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=30), end)
        params = {
            "function": "NEWS_SENTIMENT", "tickers": "CRYPTO:BTC",
            "time_from": alpha_time(cursor), "time_to": alpha_time(chunk_end),
            "sort": "EARLIEST", "limit": 1000, "apikey": api_key,
        }
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)
        payload = fetch_json(url)
        if any(k in payload for k in ["Information", "Note", "Error Message"]):
            raise RuntimeError(payload.get("Information") or payload.get("Note") or payload.get("Error Message"))
        for item in payload.get("feed", []):
            published = item.get("time_published", "")
            try:
                ts = pd.to_datetime(published, format="%Y%m%dT%H%M%S", utc=True)
            except Exception:
                continue
            ticker_items = [t for t in item.get("ticker_sentiment", []) if t.get("ticker") == "CRYPTO:BTC"]
            relevance = max([float(t.get("relevance_score", 0)) for t in ticker_items] or [0.0])
            ticker_score = [float(t.get("ticker_sentiment_score", 0)) for t in ticker_items]
            sentiment = ticker_score[0] if ticker_score else float(item.get("overall_sentiment_score", 0) or 0)
            rows.append({"timestamp": ts, "title": item.get("title", ""), "url": item.get("url", ""),
                         "source": item.get("source", ""), "relevance": relevance,
                         "sentiment": sentiment, "provider": "Alpha Vantage"})
        cursor = chunk_end
        time.sleep(1.1)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "title", "url", "source", "relevance", "sentiment", "provider"])
    out = out.drop_duplicates(subset=["url"]).sort_values("timestamp")
    out.to_csv(cache, index=False)
    return out


def attach_events(scores: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    x = scores.copy()
    x["macro_lockout"] = False
    x["macro_event"] = ""
    for _, e in macro.iterrows():
        # 60m before through 30m after scheduled high-impact releases.
        mask = (x.index >= e.timestamp - pd.Timedelta(minutes=60)) & (x.index <= e.timestamp + pd.Timedelta(minutes=30))
        x.loc[mask, "macro_lockout"] = True
        x.loc[mask, "macro_event"] = e.event_type
    x["news_score_4h"] = 0.0
    x["news_count_4h"] = 0
    if not news.empty:
        n = news.set_index("timestamp").sort_index()
        weighted = n.sentiment * n.relevance
        # News is assigned to the first completed 15m bucket at/after publication.
        agg = pd.DataFrame({"weighted": weighted, "count": 1}).resample("15min", label="right", closed="right").sum()
        agg = agg.reindex(x.index, fill_value=0)
        agg["news_score_4h"] = agg.weighted.rolling(16, min_periods=1).sum() / np.sqrt(agg["count"].rolling(16, min_periods=1).sum().clip(lower=1))
        agg["news_count_4h"] = agg["count"].rolling(16, min_periods=1).sum()
        x["news_score_4h"] = agg.news_score_4h.fillna(0)
        x["news_count_4h"] = agg.news_count_4h.fillna(0).astype(int)
    x["strong_news"] = x.news_count_4h.ge(2) & x.news_score_4h.abs().ge(0.20)
    sign = np.sign(x.direction_score)
    x["news_conflict"] = x.strong_news & (np.sign(x.news_score_4h) != sign) & sign.ne(0)
    return x


def forward_stats(x: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = cfg["backtest"]["horizons_hours"]
    cooldown = int(cfg["backtest"].get("signal_cooldown_hours", 4) * 4)
    candidate = x.permission.isin(["LONG_ONLY", "SHORT_ONLY"])
    trigger = candidate & (~candidate.shift(1, fill_value=False) | x.permission.ne(x.permission.shift()))
    # Also sample persistent regimes after cooldown without overlapping every 15m bar.
    indices = []
    last = -10**9
    for i, ok in enumerate(candidate.to_numpy()):
        if ok and (trigger.iloc[i] or i-last >= cooldown):
            indices.append(i); last = i
    base = x.iloc[indices].copy()
    policies = {
        "TECH_ONLY": pd.Series(True, index=base.index),
        "TECH_PLUS_MACRO": ~base.macro_lockout,
        "TECH_PLUS_MACRO_NEWS": ~base.macro_lockout & ~base.news_conflict,
    }
    event_rows, summary_rows = [], []
    for policy, allowed in policies.items():
        g = base.loc[allowed].copy()
        for ts, r in g.iterrows():
            loc = x.index.get_loc(ts)
            side = 1 if r.permission == "LONG_ONLY" else -1
            row = {"policy": policy, "timestamp": ts, "side": "LONG" if side == 1 else "SHORT",
                   "price": r.close, "macro_event": r.macro_event, "news_score_4h": r.news_score_4h}
            for h in horizons:
                future = x.iloc[loc+1:loc+1+h*4]
                if len(future) < h*4:
                    row[f"return_{h}h"] = np.nan; row[f"mfe_{h}h"] = np.nan; row[f"mae_{h}h"] = np.nan
                else:
                    row[f"return_{h}h"] = side*(future.close.iloc[-1]/r.close-1)
                    if side == 1:
                        row[f"mfe_{h}h"] = future.high.max()/r.close-1
                        row[f"mae_{h}h"] = future.low.min()/r.close-1
                    else:
                        row[f"mfe_{h}h"] = 1-future.low.min()/r.close
                        row[f"mae_{h}h"] = 1-future.high.max()/r.close
            event_rows.append(row)
        events = pd.DataFrame([z for z in event_rows if z["policy"] == policy])
        result = {"policy": policy, "signals": len(events), "removed_vs_tech": len(base)-len(events)}
        for h in horizons:
            s = events[f"return_{h}h"].dropna() if not events.empty else pd.Series(dtype=float)
            result[f"avg_return_{h}h"] = s.mean()
            result[f"hit_rate_{h}h"] = (s > 0).mean() if len(s) else np.nan
            result[f"avg_mae_{h}h"] = events[f"mae_{h}h"].mean() if not events.empty else np.nan
        summary_rows.append(result)
    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def event_studies(prices: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_rows = []
    for _, e in macro.iterrows():
        pos = prices.index.searchsorted(e.timestamp)
        if pos >= len(prices): continue
        p = prices.close.iloc[pos]
        row = dict(e)
        for h in [1,4,8,24]:
            q = pos+h*4
            row[f"abs_return_{h}h"] = abs(prices.close.iloc[q]/p-1) if q < len(prices) else np.nan
        macro_rows.append(row)
    news_rows = []
    for _, e in news.loc[(news.relevance >= .5) & (news.sentiment.abs() >= .15)].iterrows():
        pos = prices.index.searchsorted(e.timestamp)
        if pos >= len(prices): continue
        p = prices.close.iloc[pos]; side = np.sign(e.sentiment)
        row = {"timestamp": e.timestamp, "source": e.source, "title": e.title,
               "relevance": e.relevance, "sentiment": e.sentiment}
        for h in [1,4,8,24]:
            q = pos+h*4
            row[f"aligned_return_{h}h"] = side*(prices.close.iloc[q]/p-1) if q < len(prices) else np.nan
        news_rows.append(row)
    return pd.DataFrame(macro_rows), pd.DataFrame(news_rows)


def make_report(summary: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame, news_study: pd.DataFrame) -> str:
    lines = ["BTC FREE EVENT ABLATION REPORT", "="*40,
             f"Macro events: {len(macro)}", f"News articles: {len(news)}",
             f"High-relevance news study sample: {len(news_study)}", ""]
    if news.empty:
        lines += ["NEWS COVERAGE: FAILED/EMPTY", "Do not treat TECH_PLUS_MACRO_NEWS as validated.", ""]
    else:
        span = (news.timestamp.max()-news.timestamp.min()).days+1
        density = len(news)/max(span,1)
        lines += [f"News coverage: {news.timestamp.min()} to {news.timestamp.max()}",
                  f"News density: {density:.2f} articles/day",
                  "Coverage warning: Alpha Vantage is not a complete archive; conclusions require adequate monthly density.", ""]
    lines += [summary.to_string(index=False), "",
              "Interpretation rule:",
              "Keep a filter only if it improves out-of-sample aligned return or hit rate AND reduces adverse excursion",
              "without removing an impractical share of signals. This report is an event study, not an entry/exit backtest."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/BTCUSDT_15m.csv")
    ap.add_argument("--output", default="output_free_events")
    args = ap.parse_args()
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing Railway variable ALPHA_VANTAGE_API_KEY")
    cfg = load_config(args.config)
    data = normalize(pd.read_csv(args.input))
    start, end = data.index.min(), data.index.max()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    macro = collect_macro_events(start, end)
    news = collect_alpha_news(start, end, key, out/"alpha_btc_news_cache.csv")
    scores = attach_events(build_scores(data, cfg), macro, news)
    events, summary = forward_stats(scores.dropna(subset=["h4_bb_mid", "h1_ema50"]), cfg)
    macro_study, news_study = event_studies(data, macro, news)
    macro.to_csv(out/"free_macro_events.csv", index=False)
    news.to_csv(out/"free_news_events.csv", index=False)
    events.to_csv(out/"ablation_signal_events.csv", index=False)
    summary.to_csv(out/"ablation_summary.csv", index=False)
    macro_study.to_csv(out/"macro_event_study.csv", index=False)
    news_study.to_csv(out/"news_reliability.csv", index=False)
    report = make_report(summary, macro, news, news_study)
    (out/"validation_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
