# BTC Bull Regime V1 — initial validation

## Verdict: RESEARCH_ONLY / current specification fails holdout

This first pass evaluates two long-only 4H entry families under a daily/weekly bull-regime filter. It uses closed candles, enters at the next 4H open, resolves same-bar ambiguity adversely, and includes 5.5 bps taker fees plus 2 bps slippage per side.

Data: Binance BTCUSDT spot 4H from 2017-08-17 through 2026-08-28. Spot is used as a long-history price proxy in this pass, so perpetual funding is not yet included.

Chronological split:

- Train: 2017–2021
- Validation: 2022–2024
- Untouched holdout: 2025–2026-08-28

## Main finding

The attractive full-history results are not stable out of sample. The best-looking full-sample variant, `PULLBACK_RECLAIM + E_HALF_2R_REGIME` with minimum regime score 3, produced:

| Split | Trades | Avg net R | Win rate | Profit factor | Risk-scaled return | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Full | 30 | +2.018 | 30.0% | 3.77 | +64.6% | -8.3% |
| Train 2017–2021 | 13 | +2.550 | 15.4% | 3.83 | +29.0% | -8.3% |
| Validation 2022–2024 | 10 | +3.197 | 50.0% | 7.75 | +26.7% | -2.1% |
| Holdout 2025–now | 7 | **-0.655** | 28.6% | **0.15** | **-4.5%** | -4.9% |

The full-sample result is dominated by a few very large trend captures, including approximately +44R in 2020–2021 and +22R in 2023–2024. This is a valid trend-following payoff shape, but the edge did not persist in the holdout.

The strongest holdout result with at least five trades was `PULLBACK_RECLAIM + fixed 2R`, minimum score 3:

- 14 trades
- average net result: -0.016R
- profit factor: 0.98
- risk-scaled return: -0.37%
- maximum drawdown: -5.57%

That is effectively break-even before funding and cannot be called an edge.

## Benchmark context

During the 2025–2026 holdout:

| Benchmark | Return | Max drawdown | Exposure |
|---|---:|---:|---:|
| Buy and hold | -14.2% | -53.4% | 100% |
| Close above daily SMA200 | -2.3% | -33.3% | 45.7% |
| 4H EMA20 above EMA50 | +20.9% | -27.5% | 48.3% |

The active pullback model controlled drawdown much better than the unlevered benchmarks because it risked only 0.5–1.0% per trade, but it did not outperform the simple 4H EMA trend benchmark on return. These returns are not directly comparable without volatility normalization, so they are context rather than proof.

## Current state

At the latest completed data point (2026-08-28 04:00 UTC), the regime score is 3/4 (`CONFIRMED_BULL`). Neither entry family has a valid signal: the model classifies the environment as bullish but does not currently authorize chasing.

## Interpretation

The regime filter appears useful for reducing exposure and drawdown, but the specified 4H reclaim/retest entry rules are not robust enough. Raising the regime threshold from 2 to 4 does not repair the holdout. The runner-style exits amplify rare historical winners but are especially weak in the current holdout.

Do not connect this version to alerts or live trading.

## Next defensible experiment

Keep the regime layer, but replace the strict HH/HL plus previous-bar reclaim trigger with a small, predeclared sensitivity grid:

1. shallow versus deep pullback location;
2. one-bar reclaim versus two-bar stabilization;
3. volatility-normalized trend strength;
4. fixed 2R as the control exit, not a runner exit;
5. locked holdout retained unchanged.

If none of those nearby variants becomes positive across both validation and holdout with adequate trade count, stop the project rather than adding OI, funding, or machine learning.

## V1.1 predeclared sensitivity result

The follow-up tested 162 nearby combinations using only:

- shallow, deep, or either pullback location;
- one-bar reclaim or two-bar stabilization;
- EMA20–EMA50 separation of 0, 0.5, or 1.0 ATR;
- retest tolerance of 0.15, 0.30, or 0.50 ATR;
- minimum regime score 2, 3, or 4;
- fixed 2R exit for every variant.

The ten variants selected solely by 2017–2021 training expectancy did not persist. Nine of the ten were negative in validation; the only approximately flat validation variants were also flat or negative in holdout. The training winner fell from +0.681R per trade to -0.173R in validation and -0.174R in holdout.

Across the 77 variants with minimally adequate trade counts, only 15.6% were positive in holdout. Nine happened to be positive in both validation and holdout, but most had negative training performance and therefore were not legitimate ex-ante selections.

One broad variant was weakly positive in all three periods: `ANY + TWO_BAR_STABLE + no minimum EMA spread + 0.50 ATR tolerance + regime score >=2`. After excluding the still-open end-of-data trade, its approximate profile is:

| Split | Closed trades | Avg net R | Profit factor | Bootstrap 95% interval for mean R |
|---|---:|---:|---:|---:|
| Train | 42 | +0.156 | 1.25 | -0.273 to +0.587 |
| Validation | 47 | +0.085 | 1.13 | -0.300 to +0.531 |
| Holdout | 19 | about +0.027 | about 1.04 | includes zero by a wide margin |

This is not evidence of a tradable edge. Its confidence intervals are wide, profit factors are close to 1, and 2026 has only one completed trade, a loss of approximately -1.05R. The V1.1 verdict remains `RESEARCH_ONLY`, leaning `FAIL` for this entry family.
