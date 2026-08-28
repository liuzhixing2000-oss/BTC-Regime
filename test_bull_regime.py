import unittest

import numpy as np
import pandas as pd

from bull_regime_backtest import Config, confirmed_swings, prepare


def sample_bars(n=1500):
    idx = pd.date_range("2020-01-01 04:00", periods=n, freq="4h", tz="UTC")
    close = 10_000 * np.exp(np.linspace(0, 0.8, n) + 0.02 * np.sin(np.arange(n) / 8))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.full(n, 100.0),
    }, index=idx)


class BullRegimeTests(unittest.TestCase):
    def test_prefix_values_do_not_change_when_future_is_appended(self):
        bars = sample_bars()
        short = prepare(bars.iloc[:1200], Config())
        full = prepare(bars, Config()).iloc[:1200]
        cols = ["regime_score", "ema20", "ema50", "atr", "swing_low", "hh_hl"]
        pd.testing.assert_frame_equal(short[cols], full[cols])

    def test_swing_is_visible_only_after_confirmation_delay(self):
        bars = sample_bars(20)
        bars.iloc[8, bars.columns.get_loc("low")] *= 0.8
        swings = confirmed_swings(bars, left=2, right=2)
        self.assertFalse(np.isfinite(swings.swing_low.iloc[9]))
        self.assertTrue(np.isfinite(swings.swing_low.iloc[10]))


if __name__ == "__main__":
    unittest.main()
