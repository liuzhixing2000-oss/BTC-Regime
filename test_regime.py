from pathlib import Path
import tempfile

from btc_regime import build_scores, load_config, normalize, run_pipeline, synthetic_data


def main():
    cfg = load_config(Path(__file__).with_name("config.yaml"))
    data = synthetic_data(3000)
    scores = build_scores(data, cfg)
    assert scores.index.is_monotonic_increasing
    assert scores.direction_score.dropna().between(-100, 100).all()
    assert scores.opportunity_score.dropna().between(0, 100).all()
    assert set(scores.permission.dropna().unique()) <= {
        "LONG_ONLY", "SHORT_ONLY", "BOTH_ALLOWED", "WAIT_FOR_BREAKOUT", "EVENT_LOCKOUT", "NO_TRADE"
    }
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(cfg)
        cfg["output_dir"] = tmp
        latest = run_pipeline(data, cfg)
        assert latest["permission"] in set(scores.permission)
        expected = {"latest_regime.json", "regime_history.csv", "regime_events.csv",
                    "forward_validation.csv", "validation_summary.csv"}
        assert expected <= {p.name for p in Path(tmp).iterdir()}
    print("All BTC Regime V1 checks passed.")


if __name__ == "__main__":
    main()
