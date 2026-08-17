# BTC Regime Radar V1

第一层BTC大环境模型。它不下单，只输出市场环境、方向、机会强度、置信度、事件风险和对第二层的交易许可。

## 当前范围

- 15m数据聚合为1H和4H，避免不同周期时间错位。
- 4H布林中轨、EMA、结构高低点、突破/回踩状态机。
- 成交量、主动买卖量、OI与资金费率确认。
- Direction / Opportunity / Confidence / Event Risk四项评分。
- LONG_ONLY / SHORT_ONLY / BOTH_ALLOWED / WAIT_FOR_BREAKOUT / EVENT_LOCKOUT / NO_TRADE许可。
- 无未来数据泄漏的前向表现评估。
- 宏观和突发新闻风险的标准CSV入口（V2再接实时来源）。

## 安装与运行

```bash
pip install -r requirements.txt
python btc_regime.py download
python btc_regime.py run
```

只做本地合成数据自检：

```bash
python btc_regime.py self-test
```

读取你自己的CSV：

```bash
python btc_regime.py run --input data/BTCUSDT_15m.csv
```

CSV至少包含：`timestamp,open,high,low,close,volume`。可选：`turnover,buy_volume,open_interest,funding_rate,event_risk`。

## 输出

- `output/latest_regime.json`：当前环境状态。
- `output/regime_history.csv`：每根15m的完整评分。
- `output/regime_events.csv`：许可或结构状态发生变化的事件。
- `output/forward_validation.csv`：不同观察期的前向涨跌空间。
- `output/validation_summary.csv`：不同许可状态的统计汇总。

## 重要说明

V1只验证盘面环境层。新闻与宏观事件暂时通过`event_risk`字段进入模型，避免在尚未证明技术状态有效前引入无法归因的复杂度。输出不是买卖建议，也不应直接连接下单接口。

