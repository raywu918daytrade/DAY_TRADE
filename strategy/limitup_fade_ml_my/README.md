# limitup_fade_ml

漲停隔日開高 → 首根 **3 分 K** 下跌 → LightGBM **三分類**過濾做空。

## 規則（硬過濾）

- 前日：日報酬 ≥ 9.5%、陽線、body ≥ 50%、上影 ≤ 20%
- 今日：open > 昨收
- 觸發：`m3_std` @ 09:03，close < open
- 進場：09:03 該棒收；標籤／回測出場：做空 Triple Barrier ±3%，時間牆 13:25

## 模型

- 三分類：0=止損 / 1=震盪 / 2=止盈
- live 進場：`P(止盈) >= THRESHOLD`（預設 0.6）
- 模組：`strategy.limitup_fade_ml.down.live` → 策略名 `limitup_fade_ml_down`
- **注意**：現行 executor 只開多；本策略訊號供監控，真下空單另議

## 用法

```bash
# CLI
python -m strategy.limitup_fade_ml.train train --start_date 2026-06-01 --end_date 2026-07-31 --use_cache

# 或不帶參數：改 train.py __main__ 裡的 mode / start_date / end_date 後直接跑
python -m strategy.limitup_fade_ml.train

# 回測（規則 vs ML）
python -m strategy.limitup_fade_ml.run_backtest --start_date 2026-06-01 --end_date 2026-07-31 --use_cache
```

也可程式呼叫：`from strategy.limitup_fade_ml.train import main; main(mode="train", start_date="2024-01-01", end_date="2026-07-31")`
