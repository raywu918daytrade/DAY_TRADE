# limitup_fade_ml

漲停隔日開高 → 首根 **3 分 K** 下跌（Stage1候選）→ **09:10 延續確認**（Stage2）→
LightGBM **三分類**過濾做空。

2026-08-04 全新實作，跟舊版 `strategy/limitup_fade_ml.zip` 無關（未參考其內容）。

**2026-08-04 二次改版（兩階段觸發）**：09:03 立即進場版本已完整驗證過，結論是
邊際不穩定不建議上線（純規則扣成本五年不賺、ML門檻2024/2025 walk-forward互相
矛盾，見下方「已驗證結論」）。改成概念類似 `strategy/orb` 的兩階段設計：09:03
先當候選觸發，延到 09:10 看價格有沒有延續下跌（比09:03更低）才真的進場，用意
是過濾掉「09:03瞬間雜訊下跌、之後馬上被拉回」的假突破。

**2026-08-04 三次改版（ATR動態停利/停損）**：兩階段版本規則基準驗證發現表現
反而比09:03立即進場版差（平均報酬+0.337%→+0.149%），研判固定±3%對這批波動
遠高於一般股票的漲停股不合理——同一個3%對低波動股太寬、對高波動股太窄。改成
比照 `strategy/orb/features.py` 的 `day_atr` 寫法：用個股自身日K ATR14 當停利/
停損距離（TP=SL=1×day_atr，對稱），取代全市場統一的固定%。TW股市09:00~13:30，
持倉09:03/09:10~13:25幾乎涵蓋整個交易日，日ATR的波動尺度跟實際持倉時長對得上。

## 規則（硬過濾）

**Stage 1（候選，@09:03）**
- 前日：日報酬 ≥ 9.5%、陽線、body ≥ 5%、上影 ≤ 20%
- 今日：open > 昨收
- 觸發：`m3_std` @ 09:03（覆蓋 09:00:00~09:02:59），close < open
- 同時算日K ATR14（`day_atr`），算出 `tp_price = m3_close×(1-day_atr)`、
  `sl_price = m3_close×(1+day_atr)`（ATR14不足14天歷史的新掛牌股直接捨棄）

**Stage 2（延續確認，@09:10）**
- 09:09 那根 `db/m1`（涵蓋 09:09:00~09:09:59，即「09:10 當下最新一根已收完的
  分K」）收盤價，要比 Stage1 的 m3 收盤價更低才算延續下跌，確認才進場
  （`entry_price` = 這個確認價）；沒有延續下跌就整筆事件捨棄（視為假突破/已反彈）
  ——只決定要不要進場，不影響 `tp_price`/`sl_price`（那是從 m3_close 算的，
  維持不變，避免「基準價下移、停利更難達成」的問題）
- 標籤／回測出場：做空 Triple Barrier（動態 TP=SL=1×day_atr），時間牆 13:25，
  從 Stage2 進場時間算起

## 模型

- 三分類：0=止損 / 1=震盪 / 2=止盈
- live 進場：`P(止盈) >= THRESHOLD`（預設 0.6）
- 模組：`strategy.limitup_fade_ml.down.live` → 策略名 `limitup_fade_ml_down`
- 只做空單，沒有 `up/` 變體

## 檔案

- `config.py` — 策略常數（規則門檻、Stage1/Stage2 時間點、Triple Barrier、模型/回測參數）
- `dataset.py` — 候選挖掘（`build_gap_candidates`）、Stage1觸發判定（`attach_m3_trigger`）、
  Stage2延續確認（`attach_confirm`）、空單 Triple Barrier 標籤（`short_triple_barrier_label`）、
  端到端 `build_events()`
- `features.py` — `FEATURES` 清單 + 欄位檢查
- `train.py` — LightGBM 訓練／評估／信心度掃描（CLI）
- `predict.py` — `predict()`（測試集事件級機率，供回測用）／`predict_live()`（兩階段：
  09:03 記錄Stage1候選到待確認快取、09:10 讀快取做Stage2確認+出訊號，其餘分鐘不動作）
- `run_backtest.py` — 空單事件級回測（自寫出場模擬，`backtest/intraday_platform.py`
  的共用引擎是純多單設計，不能直接套用，只重用其 `print_trades()` 欄位格式）
- `down/live.py` — 即時交易介面，`DIRECTIONS={"down"}`
- `experiments/verify_rule_baseline.py` — 純規則（不套 ML）基準驗證，訓練前先跑

## 候選股票母體

`dataset.py::build_events()` 一開始就把日K限定在 `finmind.tick_universe.load_tick_universe()`
（固定 400 檔 + 0050）。**2026-08-04 決定不掃全市場**：`tick_universe` 是唯一每天被
`scripts/update_daily.py` 主動更新/校正的股票池，池外股票的 `db/adjustment_day` 資料沒有
保障（曾經全市場掃描時發現池外股票缺天數/異常值，例如 2832 完全不在 `db/d1` 裡、
3374/6142/5488/6934 在 `db/adjustment_day` 有整天缺K棒，`shift(1)` 算日報酬率時誤把
兩天漲幅當一天，冒出假的漲停事件）。事件稀疏是另一個問題（放寬規則門檻/拉長歷史區間），
不該用「掃描資料源不主動維護的股票」解決。

## 已驗證結論（09:03 立即進場版本，改版前）

**09:03立即進場、固定±3%版本**：純規則基準扣估計成本後，2022~2026 五年沒有一年淨賺
（最好的2026年迄今也只是打平）；ML門檻在2024/2025 walk-forward驗證互相矛盾（2024年
門檻≥0.6報酬+11.49%但0.5-0.6區間-18.95%；2025年反過來）；加特徵後連信心度分桶的單調
梯度都消失。結論是這版邊際不穩定，不建議上線。

**兩階段、固定±3%版本**：純規則基準平均報酬 +0.149%（比09:03立即進場版的+0.337%更差），
研判是固定±3%對這批高波動股不合理，才改成本文件描述的 ATR 動態版本——**ATR動態版本
目前還沒有跑過同一輪完整驗證（純規則基準/信心度分桶/walk-forward/跟limitup_fade_ml_my
的公平比較），不要假設它就比較好，要照下面「用法」重新跑一次**。

## 已知限制

- `prev_day_ret` 目前沒有合理性上限/交易日連續性檢查，即使限定在 `tick_universe` 內，
  理論上還是可能出現類似「單日缺K棒導致跨天誤算」的情況（機率低很多，因為池內股票每天
  都被主動更新），出現離群值時要先懷疑這個問題，不要當成真事件。
- 沒有放空可行性檢查（平盤下不得放空、融券額度/收費），回測跟即時訊號都沒有處理這塊，
  實際下單前需要額外驗證。
- 現行 `main/live_trader.py` 的 executor 只支援開多單，本策略訊號目前僅供監控，
  尚未加進 `.env` 的 `STRATEGY_MODULES`。
- Stage2 延續確認會篩掉一部分 Stage1 候選（已反彈的），樣本量會比 09:03 立即進場版本
  （2,176筆）更少，重新驗證時要先看夠不夠訓練。

## 用法

```bash
# 1. 先驗證純規則基準（樣本量/勝率），訓練前先確認值不值得往下做
python -m strategy.limitup_fade_ml.experiments.verify_rule_baseline --start_date 2022-01-01

# 2. 訓練
python -m strategy.limitup_fade_ml.train train --start_date 2022-01-01

# 3. 信心度門檻掃描，決定 THRESHOLD
python -m strategy.limitup_fade_ml.train confidence --use_cache --start_date 2022-01-01

# 4. 回測
python strategy/limitup_fade_ml/run_backtest.py
```
