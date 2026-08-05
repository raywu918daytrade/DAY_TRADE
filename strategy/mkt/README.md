# mkt 策略（個股 vs 大盤0050 相對強弱 / RandomForest / XGBoost / LightGBM）

## 核心假設

當大盤上漲、個股沒跟上，個股有機會補漲往大盤靠近（`ret_vs_idx`：個股從今日
開盤累積到現在的報酬率，減去0050同期的報酬率）。3分類標籤（跌=0/平=1/漲=2，
`config.py` 的 triple barrier：`TP_PCT=SL_PCT=3%`、`HOLD_BARS=30`），但策略
本身**只做多**，只在乎「漲」（class=2）這個訊號的precision，其他指標只是
輔助參考。

## 檔案結構

| 檔案 | 內容 |
|---|---|
| `config.py` | 交易相關設定（`TP_PCT`/`SL_PCT`/`HOLD_BARS`、`SESSION_START/END`、`MODEL_TYPE`、`THRESHOLD`、`ATR5_FILTER_THRESHOLD`） |
| `features.py` | 特徵工程、triple barrier 標籤、`FEATURES` 清單（單一事實來源） |
| `train.py` | RandomForest / XGBoost / LightGBM 訓練、評估（`evaluate()`/`confidence_report()`）、`_prepare_data()` cache |
| `predict.py` | 批次（`predict()`，回測用）與即時推論（`predict_live()`，正式對外入口） |
| `run_backtest.py` | 串 `predict.py` + 共用回測引擎 `backtest/intraday_platform.py` |
| `up/live.py`、`down/live.py` | 對外固定介面（2026-07-25起拆成多空各自獨立策略，取代原本單一的`live.py`），`main/live_trader.py` 透過 `STRATEGY_MODULES`（`strategy.mkt.up.live`/`strategy.mkt.down.live`）載入，兩邊都直接沿用本體的`config`/`predict`/`train` |
| `experiments/` | 一次性假設驗證腳本，不是核心pipeline的一部分 |

## 目前設定

- **股票母體**：`db/tickers/tick_universe.parquet` 固定400支（2026-07-25起，見下方「股票母體改用tick_universe」章節，取代原本的`TOP_N`動態排名設計）。
- **HOLD_BARS**：30（2026-07-25從10改成30，見下方「HOLD_BARS改為30」章節）。
- **ATR5平盤過濾門檻**：`config.ATR5_FILTER_THRESHOLD = 0.01095`（p99，2026-07-25對HOLD_BARS=30重新驗證後選定，見下方「ATR5門檻改為p99」章節）。
- **交易時段**：9:11~9:30（9:00~9:10被判定為集合競價剛結束的雜訊時段，見`config.py`）。
- **要用哪個模型**：`config.MODEL_TYPE`（讀 `.env` 的 `MKT_MODEL_TYPE`）。

## FEATURES（詳見 features.py 的 discussion note）

`ret_vs_idx`、`idx_ret_since_open`、`vol_ratio_cum/prev`、`ret_1m`、
`body_pct`、`bullish_volume_surge`、`bullish_reversal`、`bearish_divergence`、
`m3_ret`/`m3_vol_ratio`/`m5_ret`/`m5_vol_ratio`（rolling版）、
`m3_std_ret`/`m3_std_vol_ratio`/`m5_std_ret`/`m5_std_vol_ratio`（標準獨立K棒版）。

## ⚠️ 股票母體改用tick_universe（2026-07-25，重大bug修復）

**發現的問題**：mkt原本用`top_n_by_prev_day_volume()`／`TOP_N`常數，依「前
一交易日全天量排名」動態選出候選股票池（先後試過100、300）。逐月檢查
`db/m1/`才發現：這個資料夾的股票覆蓋範圍本身在整個歷史上一直在變——

| 期間 | db/m1 涵蓋股票數 |
|---|---|
| 2021初 | 333支 |
| 2022~2025 | 逐漸長到2710支（2026-06） |
| 2026-07 | 驟降到1605支（過渡月） |
| 2026-08起 | 固定400支（`data/m1_data_loader.py` 改成只收tick_universe） |

也就是說「top_n=300」在不同訓練/回測時期，實際上是從**不同大小的母體**裡
選股，越晚的窗口母體越小。這代表2026-07-25之前做的top_n=100 vs 300、以及
第一版ATR5門檻p90~p99 walk-forward驗證，都摻雜了「母體結構隨時間變化」這個
雜訊，數字不可信（那幾次驗證的完整過程還留在git歷史/舊版README裡，但不要
拿來當結論用）。

**修法**：`train.py::_prepare_data()`／`predict.py::build_prewarm_cache()`/
`predict_live()` 全部改成呼叫 `finmind.tick_universe.load_tick_universe()`
讀固定400支（399支依均量排名+0050強制併入），不再依日期/成交量動態變動，
train/predict用同一份、跟8月起`db/m1`本身收錄的母體一致。`config.py`的
`TOP_N`常數已刪除，`features.py`裡`top_n_stock_ids_by_latest_volume()`
（無其他呼叫端）已一併刪除，`top_n_by_prev_day_volume()`留著給少數還在用
它的舊experiments腳本用。

**順手修復的記憶體bug**：篩選母體的時機原本放在算完`ret_vs_idx`/
`add_idx_gap_pct()`之後，等於對用不到的~2000+支歷史股票也白算一次逐列特徵，
2026-08-05實測把process記憶體吃到55GB、卡死。已改成`load_m1()`後馬上篩選
（`add_ret_vs_idx()`/`add_idx_gap_pct()`都只依賴目標股票自己+0050，篩選
不影響結果），記憶體降到10GB以內。

## HOLD_BARS改為30（2026-07-25，取代原本的10）

**背景**：`config.py`原本記錄「2026-07-14用`experiments/ret_vs_idx_signal_check.py`
驗證過，30分鐘訊號會反轉」，所以`HOLD_BARS`定為10。但那次驗證用的是有問題
的top_n=100母體（見上一節）。

**重新驗證**：改用tick_universe固定400支、抓最近3個月，重跑同一支腳本
（`forward_minutes=5/10/15/30`）：

| forward_minutes | 相關係數 | decile0(落後最多)未來報酬 | decile9(領先最多)未來報酬 | 上漲比例差 |
|---|---|---|---|---|
| 5 | -0.0161 | +0.0115% | -0.0223% | +8.13pp |
| 10 | -0.0164 | +0.0122% | -0.0364% | +8.75pp |
| 15 | -0.0165 | +0.0111% | -0.0481% | +8.97pp |
| 30 | **-0.0198** | +0.0022% | **-0.0976%** | +8.91pp |

完全沒有反轉——相關係數在30分鐘反而最強，decile0/9的差距也沒有縮小。不確定
是母體換了還是抓的時間段（最近3個月）剛好跟原本驗證的期間market regime不同，
但用這次結果把`HOLD_BARS`改成30。

**意外的附帶效果**：label分布大幅改善。HOLD_BARS=10時「平」佔85~89%，改成
30之後：

```
target
1 (平)   64.42%
0 (跌)   17.91%
2 (漲)   17.67%
```

單純拉長觀察窗口（更容易觸及±3%），比之前討論過的任何「處理不平衡」手法
（class_weight、採樣、ATR5過濾）都更直接有效——這幾個手法其實模型訓練時
已經在用（`class_weight="balanced"`／`compute_sample_weight`），但都是在
「訓練時加權」，不像HOLD_BARS這樣從根本上改變label本身的分布。

**precision（單一近期窗口，20天測試集，XGB，尚未走完整walk-forward）**：

| 門檻 | precision | recall |
|---|---|---|
| 0.5 | 45.82% | 27.13% |
| 0.6 | 51.41% | 21.53% |
| 0.7 | 55.50% | 17.38% |
| 0.8 | 58.00% | 12.49% |

⚠️ 這只是單一最近窗口，這個策略的歷史經驗是最近窗口常常特別好看，正式的
5窗口walk-forward驗證結果見下方「ATR5門檻」章節（跟ATR5門檻選擇一起驗證）。

## ATR5門檻改為p99（2026-07-25正式採用，對HOLD_BARS=30重新驗證過）

**背景**：ATR5平盤過濾（`add_atr5()` + `ATR5_FILTER_THRESHOLD`，2026-07-23
新增）——不看類別、只看當下atr5這個數字是否 >= 固定門檻，train/test/上線
推論三邊用同一套規則。踩過的坑（跨股票/自己歷史的相對排名都太弱、「只篩平」
效果最強但依賴事後label沒法上線）記錄在`add_atr5()`/`atr5_pr_check.py`的
docstring裡，不重複貼。

第一版（top_n=300、HOLD_BARS=10）驗證選了p97，但那次的母體有前述tick_universe
bug，數字不可信。改用tick_universe固定400支後，同一天又把HOLD_BARS從10改
成30（見上一節），所以用`experiments/retest_atr5_tick_universe.py`對
**tick_universe固定400支 + HOLD_BARS=30**重新算一次。

**密度診斷**（`_prepare_data(atr5_threshold=-1.0)`拿到時段過濾後、ATR5過濾
前的完整population，共6,482,079筆）：

| 門檻 | atr5值 | 保留% | 跌% | 平% | 漲% |
|---|---|---|---|---|---|
| p90 | 0.00552 | 10.00% | 9.24% | 79.41% | 11.35% |
| p95 | 0.00707 | 5.00% | 12.56% | 73.09% | 14.35% |
| p97 | 0.00827 | 3.00% | 14.86% | 69.07% | 16.07% |
| p99 | 0.01095 | 1.00% | 19.42% | 62.21% | 18.38% |

**precision結果**（5窗口45天walk-forward，XGB round2參數）：

| 模型門檻 | p90 | p95 | p97 | p99 |
|---|---|---|---|---|
| 0.5 | 19.82% | 22.23% | 24.81% | **28.66%** |
| 0.6 | 24.32% | 25.85% | 27.76% | **30.80%** |
| 0.7 | 29.15% | 29.54% | 29.96% | **34.74%** |
| 0.8 | 32.47% | 33.93% | 30.56% | **35.09%** |

跟第一版（HOLD_BARS=10）不同：這次**p99在全部4個模型門檻都是最好**，沒有
出現p97在高門檻反而贏的情況。整體precision也比第一版好非常多（第一版最高
單點62.05%只是0.8門檻一次性數字，這次p99整條曲線19.82%~35.09%都很紮實）。

代價一樣是樣本量少：過濾前648萬筆population，p99只剩64,834筆（約1%）。但
這次precision優勢夠明確、全門檻一致獲勝，選**p99**，不再猶豫p97。

`config.ATR5_FILTER_THRESHOLD = 0.01095`，`cache/mkt_prepared.parquet`已
用p99門檻重建（利用剛才驗證時已經算好的未過濾population直接篩選，沒有
重跑一次完整pipeline），三個模型（rfc/xgb/lgbm）都已重新訓練過。

<!-- TODO: 補上這次重新驗證的密度表+precision表+最終選定的門檻 -->

## 已驗證沒幫助、別再重試的方向

| 日期 | 嘗試 | 結果 |
|---|---|---|
| 2026-07-19 | `atr7`/`range_pct` 當FEATURES | 樹拿去投機分裂，稀釋掉`ret_vs_idx`重要性，precision全面變差，已拿掉 |
| 2026-07-21 | `day_ret_vs_idx_1~10`（日K相對大盤10天落後值） | XGB precision全面持平甚至微降（0.8門檻34%→33%），已刪除 |
| 2026-07-22 | `idx_gap_pct`（0050開盤跳空缺口） | XGB precision幾乎持平微降（0.6:21%→20%、0.8:34%→33%），已從FEATURES註解掉（函式還留著） |
| 2026-07-23 | `strategy/mkt_label2/`（改2分類：漲vs非漲） | walk-forward顯示全面輸給3分類（4個門檻mean都輸），整個資料夾已刪除 |
| 2026-07-22~25 | `top_n=100/300`動態排名當流動性過濾 | ⚠️ 2026-07-25發現整個設計建立在錯誤假設上（母體隨時間漂移），相關的walk-forward結論全部作廢，已改用tick_universe固定400支，見上方章節。這個方向本身「不是沒幫助」，是設計方法錯了，跟其他列在這裡的「試過沒用」性質不同，只是放在一起方便查 |

## 目前最佳XGB參數（2026-07-21第二輪walk-forward驗證過，HOLD_BARS=10時代調的，尚未針對HOLD_BARS=30重調）

```python
n_estimators=500, max_depth=10, learning_rate=0.12770505395846093,
subsample=0.9541586311700712, colsample_bytree=0.9978337571109724,
min_child_weight=22, reg_lambda=1.847987791882633, gamma=0.2824283269109099
```

⚠️ 這組參數是HOLD_BARS=10、舊母體時代調的，HOLD_BARS=30+tick_universe下
還沒重新tune過，precision數字已經比舊版好很多（見上方HOLD_BARS章節），但
理論上重新tune還有進步空間，之後有空可以回來做。

LGBM 也調過參數（`experiments/tune_lgbm.py`，同樣是舊母體/HOLD_BARS時代），
目前還是XGB表現最好，但同樣沒有針對新設定重調過。
