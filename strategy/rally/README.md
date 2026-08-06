# rally 策略（破底翻 / RandomForest / XGBoost / LightGBM）

## 核心假設

抓「先跌後漲」的當日反彈型態：股價當天先破底、動能轉弱後又開始翻揚，賭這股
反彈動能會延續。不是規則式硬過濾，而是用機器學習分類器（RandomForest/XGBoost/
LightGBM 三選一比較）學這個型態加上一整套盤中動能/量能/位置特徵，去預測「進場
後 30 根分K內（`HOLD_BARS`），會先漲 3%（`TP_PCT`）還是先跌 3%（`SL_PCT`）」
（triple barrier 標籤，`target=1/0`，見 `features.py`）。

`breakout_signal`（第1根5分鐘K跌、第2根5分鐘K漲＝先跌後漲）原本是即時交易的硬
過濾規則，2026-07-09 驗證後拿掉了——加這道規則反而讓勝率下降 8-10 個百分點
（見 `experiments/breakout_filter_eval.py`），現在只是模型的輸入特徵之一，讓
樹模型自己決定要不要用、怎麼用，不再強制訊號一定要先跌後漲。

全天訓練（不鎖 9:00~9:30 黃金窗口），靠 `minutes_since_open` 這個特徵讓模型
自己判斷開盤動能期 vs 中午盤整期，交易時段限制交給呼叫端（回測用
`first_entry_time`/`last_entry_time`，即時交易用 `config.py` 的
`SESSION_START`/`SESSION_END`）。當沖策略，不留倉，收盤前強制平倉
（`live_trader.py` 的 `_force_close_eod`）。

## 檔案結構

| 檔案 | 內容 |
|---|---|
| `config.py` | 交易相關設定（`TP_PCT`/`SL_PCT`/`HOLD_BARS`、`THRESHOLD_BY_MODEL`、`SESSION_START/END`、`ATR_FILTER_THRESHOLD`）。只放常數，不放邏輯，且只放上線實際會用到的設定——`BREAKOUT_TRADE_START/END`（破底翻黃金窗口）2026-08-06 已搬到 `experiments/breakout_filter_eval.py`/`breakout_specialist.py`，因為只有這兩支實驗腳本在用，放在 `config.py` 容易讓人誤以為還在生效 |
| `features.py` | 特徵工程、triple barrier 標籤、`FEATURES` 清單（78個，單一事實來源）、月分區 cache（`cache/m1_rally_features/`）、固定400支 universe 過濾 |
| `train.py` | 訓練 + CLI 進入點（`entry.py` 已併入這裡，orb/mkt 沒有獨立 entry.py 也是同一個模式）。RandomForest/XGBoost/LightGBM 三選一，三模型共用 `_prepare_train_test()` |
| `validate.py` | 信心度分析（`confidence_report`）、召回率分析（`coverage_report`）、多模型×時段×信心度交叉報表（`model_hour_confidence_report`）、每小時訊號數報表（`hour_signal_report`）、特徵重要性（`feature_importance`） |
| `predict.py` | 批次（`predict()`，回測用）與即時推論（`predict_live()`，正式對外入口） |
| `run_backtest.py` | 串 `predict.py` + 共用回測引擎 `backtest/intraday_platform.py`（跟 orb 共用同一份引擎，不改引擎本身） |
| `rfc/live.py`、`xgb/live.py`、`lgbm/live.py` | 三個模型各自獨立掛成 live 策略（`rally_rfc`/`rally_xgb`/`rally_lgbm`），`main/live_trader.py` 透過 `STRATEGY_MODULES`（`.env`）載入，都直接沿用本體的 `config`/`predict`/`train`，只是各自查 `THRESHOLD_BY_MODEL` 的不同 key |
| `experiments/` | 一次性假設驗證腳本：`tune_xgb.py`/`tune_lgbm.py`（Optuna 調參）、`walk_forward_{xgb,lgbm,rfc}.py`（多窗口驗證）、`breakout_filter_eval.py`/`breakout_specialist.py`（破底翻硬過濾實驗，已有定論見上方） |

## 目前設定（2026-08-06）

- **股票母體**：`finmind.tick_universe.load_tick_universe()` 固定400支（跟 orb/mkt 一致的既有做法，池外股票資料沒人主動維護，見下方連結）。
- **訓練資料起日**：2021-01-01 起（`train.py` 的 `start_date`）。
- **交易時段**：只在 9:00~9:30 開盤黃金窗口交易，寫死在 `config.py`（不走 `.env`）——逐信心度桶比對顯示這個窗口的機率校準明顯最好，9:30 之後（尤其10點後）各模型高信心度桶精準率都開始不升反降，不值得冒險交易。
- **ATR 平盤過濾**：`config.ATR_FILTER_THRESHOLD = 0.00761`（**p99**，2021年起+固定400支母體、25,918,051筆全體樣本算出來的分位數，篩完剩約1%＝約26萬筆）。比照 `strategy/mkt/config.py::ATR5_FILTER_THRESHOLD` 的設計：用 `m1_atr`（已在 `FEATURES` 裡）當絕對門檻，篩掉波動太小、本來就不太可能觸發3%停利/停損的平盤樣本。`train.py`/`predict.py`/`validate.py` 四處同步套用同一個門檻，避免 training-serving skew。**先試過 p90，發現 LGBM 高信心度桶還是有太多持平樣本，改採 mkt 驗證過最好的 p99**（rally 這裡目前只是直接沿用 mkt 驗證出的百分位，還沒有像 mkt 當初一樣做 p90/p95/p97/p99 四個候選各自跑一輪 walk-forward，見下方「已知未竟事項」）。
- **要用哪個模型**：目前上線 **LGBM**（`.env` 的 `STRATEGY_MODULES` 指向 `strategy.rally.lgbm.live`）。三模型各自的即時交易信心度門檻寫在 `THRESHOLD_BY_MODEL = {"rfc": 0.55, "xgb": 0.60, "lgbm": 0.60}`，寫死不走 `.env`（這些值是驗證出來的結論，不是隨時能調的操作旋鈕）。

## 三模型比較結論演變（2026-08-06 當天，兩輪 ATR 門檻）

| | 整體 Accuracy/AUC（p90） | 整體 Accuracy/AUC（p99） |
|---|---|---|
| RFC | 0.5957 / 0.5711 | 0.5238 / **0.4953**（低於隨機） |
| XGB | 0.5565 / 0.5497 | 0.5142 / 0.5074 |
| LGBM | 0.5719 / 0.5574 | 0.5330 / **0.5629**（p99下反而變好） |

**加 ATR 過濾之前**：RFC 校準曲線最乾淨（precision 隨信心度單調遞增），是當時的最優選擇，曾以 `strategy.rally.rfc.live` 上線。

**加 ATR 過濾（p90）之後**：RFC 訊號量暴減（9點台30天僅110筆），XGB/LGBM 訊號量遠高於 RFC 且勝率不輸，**改選 LGBM**（訊號量最大、高信心度桶精準率乾淨遞增到98%）。

**提高到 p99 之後**：樣本量進一步減少（9點台LGBM 30天僅720筆，最高信心度桶只有23筆，數字已經偏噪），但 LGBM 整體 AUC 不降反升（0.5629，三模型中唯一在p99下表現變好的），RFC/XGB 則明顯惡化（RFC AUC甚至跌破0.5）。**維持選用 LGBM**，`ATR_FILTER_THRESHOLD` 採用 p99。

## ⚠️ RandomForest 在千萬列級資料會爆記憶體

`RandomForestClassifier(n_jobs=-1)` 在 2500萬筆×79特徵資料上 `fit()`，實測把 25.8GB
機器的 swap 吃到 94.8%（bootstrap用全部樣本+多核同時平行造樹疊加記憶體）。修法：
`n_jobs=4`（降低同時平行造樹數）+ `max_samples=0.3`（每棵樹bootstrap只抽30%），
兩者都要改才有效，只改一個效果打折。XGBoost/LightGBM 在同樣資料量下沒有這個
問題（直方圖演算法）。**以後任何策略要用 sklearn RandomForest 訓練千萬列級資料，
直接套用這個修正。**

## config.py 時間欄位格式的專案慣例

`SESSION_START`/`SESSION_END` 是 `"H:MM"` 字串（人類好讀），不是 `(h, m)`
tuple——這跟 orb/mkt/cnn/vwap_ml/vwap_dl
現有的 tuple 寫法不一致，是**刻意的、只限 rally 的例外**。轉換邏輯統一寫在
`main/state.py::_parse_hhmm()`（同時吃字串或tuple），`main/live_trader.py` 內部
邏輯完全不用改，其他策略模組不受影響。以後如果要把其他策略也改成字串格式，
一樣走這個既有的 `_parse_hhmm()`，不要各自發明轉換邏輯。

## 已知未竟事項

- `experiments/tune_xgb.py`/`tune_lgbm.py`/`walk_forward_*.py` 裡貼的參數/驗證
  結論都是舊資料集（2026-01~07、全市場股票、ATR過濾**前**）算出來的，2021起+
  固定400支+ATR過濾重訓後還沒有重新調參——目前 `train.py` 用的 XGB/LGBM 超參數
  是舊調參結果直接沿用，不保證還是最佳。
- RFC 沒有 `experiments/tune_rfc.py`（沒調過參數），`walk_forward_rfc.py` 的
  `_CANDIDATE_PARAMS` 目前是佔位（複製 `_BASELINE_PARAMS`）。
- `ATR_FILTER_THRESHOLD` 只是直接沿用 mkt 驗證出的 p99 百分位，還沒有像 mkt
  當初一樣對 rally 自己的資料做 p90/p95/p97/p99 多候選 + walk-forward 完整驗證，
  且 p99 下樣本量已經偏少（9點台30天僅約700筆），單一30天測試窗口的數字噪聲大，
  之後要更嚴謹驗證可以用 `experiments/walk_forward_*.py` 帶不同候選門檻+拉長
  test_days 重跑。
- `train.py` 的 `__main__` 進入點目前**忽略 CLI 參數**（`if __name__ ==
  "__main__":` 區塊寫死 `mode` 變數，`main()` 內的 `argparse` 分支被繞過）。
  要用 CLI 參數的話得改用 `python -c "from strategy.rally.train import main;
  main(mode='train', ...)"` 繞過，或先修掉這個寫死問題。
