# vwap_ml 策略（VWAP±2σ 通道 / m1+m3+m5 三時間框 / LightGBM+XGBoost）

2026-07-26 討論確立的第一版設計，同一天內經過多次迭代（對稱門檻、統一
label視窗、只交易回歸、加XGB、Optuna調參後選定LGBM為正式模型）。

## 核心假設

在 m1/m3/m5 三個時間框上，各自計算「今日累積 VWAP」與「累積至今的偏離
標準差（expanding std）」，超過 `config.STD_MULT` 個標準差視為候選訊號。
三個時間框**任一觸發即算候選（OR）**，不要求三者同時滿足——如果要求同時
滿足，訊號會太少，改成候選產生用 OR、分類特徵留三個時間框各自的 z-score
（`m1_vwap_z`/`m3_vwap_z`/`m5_vwap_z`），讓模型自己學組合規律（例如
「只有 m1 觸發」跟「m1+m5 同時觸發」代表的意義不同）。

候選觸發後，三分類器判斷接下來是：

| class | 意義 |
|---|---|
| 0 | 回歸 VWAP（z 跌回0附近） |
| 1 | 無訊號（盤整，兩個 barrier 在視窗內都沒碰到） |
| 2 | 延續突破（z 進一步擴大超過延續門檻） |

配合觸發時價格在 VWAP 之上還是之下（`trigger_side`），轉成多/空方向：

| 觸發位置 | 分類結果 | 方向 |
|---|---|---|
| 上軌（z>+STD_MULT） | 回歸 | 空單 |
| 上軌 | 延續 | 多單 |
| 下軌（z<-STD_MULT） | 回歸 | 多單 |
| 下軌 | 延續 | 空單 |

## Volume Profile POC 特徵

2026-07-28 新增：Point of Control (POC) 特徵，與 VWAP 無關，是 M1 bar 的 Volume Profile 累積成交量峰值。這組特徵獨立於 VWAP Z-score，用以捕捉價格行為的另一面向。

主要引入三種 POC 特徵：

1.  `close_vs_poc_m1`：**Session Expanding POC**
    -   計算方式：從當日 9:00 開盤起，逐分鐘累積成交量並生成 Volume Profile，取其中成交量最大的價位（Point of Control, POC）。該特徵為當前收盤價 `close` 相對於此 expanding session POC 的相對差異 `(close - poc) / poc`。
    -   意義：反映價格相對於當日累計市場共識價格的偏離程度。

2.  `close_vs_poc_r30`：**Rolling 30 分鐘 POC**
    -   計算方式：取過去 30 根 M1 bar（即 30 分鐘）內的成交量，生成 Volume Profile，並取其中成交量最大的價位（POC）。該特徵為當前收盤價 `close` 相對於此 rolling 30 分鐘 POC 的相對差異 `(close - poc) / poc`。
    -   意義：反映價格相對於短期內市場共識價格的偏離程度，捕捉局部支撐阻力。

3.  `close_vs_poc_d5`：**5 交易日 Lagged 全天 Session POC**
    -   計算方式：使用 5 個交易日前（`POC_LAG_TRADING_DAYS`）的整日（9:00-13:30）Volume Profile，取其 POC。該特徵為當前收盤價 `close` 相對於 5 交易日前全天 POC 的相對差異 `(close - poc) / poc`。此值在當日所有分鐘 K 棒皆相同。
    -   意義：反映價格相對於較長期的歷史重要價位的偏離程度，捕捉關鍵趨勢或支撐阻力水平。

這些特徵與 `BASELINE_FEATURES` 一起納入模型訓練，並透過 Ablation Study 評估其對模型效能的貢獻。

## ⚠️ 2026-07-26 決定：只交易「回歸」，不交易「延續」

walk-forward 驗證（`experiments/walk_forward.py`，5個45天窗口，涵蓋
2025-12~2026-07）發現兩個類別可靠度差很多：

| 類別 | 門檻0.6 precision（5窗口平均） | 門檻0.8 precision |
|---|---|---|
| 回歸 | 67.03%（std±6.50%） | 92.71%（std±4.15%，穩定） |
| 延續 | 只有18~23% | — |

延續不夠可靠，決定**只交易回歸**——`predict.py::_direction_probas()` 已
改成只把回歸機率換算成做多/做空，延續機率完全不使用、不會產生任何訊號
（上軌回歸→做空、下軌回歸→做多，其餘情況機率一律為0）。`up/live.py`/
`down/live.py` 的結構不用改，因為 `predict_live()` 回傳的 `direction`
欄位本來就已經是轉換過的實際方向，`DIRECTIONS` 過濾機制不需要知道背後
是回歸還是延續。之後如果延續的模型/特徵有改善、precision 夠高，才考慮
加回來。

`run_backtest.py` 用這個「只回歸」版本回測（2024-01-01起、最近30天）：
7筆交易、勝率71.4%、報酬率+0.60%，但 walk-forward 回測
（`experiments/walk_forward_backtest.py`）顯示5個窗口裡有2個窗口完全
沒有交易、平均報酬率只有0.09%（std=0.49%，跟均值同量級）——**precision
驗證得很穩，但轉換成實際回測報酬還沒有那麼可靠，樣本數也太少**，這部分
還需要更多驗證才能下結論。

## 2026-07-26：模型選擇——LGBM，不選 XGB

`experiments/tune_lgbm.py`/`tune_xgb.py` 各自跑滿 Optuna 40 trials（
train/val/test三段式45天切分，只優化「回歸」class=0在門檻0.6下的
precision，避免污染最終報告用的test集），比較兩者在完全沒被調參污染的
test集上、依 `trigger_side` 拆多空的表現：

| | test precision（整體） | 上軌/做空 | 下軌/做多 |
|---|---|---|---|
| XGB（調參後） | 79.22%（看似最高） | 預測數77，precision 79.22%，recall僅1.71% | **預測數0，完全沒有任何預測** |
| LGBM（調參後） | 73.51% | 預測數876，precision 72.15%，recall 17.71% | 預測數317，precision 77.29%，recall 10.23% |

XGB 的高precision是取巧解——Optuna 選了一組極保守參數
（`learning_rate≈0.0145`、`n_estimators=200`、`max_depth=4`），讓模型
幾乎不出手、只在最有把握的少數做空案例才判斷回歸，代價是**完全放棄
「做多」這個方向**（3569次真正做多機會，一次都沒抓到），recall也低到
只剩1.71%，沒有實際交易的價值。LGBM 雖然整體precision略低，但多空
兩個方向都有數百筆等級的訊號量，且precision都在70%以上，是更平衡、
更能真正部署的模型。

**決定選用 LGBM**，已把調好的參數貼回 `train.py::train_lgbm()`（取代
原本沒調過、隨手設的起始值），`.env::VWAP_ML_MODEL_TYPE=lgbm` 明確
鎖定這個選擇。用全部標準訓練流程（`start_date=2024-01-01`、最近30天
測試集）重新訓練正式模型後，在部署門檻0.6下驗證多空表現：

| 方向 | 預測數 | precision | recall |
|---|---|---|---|
| 做空（上軌回歸） | 629 | 75.52% | 19.08% |
| 做多（下軌回歸） | 305 | 80.33% | 12.32% |

比調參前（62~64%左右）明顯提升，且沒有偏廢任一方向。

⚠️ 這個precision是**信心度門檻=0.6過濾後**的數字——`model.predict()`
不設門檻直接判斷時，回歸precision只有62%左右，門檻拉到0.6才會看到
75~80%。這代表實際上線一定要靠 `config.THRESHOLD`/`.env` 的
`VWAP_ML_THRESHOLD` 過濾，不能只看模型本身的預測結果。

## ⚠️ 資料限制：VWAP 是近似值，不是精確計算

`db/m1`、`db/m1_live` 只有 `stock_id/date/open/high/low/close/volume`，
**沒有成交金額欄位**，沒辦法算「真正的」VWAP（Σ逐筆成交價×量/Σ量）。這裡
的 VWAP 是 `close × volume` 累積近似——跟 `strategy/orb/features.py`、
`strategy/rally/features.py` 的 `vwap_dev` 用同一種近似方式，是專案裡
一貫的做法，不是 vwap_ml 獨有的妥協。

z-score 的分母用「累積至今的偏離標準差（expanding std）」，不是全天
std——全天 std 會用到未來才知道的資訊（lookahead），expanding 只用當下
已經發生的 bar，即時推論當下能拿到的資訊跟訓練時一致。

## 交易時段

`SESSION_START=(9,10)`、`SESSION_END=(10,0)`。VWAP 從 9:00 開盤就開始
累積，但 9:00~9:10 視為暖機期不交易——開盤沒多久累積樣本數太少，
expanding std 還不穩定；10:00 收工，避免抓到中盤盤整雜訊。

## 檔案結構

| 檔案 | 內容 |
|---|---|
| `config.py` | 常數：`STD_MULT`、`LABEL_HORIZON_MINUTES`、`THRESHOLD`、`SESSION_START/END`、`ATR5_FILTER_THRESHOLD`、`BACKTEST_TP_PCT/SL_PCT/HOLD_BARS`、`MODEL_TYPE` |
| `features.py` | VWAP z-score/候選觸發/三分類標籤計算、`FEATURES` 清單（單一事實來源），**包含 POC 相關特徵計算** |
| `train.py` | LightGBM/XGBoost 訓練（`train_lgbm()`/`train_xgb()`）、評估（`evaluate()`/`confidence_report()`/`direction_breakdown()`）、`_prepare_data()` cache（依 `start_date` 各自獨立檔案）、`MODEL_TYPE` 切換機制 |
| `predict.py` | 批次（`predict()`，回測用）與即時推論（`predict_live()`，正式對外入口）、方向轉換（`_direction_probas()`，只用回歸） |
| `run_backtest.py` | 串 `predict.py` + 共用回測引擎 `backtest/intraday_platform.py` |
| `up/live.py`、`down/live.py` | 對外固定介面，`main/live_trader.py` 透過 `STRATEGY_MODULES` 載入，分別只送多/空訊號 |
| `experiments/walk_forward.py` | 多窗口驗證回歸 precision/recall 穩定性（一次性腳本） |
| `experiments/walk_forward_backtest.py` | 多窗口驗證實際回測報酬率穩定性（一次性腳本） |
| `experiments/tune_lgbm.py`、`experiments/tune_xgb.py` | Optuna 超參數搜尋，train/val/test三段式避免污染（一次性腳本） |
| `experiments/poc_ablation.py` | POC 特徵 Ablation Study，walk-forward 比較基線模型與含 POC 特徵模型的效能 |

支援 `lgbm`/`xgb` 兩種演算法（`config.MODEL_TYPE`，`.env` 的
`VWAP_ML_MODEL_TYPE` 覆蓋），切換機制比照 `strategy/mkt/train.py`
（`_LOAD_MODEL_BY_TYPE`/`_TRAIN_BY_TYPE` 字典＋`load_model_by_type()`，
`up/down/live.py` 都依這個常數決定要載入哪個模型）。之後要再加其他
演算法只要在 `train.py` 補一組 `train_xxx()`/`load_model_xxx()` 函式並
登記進字典，不用改 `predict.py`/`run_backtest.py`/`up/down/live.py`
任何一行。

沒有 `build_prewarm_cache()`——VWAP z-score 只需要「今天」的資料
（expanding，不看跨日歷史），不像 orb 需要跨日歷史查表，比照 rally 沒有
實作這個函式的做法，`strategy/prewarm.py` 對缺這個屬性的模組會自動當作
`{}`，不影響 `main/premarket.py::refresh_prewarm()` 的流程。

## Label 視窗

視窗大小是「觸發的那個時間框自己的後 `LABEL_HORIZON_MINUTES`（=30）
分鐘」——m1/m3/m5 的 z-score 都是每分鐘更新一次的序列，「30分鐘」對三者
直接就是「未來30根」，不用乘時間框的分鐘數（原本設計是「m1看10分鐘、
m3看30分鐘、m5看50分鐘」各自不同，2026-07-26 實測發現統一視窗的
「無訊號」比例明顯下降、樣本更平衡，改用這個版本）。仍然是用觸發的那個
時間框自己的 z-score 序列（例如 m5 觸發就看 m5_vwap_z）判斷未來走勢，
不是統一都看 m1 的 z。

兩個 barrier 分別是「z 回到 0（回歸完成）」跟「z 進一步擴大超過延續門檻
（預設 = `STD_MULT * 2`）」，不是像 orb/rally/mkt 那樣固定百分比報酬——
這裡衡量的是「價格相對 VWAP 的位置」，不是絕對報酬率。

⚠️ 2026-07-26 修正：延續門檻原本設計是 `STD_MULT + 1.0`，但這樣兩個
barrier 到觸發點的距離不對稱——觸發點 z=STD_MULT 時，回歸要走到0（距離
=STD_MULT），延續只要走到 STD_MULT+1.0（距離只有1.0）。距離不對稱代表
延續天生比較容易觸發，不是市場真的比較常延續。改成 `STD_MULT * 2`
（延續要走到 STD_MULT*2，距離跟回歸一樣都是STD_MULT）才能讓三分類比例
真正反映市場行為，不被 barrier 設計本身扭曲。

## ATR5 平盤過濾

z-score 的分母是「該股票累積至今的偏離標準差」，本來就沒什麼波動的股票
（分母趨近於0）連小幅價格雜訊都會被除出誇張的 z 值，產生假的候選訊號。
加 ATR5（True Range 5根滾動平均/當日開盤價，算法跟 `strategy/mkt` 完全
一樣）當絕對門檻，濾掉這種假候選，train/predict_live 三邊用同一套規則。

`config.ATR5_FILTER_THRESHOLD = 0.01000` 直接沿用 mkt 當時驗證出來的
p97 門檻當起點，**還沒針對 vwap_ml 自己的樣本分布重新驗證過**——mkt 的
母體（9:11~9:30時段、ret_vs_idx情境）跟 vwap_ml（9:10~10:00、VWAP偏離
情境）不是同一種資料，這個門檻對 vwap_ml 合不合適要之後用 walk-forward
重新驗證。`atr5` 本身不進 `FEATURES`（純波動幅度指標會被樹拿去投機分裂，
稀釋掉真正有方向性的訊號，見 `strategy/mkt/README.md` 的同名章節）。

## 資料載入範圍（`start_date`）

`train.py::_prepare_data()` 的 `start_date` 參數會直接限制
`load_m1()`/`load_m3()`/`load_m5()` 的載入範圍（不是先載全部歷史再事後
篩選）——`db/m1` 實際存到 2022-12（另外還有一個獨立的
`finmind.backfill_m1_history` 背景作業在補 2019~2023 的資料），全部載入
會不必要地拖慢速度、佔用大量記憶體，且訓練通常只需要近1~2年資料
（`start_date="2024-01-01"` 是目前實際使用的值）。不同 `start_date`
存在各自獨立的 cache 檔案（`cache/vwap_ml_prepared_{start_date}.parquet`，
`start_date=None` 才用預設的 `cache/vwap_ml_prepared.parquet`），彼此不
覆蓋——`train`/`evaluate`/`confidence`/`direction`/`predict()` 都要傳
同一個 `start_date` 才會讀到同一份 cache。

⚠️ `_source_mtime()`（cache 新鮮度判斷）只看「檔名代表最新月份」的檔案
時間戳，不是掃全部檔案取最大值——避免被上述 backfill 背景作業誤觸發
（它會持續改到舊月份檔案，跟近期資料無關，2026-07-26 發現並修正）。

`use_cache`（跟 `strategy/mkt/train.py` 的慣例相反）：**預設 `False`＝
無條件重新計算並覆蓋 cache**，只有明確傳 `True` 才會去比對 cache 新鮮度、
可能直接讀取——這個階段密集調整 `features.py` 的計算邏輯，cache 的
mtime比對偵測不到「程式邏輯改了」，曾經因此誤用過舊邏輯算出的cache，
改成預設不信任比較安全。

## ⚠️ 還沒驗證過、需要之後用 walk-forward 定案的參數

| 參數 | 目前預設值 | 對應參數化入口 |
|---|---|---|
| `STD_MULT`（候選觸發門檻） | 2.0（建議之後比較 1.0/1.5/2.0） | `features.py::add_vwap_features(std_mult=...)`、`train.py::_prepare_data(std_mult=...)` |
| `ATR5_FILTER_THRESHOLD` | 0.01000（沿用mkt的p97，未針對vwap_ml重算） | `features.py::make_features(atr5_threshold=...)`、`train.py::_prepare_data(atr5_threshold=...)` |
| `LABEL_HORIZON_MINUTES` | 30 | 尚未參數化，改 `config.py` 需重新產生cache |
| `BACKTEST_TP_PCT`/`SL_PCT`/`HOLD_BARS` | 3%/3%/30根（沿用orb/rally/mkt共用起始值） | 尚未參數化 |

## 已知限制

- 共用回測引擎 `backtest/intraday_platform.py` 只吃單一（做多）機率矩陣，
  沒有方向概念，`predict()` 只回傳「做多」訊號用於回測（下軌回歸這一側）；
  上軌回歸（做空）的表現目前無法用這個引擎驗證，只能靠 precision 數字
  類推。`predict_live()` 才會同時算多空兩個方向給前端顯示。回測放空是
  完全另一個規模的工程，先不做（跟 mkt 的同樣考量）。
- 沒有像 mkt 那樣自己重新排流動性名單（`top_n_by_prev_day_volume()`）——
  候選股票池依賴 `main/live_trader.py` 傳入的 `day_trade_stocks`，v1 先
  不另外實作獨立的流動性過濾。
- walk-forward 回測（`experiments/walk_forward_backtest.py`）樣本數還
  太少（5窗口共14筆交易），實際報酬率的穩定性還沒有精確度驗證那麼可信，
  下結論前建議先累積更多交易樣本或調整門檻/流動性設定再重新驗證。

## Ablation Study 結果

[此處將呈現 POC 特徵 Ablation Study 的 walk-forward 驗證結果。]