# mkt 策略（個股 vs 大盤0050 相對強弱 / RandomForest / XGBoost / LightGBM）

## 核心假設

當大盤上漲、個股沒跟上，個股有機會補漲往大盤靠近（`ret_vs_idx`：個股從今日
開盤累積到現在的報酬率，減去0050同期的報酬率）。3分類標籤（跌=0/平=1/漲=2，
`config.py` 的 triple barrier：`TP_PCT=SL_PCT=3%`、`HOLD_BARS=10`），但策略
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
| `live.py` | 對外固定介面，`main/live_trader.py` 透過 `STRATEGY_MODULES` 載入 |
| `experiments/` | 一次性假設驗證腳本，不是核心pipeline的一部分 |

## 目前設定

- **流動性過濾**：前一交易日全天量前300名（`config.TOP_N`，2026-07-25從100改成300，見下方「top_n=300」章節）。
- **ATR5平盤過濾門檻**：`config.ATR5_FILTER_THRESHOLD = 0.01000`（p97，2026-07-25從p90改成p97，見下方「ATR5門檻改為p97」章節）。
- **交易時段**：9:11~9:30（9:00~9:10被判定為集合競價剛結束的雜訊時段，見`config.py`）。
- **要用哪個模型**：`config.MODEL_TYPE`（讀 `.env` 的 `MKT_MODEL_TYPE`），目前XGB表現最好。

## 目前最佳XGB參數（2026-07-21第二輪walk-forward驗證過）

```python
n_estimators=500, max_depth=10, learning_rate=0.12770505395846093,
subsample=0.9541586311700712, colsample_bytree=0.9978337571109724,
min_child_weight=22, reg_lambda=1.847987791882633, gamma=0.2824283269109099
```

**walk-forward基準線**（5個45天窗口，`strategy/mkt/experiments/walk_forward_xgb.py`，
特徵集：含`idx_ret_since_open`、不含`idx_gap_pct`、**不含ATR5過濾**）：

| 門檻 | 漲precision（5窗口mean） |
|---|---|
| 0.5 | 9.17% |
| 0.6 | 12.05% |
| 0.7 | 15.45% |
| 0.8 | 19.68% |

⚠️ 單一窗口（尤其最近一段時期）的評估數字常常比這個mean高很多（曾看過43%），
是因為那個窗口剛好是這個策略歷史上表現特別突出的一段期間，不能只看單一窗口
就下定論，要看walk-forward的多窗口平均才可信。

LGBM 也調過參數（`experiments/tune_lgbm.py`），walk-forward顯示0.5/0.6/0.7贏
舊參數、0.8打平略輸，整體還是不如XGB，已貼回`train_lgbm()`當次佳選擇。

## FEATURES（詳見 features.py 的 discussion note）

`ret_vs_idx`、`idx_ret_since_open`、`vol_ratio_cum/prev`、`ret_1m`、
`body_pct`、`bullish_volume_surge`、`bullish_reversal`、`bearish_divergence`、
`m3_ret`/`m3_vol_ratio`/`m5_ret`/`m5_vol_ratio`（rolling版）、
`m3_std_ret`/`m3_std_vol_ratio`/`m5_std_ret`/`m5_std_vol_ratio`（標準獨立K棒版）。

## 已驗證沒幫助、別再重試的方向

| 日期 | 嘗試 | 結果 |
|---|---|---|
| 2026-07-19 | `atr7`/`range_pct` 當FEATURES | 樹拿去投機分裂，稀釋掉`ret_vs_idx`重要性，precision全面變差，已拿掉 |
| 2026-07-21 | `day_ret_vs_idx_1~10`（日K相對大盤10天落後值） | XGB precision全面持平甚至微降（0.8門檻34%→33%），已刪除 |
| 2026-07-22 | `idx_gap_pct`（0050開盤跳空缺口） | XGB precision幾乎持平微降（0.6:21%→20%、0.8:34%→33%），已從FEATURES註解掉（函式還留著） |
| 2026-07-22 | `top_n=300`（放寬流動性過濾，**當時還沒有ATR5過濾**） | 全面且明顯輸給top_n=100（5窗口每個門檻都輸）。⚠️ 2026-07-25搭配ATR5過濾重測後結論反過來了，top_n=300已改為正式設定，見下方「top_n=300（流動性過濾放寬）」章節，這行結論已過期、不要照抄 |
| 2026-07-23 | `strategy/mkt_label2/`（改2分類：漲vs非漲） | walk-forward顯示全面輸給3分類（4個門檻mean都輸），整個資料夾已刪除 |

## ATR5 平盤過濾（2026-07-23新增，已驗證precision確實提升，目前正式採用）

**動機**：「平」佔比過高（~98%），想篩掉「本來就沒什麼波動、幾乎注定不會動」
的樣本，讓訓練資料的漲/跌密度提高。

**踩過的坑**：
- 「跨股票、同一分鐘相對排名」——沒辦法濾掉「整個市場都平靜」的情況（永遠有
  排名前10%），效果弱（PR>=0.9時漲密度只到5.41%）。
- 「同一支股票、自己的歷史分布」相對排名——比跨股票排名更弱（只到1.82%），
  因為把「這支股票本來就比較活潑」這個有效的絕對訊號也一起normalize掉了。
- 「只篩平、跌漲全部保留」（參考`strategy/cnn/experiments/atr5_flat_filter_check.py`
  的做法）——效果最強（p90時漲密度到10.76%），但這個規則依賴事後才知道的
  label，**沒辦法在即時推論時重現**（不知道會不會變成漲/跌，就沒辦法只篩平），
  只能當訓練集的hard negative mining技巧，不能當上線的候選篩選規則。

**目前採用的版本**：絕對門檻＋跌/平/漲三類都篩（`add_atr5()` + 
`ATR5_FILTER_THRESHOLD`，見`config.py`）——不看類別、只看當下atr5這個數字是
否 >= 固定門檻（0.00748，全體樣本atr5的p90分位數，2026-07-23算出來的固定
數字，不是動態重算），train/test/上線推論三邊用同一套規則，可以真的部署。

`atr5`本身**不進FEATURES**，只當過濾用（避免重演atr7/range_pct被樹拿去
投機分裂的問題）。

**密度診斷結果**（`experiments/atr5_pr_check.py::run_absolute_uniform()`）：

| 保留% | 跌% | 平% | 漲% |
|---|---|---|---|
| 100%（不篩） | 0.73% | 98.00% | 1.27% |
| 25%（p75） | 2.53% | 93.39% | 4.08% |
| 10%（p90，目前採用） | 4.61% | 88.93% | 6.47% |

**precision驗證結果**（2026-07-23，接進`_prepare_data()`後，用現行XGB
round2參數重新跑一次5窗口45天walk-forward，對照上面「目前最佳XGB參數」
那段記錄的baseline）：

| 門檻 | baseline（無ATR5過濾） | ATR5過濾版 |
|---|---|---|
| 0.5 | 9.17% | 17.13% |
| 0.6 | 12.05% | 22.98% |
| 0.7 | 15.45% | 27.77% |
| 0.8 | 19.68% | 32.79% |

四個門檻precision都翻倍以上，密度改善真的轉化成precision提升，不是像
`day_ret_vs_idx`/`idx_gap_pct`那樣密度看起來有變化但precision沒跟著變好。

⚠️ 但波動比baseline大很多（std落在9~22%之間），主要來自5個窗口裡最早一個
（2025-12-10~2026-01-24）表現異常差（threshold=0.7/0.8時precision直接是
0%，n只有21/7筆）——查過訓練樣本數，這個窗口的訓練集（62,702筆，漲樣本
3,477筆）確實是5個窗口裡最少的（ATR5過濾把整體資料砍掉一大塊，早期窗口
能用的歷史更少），後面4個窗口一致穩定變好、越晚越強。不是100%排除「單純
那段期間市場狀況不利」的可能，但資料量偏少是合理的部分解釋。實務上部署
要有心理準備：資料量不足的早期階段，表現可能不如平均值穩定，隨資料累積
應該會改善。

已正式採用，`train.py`/`predict.py`都已接上。

⚠️ **2026-07-25更新**：以下這整段記錄的是2026-07-23當時的決策過程（門檻
0.00748、p90），數字本身還是有效的歷史紀錄，但`ATR5_FILTER_THRESHOLD`
實際數值後來改了，目前正式設定是**p97(0.01000)**，見下面「ATR5門檻改為
p97」章節，不要照抄這裡的0.00748。

## top_n=300（流動性過濾放寬，2026-07-25正式採用）

**動機**：原本想驗證「放寬top_n能不能平衡label比例」（跌/平/漲密度），
用`experiments/walk_forward_top_n.py`固定住已經正式採用的ATR5 p90過濾，
只改top_n=100→300重跑一次5窗口45天walk-forward。

**結論先講**：⚠️ **label比例本身沒有被「平衡」**——跌/平/漲密度兩者幾乎
一樣（top100: 4.76%/88.93%/6.31%、top300: 4.52%/89.22%/6.26%），放寬
top_n只是把整體樣本量從107,059筆增加到238,664筆，比例結構沒變。「加
top_n來平衡label」這個假設不成立，是靠絕對樣本量變多在幫忙，不是密度
改善。

**precision結果**（分門檻，5窗口mean，對照上面ATR5過濾版baseline）：

| 門檻 | top100（=ATR5過濾版baseline） | top300 | top300贏幾窗 |
|---|---|---|---|
| 0.5 | 17.13% | 14.50% | 2/5（輸） |
| 0.6 | 22.98% | 21.71% | 2/5（輸） |
| 0.7 | 27.77% | 30.83% | 4/5（贏） |
| 0.8 | 32.79% | 42.11%（std從22.31%降到6.07%） | 3/5（贏，明顯更穩） |

低門檻(0.5/0.6)top300反而略輸；高門檻(0.7/0.8)top300明顯贏、且std大幅
下降。最關鍵的發現：**解決了ATR5過濾版最早一個窗口
（2025-12-10~2026-01-24）precision掉到0%的問題**——那個窗口實際漲樣本
從460筆變884筆（top_n放寬讓候選股票池變大，同一段時間能收集到更多漲/跌
樣本），threshold=0.7/0.8的precision從0%/0%變成15.38%/40.00%。這證實了
之前的懷疑：早期窗口表現差主要是**訓練樣本量不足**造成的，不是那段期間
市場狀況特別不利。

**權衡後決定**：整體高門檻(0.7/0.8)更常用、且穩定性(std)改善更重要，
改採**top_n=300**為正式設定（`config.TOP_N`，`_prepare_data()`/
`build_prewarm_cache()`/`top_n_stock_ids_by_latest_volume()`都已改用
這個常數，`cache/mkt_prepared.parquet`已重建為top_n=300版本，原本
top_n=100的cache保留在`cache/mkt_prepared_top100.parquet`供對照）。
三個模型（rfc/xgb/lgbm）都已用新設定重新訓練過。

## ATR5門檻改為p97（2026-07-25正式採用，取代原本的p90）

**動機**：top_n改成300之後，想順便試試看拉高ATR5過濾門檻能不能讓
precision更好——用新增的`experiments/walk_forward_atr5_threshold.py`
（固定top_n=300，只改`atr5_threshold`），比較目前(0.00748，2026-07-23對
top_n=100算出來的p90舊值，沿用至今)跟針對top_n=300population重算的
p95(0.00856)/p97(0.01000)/p99(0.01318)。

⚠️ 過程中發現並修正了`atr5_pr_check.py::run_absolute_uniform()`的一個
bug：自從ATR5過濾接進`_prepare_data()`之後，這支診斷腳本直接拿
`_prepare_data()`的cache當「全體」population算百分位，但那份cache其實
已經被目前的ATR5門檻篩過一次了——等於在已篩過的資料上再篩一次，百分位
的分母是錯的。已改成直接對「時段過濾後、ATR5過濾前」的原始population
算百分位，`_prepare_data()`也新增了`atr5_threshold`參數（非預設值時完全
跳過cache讀寫，避免弄髒正式cache），才能公平測試更高的候選門檻。

**precision結果**（分模型門檻，5窗口mean）：

| 模型門檻 | 目前(0.00748) | p95(0.00856) | p97(0.01000) | p99(0.01318) |
|---|---|---|---|---|
| 0.5 | 14.50% | 17.27% | 18.52% | 28.08% |
| 0.6 | 21.71% | 23.58% | 27.33% | 32.81% |
| 0.7 | 30.83% | 30.70% | 38.90% | 47.56% |
| 0.8 | 42.11% | 47.12% | **62.05%** | 60.05% |

門檻越高precision越好，p99在0.5/0.6/0.7全面最好，p97在0.8最好，全部候選
都明顯贏過舊門檻。但樣本量代價很大（ATR5過濾前總量353萬筆）：目前門檻
剩238,664筆 → p95剩160,199筆 → p97剩94,736筆 → p99只剩30,354筆。p99在
高模型門檻(0.8)時，單一45天窗口的候選數常常只有個位數(2/4/13/11/71)，
precision數字統計上不穩定，訊號量太少對日內策略而言可能反而不利（就算
勝率高，一天篩不出幾支候選標的）。

**權衡後決定**：選**p97**（樣本量還有9.5萬多、0.8門檻precision全部候選
最高62.05%），改為正式設定。`config.ATR5_FILTER_THRESHOLD = 0.01000`，
`cache/mkt_prepared.parquet`已用新門檻重建，三個模型都已重新訓練過（用
`start_date=2024-01-01`——⚠️ 這個日期比實際資料最早的2024-02-01還早，
等於沒有濾掉任何資料，數字應該跟不設start_date完全一樣，不是特別窄化過
的訓練集）。
