# main/ — 即時交易進入點

`main/live_trader.py` 是 Render web service 的啟動進入點（`python -m main.live_trader`），只管流程編排；實際工作拆到這個資料夾底下幾支各司其職的檔案。

## 模組分工

| 檔案 | 職責 |
|---|---|
| `config.py` | 讀 `.env` 的設定常數（`STRATEGY_MODULES`/`TRADE_MODE`/`FORCE_CLOSE_*`/`CONSENSUS_TOP_N`），以及 settings.json 優先、.env 當 fallback 的 `TOTAL_CAPITAL` |
| `state.py` | `AppState`：跨執行緒共用的執行期狀態；`state.strategies` 是 `{策略名: StrategyState}`，可同時掛多個策略 |
| `strategy_loader.py` | `load_strategies(module_paths)`：動態載入多個策略模組，策略名取模組路徑「`strategy.` 跟 `.live` 中間那段」接起來（`strategy.orb.xgb.live` → `orb_xgb`），撞名會直接 raise |
| `premarket.py` | 盤前資料準備：`refresh_tickers()`（當沖候選清單）/ `refresh_day()`（日K，均量過濾）/ `refresh_prewarm()`（各策略盤前快取） |
| `backfill.py` | `run_startup_backfill()`：開機時若 `db/m1_live/` 已有今日資料，立刻跑一次推論填 SSE monitoring，不用等下一分鐘 |
| `collector.py` | `start_collector()`：分K收集器背景執行緒，包一層自動重試（富邦 WebSocket，`fubon/marketdata_ws.py`） |
| `live_trader.py` | 主程式：建立 `AppState`、載入策略、`_startup()`/`on_minute()`/`_daily_refresh()`/`_force_close_eod()` 四個流程函式、啟動背景執行緒 + uvicorn |

`backfill.py` 跟 `fubon/marketdata_ws.py::_backfill_intraday()` 是不同層級的東西：那支用 REST 補「WebSocket 連線前」缺的原始分K（寫進 `db/m1_live/`），這支是「已經在 `db/m1_live/` 的資料」拿來跑一次推論（填前端監控畫面），兩者互不重疊。

## 多策略架構

一個 process 可以同時掛多個策略（`.env` 的 `STRATEGY_MODULES`，逗號分隔）。每個策略模組（例如 `strategy/orb/xgb/live.py`）要暴露同一組介面：`load_model()` / `predict_live(...)` / `SESSION_START` / `SESSION_END` / `THRESHOLD` / `DIRECTIONS`，包成 `StrategyState` 存進 `state.strategies`（key=策略名）。

**「同一策略、不同模型/方向」用巢狀資料夾掛成獨立策略**（不是共用一個策略內部切換）：例如 `strategy/orb/xgb/live.py`、`strategy/orb/lgbm/live.py`、`strategy/orb/rfc/live.py` 是同一個 orb 策略、三個演算法各自的變體；`strategy/mkt/up/live.py`、`strategy/mkt/down/live.py` 是同一個 mkt 模型、只是各自只送多單或空單訊號。這些變體檔案都只是薄包裝，`load_model()`/`THRESHOLD`/`DIRECTIONS` 各自寫死，`predict_live()`/`SESSION_START`/`SESSION_END` 沿用策略本體（`strategy/{name}/predict.py`、`config.py`）的實作，不重複寫特徵/推論邏輯。

**`DIRECTIONS`**（`{"up"}` / `{"down"}` / `{"up","down"}`）：這個策略允許送出的訊號方向。`orb`/`rally` 目前都只有做多訊號堪用（Triple Barrier 二分類，「跌」代表進場失敗，不是放空訊號）；`mkt` 是3分類模型，`up`/`down` 都送。`on_minute()` 用這個過濾 `predict_live()` 回傳結果，不屬於這個集合的方向不會送到前端；共識訊號比對也依方向分開算（「A策略看漲、B策略看跌」是分歧不是共識）。

**「同模型不同變體」共用一次推論結果，不重算**：`strategy/mkt/up`、`strategy/mkt/down` 這種指向同一個底層模型的變體，`on_minute()` 每分鐘用 `(predict_live函式, model物件)` 當 key 快取這一分鐘的推論結果——第一個變體算完存快取，第二個變體 key 相同就直接沿用，不重跑一次推論。這個 key 要能對得上的前提是兩邊的 `model` 物件真的是同一個，見 `strategy/mkt/train.py::load_model_by_type()` 的記憶體快取（同一個 `model_type` 只從磁碟 `joblib.load()` 一次）；`orb_xgb`/`orb_lgbm` 這種真正不同模型的，`model` 物件不同，不會被誤判成可以共用。快取只在單次 `on_minute()` 呼叫內有效（區域變數），下一分鐘重新算。

## 開機流程

```
建立 state = AppState()
  → strategy_loader.load_strategies(STRATEGY_MODULES)
      填 state.strategies = {策略名: StrategyState}
  → set_strategies(...)                    登記給前端 GET /strategies 查詢（含 directions，前端依此上色/排序）
  → threading.Thread(_startup)             背景執行緒，不阻塞下面的 uvicorn 啟動：
        → 逐一 state.strategies[name].model = load_model()
        → premarket.refresh_tickers(state)      當沖候選清單（可能要跑好幾分鐘，見下方說明）
        → premarket.refresh_day(state)          日K（均量過濾），僅在有候選清單時執行
        → premarket.refresh_prewarm(state)      各策略盤前快取
        → backfill.run_startup_backfill(state)
        → 建立交易引擎（TRADE_MODE != "off" 才建），存進 state.executor
  → 啟動其他背景執行緒：_daily_refresh / _force_close_eod / collector.start_collector
  → uvicorn 主執行緒（阻塞，立刻開始接受連線）
```

**`_startup()` 為什麼是背景執行緒**（2026-07-25）：`refresh_tickers()` 需要對候選股逐支呼叫富邦 API 確認 `canBuyDayTrade`（見 `fubon/subscribe_list.py::_filter_day_tradable()`），上千支股票、每支間隔 0.25 秒，全部跑完要 3~4 分鐘。這段邏輯以前寫在模組最外層、卡在 `uvicorn.Server().run()` 前面，`uvicorn` 真正開始監聽前，前端連 `GET /strategies`、SSE 一律連不上（不是斷線，是伺服器還沒起來）。搬進背景執行緒後 `uvicorn` 立刻起來，這些慢資料背景補齊；`predict_live()` 本來就有 `model=None` 時自動載入的 fallback，`collector`/`on_minute()` 提早開始跑也不會出錯，只是那幾次呼叫會各自重新載入一次模型。

`set_strategies()` 刻意留在 `_startup()` 之外、最上層——這是前端策略清單唯一的資料來源（`GET /strategies`，一次性 API 呼叫，不透過 SSE，也不會因為 SSE 重連而自動重打），要讓它不受 `_startup()` 拖慢。

## 背景執行緒

| 執行緒 | 排程 | 做什麼 |
|---|---|---|
| `_startup` | 開機跑一次 | 模型載入／當沖標的清單／日K／盤前快取／交易引擎（見上方說明） |
| `_daily_refresh` | 每 60 秒檢查一次 | 06:00 呼叫 `premarket.refresh_tickers`+`refresh_day`；08:45 呼叫 `premarket.refresh_prewarm` |
| `_force_close_eod` | 每 30 秒檢查一次 | 到 `force_close_time`（預設 13:25，可由前端即時改）強制平倉所有本系統當沖部位 |
| `collector.start_collector` | 持續執行（阻塞） | 富邦 WebSocket 收即時分K，寫入 `db/m1_live/`，每分鐘固定觸發一次 `on_minute` callback（不管這分鐘有沒有新訊息，保底機制） |

## 每分鐘流程（`on_minute`，由 collector 呼叫）

```
on_minute(minute_str, df)
  → load_m1_live(date_str)              載入今日分K
  → push_candles(...) / push_quote(...) 推前端 K 線圖、頁首固定追蹤報價
  → for s in state.strategies.values():
      (h,m) 不在 s.session_start~session_end → 跳過這個策略
      threshold = 前端 settings 全域門檻 or s.threshold
      查 _infer_cache[(predict_live函式,model物件)]：命中就沿用，沒有才真的呼叫 s.predict_live(...)
      依 s.directions 過濾方向 → push_monitoring / push_inference_log / push_signals
  → 共識訊號：up/down 分開比對，≥2個策略在同一分鐘的前N名重疊（同方向）才算 → push_consensus_signals
  → state.executor.reconcile(all_signals, prices)   下單（永豐 / paper / 富邦，依 TRADE_MODE）
```

## 狀態管理原則

`AppState.strategies` 裡每個 `StrategyState` 的 `model`/`prewarm_cache` 會在 `_startup()`/`_daily_refresh` 重新賦值；`session_start`/`session_end`/`load_model`/`predict_live`/`threshold`/`directions` 都是開機時設定一次、之後只讀（來自對應策略模組的常數，不會執行期變動）。`premarket.py` 的函式不吞例外，錯誤處理與 log 訊息留在呼叫端（`_startup()` 跟 `_daily_refresh()` 對同一種失敗印的訊息、做的 fallback 不同）。
