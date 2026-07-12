# main/ — 即時交易進入點

`main/live_trader.py` 是 Render web service 的啟動進入點（`python -m main.live_trader`），只管流程編排；實際工作拆到這個資料夾底下幾支各司其職的檔案。

## 模組分工

| 檔案 | 職責 |
|---|---|
| `config.py` | 讀 `.env` 的設定常數（`STRATEGY_MODULE`/`TRADE_MODE`/`THRESHOLD`/`FORCE_CLOSE_*`/`M1_COLLECTOR`），以及 settings.json 優先、.env 當 fallback 的 `TOTAL_CAPITAL` |
| `state.py` | `AppState`：跨執行緒共用的執行期狀態（策略介面、盤前資料、交易引擎），取代模組級 global 變數 |
| `strategy_loader.py` | `load_strategy(module_path)`：動態載入策略模組（切換策略/模型不用改程式碼，改 `.env` 的 `STRATEGY_MODULE` 就好） |
| `premarket.py` | 盤前資料準備：`refresh_tickers()`（當沖候選清單）/ `refresh_day()`（日K，均量過濾）/ `refresh_prewarm()`（策略盤前快取） |
| `backfill.py` | `run_startup_backfill()`：開機時若 `db/m1_live/` 已有今日資料，立刻跑一次推論填 SSE monitoring，不用等下一分鐘 |
| `collector.py` | `start_collector()`：分K收集器，依 `M1_COLLECTOR` 切換 REST（Fugle，預設）或富邦 WebSocket |
| `live_trader.py` | 主程式：建立 `AppState`、依序呼叫上面幾支模組、`on_minute()`/`_daily_refresh()`/`_force_close_eod()` 三個流程函式、啟動背景執行緒 + uvicorn |

`backfill.py` 跟 `fubon/marketdata_ws.py::_backfill_intraday()` 是不同層級的東西：那支用 REST 補「WebSocket 連線前」缺的原始分K（寫進 `db/m1_live/`），這支是「已經在 `db/m1_live/` 的資料」拿來跑一次推論（填前端監控畫面），兩者互不重疊。

## 開機流程

```
建立 state = AppState()
  → strategy_loader.load_strategy(STRATEGY_MODULE)
      填 state.strategy_module / session_start / session_end / load_model / predict_live
  → state.model = state.load_model()
  → premarket.refresh_tickers(state)      當沖候選清單
  → premarket.refresh_day(state)          日K（均量過濾），僅在有候選清單時執行
  → premarket.refresh_prewarm(state)      策略盤前快取（不需要的策略回傳空 dict）
  → backfill.run_startup_backfill(state, THRESHOLD)
  → 建立交易引擎（TRADE_MODE != "off" 才建），存進 state.executor
  → 啟動背景執行緒：_daily_refresh / _force_close_eod / collector.start_collector
  → uvicorn 主執行緒（阻塞）
```

## 背景執行緒

| 執行緒 | 排程 | 做什麼 |
|---|---|---|
| `_daily_refresh` | 每 60 秒檢查一次 | 06:00 呼叫 `premarket.refresh_tickers`+`refresh_day`；08:45 呼叫 `premarket.refresh_prewarm` |
| `_force_close_eod` | 每 30 秒檢查一次 | 到 `force_close_time`（預設 13:25，可由前端即時改）強制平倉所有本系統當沖部位 |
| `collector.start_collector` | 持續執行（阻塞） | 每分鐘（REST 輪詢）或即時（富邦 WebSocket 推送）取得分K，寫入 `db/m1_live/`，並呼叫 `on_minute` callback |

## 每分鐘流程（`on_minute`，由 collector 呼叫）

```
on_minute(minute_str, df)
  → (h,m) < session_start           → return（尚未開盤）
  → (h,m) > session_end             → 只跑 reconcile([]) 做 SL/TP，不開倉，然後 return
  → load_m1_live(date_str)          載入今日分K
  → push_candles(...)               推前端 K 線圖
  → state.predict_live(...)         模型推論（threshold=0，拿到所有股票機率）
  → push_monitoring / push_inference_log
  → update_positions_price(...)     持倉浮動損益
  → signals = [proba >= threshold]
  → push_signals(...)
  → state.executor.reconcile(signals, prices)   下單（永豐 / paper / 富邦，依 TRADE_MODE）
```

## 狀態管理原則

`AppState` 裡只有 `tickers`/`day_trade_stocks`/`day`/`prewarm_cache` 這 4 個欄位會在執行期被 `_daily_refresh` 重新賦值；`model`/`executor`/策略介面（`session_start`/`session_end`/`load_model`/`predict_live`）都是開機時設定一次、之後只讀。`premarket.py` 的函式不吞例外，錯誤處理與 log 訊息留在呼叫端（`live_trader.py` 的 bootstrap 區塊跟 `_daily_refresh()` 對同一種失敗印的訊息、做的 fallback 不同）。
