# CLAUDE.md

## 重要慣例：分K/日K 的 `date` 欄位轉 Unix timestamp

專案裡所有分K/日K資料的 `date` 欄位（Fugle API 回傳、`db/m1_live/`、`db/m1/`、
`db/fugle_day/` 存的）都是 **naive datetime／字串，但代表台北本地時間（UTC+8）**，
不是 UTC。

**絕對不要**用 `calendar.timegm(dt.timetuple())`，或未先 `tz_localize` 就呼叫
`pd.Timestamp.timestamp()` / `datetime.timestamp()`，把這種欄位轉成 Unix epoch——
這些寫法都會把台北時間誤當成 UTC，導致算出來的 timestamp **多算 8 小時**。

**正確做法**：一律呼叫 `api.py` 裡的 `tw_naive_to_epoch(dt)`（同時接受 python
`datetime` 或 pandas `Timestamp`）。任何新的 API 端點或程式碼，只要要把這類
`date` 欄位轉成給前端圖表（例如 lightweight-charts）用的 Unix timestamp，
都要用這支函式，不要自己重寫轉換邏輯。

```python
from api import tw_naive_to_epoch  # live_trader.py 等外部模組
# 或在 api.py 內部直接呼叫 tw_naive_to_epoch(dt)

ts = tw_naive_to_epoch(row["date"])  # 正確：回傳真正的 UTC epoch 秒數
```

已知修過這個 bug 的地方（2026-07-05 發現並修正）：
- `api.py` `GET /chart/{stock_id}/candles/history`
- `live_trader.py` `on_minute()` 推送即時 K 線給 `GET /chart/{stock_id}/candles`

下次新增任何會回傳 K 線／時間序列給前端的 API，記得檢查有沒有踩到同一個問題。
