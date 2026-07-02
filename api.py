"""
即時交易監控 API（FastAPI + Swagger）

Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc

端點：
    GET  /health                     健康檢查
    GET  /dashboard/summary          上方資訊列統計
    GET  /signals/today              左邊訊號列表
    GET  /signals/{stock_id}/detail  右邊訊號詳情
    GET  /chart/{stock_id}/candles   中間 K 圖
    GET  /positions                  下方持倉區
    GET  /trades                     下方成交記錄
    WS   /ws                         即時推送
"""

import asyncio
import json
import threading
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_TW = timezone(timedelta(hours=8))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="just1stock 即時交易監控",
    version="1.0.0",
    description="當沖模型訊號監控平台 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic response models（Swagger 自動生成文件）───────────────────────────


class DashboardSummary(BaseModel):
    open_trades: int = 0
    holding: int
    closed: int  # 已平倉回合數（永豐 FIFO）
    win_rate: Optional[float] = None
    today_pnl_pct: float
    today_pnl_amt: float = 0.0
    total_capital: float = 0.0
    used_quota: float = 0.0
    risk_rejected: int
    errors: int
    last_updated: str


class SignalRecord(BaseModel):
    time: str
    stock_id: str
    name: str
    direction: str  # "buy" | "sell"
    score: int  # proba * 100，0~100
    status: str  # signal_only / risk_pass / sent / filled / holding / closed / failed
    pnl_pct: Optional[float] = None


class Candle(BaseModel):
    time: int | str  # Unix timestamp (int) 或舊格式字串
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


class CandleResponse(BaseModel):
    stock_id: str
    candles: list[Candle]


class LifecycleEvent(BaseModel):
    time: str
    event: str
    detail: Optional[str] = None


class SignalDetail(BaseModel):
    stock_id: str
    name: str
    direction: str
    signal_time: str
    signal_price: float
    filled_avg: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    stop_loss: float
    take_profit: float
    exit_rule: Optional[str] = None
    signal_reasons: list[str] = []
    lifecycle: list[LifecycleEvent] = []


class Position(BaseModel):
    stock_id: str
    name: str
    quantity: int = 0
    pnl_pct: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float


class TradeRecord(BaseModel):
    order_id: str = ""
    time: str
    stock_id: str
    direction: str
    price: float
    quantity: int
    filled: int = 0
    status: str  # FILLED / PARTIAL / SENT / FAILED
    broker_response: Optional[str] = None


# ── 使用者設定（settings.json + HF Hub 備份，覆蓋 .env 預設值）─────────────

import os as _os

_SETTINGS_PATH = Path(__file__).parent / "settings.json"
_HF_SETTINGS_FILENAME = "day_trade/settings.json"
_settings_cache: dict | None = None  # in-memory cache，避免每次 reconcile 讀磁碟


def _hf_download_settings() -> dict:
    """從 HF Hub 拉 settings.json（Render 重啟後磁碟遺失時用）"""
    repo_id = _os.environ.get("HF_REPO_ID", "")
    token = _os.environ.get("HF_TOKEN", "") or None
    if not repo_id:
        return {}
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id,
            filename=_HF_SETTINGS_FILENAME,
            repo_type="dataset",
            token=token,
        )
        data = json.loads(Path(local).read_text(encoding="utf-8"))
        print(f"[設定] 從 HF Hub 載入 settings: {data}")
        return data
    except Exception as e:
        print(f"[設定] HF Hub 無設定檔（首次或未上傳）: {e}")
        return {}


def _hf_upload_settings(data: dict) -> None:
    """把 settings.json 非同步上傳到 HF Hub（寫完本地後背景執行）"""
    repo_id = _os.environ.get("HF_REPO_ID", "")
    token = _os.environ.get("HF_TOKEN", "") or None
    if not repo_id:
        return

    def _upload():
        try:
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=json.dumps(data, ensure_ascii=False, indent=2).encode(),
                path_in_repo=_HF_SETTINGS_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="update settings",
            )
            print("[設定] settings.json 已同步至 HF Hub")
        except Exception as e:
            print(f"[設定] HF Hub 上傳失敗: {e}")

    threading.Thread(target=_upload, daemon=True).start()


def _load_settings() -> dict:
    """讀設定：優先 in-memory cache → 本地 settings.json → HF Hub"""
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache
    # 本地有就用本地
    try:
        _settings_cache = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return _settings_cache
    except Exception:
        pass
    # 本地沒有（Render 重啟）→ 從 HF Hub 拉
    _settings_cache = _hf_download_settings()
    if _settings_cache:
        # 存回本地，避免同一次運行重複下載
        _SETTINGS_PATH.write_text(json.dumps(_settings_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return _settings_cache


def _save_settings(data: dict) -> None:
    global _settings_cache
    _settings_cache = data
    _SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _hf_upload_settings(data)  # 背景同步至 HF Hub


def get_setting(key: str, default=None):
    """取使用者設定值；cache/settings.json 優先，找不到回傳 default（通常來自 .env）"""
    return _load_settings().get(key, default)


# ── In-memory state ──────────────────────────────────────────────────────────

_lock = threading.Lock()
_today_date: date = None

_SUMMARY_DEFAULT: dict = {
    "open_trades": 0,  # 買進成交次數（開倉），即時計
    "holding": 0,  # 目前持倉部位數
    "closed": 0,  # 已平倉回合數（永豐 FIFO 配對，sync_from_broker 更新）
    "wins": 0,
    "win_rate": None,
    "today_pnl_pct": 0.0,
    "today_pnl_amt": 0.0,  # 今日已實現損益（元，估算）
    "total_capital": 0.0,  # 當沖總額度（元），來自 TOTAL_CAPITAL 環境變數
    "used_quota": 0.0,  # 今日已用額度 = 買入金額 + 賣出金額
    "risk_rejected": 0,
    "errors": 0,
    "last_updated": "",
}
_summary: dict = dict(_SUMMARY_DEFAULT)

_collector_status: str = "stopped"  # "running" | "stopped" | "error"

_signals: list = []  # 今日訊號列表
_positions: dict = {}  # {stock_id: Position dict}
_trades: list = []  # 今日原始成交事件（買/賣各一筆）
_completed_trades: list = []  # 今日完整回合（進出配對，含損益）
_candles: dict = {}  # {stock_id: [Candle dict, ...]}
_signal_detail: dict = {}  # {stock_id: SignalDetail dict}
_monitoring: dict = {}  # {stock_id: {stock_id, name, proba, price, is_signal, minute}}

# ── SSE 廣播 ─────────────────────────────────────────────────────────────────

_sse_clients: set[asyncio.Queue] = set()
_event_loop: asyncio.AbstractEventLoop = None


@app.on_event("startup")
async def _capture_loop():
    global _event_loop
    _event_loop = asyncio.get_running_loop()


def _broadcast(data: dict):
    """從同步執行緒安全地推送給所有 SSE 客戶端"""
    if not _event_loop or not _sse_clients:
        return

    async def _enqueue_all():
        for q in list(_sse_clients):
            await q.put(data)

    asyncio.run_coroutine_threadsafe(_enqueue_all(), _event_loop)


# ── Public push functions（由 live_trader.py 呼叫）───────────────────────────


def _reset_if_new_day():
    global _today_date, _summary, _signals, _trades, _completed_trades, _candles, _signal_detail, _positions, _monitoring
    today = datetime.now(_TW).date()
    if _today_date != today:
        _today_date = today
        _signals.clear()
        _trades.clear()
        _completed_trades.clear()
        _candles.clear()
        _signal_detail.clear()
        _positions.clear()
        _monitoring.clear()
        _summary = dict(_SUMMARY_DEFAULT)


def push_monitoring(minute_str: str, all_results: list, threshold: float):
    """推入所有監控股票的最新推論結果（threshold=0 全部），由 on_minute 呼叫"""
    print(f"[push_monitoring] {minute_str} 接收 {len(all_results)} 支股票的推論結果", flush=True)
    with _lock:
        for r in all_results:
            _monitoring[r["stock_id"]] = {
                "stock_id": r["stock_id"],
                "name": r.get("name", r["stock_id"]),
                "proba": round(r["proba"], 4),
                "price": r["price"],
                "direction": r.get("direction", "buy"),
                "is_signal": r["proba"] >= threshold,
                "minute": minute_str[11:16],
            }
        data = sorted(_monitoring.values(), key=lambda x: -x["proba"])
        _broadcast({"type": "monitoring", "minute": minute_str[11:16], "data": data})


def push_signals(minute_str: str, signals: list):
    """每分K推入新訊號（由 on_minute 呼叫）"""
    print(f"[push_signals] {minute_str} 產生 {len(signals)} 筆訊號", flush=True)
    with _lock:
        _reset_if_new_day()
        for s in signals:
            record = {
                "time": minute_str[11:16],
                "stock_id": s["stock_id"],
                "name": s.get("name", s["stock_id"]),
                "direction": "buy",
                "score": int(s["proba"] * 100),
                "status": "signal_only",
                "pnl_pct": None,
            }
            _signals.append(record)
            _summary["today_signals"] += 1
            _summary["last_updated"] = minute_str[11:]

            # 初始化詳情
            _signal_detail[s["stock_id"]] = {
                "stock_id": s["stock_id"],
                "name": s.get("name", s["stock_id"]),
                "direction": "buy",
                "signal_time": minute_str[11:],
                "signal_price": s["price"],
                "filled_avg": None,
                "current_price": s["price"],
                "unrealized_pnl_pct": None,
                "stop_loss": round(s["price"] * 0.97, 2),
                "take_profit": round(s["price"] * 1.03, 2),
                "exit_rule": None,
                "signal_reasons": [],
                "lifecycle": [
                    {"time": minute_str[11:], "event": "產生訊號", "detail": f"模型分數 {int(s['proba']*100)}"}
                ],
            }

        _broadcast({"type": "signals", "minute": minute_str, "data": signals})


def push_candles(stock_id: str, candles: list):
    """推入 K 線資料（index=datetime, open/high/low/close/volume/vwap）"""
    print(f"[push_candles] {stock_id} 推入 {len(candles)} 根 K 線", flush=True)
    with _lock:
        _candles[stock_id] = candles
        _broadcast({"type": "candles", "stock_id": stock_id})


def push_position(position: dict):
    """新增或更新持倉"""
    with _lock:
        sid = position["stock_id"]
        _positions[sid] = position
        _summary["holding"] = len(_positions)
        # 更新訊號列表狀態
        for r in _signals:
            if r["stock_id"] == sid:
                r["status"] = "holding"
                r["pnl_pct"] = position.get("pnl_pct")
        _broadcast({"type": "position", "data": position})


def push_trade(trade: dict):
    """推入成交記錄；open_trades 即時計成交買進，平倉數由永豐 FIFO（closed）為準"""
    with _lock:
        _trades.append(trade)
        status = trade.get("status", "")
        if trade.get("direction") == "buy" and status in {"FILLED", "PARTIAL"}:
            _summary["open_trades"] += 1
        sid = trade["stock_id"]
        # 更新生命週期
        if sid in _signal_detail:
            _signal_detail[sid]["lifecycle"].append(
                {
                    "time": trade["time"],
                    "event": "成交",
                    "detail": f"{trade['quantity']} 股 @ {trade['price']}",
                }
            )
            _signal_detail[sid]["filled_avg"] = trade["price"]
        _broadcast({"type": "trade", "data": trade})


def close_position(stock_id: str, pnl_pct: float, exit_reason: str = "", exit_price: float = 0.0):
    """平倉：移除持倉、更新統計、記錄完整回合"""
    from datetime import datetime

    with _lock:
        pos = _positions.pop(stock_id, {})
        _summary["holding"] = len(_positions)
        _summary["closed"] += 1
        if pnl_pct > 0:
            _summary["wins"] += 1
        closed = _summary["closed"]
        _summary["win_rate"] = round(_summary["wins"] / closed * 100, 1) if closed else None
        prev = _summary["today_pnl_pct"] * (closed - 1)
        _summary["today_pnl_pct"] = round((prev + pnl_pct) / closed, 4)
        if pnl_amt is not None:
            _summary["today_pnl_amt"] = round(_summary.get("today_pnl_amt", 0) + pnl_amt, 0)

        # 完整回合記錄
        entry = pos.get("entry_price", 0)
        qty = pos.get("quantity", 0)
        pnl_amt = round((exit_price - entry) * qty * 1000, 0) if entry and exit_price else None
        _completed_trades.append(
            {
                "time": datetime.now(_TW).strftime("%H:%M:%S"),
                "stock_id": stock_id,
                "name": pos.get("name", stock_id),
                "quantity": qty,
                "entry_price": entry,
                "exit_price": exit_price or None,
                "pnl_pct": round(pnl_pct, 4),
                "pnl_amt": pnl_amt,
                "exit_reason": exit_reason,
            }
        )
        _broadcast({"type": "completed_trade", "data": _completed_trades[-1]})
        for r in _signals:
            if r["stock_id"] == stock_id:
                r["status"] = "closed"
                r["pnl_pct"] = pnl_pct
        if stock_id in _signal_detail:
            _signal_detail[stock_id]["exit_rule"] = exit_reason
        _broadcast({"type": "closed", "stock_id": stock_id, "pnl_pct": pnl_pct})


def push_summary_update(updates: dict):
    """直接更新 _summary 的部分欄位（重啟同步用）"""
    with _lock:
        _summary.update(updates)
        _broadcast({"type": "summary", "data": dict(_summary)})


def sync_broker_snapshot(
    positions: list[dict],
    trades: list[dict],
    completed_trades: list[dict],
    summary_updates: dict,
):
    """用永豐目前狀態重建 dashboard 快取；只覆蓋 broker 相關區塊。"""
    with _lock:
        _reset_if_new_day()

        _positions.clear()
        for pos in positions:
            _positions[pos["stock_id"]] = pos

        _trades.clear()
        _trades.extend(trades)

        _completed_trades.clear()
        _completed_trades.extend(completed_trades)

        filled = {"FILLED", "PARTIAL"}
        open_trades = sum(1 for t in _trades if t.get("direction") == "buy" and t.get("status") in filled)
        _summary.update(
            {
                "holding": len(_positions),
                "open_trades": open_trades,
                # close_trades / closed 由 summary_updates 傳入（FIFO 回合數），不在此重算
            }
        )
        _summary.update(summary_updates)

        snapshot = {
            "summary": dict(_summary),
            "positions": list(_positions.values()),
            "trades": list(reversed(_trades)),
            "completed_trades": list(reversed(_completed_trades)),
        }

    _broadcast({"type": "summary", "data": snapshot["summary"]})
    _broadcast({"type": "broker_snapshot", "data": snapshot})


def update_positions_price(price_map: dict):
    """
    每分鐘更新持倉卡片的現價與浮動損益，price_map = {stock_id: current_price}。
    今日損益（_summary["today_pnl_pct"]）僅由 close_position() 在平倉時更新（已實現）。
    """
    with _lock:
        if not _positions:
            return
        for sid, pos in _positions.items():
            price = price_map.get(sid)
            if price is None or pos.get("entry_price", 0) <= 0:
                continue
            entry = pos["entry_price"]
            pos["current_price"] = price
            pos["pnl_pct"] = round((price - entry) / entry * 100, 4)
            _broadcast({"type": "position", "data": dict(pos)})


# ── REST Endpoints ────────────────────────────────────────────────────────────


def set_collector_status(status: str):
    """由 live_trader.py 呼叫，更新 collector 狀態（'running' | 'stopped' | 'error'）"""
    global _collector_status
    _collector_status = status


_COLLECTOR_MSG = {
    "running": "資料流正常",
    "stopped": "盤後或尚未啟動",
    "error": "資料流中斷",
}


@app.get("/settings", tags=["系統"], summary="取得使用者設定")
def settings_get():
    """回傳 settings.json 的內容（若檔案不存在回傳空物件）"""
    return _load_settings()


@app.post("/settings", tags=["系統"], summary="儲存使用者設定")
async def settings_post(request: Request):
    """
    更新 settings.json 並立即套用至儀表板。
    目前支援欄位：
      total_capital (float)：當沖總額度（元）
    """
    body: dict = await request.json()
    current = _load_settings()
    for k, v in body.items():
        if v is None:
            current.pop(k, None)  # null → 刪除 key，回落 .env 預設
        else:
            current[k] = v
    _save_settings(current)
    if "total_capital" in body and body["total_capital"] is not None:
        push_summary_update({"total_capital": float(body["total_capital"])})
    return {"ok": True, "settings": current}


@app.get("/health", tags=["系統"], summary="健康檢查")
def health():
    print(
        f"[GET /health] sse_clients={len(_sse_clients)} signals={len(_signals)} positions={len(_positions)}", flush=True
    )
    with _lock:
        last_signal = _signals[-1]["time"] if _signals else None
    return {
        "status": "ok",
        "collector": _collector_status,
        "message": _COLLECTOR_MSG.get(_collector_status, _collector_status),
        "sse_clients": len(_sse_clients),
        "ws_clients": len(_sse_clients),
        "last_signal_at": last_signal,
    }


@app.get(
    "/dashboard/summary",
    response_model=DashboardSummary,
    tags=["儀表板"],
    summary="上方資訊列統計",
)
def dashboard_summary():
    print(
        f"[GET /dashboard/summary] 回傳摘要：holding={_summary.get('holding')} pnl={_summary.get('today_pnl_pct')}%",
        flush=True,
    )
    with _lock:
        return dict(_summary)


@app.get(
    "/signals/today",
    response_model=list[SignalRecord],
    tags=["訊號"],
    summary="今日訊號列表（左邊訊號區）",
)
def signals_today():
    print(f"[GET /signals/today] 回傳 {len(_signals)} 筆訊號", flush=True)
    with _lock:
        return list(reversed(_signals))  # 最新在上


@app.get(
    "/signals/{stock_id}/detail",
    response_model=SignalDetail,
    tags=["訊號"],
    summary="訊號詳情（右邊詳情區，依選定股票）",
)
def signal_detail(stock_id: str):
    with _lock:
        detail = _signal_detail.get(stock_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"無 {stock_id} 的訊號資料")
    return detail


@app.get(
    "/chart/{stock_id}/candles",
    response_model=CandleResponse,
    tags=["圖表"],
    summary="K 線資料（中間 K 圖，依選定股票）",
)
def chart_candles(stock_id: str):
    with _lock:
        candles = list(_candles.get(stock_id, []))
    print(f"[GET /chart/{stock_id}/candles] 回傳 {len(candles)} 根 K 線", flush=True)
    return {"stock_id": stock_id, "candles": candles}


@app.get("/monitoring", tags=["監控"], summary="監控中股票的最新推論結果（依信心度排序）")
def get_monitoring():
    print(f"[GET /monitoring] 回傳 {len(_monitoring)} 支監控股票", flush=True)
    with _lock:
        return sorted(_monitoring.values(), key=lambda x: -x["proba"])


@app.get("/completed_trades", tags=["成交"], summary="今日已完成回合（進出配對，含損益）")
def completed_trades():
    with _lock:
        return list(reversed(_completed_trades))  # 最新在上


@app.get("/failed_orders", tags=["成交"], summary="今日失敗委託（含錯誤訊息）")
def failed_orders():
    with _lock:
        return list(reversed([t for t in _trades if t.get("status") == "FAILED"]))


def push_completed_trades_from_broker(closed_list: list, name_lookup=None):
    """重啟後從永豐重建今日已平倉記錄（get_closed_today 回傳）。
    name_lookup: callable(stock_id) → str，可傳入 _tickers.get 查公司名。
    """
    _lookup = name_lookup or (lambda sid: sid)
    with _lock:
        _completed_trades.clear()
        for c in closed_list:
            entry = c.get("buy_avg", 0)
            ex = c.get("sell_avg", 0)
            qty = c.get("quantity", 0)
            pnl_pct = c.get("pnl_pct", 0.0)
            pnl_amt = round((ex - entry) * qty * 1000, 0) if entry and ex else None
            sid = c["stock_id"]
            _completed_trades.append(
                {
                    "time": "-",
                    "stock_id": sid,
                    "name": _lookup(sid),
                    "quantity": qty,
                    "entry_price": entry,
                    "exit_price": ex,
                    "pnl_pct": pnl_pct,
                    "pnl_amt": pnl_amt,
                    "exit_reason": "broker_sync",
                }
            )


@app.get(
    "/positions",
    response_model=list[Position],
    tags=["持倉"],
    summary="當前持倉（下方持倉區）",
)
def positions():
    print(f"[GET /positions] 回傳 {len(_positions)} 檔持倉", flush=True)
    with _lock:
        return list(_positions.values())


@app.get(
    "/trades",
    response_model=list[TradeRecord],
    tags=["成交"],
    summary="今日成交記錄（下方成交區）",
)
def trades():
    print(f"[GET /trades] 回傳 {len(_trades)} 筆成交", flush=True)
    with _lock:
        return list(reversed(_trades))  # 最新在上


# ── SSE ──────────────────────────────────────────────────────────────────────


@app.get("/stream", tags=["即時推送"], summary="SSE 即時推送（訊號 / 持倉 / 成交）")
async def event_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue()
    _sse_clients.add(queue)

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=5)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # 保持連線，同時讓 is_disconnected 有機會偵測
        finally:
            _sse_clients.discard(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Start（由 live_trader.py 呼叫）───────────────────────────────────────────


def get_uvicorn_config(host: str = "0.0.0.0", port: int = 8000) -> uvicorn.Config:
    return uvicorn.Config(app=app, host=host, port=port, log_level="warning")
