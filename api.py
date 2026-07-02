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
from datetime import date
from typing import Optional

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
    today_signals: int
    sent_orders: int
    filled: int
    holding: int
    closed: int
    win_rate: Optional[float] = None
    today_pnl_pct: float
    risk_rejected: int
    errors: int
    last_updated: str

class SignalRecord(BaseModel):
    time: str
    stock_id: str
    name: str
    direction: str          # "buy" | "sell"
    score: int              # proba * 100，0~100
    status: str             # signal_only / risk_pass / sent / filled / holding / closed / failed
    pnl_pct: Optional[float] = None

class Candle(BaseModel):
    time: int | str   # Unix timestamp (int) 或舊格式字串
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
    time: str
    stock_id: str
    direction: str
    price: float
    quantity: int
    status: str             # FILLED / PARTIAL / SENT / FAILED
    broker_response: Optional[str] = None

# ── In-memory state ──────────────────────────────────────────────────────────

_lock = threading.Lock()
_today_date: date = None

_SUMMARY_DEFAULT: dict = {
    "today_signals": 0,
    "sent_orders": 0,
    "filled": 0,
    "holding": 0,
    "closed": 0,
    "wins": 0,
    "win_rate": None,
    "today_pnl_pct": 0.0,
    "risk_rejected": 0,
    "errors": 0,
    "last_updated": "",
}
_summary: dict = dict(_SUMMARY_DEFAULT)

_collector_status: str = "stopped"   # "running" | "stopped" | "error"

_signals: list = []            # 今日訊號列表
_positions: dict = {}          # {stock_id: Position dict}
_trades: list = []             # 今日原始成交事件（買/賣各一筆）
_completed_trades: list = []   # 今日完整回合（進出配對，含損益）
_candles: dict = {}            # {stock_id: [Candle dict, ...]}
_signal_detail: dict = {}      # {stock_id: SignalDetail dict}
_monitoring: dict = {}         # {stock_id: {stock_id, name, proba, price, is_signal, minute}}

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
    today = date.today()
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
    """推入成交記錄"""
    with _lock:
        _trades.append(trade)
        _summary["filled"] = len(_trades)
        sid = trade["stock_id"]
        # 更新生命週期
        if sid in _signal_detail:
            _signal_detail[sid]["lifecycle"].append({
                "time": trade["time"],
                "event": "成交",
                "detail": f"{trade['quantity']} 股 @ {trade['price']}",
            })
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

        # 完整回合記錄
        entry = pos.get("entry_price", 0)
        qty = pos.get("quantity", 0)
        pnl_amt = round((exit_price - entry) * qty * 1000, 0) if entry and exit_price else None
        _completed_trades.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "stock_id": stock_id,
            "name": pos.get("name", stock_id),
            "quantity": qty,
            "entry_price": entry,
            "exit_price": exit_price or None,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_amt": pnl_amt,
            "exit_reason": exit_reason,
        })
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

        _summary.update({
            "holding": len(_positions),
            "sent_orders": len(_trades),
            "filled": sum(1 for t in _trades if t.get("status") in {"FILLED", "PARTIAL"}),
            "closed": len(_completed_trades),
        })
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
    "error":   "資料流中斷",
}

@app.get("/health", tags=["系統"], summary="健康檢查")
def health():
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
    with _lock:
        return dict(_summary)


@app.get(
    "/signals/today",
    response_model=list[SignalRecord],
    tags=["訊號"],
    summary="今日訊號列表（左邊訊號區）",
)
def signals_today():
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
    return {"stock_id": stock_id, "candles": candles}


@app.get("/monitoring", tags=["監控"], summary="監控中股票的最新推論結果（依信心度排序）")
def get_monitoring():
    with _lock:
        return sorted(_monitoring.values(), key=lambda x: -x["proba"])


@app.get("/completed_trades", tags=["成交"], summary="今日已完成回合（進出配對，含損益）")
def completed_trades():
    with _lock:
        return list(reversed(_completed_trades))  # 最新在上


def push_completed_trades_from_broker(closed_list: list):
    """重啟後從永豐重建今日已平倉記錄（get_closed_today 回傳）"""
    with _lock:
        _completed_trades.clear()
        for c in closed_list:
            entry = c.get("buy_avg", 0)
            ex = c.get("sell_avg", 0)
            qty = c.get("quantity", 0)
            pnl_pct = c.get("pnl_pct", 0.0)
            pnl_amt = round((ex - entry) * qty * 1000, 0) if entry and ex else None
            _completed_trades.append({
                "time": "-",
                "stock_id": c["stock_id"],
                "name": c["stock_id"],
                "quantity": qty,
                "entry_price": entry,
                "exit_price": ex,
                "pnl_pct": pnl_pct,
                "pnl_amt": pnl_amt,
                "exit_reason": "broker_sync",
            })


@app.get(
    "/positions",
    response_model=list[Position],
    tags=["持倉"],
    summary="當前持倉（下方持倉區）",
)
def positions():
    with _lock:
        return list(_positions.values())


@app.get(
    "/trades",
    response_model=list[TradeRecord],
    tags=["成交"],
    summary="今日成交記錄（下方成交區）",
)
def trades():
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
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"   # 保持連線
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
