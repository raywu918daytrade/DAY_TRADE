"""
即時交易監控 API（FastAPI + Swagger）

Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc

端點：
    GET  /health                     健康檢查
    GET  /dashboard/summary          上方資訊列統計
    GET  /signals/today              左邊訊號列表
    GET  /consensus/today            今日「多個策略前N名重疊」訊號列表
    GET  /conflict/today             今日「多空衝突（反轉）」訊號列表
    GET  /strategies                 目前啟用的策略清單（幾個模型、各自session範圍）
    GET  /signals/{stock_id}/detail  右邊訊號詳情
    GET  /chart/{stock_id}/candles   中間 K 圖
    GET  /chart/{stock_id}/candles/history   歷史K線（任意過去日期，交易回放用）
    GET  /quote/{stock_id}           固定追蹤清單股票的最新報價/漲跌幅（見 push_quote()）
    GET  /positions                  下方持倉區
    GET  /trades                     下方成交記錄
    GET  /api/inference/history/dates        可回放日期清單（HF Hub 上實際有資料的日期）
    GET  /api/inference/history      每分鐘推論歷史記錄（給前端分析用，今日記憶體/過去日期 HF Hub）
    WS   /ws                         即時推送
"""

import asyncio
import json
import threading
import time as _time_mod
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_TW = timezone(timedelta(hours=8))


def tw_naive_to_epoch(dt) -> int:
    """把「naive datetime，但實際代表台北本地時間」的值轉成正確的 UTC epoch 秒數。

    專案裡分K的 date 欄位（Fugle 回傳、db/m1_live 存的、db/m1 存的）一律是這種格式：
    看起來沒有時區資訊，但實際上代表台北時間。絕對不能直接用 calendar.timegm() 或
    .timestamp()（兩者都會把「台北時間」誤當成「UTC 時間」，導致 timestamp 多算 8 小時），
    一律透過這支函式轉換，才不會每個新端點各自重犯同一個 bug。

    接受 python datetime 或 pandas Timestamp（兩者的 tz 標記方式不同，這裡統一處理）。
    """
    if hasattr(dt, "tz_localize"):  # pandas Timestamp
        return int(dt.tz_localize(_TW).timestamp())
    return int(dt.replace(tzinfo=_TW).timestamp())


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

from pattern.pattern_api import router as pattern_router

app.include_router(pattern_router)

_SKIP_LOG_PATHS = {"/stream", "/health"}

@app.middleware("http")
async def _log_http(request: Request, call_next):
    """前端 → 後端每次 API 呼叫記錄到系統 log（/stream SSE 和 /health 除外）"""
    if request.url.path in _SKIP_LOG_PATHS:
        return await call_next(request)
    t0 = _time_mod.time()
    response = await call_next(request)
    elapsed_ms = int((_time_mod.time() - t0) * 1000)
    path = request.url.path
    qs = f"?{request.url.query}" if request.url.query else ""
    append_system_log(
        f"{request.method} {path}{qs} → {response.status_code} ({elapsed_ms}ms)",
        level="info" if response.status_code < 400 else "error",
    )
    return response

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


class InferenceRecord(BaseModel):
    date: str
    time: str
    stock_id: str
    name: str
    proba: float
    price: Optional[float] = None
    direction: str = "buy"
    is_signal: bool = False


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
    "total_capital": 0.0,  # 當沖總額度（元），使用者設定值，來自 settings.json / TOTAL_CAPITAL 環境變數
    "used_quota": 0.0,  # 今日已用額度 = 買入金額 + 賣出金額
    "broker_balance": None,  # 永豐帳戶現金餘額（元），僅供顯示參考，非官方當沖核准額度
    "today_signals": 0,
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
_quotes: dict = {}  # {stock_id: {stock_id, price, prev_close, change_pct, minute}}，見 push_quote()
_signal_detail: dict = {}  # {stock_id: SignalDetail dict}
_monitoring: dict = {}  # {stock_id: {stock_id, name, proba, price, is_signal, minute}}
_consensus_signals: list = []  # 今日「多個策略的前N名重疊」記錄，見 push_consensus_signals()
_conflict_signals: list = []  # 今日「多空衝突（反轉）」記錄，見 push_conflict_signals()
_vwap_breakout_signals: list = []  # 今日「VWAP突破/跌破」記錄，見 push_vwap_breakout()
_strategies_registry: dict = {"strategies": [], "consensus_top_n": 0}  # 見 set_strategies()
_force_close_queue: set = set()  # 前端觸發的立即平倉股票清單

from collections import OrderedDict as _OrderedDict

# 每分鐘推論記錄（全部監控股票，threshold=0）：今日累積在記憶體，背景同步整天到 HF Hub
_inference_buffer: list = []
_inference_date: date = None
_inference_io_lock = threading.Lock()  # 序列化背景寫檔/上傳，避免同分鐘重疊
_inference_hf_cache: _OrderedDict = _OrderedDict()  # 過去日期查詢快取（date_str -> DataFrame），LRU 上限 14 天
_INFERENCE_HF_CACHE_MAX = 14

# ── Log 緩衝區 ────────────────────────────────────────────────────────────────
from collections import deque as _deque
import json as _json_mod

_system_logs: _deque = _deque(maxlen=500)  # 程式/系統事件
_trade_logs: _deque = _deque(maxlen=500)   # 交易事件（補充 _trades 的詳細說明）
_LOG_DIR = Path(__file__).parent / "logs"


def _write_log_file(entry: dict, cat: str):
    """Append one log entry to today's JSONL file (logs/YYYY-MM-DD.jsonl)."""
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        today = datetime.now(_TW).strftime("%Y-%m-%d")
        path = _LOG_DIR / f"{today}.jsonl"
        line = _json_mod.dumps({"cat": cat, **entry}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception as _e:
        print(f"[LOG FILE ERROR] {_e}", flush=True)


def _read_log_file() -> list[dict]:
    """Read today's JSONL log file; returns [] on any error."""
    try:
        today = datetime.now(_TW).strftime("%Y-%m-%d")
        path = _LOG_DIR / f"{today}.jsonl"
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        entries.append(_json_mod.loads(_line))
                    except Exception:
                        pass
        return entries
    except Exception:
        return []

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


def set_strategies(strategies: list[dict], consensus_top_n: int = 0):
    """由 main/live_trader.py 開機時呼叫一次，登記目前有哪些策略在跑，供前端
    GET /strategies 查詢——例如要動態渲染幾組監控面板、每組叫什麼名字，不用
    等第一筆 monitoring/signals 事件送到才知道有幾個模型。

    strategies: [{"name":..., "session_start":"HH:MM", "session_end":"HH:MM",
    "directions": ["up"]|["down"]|["up","down"]}, ...]——directions 給前端依此
    幫策略欄位上色（2026-07-23討論：orb/rally 目前只做多，mkt 多空都送）。
    """
    global _strategies_registry
    _strategies_registry = {"strategies": strategies, "consensus_top_n": consensus_top_n}


# ── Public push functions（由 live_trader.py 呼叫）───────────────────────────


def _reset_if_new_day():
    global _today_date, _summary, _signals, _trades, _completed_trades, _candles, _quotes, _signal_detail, _positions, _monitoring, _consensus_signals, _conflict_signals, _vwap_breakout_signals
    today = datetime.now(_TW).date()
    if _today_date != today:
        _today_date = today
        _signals.clear()
        _trades.clear()
        _completed_trades.clear()
        _candles.clear()
        _quotes.clear()
        _signal_detail.clear()
        _positions.clear()
        _system_logs.clear()
        _trade_logs.clear()
        _monitoring.clear()
        _consensus_signals.clear()
        _conflict_signals.clear()
        _vwap_breakout_signals.clear()
        _summary = dict(_SUMMARY_DEFAULT)


def push_monitoring(minute_str: str, all_results: list, threshold: float, strategy: str = ""):
    """推入所有監控股票的最新推論結果（threshold=0 全部），由 on_minute 呼叫。

    strategy: 產生這批結果的策略名稱（main/state.py 的 StrategyState.name）。
    可能同時有多個策略在跑（見 main/config.py 的 STRATEGY_MODULES），_monitoring
    的 key 因此是 f"{strategy}:{stock_id}" 不是單純 stock_id，避免不同策略
    對同一支股票的推論互相覆蓋掉；每筆資料也帶 "strategy" 欄位，前端目前還是
    混在一起顯示（還沒做分策略 UI），之後要分開顯示時直接篩這個欄位。
    strategy 留空（沒有策略概念的舊呼叫端）就退回單純用 stock_id 當 key。
    """
    print(f"[push_monitoring] {minute_str} [{strategy or '-'}] 接收 {len(all_results)} 支股票的推論結果", flush=True)
    cur_dt = datetime.strptime(minute_str, "%Y-%m-%d %H:%M:%S")
    date_str = minute_str[:10]
    with _lock:
        for r in all_results:
            key = f"{strategy}:{r['stock_id']}" if strategy else r["stock_id"]
            _monitoring[key] = {
                "stock_id": r["stock_id"],
                "strategy": strategy,
                "name": r.get("name", r["stock_id"]),
                "proba": round(r["proba"], 4),
                "price": r["price"],
                "volume": r.get("volume"),  # 這一分鐘的成交量（不是累積量），純參考用
                "direction": r.get("direction", "buy"),
                "is_signal": r["proba"] >= threshold,
                "minute": minute_str[11:16],
            }
        # 清掉超過 3 分鐘沒被本輪 all_results 更新到的股票：這種股票已經不在目前
        # 實際被推論的候選清單裡（例如 day_trade_stocks 名單變動、盤中被剔除），
        # 若不清掉，它舊的（通常偏高的）proba 會永遠卡在排行榜頂端，跟 on_minute
        # 印出的 top5 log 對不起來，且前端「監控中」清單會顯示早就過期的股票。
        stale_keys = [
            key for key, v in _monitoring.items()
            if (cur_dt - datetime.strptime(f"{date_str} {v['minute']}", "%Y-%m-%d %H:%M")).total_seconds() > 180
        ]
        for key in stale_keys:
            del _monitoring[key]
        data = sorted(_monitoring.values(), key=lambda x: -x["proba"])
        _broadcast({"type": "monitoring", "minute": minute_str[11:16], "data": data})


_INFERENCE_LOCAL_DIR = Path(__file__).parent / "db" / "inference_live"
_HF_INFERENCE_PREFIX = "day_trade/inference"


def push_inference_log(minute_str: str, all_results: list, threshold: float = 0.0, strategy: str = ""):
    """每分鐘全部監控股票的推論結果落地：記憶體累積今日資料 + 背景寫本地 parquet + 同步至 HF Hub。
    由 on_minute 在 push_monitoring 之後呼叫，供 /api/inference/history 分析查詢用。

    strategy: 見 push_monitoring() 的說明，附加成一個欄位（純新增欄位，parquet/
    下游查詢不會因為多這欄而壞掉）。
    """
    global _inference_buffer, _inference_date
    if not all_results:
        return
    date_str = minute_str[:10]
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    with _lock:
        if _inference_date != day:
            _inference_date = day
            _inference_buffer = []
        for r in all_results:
            _inference_buffer.append(
                {
                    "date": date_str,
                    "time": minute_str[11:16],
                    "strategy": strategy,
                    "stock_id": r["stock_id"],
                    "name": r.get("name", r["stock_id"]),
                    "proba": round(float(r["proba"]), 4),
                    "price": r.get("price"),
                    "direction": r.get("direction", "buy"),
                    "is_signal": bool(r["proba"] >= threshold),
                }
            )
        rows = list(_inference_buffer)
    _persist_inference_log(date_str, rows)


def _persist_inference_log(date_str: str, rows: list) -> None:
    """背景執行緒：整天推論記錄覆寫本地 parquet，並同步到 HF Hub（無 HF_REPO_ID 時只寫本地）"""

    def _write():
        with _inference_io_lock:
            try:
                import pandas as pd

                df = pd.DataFrame(rows)
                _INFERENCE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                local_path = _INFERENCE_LOCAL_DIR / f"{date_str}.parquet"
                df.to_parquet(local_path, index=False, compression="zstd")

                repo_id = _os.environ.get("HF_REPO_ID", "")
                token = _os.environ.get("HF_TOKEN", "") or None
                if not repo_id:
                    return
                from huggingface_hub import HfApi

                HfApi().upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"{_HF_INFERENCE_PREFIX}/{date_str}.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"inference log {date_str}",
                )
                print(f"[推論記錄] {date_str} 已同步至 HF Hub（{len(df)} 筆）", flush=True)
            except Exception as e:
                print(f"[推論記錄] 寫入/上傳失敗: {e}", flush=True)

    threading.Thread(target=_write, daemon=True).start()


def _load_inference_day(date_str: str):
    """取得指定日期的推論記錄 DataFrame：今日走記憶體 buffer，過去日期從 HF Hub 下載（含 LRU cache）"""
    import pandas as pd

    today_str = datetime.now(_TW).strftime("%Y-%m-%d")
    if date_str == today_str:
        with _lock:
            rows = list(_inference_buffer)
        return pd.DataFrame(rows)

    if date_str in _inference_hf_cache:
        _inference_hf_cache.move_to_end(date_str)
        return _inference_hf_cache[date_str]

    repo_id = _os.environ.get("HF_REPO_ID", "")
    token = _os.environ.get("HF_TOKEN", "") or None
    df = pd.DataFrame()
    if repo_id:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                repo_id=repo_id,
                filename=f"{_HF_INFERENCE_PREFIX}/{date_str}.parquet",
                repo_type="dataset",
                token=token,
            )
            df = pd.read_parquet(local)
        except Exception as e:
            print(f"[推論記錄] {date_str} HF Hub 無資料: {e}", flush=True)

    _inference_hf_cache[date_str] = df
    _inference_hf_cache.move_to_end(date_str)
    while len(_inference_hf_cache) > _INFERENCE_HF_CACHE_MAX:
        _inference_hf_cache.popitem(last=False)
    return df


def push_signals(minute_str: str, signals: list, strategy: str = ""):
    """每分K推入新訊號（由 on_minute 呼叫）。

    strategy: 見 push_monitoring() 的說明。_signal_detail 目前還是單純用
    stock_id 當 key（跟交易回合/成交配對邏輯綁在一起，見下面 update_signal_status
    這類函式），如果兩個策略同一天對同一支股票都發訊號，後者會覆蓋前者的
    詳情——TRADE_MODE=off（交易先暫停）時不影響監控顯示，但之後要多策略
    同時「真的下單」時，這裡要重新設計成 (strategy, stock_id) 當 key，不要
    假設現在已經處理好。
    """
    print(f"[push_signals] {minute_str} [{strategy or '-'}] 產生 {len(signals)} 筆訊號", flush=True)
    with _lock:
        _reset_if_new_day()
        for s in signals:
            record = {
                "time": minute_str[11:16],
                "strategy": strategy,
                "stock_id": s["stock_id"],
                "name": s.get("name", s["stock_id"]),
                "direction": s.get("direction", "buy"),
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
                "strategy": strategy,
                "name": s.get("name", s["stock_id"]),
                "direction": s.get("direction", "buy"),
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

        _broadcast({"type": "signals", "minute": minute_str, "strategy": strategy, "data": signals})


def push_consensus_signals(minute_str: str, consensus: list):
    """推入「多個策略的前N名重疊」訊號（由 main/live_trader.py 的 on_minute() 在
    所有策略都跑完那分鐘的推論後呼叫，見 CONSENSUS_TOP_N 設定）。

    consensus 每筆結構：{"stock_id", "name", "strategies": [策略名,...],
    "probas": {策略名: proba}}——同一支股票同時出現在2個以上策略的前N名，
    才會被列進來，代表「多個模型都看好」，給前端額外標記參考用，跟個別
    策略自己的 monitoring/signals 是分開的資訊，不會互相取代。
    """
    print(f"[push_consensus_signals] {minute_str} 產生 {len(consensus)} 筆重疊訊號", flush=True)
    with _lock:
        _reset_if_new_day()
        for c in consensus:
            _consensus_signals.append({**c, "time": minute_str[11:16]})
        _broadcast({"type": "consensus_signals", "minute": minute_str[11:16], "data": consensus})


def push_conflict_signals(minute_str: str, conflicts: list):
    """推入「多空衝突（反轉）」訊號（由 main/live_trader.py 的 on_minute() 在
    所有策略都跑完那分鐘的推論後呼叫，見 main/config.py::CONFLICT_THRESHOLD）。

    conflicts 每筆結構：{"stock_id", "name", "up_proba", "up_strategy",
    "down_proba", "down_strategy"}——同一支股票，多方最高信心度、空方最高
    信心度都超過門檻，代表模型之間對這支股票方向嚴重分歧，跟 push_consensus_signals()
    （要求同方向）邏輯相反，分開推送，不取代。
    """
    print(f"[push_conflict_signals] {minute_str} 產生 {len(conflicts)} 筆多空衝突訊號", flush=True)
    with _lock:
        _reset_if_new_day()
        for c in conflicts:
            _conflict_signals.append({**c, "time": minute_str[11:16]})
        _broadcast({"type": "conflict_signals", "minute": minute_str[11:16], "data": conflicts})


def push_vwap_breakout(minute_str: str, breakouts: list):
    """推入「VWAP突破/跌破」訊號（main/live_trader.py 的 on_minute() 每分鐘偵測，
    見該函式內的說明）。

    只判斷「剛好跨過那條線」的那一分鐘（跟上一分鐘比，收盤價相對VWAP的
    位置有沒有變號），不是「目前在VWAP上/下方」這種持續狀態——不然同一支
    股票只要一直維持在VWAP上方，會每分鐘都被判定成「突破」，訊號就沒有
    意義了。只涵蓋當沖候選(~400)這個即時收集1分K的股票池（db/m1_live），
    全市場其他股票盤中沒有即時資料可以判斷，2026-08-12跟使用者確認過
    不做全市場版本（見股票清單欄「當沖候選/全市場」切換是另一個獨立
    查詢功能，跟這個逐分鐘自動偵測的機制不一樣）。

    breakouts 每筆結構：{"stock_id", "name", "direction"("up"=突破/"down"=
    跌破), "price", "vwap"}
    """
    print(f"[push_vwap_breakout] {minute_str} 產生 {len(breakouts)} 筆VWAP突破訊號", flush=True)
    with _lock:
        _reset_if_new_day()
        for b in breakouts:
            _vwap_breakout_signals.append({**b, "time": minute_str[11:16]})
        _broadcast({"type": "vwap_breakout", "minute": minute_str[11:16], "data": breakouts})


def push_candles(stock_id: str, candles: list):
    """推入 K 線資料（index=datetime, open/high/low/close/volume/vwap）"""
    with _lock:
        _candles[stock_id] = candles
        _broadcast({"type": "candles", "stock_id": stock_id})


def push_quote(stock_id: str, price: float, prev_close: float | None, minute_str: str = ""):
    """推入單一股票的即時報價與漲跌幅，給頁首固定追蹤清單用（例如0050這種ETF，
    不經過策略推論篩選）。change_pct 基準是 prev_close（前一交易日收盤），不是
    當日開盤，跟大盤/新聞看到的算法一致。由 main/live_trader.py 的 on_minute()
    針對 WATCHLIST_QUOTES（見 main/config.py）逐一呼叫。"""
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else None
    with _lock:
        _quotes[stock_id] = {
            "stock_id": stock_id,
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "minute": minute_str,
        }
        _broadcast({"type": "quote", "stock_id": stock_id, "data": _quotes[stock_id]})


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
        entry = pos.get("entry_price", 0)
        qty = pos.get("quantity", 0)
        pnl_amt = round((exit_price - entry) * qty * 1000, 0) if entry and exit_price else None

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


def _log_ts() -> str:
    return datetime.now(_TW).strftime("%m/%d %H:%M:%S")


def append_system_log(message: str, level: str = "info"):
    """寫入系統日誌環形緩衝區並廣播。level: info / warning / error"""
    entry = {"time": _log_ts(), "level": level, "msg": message}
    _system_logs.append(entry)
    _write_log_file(entry, "system")
    _broadcast({"type": "log", "cat": "system", **entry})


def append_trade_log(stock_id: str, action: str, detail: str, status: str = ""):
    """寫入交易日誌環形緩衝區並廣播。action: buy/sell/cancel/sltp/force"""
    entry = {"time": _log_ts(), "stock_id": stock_id, "action": action,
             "detail": detail, "status": status}
    _trade_logs.append(entry)
    _write_log_file(entry, "trade")
    _broadcast({"type": "log", "cat": "trade", **entry})


def push_alert(message: str, level: str = "warning"):
    """推送系統通知到前端（SSE）並寫入系統日誌。level: info / warning / error"""
    ts = _log_ts()
    print(f"[ALERT:{level.upper()}] {message}", flush=True)
    entry = {"time": ts, "level": level, "msg": message}
    _system_logs.append(entry)
    _write_log_file(entry, "system")
    _broadcast({"type": "alert", "level": level, "message": message, "time": ts})
    _broadcast({"type": "log", "cat": "system", **entry})


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


_collector_coverage: dict = {"arrived": 0, "total": 0}


def set_collector_coverage(arrived: int, total: int):
    """由 fubon/marketdata_ws.py 呼叫：這一輪收盤後等涵蓋率/逾時二選一觸發時，
    實際等到多少支股票回報（不含還沒到的），供前端顯示即時涵蓋率用。"""
    global _collector_coverage
    _collector_coverage = {"arrived": arrived, "total": total}


_COLLECTOR_MSG = {
    "running": "資料流正常",
    "stopped": "盤後或尚未啟動",
    "error": "資料流中斷",
}


@app.get("/settings", tags=["系統"], summary="取得使用者設定")
def settings_get():
    """回傳 settings.json 的內容（若檔案不存在回傳空物件）"""
    return _load_settings()


# 設定儲存密碼（不要在原始碼裡放明碼，只放 hash；即使有原始碼也不會直接看到密碼）
# 注意：這只是「防止隨手誤改」的簡單防呆，不是真正的安全機制——6碼數字密碼
# 只有 100 萬種組合，任何人拿到原始碼都能在本機瞬間窮舉出這個 hash 對應的明碼，
# 擋不住蓄意破解，只能擋住看原始碼隨手改的情況。
_SETTINGS_PASSWORD_HASH = "f768ba2c8fe0938b19ee56666f5d16be47520cff612b6d75c5bf35bc0771fbd5"


def _check_settings_password(password) -> bool:
    import hashlib

    if not isinstance(password, str):
        return False
    return hashlib.sha256(password.encode()).hexdigest() == _SETTINGS_PASSWORD_HASH


@app.post("/settings", tags=["系統"], summary="儲存使用者設定（需密碼）")
async def settings_post(request: Request):
    """
    更新 settings.json 並立即套用至儀表板，需帶 password 欄位才能儲存。
    目前支援欄位：
      password (str)：必填，儲存密碼
      total_capital (float)：當沖總額度（元）
    """
    body: dict = await request.json()
    if not _check_settings_password(body.get("password")):
        raise HTTPException(status_code=403, detail="密碼錯誤，未儲存")

    current = _load_settings()
    for k, v in body.items():
        if k == "password":
            continue  # 密碼只用來驗證，不寫進 settings.json
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
        "coverage": _collector_coverage,
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
    "/consensus/today",
    tags=["訊號"],
    summary="今日「多個策略前N名重疊」訊號列表",
)
def consensus_today():
    print(f"[GET /consensus/today] 回傳 {len(_consensus_signals)} 筆重疊訊號", flush=True)
    with _lock:
        return list(reversed(_consensus_signals))  # 最新在上


@app.get(
    "/conflict/today",
    tags=["訊號"],
    summary="今日「多空衝突（反轉）」訊號列表",
)
def conflict_today():
    print(f"[GET /conflict/today] 回傳 {len(_conflict_signals)} 筆多空衝突訊號", flush=True)
    with _lock:
        return list(reversed(_conflict_signals))  # 最新在上


@app.get(
    "/vwap_breakout/today",
    tags=["訊號"],
    summary="今日「VWAP突破/跌破」訊號列表",
)
def vwap_breakout_today():
    print(f"[GET /vwap_breakout/today] 回傳 {len(_vwap_breakout_signals)} 筆VWAP突破訊號", flush=True)
    with _lock:
        return list(reversed(_vwap_breakout_signals))  # 最新在上


@app.get(
    "/strategies",
    tags=["訊號"],
    summary="目前啟用的策略清單（給前端動態渲染幾個模型的面板用）",
)
def get_strategies():
    return _strategies_registry


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


@app.get(
    "/quote/{stock_id}",
    tags=["圖表"],
    summary="固定追蹤清單股票的最新報價與漲跌幅（不經過策略候選篩選，如0050，見 push_quote()）",
)
def get_quote(stock_id: str):
    with _lock:
        q = _quotes.get(stock_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"尚無 {stock_id} 的報價資料（可能還沒開盤或非追蹤清單）")
    return q


@app.get(
    "/chart/{stock_id}/candles/history",
    response_model=CandleResponse,
    tags=["圖表"],
    summary="歷史K線資料（任意過去日期，交易回放用）",
)
def chart_candles_history(
    stock_id: str,
    date: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
):
    """date: 必填，格式 YYYY-MM-DD（歷史日期，非當天，當天請用 /chart/{stock_id}/candles）。
    start_time / end_time: 選填，格式 HH:MM，篩選時間區間（含頭尾），例如 09:00~10:00。
    interval: 選填，預設 "1m"，目前只支援 1 分鐘K。

    資料來源：Fugle /historical/candles，僅能查近30日資料（Fugle API 本身限制，
    無法指定 from/to），超過30日的日期會回 404。
    """
    if interval != "1m":
        raise HTTPException(status_code=400, detail="目前只支援 interval=1m")

    import pandas as pd

    from data.m1_data_loader import _download_m1

    try:
        df = _download_m1(stock_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向 Fugle 取得歷史分K失敗: {e}")
    if df.empty:
        raise HTTPException(status_code=404, detail=f"{stock_id} 查無歷史分K資料")

    df["date"] = pd.to_datetime(df["date"])
    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    df = df[(df["date"] >= day_start) & (df["date"] < day_end)]
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{stock_id} 在 {date} 沒有分K資料（Fugle 歷史 API 僅能查近30日，或該日非交易日）",
        )

    df["_minute"] = df["date"].dt.strftime("%H:%M")
    if start_time:
        df = df[df["_minute"] >= start_time]
    if end_time:
        df = df[df["_minute"] <= end_time]
    df = df.sort_values("date")

    candles = [
        {
            "time": tw_naive_to_epoch(row["date"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        }
        for _, row in df.iterrows()
    ]
    print(
        f"[GET /chart/{stock_id}/candles/history] date={date} start_time={start_time} end_time={end_time} "
        f"回傳 {len(candles)} 根 K 線",
        flush=True,
    )
    return {"stock_id": stock_id, "candles": candles}


@app.get("/monitoring", tags=["監控"], summary="監控中股票的最新推論結果（依信心度排序）")
def get_monitoring():
    """2026-07-29發現的bug：main/backfill.py::run_startup_backfill()重啟補載時
    原本沒檢查策略自己的session時段，可能把過期很久（例如收盤後才重啟）的
    監控結果寫進_monitoring；該策略當天之後不會再被呼叫，push_monitoring()
    內建的過期清理（見該函式，180秒門檻）也就沒有機會再觸發，舊資料會一直
    卡著、被這支端點原封不動吐給前端，讓人誤以為是即時訊號、可能因此手動
    進場。backfill.py的根因已經修了，這裡額外加一層防護：回傳前只保留
    minute距離現在不超過180秒（跟push_monitoring()同一個門檻）的項目，就算
    未來還有其他沒想到的原因造成資料卡住，前端也不會被誤導成看到「還在跑」
    的假象。"""
    now = datetime.now(_TW).replace(tzinfo=None)
    today_str = now.strftime("%Y-%m-%d")
    with _lock:
        fresh = [
            v
            for v in _monitoring.values()
            if (now - datetime.strptime(f"{today_str} {v['minute']}", "%Y-%m-%d %H:%M")).total_seconds() <= 180
        ]
        print(f"[GET /monitoring] 回傳 {len(fresh)}/{len(_monitoring)} 支監控股票（濾掉過期）", flush=True)
        return sorted(fresh, key=lambda x: -x["proba"])


@app.get("/completed_trades", tags=["成交"], summary="今日已完成回合（進出配對，含損益）")
def completed_trades():
    with _lock:
        return list(reversed(_completed_trades))  # 最新在上


@app.get(
    "/api/inference/history/dates",
    tags=["分析"],
    summary="可回放日期清單（HF Hub 上實際有推論記錄的日期，由新到舊）",
)
def inference_history_dates():
    """回傳 day_trade/inference/ 底下實際存在的日期清單，前端可用來決定回放功能的日期選單，
    避免選到沒有資料的日期。回傳格式：{"dates": ["2026-07-09", "2026-07-08", ...]}
    """
    repo_id = _os.environ.get("HF_REPO_ID", "")
    token = _os.environ.get("HF_TOKEN", "") or None
    if not repo_id:
        return {"dates": []}

    import re as _re

    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:
        print(f"[GET /api/inference/history/dates] HF Hub 查詢失敗: {e}", flush=True)
        return {"dates": []}

    pattern = _re.compile(rf"{_re.escape(_HF_INFERENCE_PREFIX)}/(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$")
    dates = sorted(
        {m.group(1) for f in files if (m := pattern.match(f))},
        reverse=True,
    )
    print(f"[GET /api/inference/history/dates] 回傳 {len(dates)} 個可回放日期", flush=True)
    return {"dates": dates}


@app.get(
    "/api/inference/history",
    response_model=list[InferenceRecord],
    tags=["分析"],
    summary="每分鐘推論歷史記錄（給前端分析使用）",
)
def inference_history(
    date: Optional[str] = None,
    stock_id: Optional[str] = None,
    min_proba: float = 0.0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 5000,
):
    """date: 選填，格式 YYYY-MM-DD，預設今日；今日走記憶體，過去日期從 HF Hub 下載（需已設定 HF_REPO_ID）。
    stock_id: 選填，只回傳單一股票。
    min_proba: 選填，只回傳 proba >= min_proba 的記錄（例如只看有效訊號）。
    start_time / end_time: 選填，格式 HH:MM，篩選時間區間（含頭尾），例如 09:00~09:30。
    limit: 回傳筆數上限，預設 5000，依 time 排序後截斷。
    """
    date_str = date or datetime.now(_TW).strftime("%Y-%m-%d")
    df = _load_inference_day(date_str)
    if df.empty:
        return []
    if stock_id:
        df = df[df["stock_id"] == stock_id]
    if min_proba > 0:
        df = df[df["proba"] >= min_proba]
    if start_time:
        df = df[df["time"] >= start_time]
    if end_time:
        df = df[df["time"] <= end_time]
    df = df.sort_values(["time", "stock_id"]).head(limit)
    print(
        f"[GET /api/inference/history] date={date_str} stock_id={stock_id} min_proba={min_proba} "
        f"start_time={start_time} end_time={end_time} 回傳 {len(df)} 筆",
        flush=True,
    )
    return df.to_dict("records")


@app.get("/failed_orders", tags=["成交"], summary="今日失敗委託（含錯誤訊息）")
def failed_orders():
    with _lock:
        return list(reversed([t for t in _trades if t.get("status") == "FAILED"]))


@app.get("/api/logs", tags=["系統"], summary="今日程式日誌與交易日誌（各最多500筆）")
def get_logs(cat: str = "all"):
    """
    cat=system  → 程式/系統事件（限流、錯誤、reconcile）
    cat=trade   → 交易事件（買/賣/SL/TP/force close）
    cat=all     → 兩者合併，依時間倒序

    優先讀今日 JSONL 檔（重啟不遺失）；檔案不存在時 fallback 到記憶體緩衝。
    """
    entries = _read_log_file()
    if not entries:
        # 檔案尚不存在（例如剛啟動當天第一次寫前），fallback 到記憶體
        with _lock:
            sys_list = list(_system_logs)
            trd_list = list(_trade_logs)
        entries = [{"cat": "system", **e} for e in sys_list] + [{"cat": "trade", **e} for e in trd_list]
    if cat == "system":
        result = [e for e in entries if e.get("cat") == "system"]
    elif cat == "trade":
        result = [e for e in entries if e.get("cat") == "trade"]
    else:
        result = entries
    return list(reversed(sorted(result, key=lambda x: x.get("time", ""), reverse=False)))[:500]


def get_force_close_queue() -> set:
    return set(_force_close_queue)


def clear_force_close(sids: list):
    _force_close_queue.difference_update(sids)


_close_now_fn = None  # live_trader.py 啟動後注入


def register_close_now(fn):
    global _close_now_fn
    _close_now_fn = fn


@app.post("/close_now", tags=["交易"], summary="立即平倉指定股票（市價單）")
async def close_now(request: Request):
    """body: {"stock_ids": ["1409", "2367"]}"""
    import threading
    body = await request.json()
    sids = body.get("stock_ids", [])
    print(f"[CLOSE NOW] 前端觸發立即平倉: {sids}", flush=True)
    if _close_now_fn:
        threading.Thread(target=_close_now_fn, args=(sids,), daemon=True).start()
        return {"ok": True, "executing": sids}
    # fallback：executor 未就緒時排隊等下一分鐘
    with _lock:
        _force_close_queue.update(sids)
    return {"ok": True, "queued": list(_force_close_queue)}


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
