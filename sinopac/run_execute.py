import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_TW = timezone(timedelta(hours=8))

# ── sys.path 注入（支援 Render 或其他環境）─────────────────────────────
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sinopac import sinopac_api as trade_api  # 相對於項目根目錄
except (ImportError, ModuleNotFoundError):
    try:
        import sinopac_api as trade_api  # 相對於 sinopac/ 目錄
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[ERROR] 永豐模組無法載入: {e}", flush=True)
        trade_api = None

try:
    from sinopac.broker_client import BrokerClient
except (ImportError, ModuleNotFoundError):
    try:
        from broker_client import BrokerClient
    except (ImportError, ModuleNotFoundError):
        BrokerClient = None  # type: ignore

try:
    from api import _positions as _api_positions

    def _get_pos_entry(sid: str) -> float:
        return _api_positions.get(sid, {}).get("entry_price", 0.0)

except (ImportError, ModuleNotFoundError):

    def _get_pos_entry(_):
        return 0.0


try:
    from api import push_trade as _push_trade
    from api import push_position as _push_pos
    from api import close_position as _close_pos
    from api import push_summary_update as _push_summary
    from api import push_completed_trades_from_broker as _push_closed_sync
    from api import sync_broker_snapshot as _sync_broker_snapshot
    from api import get_setting as _get_setting
    from api import append_system_log as _log_sys
    from api import append_trade_log as _log_trade
except (ImportError, ModuleNotFoundError):

    def _push_trade(*_):
        pass

    def _push_pos(*_):
        pass

    def _close_pos(*_):
        pass

    def _push_summary(*_):
        pass

    def _push_closed_sync(*_):
        pass

    def _sync_broker_snapshot(*_):
        pass

    def _get_setting(key, default=None):
        return default

    def _log_sys(msg, level="info"):
        pass

    def _log_trade(sid, action, detail, status=""):
        pass


def _now() -> str:
    return datetime.now(_TW).strftime("%m/%d %H:%M:%S")


def _to_bool(v, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "no", "off", "")


def _fmt_time(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%m/%d %H:%M:%S")
    text = str(value or "")
    # "2026-07-03 11:20:06..." → "07/03 11:20:06"
    if len(text) >= 19:
        return text[5:7] + "/" + text[8:10] + " " + text[11:19]
    return text


class PaperTrader:
    """
    本機模擬交易器。不連接任何券商 API，持倉狀態存在 paper_state.json。

    用途：在沒有券商帳號、或不想冒風險時，驗證交易邏輯是否正確。
    每次 reconcile() 後會把持倉寫回 JSON，重啟不遺失。
    同時呼叫 api.push_*，讓 dashboard /positions 與 /trades 端點有資料。
    """

    def __init__(self, total_capital: float = 1_000_000, state_path: Path | None = None):
        self.total_capital = total_capital
        self._path = state_path or Path(__file__).parent / "paper_state.json"
        self._positions: dict = {}
        if self._path.exists():
            self._positions = json.loads(self._path.read_text())

    @property
    def positions(self) -> dict:
        return dict(self._positions)

    def reconcile(self, signals: list[dict], prices: dict | None = None):
        """
        依本分鐘模型訊號調整模擬持倉。

        signals 是 predict_live() 回傳的清單，每筆含 stock_id / price / proba / name。
        - 在 signals 中但不在持倉 → 模擬開倉
        - 在持倉中但不在 signals → 模擬平倉
        - 兩者皆有 → 不動（不重複下單）
        """
        target = {s["stock_id"]: s for s in signals}
        to_open = set(target) - set(self._positions)
        to_close: set = set()  # 出場由 SL/TP 決定

        # 停損/停利：用傳入的現價檢查（不另打 API）
        if prices:
            sl_pct = float(_get_setting("stop_loss_pct") or 3.0) / 100
            tp_pct = float(_get_setting("take_profit_pct") or 3.0) / 100
            for sid, pos in self._positions.items():
                if sid in to_close:
                    continue
                price = prices.get(sid)
                if not price:
                    continue
                avg = pos["avg_price"]
                pnl = (price - avg) / avg if avg else 0
                if pnl <= -sl_pct:
                    to_close.add(sid)
                    print(f"[PAPER SL] {sid} 停損 {pnl*100:.2f}%", flush=True)
                elif pnl >= tp_pct:
                    to_close.add(sid)
                    print(f"[PAPER TP] {sid} 停利 {pnl*100:.2f}%", flush=True)

        if not to_open and not to_close:
            return

        # 當沖額度：買入 + 賣出 各佔一半
        budget = self.total_capital / 2 / max(len(target), 1)  # PaperTrader: 沿用 /2 邏輯

        for sid in sorted(to_close):
            pos = self._positions.pop(sid)
            entry = pos["avg_price"]
            print(f"[PAPER CLOSE] {sid} {pos['quantity']}張 (entry={entry:.2f})")
            _push_trade(
                {
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "sell",
                    "price": entry,  # exit price unknown without feed; use entry as placeholder
                    "quantity": pos["quantity"],
                    "status": "FILLED",
                    "broker_response": "paper",
                }
            )
            _close_pos(sid, 0.0)

        for sid in sorted(to_open):
            sig = target[sid]
            price = sig.get("price", 0)
            if price <= 0:
                print(f"[PAPER SKIP] {sid} 無報價")
                continue
            lots = int(budget / (price * 1000))
            if lots < 1:
                print(f"[PAPER SKIP] {sid} 資金不足一張 (價={price:.2f})")
                continue
            self._positions[sid] = {"quantity": lots, "avg_price": price}
            print(f"[PAPER OPEN] {sid} {lots}張 @ {price:.2f}")
            _push_trade(
                {
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "buy",
                    "price": price,
                    "quantity": lots,
                    "status": "FILLED",
                    "broker_response": "paper",
                }
            )
            _push_pos(
                {
                    "stock_id": sid,
                    "name": sig.get("name", sid),
                    "pnl_pct": 0.0,
                    "entry_price": price,
                    "current_price": price,
                    "stop_loss": round(price * 0.97, 2),
                    "take_profit": round(price * 1.03, 2),
                }
            )

        self._path.write_text(json.dumps(self._positions, ensure_ascii=False, indent=2))

    def force_close_own_positions(self):
        """強制平倉本系統所有 paper 持倉（模式與 LiveTrader 介面一致）"""
        self.reconcile([])


class LiveTrader:
    """
    透過永豐證券 Shioaji API 真實（或模擬帳號）下單。

    simulation=True  → 永豐模擬環境（帳號相同，單子不真實成交）
    simulation=False → 正式帳號，會真的扣款與交割

    連線是 lazy 的：第一次 reconcile() 時才登入，之後保持同一個 api 物件。
    登入失敗（例如雲端 IP 被擋）會印錯誤並直接 return，不影響 dashboard。
    """

    def __init__(self, total_capital: float = 1_000_000, simulation: bool = True, name_lookup=None):
        # 當沖總額度：當日買賣合計金額共用同一個額度，買進、平倉都會扣，不因平倉歸還。
        self.total_capital = total_capital
        self.simulation = simulation
        self._api = None
        self._broker: "BrokerClient | None" = None  # 連線後建立
        self._positions_synced = False  # 啟動後第一次 reconcile 時同步現有持倉
        self._name_lookup = name_lookup or (lambda sid: sid)
        self._used_quota = 0.0  # 今日已用額度；重啟由 sync_from_broker 恢復

    def _connect(self) -> bool:
        if self._api is not None:
            return True
        # 檢查 trade_api 是否正確載入
        if trade_api is None:
            print(f"[LIVE CONNECT] ✗ 永豐模組未載入，無法連線", flush=True)
            return False
        mode = "模擬" if self.simulation else "正式"
        print(f"[LIVE CONNECT] 正在連接永豐 {mode}環境...", flush=True)
        try:
            self._api = trade_api.test() if self.simulation else trade_api.login()
            self._api.set_order_callback(self._on_order_event)
            if BrokerClient is not None:
                self._broker = BrokerClient(self._api)
            print(f"[LIVE CONNECT] 永豐 {mode}環境連線成功 ✓", flush=True)
            return True
        except Exception as e:
            print(f"[LIVE CONNECT] ✗ 登入失敗 ({mode}環境): {e}", flush=True)
            return False

    def _reset_on_disconnect(self, e: Exception, context: str):
        """偵測到連線中斷時重置 API，下次 reconcile 自動重連。"""
        err = str(e).lower()
        if any(k in err for k in ["connection", "timeout", "disconnected", "socket", "eof"]):
            print(f"[LIVE] {context}: 連線中斷，下次將重連: {e}")
            try:
                self._api.logout()
            except Exception:
                pass
            self._api = None
        else:
            print(f"[LIVE] {context}: {e}")

    def sync_from_broker(self) -> bool:
        """重啟後從永豐只讀同步持倉、今日委託與已平倉到 dashboard 快取。"""
        print(f"[LIVE SYNC] 啟動：正在從永豐同步持倉、委託與已平倉...", flush=True)
        if not self._connect():
            print(f"[LIVE SYNC] ✗ 連線失敗，同步中止", flush=True)
            return False

        try:
            current = trade_api.get_positions(self._api, day_trade_only=True)
        except Exception as e:
            print(f"[LIVE SYNC] 取得持倉失敗，略過同步: {e}")
            return False  # 取不到持倉就不同步，避免把 dashboard 清空

        try:
            if current:
                contracts = [self._api.Contracts.Stocks[sid] for sid in current]
                snapshots = {s.code: s.close for s in self._api.snapshots(contracts)}
            else:
                snapshots = {}
        except Exception as e:
            print(f"[LIVE SYNC] 取得持倉報價失敗: {e}")
            snapshots = {}

        positions = []
        for sid, pos in sorted(current.items()):
            avg = pos["avg_price"]
            qty = pos["quantity"]
            curr_price = snapshots.get(sid, avg)
            pnl = round((curr_price - avg) / avg * 100, 4) if avg else 0.0
            positions.append(
                {
                    "stock_id": sid,
                    "name": self._name_lookup(sid),
                    "quantity": qty,
                    "pnl_pct": pnl,
                    "entry_price": avg,
                    "current_price": curr_price,
                    "stop_loss": round(avg * 0.97, 2),
                    "take_profit": round(avg * 1.03, 2),
                }
            )

        try:
            orders = trade_api.list_orders_today(self._api)
        except Exception as e:
            print(f"[LIVE SYNC] 今日委託同步失敗: {e}")
            orders = []

        trades = [
            {
                "order_id": o.get("order_id", ""),
                "time": _fmt_time(o.get("time")),
                "stock_id": o["stock_id"],
                "direction": o["direction"],
                "price": o["price"],
                "quantity": o["quantity"],
                "filled": o.get("filled", 0),
                "status": o["status"],
                "broker_response": o.get("msg") or o["status"],
            }
            for o in orders
        ]

        try:
            closed_today = trade_api.get_closed_today(self._api)
        except Exception as e:
            print(f"[LIVE SYNC] 今日已平倉同步失敗: {e}")
            closed_today = []

        completed_trades = []
        for c in closed_today:
            entry = c.get("buy_avg", 0)
            exit_price = c.get("sell_avg", 0)
            qty = c.get("quantity", 0)
            pnl_pct = c.get("pnl_pct", 0.0)
            pnl_amt = round((exit_price - entry) * qty * 1000, 0) if entry and exit_price else None
            completed_trades.append(
                {
                    "time": c.get("sell_time", "-"),
                    "stock_id": c["stock_id"],
                    "name": self._name_lookup(c["stock_id"]),
                    "quantity": qty,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_amt": pnl_amt,
                    "exit_reason": "broker_sync",
                }
            )

        wins = sum(1 for c in completed_trades if c["pnl_pct"] > 0)
        closed = len(completed_trades)  # FIFO 配對回合數，最準確
        avg_pnl = round(sum(c["pnl_pct"] for c in completed_trades) / closed, 4) if closed else 0.0
        total_pnl_amt = round(sum(c.get("pnl_amt") or 0 for c in completed_trades), 0)

        # 已用額度 = 當日買賣合計成交金額（買進、平倉都扣，不因平倉歸還）
        # 直接從當日委託重算（非累加），避免重複扣款；未成交單 filled=0 自動不計入
        used_quota = round(
            sum(t["price"] * t["filled"] * 1000 for t in trades),
            0,
        )
        # 開倉成交 = FILLED 買進筆數；平倉回合 = FIFO 配對數（closed）
        filled = {"FILLED", "PARTIAL"}
        open_count = sum(1 for t in trades if t.get("direction") == "buy" and t.get("status") in filled)
        failed_count = sum(1 for t in trades if t.get("status") == "FAILED")
        last_updated = _fmt_time(orders[-1]["time"]) if orders else _now()

        try:
            broker_balance = trade_api.get_account_balance(self._api)
        except Exception as e:
            print(f"[LIVE SYNC] 查詢永豐帳戶餘額失敗: {e}", flush=True)
            broker_balance = None

        _sync_broker_snapshot(
            positions,
            trades,
            completed_trades,
            {
                "wins": wins,
                "win_rate": round(wins / closed * 100, 1) if closed else None,
                "today_pnl_pct": avg_pnl,
                "today_pnl_amt": total_pnl_amt,
                "open_trades": open_count,
                "closed": closed,
                "total_capital": self.total_capital,
                "used_quota": used_quota,
                "broker_balance": broker_balance,
                "errors": failed_count,
                "last_updated": last_updated,
            },
        )
        self._used_quota = used_quota  # 重啟恢復已用額度
        self._positions_synced = True
        self._startup_prices = snapshots  # 供 startup_sltp_check() 使用
        print(
            f"[LIVE SYNC] 啟動同步完成：持倉 {len(positions)}、"
            f"今日委託 {len(trades)}、已平倉 {closed}、已用額度 {used_quota/10000:.1f}萬"
        )
        return True

    def startup_sltp_check(self):
        """重啟後立刻檢查 SL/TP，不等下一分鐘 on_minute。
        模型只管進場，出場完全靠 SL/TP；故直接呼叫 reconcile([])，
        SL/TP 區塊會自動拉 broker 現價判斷是否需平倉。
        """
        try:
            current = trade_api.get_positions(self._api, day_trade_only=True)
        except Exception as e:
            print(f"[STARTUP SL/TP] 取得持倉失敗: {e}", flush=True)
            return
        if not current:
            print("[STARTUP SL/TP] 無持倉，略過", flush=True)
            return
        sl_pct = float(_get_setting("stop_loss_pct") or 3.0) / 100
        tp_pct = float(_get_setting("take_profit_pct") or 3.0) / 100
        print(
            f"[STARTUP SL/TP] 持倉 {list(current.keys())}  SL={sl_pct*100:.0f}%  TP={tp_pct*100:.0f}%  "
            f"→ 開始即時 SL/TP 檢查",
            flush=True,
        )
        self.reconcile([])

    def _on_order_event(self, state, msg):
        """永豐成交回報（SDEAL）→ 買進更新持倉均價；賣出立即移除持倉並記錄已平倉"""
        import shioaji as sj

        if state != sj.OrderState.StockDeal:
            return
        try:
            sid = msg.get("code") or msg["contract"]["code"]
            fill_price = float(msg["price"])
            fill_qty = int(msg["quantity"])
            action = str(msg.get("action", "")).lower()
            direction = "buy" if "buy" in action else "sell"
            print(f"[LIVE FILL] {sid} {direction} {fill_qty}張 @ {fill_price} ✓", flush=True)
            _log_trade(sid, f"fill_{direction}", f"{fill_qty}張@{fill_price} 成交", "FILLED")
            _push_trade(
                {
                    "order_id": msg.get("id", ""),
                    "time": _now(),
                    "stock_id": sid,
                    "direction": direction,
                    "price": fill_price,
                    "quantity": fill_qty,
                    "status": "FILLED",
                    "broker_response": f"fill@{fill_price}",
                }
            )
            if direction == "sell":
                if self._broker:
                    self._broker.on_fill(sid, "sell")
                entry_price = _get_pos_entry(sid)
                pnl_pct = round((fill_price - entry_price) / entry_price * 100, 4) if entry_price else 0.0
                print(f"[LIVE FILL] 賣出成交確認 {sid} entry={entry_price} fill={fill_price} pnl={pnl_pct:.2f}%", flush=True)
                _close_pos(sid, pnl_pct, exit_price=fill_price, exit_reason="filled")
            else:
                # 買進成交：更新 dashboard 持倉的實際進場均價
                try:
                    current = trade_api.get_positions(self._api, day_trade_only=True)
                    if sid in current:
                        pos = current[sid]
                        avg = pos["avg_price"]
                        qty = pos["quantity"]
                        snap = self._api.snapshots([self._api.Contracts.Stocks[sid]])[0].close
                        pnl = round((snap - avg) / avg * 100, 4) if avg else 0.0
                        print(f"[LIVE FILL] 更新持倉：{sid} avg={avg} qty={qty} pnl={pnl}%", flush=True)
                        _push_pos(
                            {
                                "stock_id": sid,
                                "name": self._name_lookup(sid),
                                "quantity": qty,
                                "pnl_pct": pnl,
                                "entry_price": avg,
                                "current_price": snap,
                                "stop_loss": round(avg * 0.97, 2),
                                "take_profit": round(avg * 1.03, 2),
                            }
                        )
                except Exception as e:
                    print(f"[LIVE FILL] ✗ 更新持倉失敗: {e}", flush=True)
        except Exception as e:
            print(f"[LIVE FILL] ✗ 解析回報失敗: {e} | {msg}", flush=True)

    def _sync_positions_with_broker(self, current: dict):
        """每次 reconcile 呼叫：以永豐目前持倉為準，修正 dashboard 偏移。
        - broker 有、dashboard 沒有 → 補進去（可能是 SDEAL 漏接）
        - dashboard 有、broker 沒有 → 移除（broker 已平倉但通知未收到）
        - 兩者皆有但均價不同 → 更新 entry_price（fill 後均價修正）
        """
        print(f"[SYNC POS] 與永豐同步持倉：broker={len(current)} dashboard={len(_api_positions)}", flush=True)
        # 補進 broker 有但 dashboard 沒有的持倉
        for sid, pos in current.items():
            dash = _api_positions.get(sid, {})
            avg = pos["avg_price"]
            qty = pos["quantity"]
            if sid not in _api_positions or abs(dash.get("entry_price", 0) - avg) > 0.01:
                print(f"[SYNC POS] 補進 {sid} {qty}張 @ {avg}", flush=True)
                _push_pos(
                    {
                        "stock_id": sid,
                        "name": self._name_lookup(sid),
                        "quantity": qty,
                        "pnl_pct": dash.get("pnl_pct", 0.0),
                        "entry_price": avg,
                        "current_price": dash.get("current_price", avg),
                        "stop_loss": round(avg * 0.97, 2),
                        "take_profit": round(avg * 1.03, 2),
                    }
                )
        # 移除 broker 已不存在的 ghost 持倉（賣單已成交）
        # 只有 broker 有回傳至少一筆時才移除，避免 API 短暫異常誤刪全部持倉
        if current:
            ghost_sids = [sid for sid in list(_api_positions.keys()) if sid not in current]
            if ghost_sids:
                # 查永豐今日已平倉取得真實出場價與損益（不用估算限價）
                try:
                    closed_map = {c["stock_id"]: c for c in trade_api.get_closed_today(self._api)}
                except Exception as e:
                    print(f"[SYNC POS] get_closed_today 失敗: {e}", flush=True)
                    closed_map = {}
                for sid in ghost_sids:
                    if self._broker:
                        self._broker.on_position_closed(sid)
                    c = closed_map.get(sid, {})
                    pnl_pct = c.get("pnl_pct", 0.0)
                    exit_price = c.get("sell_avg") or 0.0
                    print(f"[SYNC POS] 移除已平倉 {sid} pnl={pnl_pct:.2f}% exit={exit_price}", flush=True)
                    _close_pos(sid, pnl_pct, exit_price=exit_price, exit_reason="broker_closed")

    def reconcile(self, signals: list[dict], prices: dict | None = None):
        """
        依本分鐘模型訊號調整券商實際持倉。

        先從永豐取得目前持倉（每分鐘同步，確保 dashboard 與 broker 不偏移），
        再與 signals 做 diff：
        - 在 signals 中但不在券商持倉 → 取快照報價 → 下限價買單
        - 在券商持倉中但不在 signals → 下限價賣單
        下單後同步更新 api.py 的 in-memory 狀態，供 dashboard 顯示。
        """
        _log_sys(f"reconcile 開始：{len(signals)} 筆訊號", "info")
        print(f"[RECONCILE] 開始調和：收到 {len(signals)} 筆訊號", flush=True)
        if not self._connect():
            print(f"[RECONCILE] ✗ 連線失敗，中止", flush=True)
            return

        # 每次 reconcile 重新讀取設定，允許前端即時變更 total_capital 立即生效
        cap_override = _get_setting("total_capital")
        if cap_override is not None:
            self.total_capital = float(cap_override)

        sig_map = {s["stock_id"]: s for s in signals}
        target = set(sig_map)

        try:
            current = trade_api.get_positions(self._api, day_trade_only=True)
        except Exception as e:
            self._reset_on_disconnect(e, "取得持倉")
            return

        # 每次 reconcile 都以永豐為準修正 dashboard（防止 SDEAL 漏接造成偏移）
        self._sync_positions_with_broker(current)

        # 取消上分鐘所有 SENT 未成交買/賣單（BrokerClient 負責更新追蹤）
        cancelled = {"buy": [], "sell": []}  # 預設空，cancel 失敗時不影響後續
        try:
            if self._broker:
                cancelled = self._broker.cancel_all_orders()
            else:
                cancelled = trade_api.cancel_sent_orders(self._api)
            if cancelled["buy"]:
                restore = sum(cost for _, cost in cancelled["buy"])
                self._used_quota = max(0, self._used_quota - restore)
                sids = [sid for sid, _ in cancelled["buy"]]
                print(f"[CANCEL BUY]  取消未成交買單: {sids}，還原額度 {restore/10000:.1f}萬")
                _log_sys(f"取消買單: {sids}，還原 {restore/10000:.1f}萬", "info")
            if cancelled["sell"]:
                print(f"[CANCEL SELL] 取消未成交賣單: {cancelled['sell']}，本分鐘重新掛")
                _log_sys(f"取消賣單，重掛: {cancelled['sell']}", "info")
        except Exception as e:
            _log_sys(f"取消委託失敗: {e}", "error")
            print(f"[CANCEL] 取消委託失敗: {e}")

        # 取消後重新取得委託狀態（PARTIAL 單仍保留不動）
        pending_orders: list = []
        try:
            pending_orders = trade_api.list_orders_today(self._api)
            pending_buy = {
                o["stock_id"] for o in pending_orders if o["status"] in {"SENT", "PARTIAL"} and o["direction"] == "buy"
            }
            # pending_sell: 以 BrokerClient 自追蹤為準（比 broker 即時狀態更可靠）
            pending_sell = self._broker.pending_sell_sids if self._broker else {
                o["stock_id"] for o in pending_orders if o["status"] in {"SENT", "PARTIAL"} and o["direction"] == "sell"
            }
        except Exception:
            pending_buy = set()
            pending_sell = set()

        # 取消確認後立即更新本地追蹤（BrokerClient 已內部清理 _sell_orders）
        if cancelled["buy"]:
            pending_buy -= {sid for sid, _ in cancelled["buy"]}
        # cancelled["sell"] 的 _broker._sell_orders 已在 cancel_all_orders 內部清理

        # 重啟後第一次 reconcile：把現有持倉推回 dashboard（使用永豐實際均價）
        if not self._positions_synced:
            self._positions_synced = True
            if current:
                try:
                    contracts = [self._api.Contracts.Stocks[sid] for sid in current]
                    snaps = {s.code: s.close for s in self._api.snapshots(contracts)}
                except Exception:
                    snaps = {}
                for sid, pos in current.items():
                    avg = pos["avg_price"]
                    qty = pos["quantity"]
                    curr_price = snaps.get(sid, avg)
                    pnl = round((curr_price - avg) / avg * 100, 4) if avg else 0.0
                    _push_pos(
                        {
                            "stock_id": sid,
                            "name": self._name_lookup(sid),
                            "quantity": qty,
                            "pnl_pct": pnl,
                            "entry_price": avg,
                            "current_price": curr_price,
                            "stop_loss": round(avg * 0.97, 2),
                            "take_profit": round(avg * 1.03, 2),
                        }
                    )
                print(f"[LIVE SYNC] 同步 {len(current)} 筆現有持倉到 dashboard（含永豐均價）")

            # 同步今日已平倉（從永豐 list_trades 重建，僅今日）
            try:
                closed_today = trade_api.get_closed_today(self._api)
                if closed_today:
                    avg_pnl = sum(c["pnl_pct"] for c in closed_today) / len(closed_today)
                    wins = sum(1 for c in closed_today if c["pnl_pct"] > 0)
                    n = len(closed_today)
                    _push_summary(
                        {
                            "closed": n,
                            "wins": wins,
                            "win_rate": round(wins / n * 100, 1),
                            "today_pnl_pct": round(avg_pnl, 4),
                        }
                    )
                    _push_closed_sync(closed_today)
                    print(f"[LIVE SYNC] 今日已平倉 {n} 筆，勝率 {wins}/{n}，均損益 {avg_pnl:.2f}%")
            except Exception as e:
                print(f"[LIVE SYNC] 已平倉同步失敗: {e}")

        open_enabled = _to_bool(_get_setting("open_enabled"), default=True)
        close_enabled = _to_bool(_get_setting("close_enabled"), default=True)
        # 模型只管進場；出場完全由 SL/TP（下方）決定，不因訊號消失而平倉
        to_open = (target - set(current) - pending_buy) if open_enabled else set()
        to_close: set = set()

        # 前端觸發的立即平倉（市價單）
        try:
            from api import get_force_close_queue, clear_force_close
            forced = get_force_close_queue() & set(current) - pending_sell
            if forced:
                to_close.update(forced)
                clear_force_close(list(forced))
                print(f"[FORCE NOW] 前端觸發立即平倉: {forced}", flush=True)
        except ImportError:
            pass

        # 停損/停利：跟模型訊號無關，純粹用現價比對進場均價
        # prices 只含 _day_trade_stocks（前500支），持倉若不在其中需補拉 broker snapshot
        if close_enabled:
            sl_pct = float(_get_setting("stop_loss_pct") or 3.0) / 100
            tp_pct = float(_get_setting("take_profit_pct") or 3.0) / 100
            missing = [
                sid for sid in current
                if sid not in (prices or {}) and sid not in to_close and sid not in pending_sell
            ]
            extra_prices: dict = {}
            if missing:
                try:
                    contracts = [self._api.Contracts.Stocks[sid] for sid in missing]
                    extra_prices = {s.code: s.close for s in self._api.snapshots(contracts)}
                    print(f"[SL/TP] 補拉 {len(missing)} 支不在監控清單的持倉現價", flush=True)
                except Exception as e:
                    print(f"[SL/TP] 補拉現價失敗: {e}", flush=True)
            all_prices = {**(prices or {}), **extra_prices}
            for sid, pos in current.items():
                if sid in to_close or sid in pending_sell:
                    continue
                price = all_prices.get(sid)
                if not price:
                    continue
                avg = pos["avg_price"]
                pnl = (price - avg) / avg if avg else 0
                if pnl <= -sl_pct:
                    to_close.add(sid)
                    msg = f"{sid} 停損 {pnl*100:.2f}% ≤ -{sl_pct*100:.0f}%（現價={price}）"
                    print(f"[SL] {msg}", flush=True)
                    _log_trade(sid, "sltp", msg, "SL")
                elif pnl >= tp_pct:
                    to_close.add(sid)
                    msg = f"{sid} 停利 {pnl*100:.2f}% ≥ +{tp_pct*100:.0f}%（現價={price}）"
                    print(f"[TP] {msg}", flush=True)
                    _log_trade(sid, "sltp", msg, "TP")

        print(
            f"[RECONCILE] 交易計畫：開倉 {len(to_open)}、平倉 {len(to_close)}（含SL/TP）、待核實 買={len(pending_buy)} 賣={len(pending_sell)}",
            flush=True,
        )
        # 已用額度 = 當日買賣合計成交金額（買進、平倉都扣，不因平倉歸還）+ 仍 SENT 中未成交買單（預先保留）
        # 直接從當日委託重算（非累加），避免重複扣款；未成交單 filled=0 自動不計入
        self._used_quota = round(
            sum(o["price"] * o.get("filled", 0) * 1000 for o in pending_orders), 0
        )
        _pending_buy_reserve = round(sum(
            o["price"] * max(0, o["quantity"] - o.get("filled", 0)) * 1000
            for o in pending_orders
            if o["status"] in {"SENT", "PARTIAL"} and o["direction"] == "buy"
        ), 0)
        if _pending_buy_reserve > 0:
            print(f"[QUOTA] SENT 買單保留額度 {_pending_buy_reserve/10000:.1f}萬（cancel 失敗或尚未確認）", flush=True)
        self._used_quota = round(self._used_quota + _pending_buy_reserve, 0)
        available = max(0.0, self.total_capital - self._used_quota)
        try:
            broker_balance = trade_api.get_account_balance(self._api)
        except Exception as e:
            print(f"[RECONCILE] 查詢永豐帳戶餘額失敗: {e}", flush=True)
            broker_balance = None
        _push_summary({"used_quota": self._used_quota, "total_capital": self.total_capital,
                       "available": available, "broker_balance": broker_balance})

        if not to_open and not to_close:
            print(f"[RECONCILE] 無新增交易，完成", flush=True)
            return

        # 買新倉前先保留「目前持倉之後平倉」也要再扣一次額度的空間，
        # 避免把 available 全部買光，導致持倉平倉時卡在額度不足送不出賣單
        _open_position_reserve = round(
            sum(p["avg_price"] * p["quantity"] * 1000 for p in current.values()), 0
        )
        safe_buy_room = max(0.0, available - _open_position_reserve)
        if _open_position_reserve > 0:
            print(
                f"[QUOTA] 持倉平倉預留 {_open_position_reserve/10000:.1f}萬，買進安全額度剩 {safe_buy_room/10000:.1f}萬",
                flush=True,
            )

        # 買入預算 = 買進安全額度 / 本分鐘訊號支數（實際仍受 safe_buy_room 限制，見下方 min）
        budget = safe_buy_room / max(len(target), 1)
        snapshots: dict[str, float] = {}

        if to_open:
            try:
                contracts = [self._api.Contracts.Stocks[sid] for sid in to_open]
                snapshots = {s.code: s.close for s in self._api.snapshots(contracts)}
            except Exception as e:
                self._reset_on_disconnect(e, "取得快照")

        cfg_min_price = _get_setting("min_price")
        cfg_max_price = _get_setting("max_price")
        cfg_max_budget_stock = _get_setting("max_budget_per_stock")  # 萬
        # 手續費 + 證交稅緩衝（買+賣 ≈ 0.3%），讓實際資金留有餘裕
        fee_rate = float(_get_setting("fee_rate") or 0.003)
        lot_cost_factor = 1 + fee_rate  # 每張有效成本倍數

        for sid in sorted(to_open, key=lambda s: sig_map.get(s, {}).get("proba", 0), reverse=True):
            try:
                price = snapshots.get(sid, 0)
                if price <= 0:
                    print(f"[LIVE SKIP] {sid} 無報價")
                    continue

                # 股價過濾
                if cfg_min_price is not None and price < float(cfg_min_price):
                    print(f"[LIVE SKIP] {sid} 股價 {price} 低於下限 {cfg_min_price}")
                    continue
                if cfg_max_price is not None and price > float(cfg_max_price):
                    print(f"[LIVE SKIP] {sid} 股價 {price} 超過上限 {cfg_max_price}")
                    continue

                # 單股預算：優先用 max_budget_per_stock 設定，否則均分買入上限
                if cfg_max_budget_stock is not None:
                    stock_budget = float(cfg_max_budget_stock) * 10000
                else:
                    stock_budget = budget  # safe_buy_room / N
                stock_budget = min(stock_budget, safe_buy_room)  # 不超過扣除平倉預留後的安全額度

                # 每張有效成本含手續費緩衝（確保不因小數截斷而超額）
                effective_lot_cost = price * 1000 * lot_cost_factor
                lots = int(stock_budget / effective_lot_cost)
                if lots < 1:
                    print(f"[LIVE SKIP] {sid} 額度不足（買進安全額度 {safe_buy_room/10000:.1f}萬，需 {price/10:.1f}萬/張）")
                    continue

                if self._broker:
                    trade = self._broker.buy(sid, lots)
                else:
                    trade = trade_api.open(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                buy_cost = price * lots * 1000
                # 本分鐘內給後續股票用；下次 reconcile 從 broker 重算。
                # 新買的部位之後也要平倉，safe_buy_room 要多扣一次（買進本身 + 預留平倉）
                available -= buy_cost
                safe_buy_room = max(0.0, safe_buy_room - buy_cost * 2)
                print(
                    f"[LIVE OPEN] {sid} {lots}張 @ {price} → {status} [{order_id}]"
                    f"（剩餘可用 {available/10000:.1f}萬，買進安全額度 {safe_buy_room/10000:.1f}萬）"
                )
                _log_trade(sid, "buy", f"{lots}張@{price} [{order_id}]", status)
                _push_trade(
                    {
                        "order_id": order_id,
                        "time": _now(),
                        "stock_id": sid,
                        "direction": "buy",
                        "price": price,
                        "quantity": lots,
                        "status": status,
                        "broker_response": status,
                    }
                )
                sig = sig_map.get(sid, {})
                _push_pos(
                    {
                        "stock_id": sid,
                        "name": sig.get("name", sid),
                        "quantity": lots,
                        "pnl_pct": 0.0,
                        "entry_price": price,
                        "current_price": price,
                        "stop_loss": round(price * 0.97, 2),
                        "take_profit": round(price * 1.03, 2),
                    }
                )
            except Exception as e:
                print(f"[LIVE ERROR] 開倉 {sid}: {e}")

        for sid in sorted(to_close):
            try:
                pos = current[sid]
                lots = pos["quantity"]
                yd_qty = pos.get("yd_quantity", 0) or 0
                day_trade = (yd_qty == 0)
                if self._broker:
                    trade = self._broker.sell(sid, lots, day_trade=day_trade)
                elif day_trade:
                    trade = trade_api.close(self._api, sid, lots)
                else:
                    trade = trade_api.close_normal(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                limit_price = float(trade.order.price) if hasattr(trade, "order") else pos["avg_price"]
                print(
                    f"[LIVE CLOSE] {sid} {lots}張 {'當沖' if day_trade else '普通'}限價@{limit_price} → {status} [{order_id}]"
                    f"（等待成交確認後從持倉移除）"
                )
                _log_trade(sid, "sell", f"{lots}張@{limit_price} {'當沖' if day_trade else '普通'} [{order_id}]", status)
                _push_trade(
                    {
                        "order_id": order_id,
                        "time": _now(),
                        "stock_id": sid,
                        "direction": "sell",
                        "price": limit_price,
                        "quantity": lots,
                        "status": status,
                        "broker_response": status,
                    }
                )
                # 不在此呼叫 _close_pos：持倉留在 dashboard 直到 broker 確認成交
                # BrokerClient 已在 sell() 內部追蹤 _sell_orders，不需額外 add
            except Exception as e:
                print(f"[LIVE ERROR] 平倉 {sid}: {e}")

        # 同步最新額度到 dashboard
        _push_summary(
            {
                "used_quota": self._used_quota,
                "total_capital": self.total_capital,
            }
        )
        print(
            f"[RECONCILE] 本分鐘調和完成：已用額度 {self._used_quota/10000:.1f}萬，目前持倉 {len(_api_positions)}",
            flush=True,
        )

    def force_close_own_positions(self):
        """13:25 強制平倉：只平本系統今日開的當沖倉，不動其他長期持倉。
        判斷依據：_api_positions（本系統追蹤的持倉），非永豐帳戶全部持倉。
        """
        _log_sys("盤後強制平倉程序啟動", "warning")
        print(f"[FORCE CLOSE] 盤後強制平倉程序啟動", flush=True)
        if not self._connect():
            print(f"[FORCE CLOSE] ✗ 連線失敗，中止", flush=True)
            return
        own_stocks = list(_api_positions.keys())
        if not own_stocks:
            print(f"[FORCE CLOSE] 無本系統持倉，略過", flush=True)
            return
        print(f"[FORCE CLOSE] 準備平倉 {len(own_stocks)} 支：{own_stocks}", flush=True)
        try:
            current = trade_api.get_positions(self._api, day_trade_only=True)
        except Exception as e:
            print(f"[FORCE CLOSE] ✗ 取得持倉失敗: {e}", flush=True)
            return
        for sid in own_stocks:
            if sid not in current:
                continue
            try:
                pos = current[sid]
                lots = pos["quantity"]
                avg_price = pos["avg_price"]
                yd_qty = pos.get("yd_quantity", 0) or 0
                day_trade = (yd_qty == 0)
                if self._broker:
                    # 強制平倉用現價單（close_at_price），確保收盤前成交
                    trade = self._broker.sell(sid, lots, day_trade=day_trade, market=True)
                elif day_trade:
                    trade = trade_api.close_at_price(self._api, sid, lots, day_trade=True)
                else:
                    trade = trade_api.close_at_price(self._api, sid, lots, day_trade=False)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                exit_price = float(trade.order.price) if hasattr(trade, "order") else avg_price
                entry_price = _get_pos_entry(sid)
                pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price else 0.0
                print(f"[FORCE CLOSE] {sid} {lots}張 {'當沖' if day_trade else '普通'}@{exit_price} → {status} [{order_id}]")
                _log_trade(sid, "force", f"{lots}張@{exit_price} 強制平倉 [{order_id}]", status)
                _push_trade(
                    {
                        "order_id": order_id,
                        "time": _now(),
                        "stock_id": sid,
                        "direction": "sell",
                        "price": exit_price,
                        "quantity": lots,
                        "status": status,
                        "broker_response": "force_close_eod",
                    }
                )
                _close_pos(sid, pnl_pct, exit_reason="force_close_eod", exit_price=exit_price)
            except Exception as e:
                print(f"[FORCE CLOSE] 平倉 {sid} 失敗: {e}")

    def close_stock_now(self, sids: list):
        """前端「立即平倉」按鈕觸發，背景執行緒直接現價賣出，不等下一分鐘 reconcile。
        BrokerClient.sell() 會先取消同股舊賣單再下新單，避免集保重複凍結。
        """
        if not self._connect():
            print(f"[CLOSE NOW] 連線失敗", flush=True)
            return
        try:
            current = trade_api.get_positions(self._api)  # 含 yd 持股
        except Exception as e:
            print(f"[CLOSE NOW] 取得持倉失敗: {e}", flush=True)
            return
        for sid in sids:
            pos = current.get(sid)
            if not pos:
                print(f"[CLOSE NOW] {sid} 無持倉，略過", flush=True)
                continue
            lots = pos["quantity"]
            yd_qty = pos.get("yd_quantity", 0) or 0
            day_trade = (yd_qty == 0)
            try:
                if self._broker:
                    # sell() 內部先取消舊賣單再以現價掛限價（close_at_price）
                    trade = self._broker.sell(sid, lots, day_trade=day_trade, market=True)
                else:
                    trade = trade_api.close_at_price(self._api, sid, lots, day_trade=day_trade)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                limit_price = float(trade.order.price) if hasattr(trade, "order") else 0
                label = "當沖" if day_trade else "普通"
                print(f"[CLOSE NOW] {sid} {lots}張 現價{label}賣出@{limit_price} → {status} [{order_id}]", flush=True)
                _push_trade({
                    "order_id": order_id,
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "sell",
                    "price": limit_price,
                    "quantity": lots,
                    "status": status,
                    "broker_response": "close_now",
                })
            except Exception as e:
                print(f"[CLOSE NOW] {sid} 平倉失敗: {e}", flush=True)

    def logout(self):
        """登出永豐 API（程序結束前呼叫）。"""
        if self._api:
            try:
                self._api.logout()
            except Exception:
                pass
            self._api = None


def make_executor(mode: str, total_capital: float = 1_000_000, name_lookup=None):
    """
    mode:
      paper → PaperTrader（本機 JSON，不碰 API）
      sim   → LiveTrader(simulation=True)（永豐模擬環境）
      live  → LiveTrader(simulation=False)（正式下單）
    name_lookup: callable(sid) → str，用於查詢公司名稱
    """
    if mode == "paper":
        return PaperTrader(total_capital)
    if mode == "sim":
        return LiveTrader(total_capital, simulation=True, name_lookup=name_lookup)
    if mode == "live":
        return LiveTrader(total_capital, simulation=False, name_lookup=name_lookup)
    raise ValueError(f"未知 mode: {mode!r}，應為 paper / sim / live")
