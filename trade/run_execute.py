import json
from datetime import datetime
from pathlib import Path

try:
    from trade import trade_api  # imported from project root
except ImportError:
    import trade_api  # run directly from trade/ directory

try:
    from api import push_trade as _push_trade
    from api import push_position as _push_pos
    from api import close_position as _close_pos
    from api import push_summary_update as _push_summary
except ImportError:
    # standalone / unit-test fallback
    def _push_trade(*_): pass
    def _push_pos(*_): pass
    def _close_pos(*_): pass
    def _push_summary(*_): pass


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


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

    def reconcile(self, signals: list[dict]):
        """
        依本分鐘模型訊號調整模擬持倉。

        signals 是 predict_live() 回傳的清單，每筆含 stock_id / price / proba / name。
        - 在 signals 中但不在持倉 → 模擬開倉
        - 在持倉中但不在 signals → 模擬平倉
        - 兩者皆有 → 不動（不重複下單）
        """
        target = {s["stock_id"]: s for s in signals}
        to_open = set(target) - set(self._positions)
        to_close = set(self._positions) - set(target)

        if not to_open and not to_close:
            return

        budget = self.total_capital / max(len(target), 1)

        for sid in sorted(to_close):
            pos = self._positions.pop(sid)
            entry = pos["avg_price"]
            print(f"[PAPER CLOSE] {sid} {pos['quantity']}張 (entry={entry:.2f})")
            _push_trade({
                "time": _now(),
                "stock_id": sid,
                "direction": "sell",
                "price": entry,       # exit price unknown without feed; use entry as placeholder
                "quantity": pos["quantity"],
                "status": "FILLED",
                "broker_response": "paper",
            })
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
            _push_trade({
                "time": _now(),
                "stock_id": sid,
                "direction": "buy",
                "price": price,
                "quantity": lots,
                "status": "FILLED",
                "broker_response": "paper",
            })
            _push_pos({
                "stock_id": sid,
                "name": sig.get("name", sid),
                "pnl_pct": 0.0,
                "entry_price": price,
                "current_price": price,
                "stop_loss": round(price * 0.97, 2),
                "take_profit": round(price * 1.03, 2),
            })

        self._path.write_text(json.dumps(self._positions, ensure_ascii=False, indent=2))


class LiveTrader:
    """
    透過永豐證券 Shioaji API 真實（或模擬帳號）下單。

    simulation=True  → 永豐模擬環境（帳號相同，單子不真實成交）
    simulation=False → 正式帳號，會真的扣款與交割

    連線是 lazy 的：第一次 reconcile() 時才登入，之後保持同一個 api 物件。
    登入失敗（例如雲端 IP 被擋）會印錯誤並直接 return，不影響 dashboard。
    """

    def __init__(self, total_capital: float = 1_000_000, simulation: bool = True):
        self.total_capital = total_capital
        self.simulation = simulation
        self._api = None
        self._positions_synced = False  # 啟動後第一次 reconcile 時同步現有持倉

    def _connect(self) -> bool:
        if self._api is not None:
            return True
        try:
            self._api = trade_api.test() if self.simulation else trade_api.login()
            self._api.set_order_callback(self._on_order_event)
            return True
        except Exception as e:
            print(f"[LIVE] 登入失敗: {e}")
            return False

    def _on_order_event(self, state, msg):
        """永豐成交回報（SDEAL）→ 更新 dashboard 進場均價"""
        import shioaji as sj
        if state != sj.OrderState.StockDeal:
            return
        try:
            sid = msg.get("code") or msg["contract"]["code"]
            fill_price = float(msg["price"])
            fill_qty = int(msg["quantity"])
            action = str(msg.get("action", "")).lower()
            print(f"[LIVE FILL] {sid} {action} {fill_qty}張 @ {fill_price}")
            _push_trade({
                "time": _now(),
                "stock_id": sid,
                "direction": "buy" if "buy" in action else "sell",
                "price": fill_price,
                "quantity": fill_qty,
                "status": "FILLED",
                "broker_response": f"fill@{fill_price}",
            })
            # 更新 dashboard 持倉的實際進場均價
            try:
                current = trade_api.get_positions(self._api)
                if sid in current:
                    pos = current[sid]
                    avg = pos["avg_price"]
                    qty = pos["quantity"]
                    snap = self._api.snapshots([self._api.Contracts.Stocks[sid]])[0].close
                    pnl = round((snap - avg) / avg * 100, 4) if avg else 0.0
                    _push_pos({
                        "stock_id": sid,
                        "name": sid,
                        "quantity": qty,
                        "pnl_pct": pnl,
                        "entry_price": avg,
                        "current_price": snap,
                        "stop_loss": round(avg * 0.97, 2),
                        "take_profit": round(avg * 1.03, 2),
                    })
            except Exception as e:
                print(f"[LIVE FILL] 更新持倉失敗: {e}")
        except Exception as e:
            print(f"[LIVE FILL] 解析回報失敗: {e} | {msg}")

    def reconcile(self, signals: list[dict]):
        """
        依本分鐘模型訊號調整券商實際持倉。

        先從永豐取得目前持倉，再與 signals 做 diff：
        - 在 signals 中但不在券商持倉 → 取快照報價 → 下限價買單
        - 在券商持倉中但不在 signals → 下限價賣單
        下單後同步更新 api.py 的 in-memory 狀態，供 dashboard 顯示。
        """
        if not self._connect():
            return

        sig_map = {s["stock_id"]: s for s in signals}
        target = set(sig_map)

        try:
            current = trade_api.get_positions(self._api)
        except Exception as e:
            print(f"[LIVE] 取得持倉失敗: {e}")
            return

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
                    _push_pos({
                        "stock_id": sid,
                        "name": sid,
                        "quantity": qty,
                        "pnl_pct": pnl,
                        "entry_price": avg,
                        "current_price": curr_price,
                        "stop_loss": round(avg * 0.97, 2),
                        "take_profit": round(avg * 1.03, 2),
                    })
                print(f"[LIVE SYNC] 同步 {len(current)} 筆現有持倉到 dashboard（含永豐均價）")

            # 同步今日已平倉（從永豐 list_trades 重建，僅今日）
            try:
                closed_today = trade_api.get_closed_today(self._api)
                if closed_today:
                    avg_pnl = sum(c["pnl_pct"] for c in closed_today) / len(closed_today)
                    _push_summary({
                        "closed": len(closed_today),
                        "today_pnl_pct": round(avg_pnl, 4),
                    })
                    print(f"[LIVE SYNC] 今日已平倉 {len(closed_today)} 筆，均損益 {avg_pnl:.2f}%")
            except Exception as e:
                print(f"[LIVE SYNC] 已平倉同步失敗: {e}")

        to_open = target - set(current)
        to_close = set(current) - target

        if not to_open and not to_close:
            return

        budget = self.total_capital / max(len(target), 1)
        snapshots: dict[str, float] = {}

        if to_open:
            try:
                contracts = [self._api.Contracts.Stocks[sid] for sid in to_open]
                snapshots = {s.code: s.close for s in self._api.snapshots(contracts)}
            except Exception as e:
                print(f"[LIVE] 取得報價失敗: {e}")

        for sid in sorted(to_open):
            try:
                price = snapshots.get(sid, 0)
                if price <= 0:
                    print(f"[LIVE SKIP] {sid} 無報價")
                    continue
                lots = int(budget / (price * 1000))
                if lots < 1:
                    print(f"[LIVE SKIP] {sid} 資金不足一張")
                    continue
                trade = trade_api.open(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                print(f"[LIVE OPEN] {sid} {lots}張 → {status}")
                _push_trade({
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "buy",
                    "price": price,
                    "quantity": lots,
                    "status": status,
                    "broker_response": status,
                })
                sig = sig_map.get(sid, {})
                _push_pos({
                    "stock_id": sid,
                    "name": sig.get("name", sid),
                    "quantity": lots,
                    "pnl_pct": 0.0,
                    "entry_price": price,
                    "current_price": price,
                    "stop_loss": round(price * 0.97, 2),
                    "take_profit": round(price * 1.03, 2),
                })
            except Exception as e:
                print(f"[LIVE ERROR] 開倉 {sid}: {e}")

        for sid in sorted(to_close):
            try:
                lots = current[sid]["quantity"]
                trade = trade_api.close(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                print(f"[LIVE CLOSE] {sid} {lots}張 → {status}")
                _push_trade({
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "sell",
                    "price": 0,          # fill price unknown until confirmed
                    "quantity": lots,
                    "status": status,
                    "broker_response": status,
                })
                _close_pos(sid, 0.0)
            except Exception as e:
                print(f"[LIVE ERROR] 平倉 {sid}: {e}")

    def logout(self):
        """登出永豐 API（程序結束前呼叫）。"""
        if self._api:
            try:
                self._api.logout()
            except Exception:
                pass
            self._api = None


def make_executor(mode: str, total_capital: float = 1_000_000):
    """
    mode:
      paper → PaperTrader（本機 JSON，不碰 API）
      sim   → LiveTrader(simulation=True)（永豐模擬環境）
      live  → LiveTrader(simulation=False)（正式下單）
    """
    if mode == "paper":
        return PaperTrader(total_capital)
    if mode == "sim":
        return LiveTrader(total_capital, simulation=True)
    if mode == "live":
        return LiveTrader(total_capital, simulation=False)
    raise ValueError(f"未知 mode: {mode!r}，應為 paper / sim / live")
