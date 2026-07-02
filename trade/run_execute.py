import json
from datetime import datetime
from pathlib import Path

try:
    from trade import trade_api  # imported from project root
except ImportError:
    import trade_api  # run directly from trade/ directory

try:
    from api import _positions as _api_positions
    def _get_pos_entry(sid: str) -> float:
        return _api_positions.get(sid, {}).get("entry_price", 0.0)
except ImportError:
    def _get_pos_entry(_): return 0.0

try:
    from api import push_trade as _push_trade
    from api import push_position as _push_pos
    from api import close_position as _close_pos
    from api import push_summary_update as _push_summary
    from api import push_completed_trades_from_broker as _push_closed_sync
    from api import sync_broker_snapshot as _sync_broker_snapshot
    from api import get_setting as _get_setting
except ImportError:
    def _push_trade(*_): pass
    def _push_pos(*_): pass
    def _close_pos(*_): pass
    def _push_summary(*_): pass
    def _push_closed_sync(*_): pass
    def _sync_broker_snapshot(*_): pass
    def _get_setting(key, default=None): return default


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_time(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    text = str(value or "")
    return text[11:19] if len(text) >= 19 else text


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

        # 當沖額度：買入 + 賣出 各佔一半
        budget = self.total_capital / 2 / max(len(target), 1)

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
        self.total_capital = total_capital
        self.simulation = simulation
        self._api = None
        self._positions_synced = False  # 啟動後第一次 reconcile 時同步現有持倉
        self._name_lookup = name_lookup or (lambda sid: sid)
        self._used_quota = 0.0  # 今日已用額度（買入+賣出金額累計）；重啟由 sync_from_broker 恢復

    def _connect(self) -> bool:
        if self._api is not None:
            return True
        try:
            self._api = trade_api.test() if self.simulation else trade_api.login()
            self._api.set_order_callback(self._on_order_event)
            print("[LIVE] 永豐 API 連線成功")
            return True
        except Exception as e:
            print(f"[LIVE] 登入失敗: {e}")
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
        if not self._connect():
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
            positions.append({
                "stock_id": sid,
                "name": self._name_lookup(sid),
                "quantity": qty,
                "pnl_pct": pnl,
                "entry_price": avg,
                "current_price": curr_price,
                "stop_loss": round(avg * 0.97, 2),
                "take_profit": round(avg * 1.03, 2),
            })

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
            completed_trades.append({
                "time": c.get("sell_time", "-"),
                "stock_id": c["stock_id"],
                "name": self._name_lookup(c["stock_id"]),
                "quantity": qty,
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_amt": pnl_amt,
                "exit_reason": "broker_sync",
            })

        wins = sum(1 for c in completed_trades if c["pnl_pct"] > 0)
        closed = len(completed_trades)   # FIFO 配對回合數，最準確
        avg_pnl = (
            round(sum(c["pnl_pct"] for c in completed_trades) / closed, 4)
            if closed else 0.0
        )
        total_pnl_amt = round(sum(c.get("pnl_amt") or 0 for c in completed_trades), 0)

        # 今日已用額度 = 所有成交的買入金額 + 賣出金額（當沖買賣都佔額度）
        filled_set = {"FILLED", "PARTIAL"}
        used_quota = round(sum(
            t["price"] * t.get("filled", t.get("quantity", 0)) * 1000
            for t in trades
            if t.get("status") in filled_set
        ), 0)
        # 開倉成交 = FILLED 買進筆數；平倉回合 = FIFO 配對數（closed）
        filled = {"FILLED", "PARTIAL"}
        open_count = sum(1 for t in trades if t.get("direction") == "buy" and t.get("status") in filled)
        failed_count = sum(1 for t in trades if t.get("status") == "FAILED")
        last_updated = _fmt_time(orders[-1]["time"]) if orders else _now()

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
                "errors": failed_count,
                "last_updated": last_updated,
            },
        )
        self._used_quota = used_quota   # 重啟恢復已用額度
        self._positions_synced = True
        print(
            f"[LIVE SYNC] 啟動同步完成：持倉 {len(positions)}、"
            f"今日委託 {len(trades)}、已平倉 {closed}、已用額度 {used_quota/10000:.1f}萬"
        )
        return True

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
                "order_id": msg.get("id", ""),
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
                current = trade_api.get_positions(self._api, day_trade_only=True)
                if sid in current:
                    pos = current[sid]
                    avg = pos["avg_price"]
                    qty = pos["quantity"]
                    snap = self._api.snapshots([self._api.Contracts.Stocks[sid]])[0].close
                    pnl = round((snap - avg) / avg * 100, 4) if avg else 0.0
                    _push_pos({
                        "stock_id": sid,
                        "name": self._name_lookup(sid),
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

    def _sync_positions_with_broker(self, current: dict):
        """每次 reconcile 呼叫：以永豐目前持倉為準，修正 dashboard 偏移。
        - broker 有、dashboard 沒有 → 補進去（可能是 SDEAL 漏接）
        - dashboard 有、broker 沒有 → 移除（broker 已平倉但通知未收到）
        - 兩者皆有但均價不同 → 更新 entry_price（fill 後均價修正）
        """
        # 補進 broker 有但 dashboard 沒有的持倉
        for sid, pos in current.items():
            dash = _api_positions.get(sid, {})
            avg = pos["avg_price"]
            qty = pos["quantity"]
            if sid not in _api_positions or abs(dash.get("entry_price", 0) - avg) > 0.01:
                _push_pos({
                    "stock_id": sid,
                    "name": self._name_lookup(sid),
                    "quantity": qty,
                    "pnl_pct": dash.get("pnl_pct", 0.0),
                    "entry_price": avg,
                    "current_price": dash.get("current_price", avg),
                    "stop_loss": round(avg * 0.97, 2),
                    "take_profit": round(avg * 1.03, 2),
                })
        # 移除 broker 已不存在的 ghost 持倉
        # 只有 broker 有回傳至少一筆時才移除，避免 API 短暫異常誤刪全部持倉
        if current:
            for sid in list(_api_positions.keys()):
                if sid not in current:
                    _close_pos(sid, 0.0, exit_reason="broker_closed")

    def reconcile(self, signals: list[dict]):
        """
        依本分鐘模型訊號調整券商實際持倉。

        先從永豐取得目前持倉（每分鐘同步，確保 dashboard 與 broker 不偏移），
        再與 signals 做 diff：
        - 在 signals 中但不在券商持倉 → 取快照報價 → 下限價買單
        - 在券商持倉中但不在 signals → 下限價賣單
        下單後同步更新 api.py 的 in-memory 狀態，供 dashboard 顯示。
        """
        if not self._connect():
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

        # 取消上分鐘所有 SENT 未成交單：
        #   買單：訊號時效過，取消後若訊號還在本分鐘重新以新價掛
        #   賣單：取消後持倉仍在 current → to_close 自動重掛新價
        try:
            cancelled = trade_api.cancel_sent_orders(self._api)
            if cancelled["buy"]:
                # 從委託單本身的價格×數量還原額度（未成交，不在 _api_positions 裡）
                restore = sum(cost for _, cost in cancelled["buy"])
                self._used_quota = max(0, self._used_quota - restore)
                sids = [sid for sid, _ in cancelled["buy"]]
                print(f"[CANCEL BUY]  取消未成交買單: {sids}，還原額度 {restore/10000:.1f}萬")
            if cancelled["sell"]:
                print(f"[CANCEL SELL] 取消未成交賣單: {cancelled['sell']}，本分鐘重新掛")
        except Exception as e:
            print(f"[CANCEL] 取消委託失敗: {e}")

        # 取消後重新取得委託狀態（PARTIAL 單仍保留不動）
        try:
            pending_orders = trade_api.list_orders_today(self._api)
            pending_buy = {
                o["stock_id"] for o in pending_orders
                if o["status"] in {"SENT", "PARTIAL"} and o["direction"] == "buy"
            }
            pending_sell = {
                o["stock_id"] for o in pending_orders
                if o["status"] in {"SENT", "PARTIAL"} and o["direction"] == "sell"
            }
        except Exception:
            pending_buy = set()
            pending_sell = set()

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
                        "name": self._name_lookup(sid),
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
                    wins = sum(1 for c in closed_today if c["pnl_pct"] > 0)
                    n = len(closed_today)
                    _push_summary({
                        "closed": n,
                        "wins": wins,
                        "win_rate": round(wins / n * 100, 1),
                        "today_pnl_pct": round(avg_pnl, 4),
                    })
                    _push_closed_sync(closed_today)
                    print(f"[LIVE SYNC] 今日已平倉 {n} 筆，勝率 {wins}/{n}，均損益 {avg_pnl:.2f}%")
            except Exception as e:
                print(f"[LIVE SYNC] 已平倉同步失敗: {e}")

        # 排除已有委託中的買/賣單，防止重複下單與「集保餘股不足」
        open_enabled  = _get_setting("open_enabled")  if _get_setting("open_enabled")  is not None else True
        close_enabled = _get_setting("close_enabled") if _get_setting("close_enabled") is not None else True
        to_open  = (target - set(current) - pending_buy)  if open_enabled  else set()
        to_close = (set(current) - target - pending_sell) if close_enabled else set()

        if not to_open and not to_close:
            return

        # 當沖額度：買入 + 賣出 各佔一半，所以買入預算 = total_capital / 2 / 持倉支數
        budget = self.total_capital / 2 / max(len(target), 1)
        snapshots: dict[str, float] = {}

        if to_open:
            try:
                contracts = [self._api.Contracts.Stocks[sid] for sid in to_open]
                snapshots = {s.code: s.close for s in self._api.snapshots(contracts)}
            except Exception as e:
                self._reset_on_disconnect(e, "取得快照")

        # 迴圈前算一次可用額度，之後每開一倉就扣，避免第 2 張高估
        open_reserved = sum(
            _api_positions.get(s, {}).get("entry_price", 0)
            * _api_positions.get(s, {}).get("quantity", 0) * 1000
            for s in _api_positions
        )
        available = self.total_capital - self._used_quota - open_reserved

        cfg_min_price        = _get_setting("min_price")
        cfg_max_price        = _get_setting("max_price")
        cfg_max_budget_stock = _get_setting("max_budget_per_stock")  # 萬
        # 手續費 + 證交稅緩衝（買+賣 ≈ 0.3%），讓實際資金留有餘裕
        fee_rate = float(_get_setting("fee_rate") or 0.003)
        lot_cost_factor = 1 + fee_rate   # 每張有效成本倍數

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

                # 單股預算：均分額度 → 單股設定上限 → 可用額度上限（三取最小）
                stock_budget = budget
                if cfg_max_budget_stock is not None:
                    stock_budget = min(stock_budget, float(cfg_max_budget_stock) * 10000)
                stock_budget = min(stock_budget, available / 2)

                # 每張有效成本含手續費緩衝（確保不因小數截斷而超額）
                effective_lot_cost = price * 1000 * lot_cost_factor
                lots_by_budget = int(stock_budget / effective_lot_cost)
                lots_by_quota  = int((available / 2) / effective_lot_cost)
                lots = min(lots_by_budget, lots_by_quota)
                if lots < 1:
                    print(f"[LIVE SKIP] {sid} 額度不足（可用 {available/10000:.1f}萬，需 {price*2/10:.1f}萬/張）")
                    continue

                trade = trade_api.open(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                buy_cost = price * lots * 1000
                self._used_quota += buy_cost   # 買入已用
                available -= buy_cost * 2      # 扣掉 買 + 預留賣，給下一支用
                print(f"[LIVE OPEN] {sid} {lots}張 @ {price} → {status} [{order_id}]"
                      f"（剩餘可用 {available/10000:.1f}萬）")
                _push_trade({
                    "order_id": order_id,
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
                avg_price = current[sid]["avg_price"]
                trade = trade_api.close(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                # 用委託限價估算出場價（SDEAL 回報後 _on_order_event 會更新）
                exit_price = float(trade.order.price) if hasattr(trade, "order") else avg_price
                entry_price = _get_pos_entry(sid)
                pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price else 0.0
                sell_cost = exit_price * lots * 1000
                self._used_quota += sell_cost  # 賣出佔用額度
                print(f"[LIVE CLOSE] {sid} {lots}張 @ {exit_price} → {status} [{order_id}]"
                      f"（額度已用 {self._used_quota/10000:.1f}萬）")
                _push_trade({
                    "order_id": order_id,
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "sell",
                    "price": exit_price,
                    "quantity": lots,
                    "status": status,
                    "broker_response": status,
                })
                _close_pos(sid, pnl_pct, exit_price=exit_price)
            except Exception as e:
                print(f"[LIVE ERROR] 平倉 {sid}: {e}")

        # 同步最新額度到 dashboard
        _push_summary({
            "used_quota": self._used_quota,
            "total_capital": self.total_capital,
        })

    def force_close_own_positions(self):
        """13:25 強制平倉：只平本系統今日開的當沖倉，不動其他長期持倉。
        判斷依據：_api_positions（本系統追蹤的持倉），非永豐帳戶全部持倉。
        """
        if not self._connect():
            return
        own_stocks = list(_api_positions.keys())
        if not own_stocks:
            print("[FORCE CLOSE] 無本系統持倉，略過")
            return
        try:
            current = trade_api.get_positions(self._api, day_trade_only=True)
        except Exception as e:
            print(f"[FORCE CLOSE] 取得持倉失敗: {e}")
            return
        for sid in own_stocks:
            if sid not in current:
                continue
            try:
                lots = current[sid]["quantity"]
                avg_price = current[sid]["avg_price"]
                trade = trade_api.close(self._api, sid, lots)
                status = trade_api.normalize_status(trade.status.status)
                order_id = getattr(getattr(trade, "order", None), "id", "") or ""
                exit_price = float(trade.order.price) if hasattr(trade, "order") else avg_price
                entry_price = _get_pos_entry(sid)
                pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price else 0.0
                print(f"[FORCE CLOSE] {sid} {lots}張 @ {exit_price} → {status} [{order_id}]")
                _push_trade({
                    "order_id": order_id,
                    "time": _now(),
                    "stock_id": sid,
                    "direction": "sell",
                    "price": exit_price,
                    "quantity": lots,
                    "status": status,
                    "broker_response": "force_close_eod",
                })
                _close_pos(sid, pnl_pct, exit_reason="force_close_eod", exit_price=exit_price)
            except Exception as e:
                print(f"[FORCE CLOSE] 平倉 {sid} 失敗: {e}")

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
