"""
BrokerClient：永豐 Shioaji API 薄包裝層。

職責：
  - 買進 / 賣出 / 市價賣出 的統一入口
  - 賣出前自動取消同股舊賣單（防止重複委託）
  - 追蹤本系統已送出的賣單（_sell_orders），不依賴 broker 即時狀態
  - 每分鐘 cancel_all_orders() 取消舊買/賣單並清理追蹤
  - 成交或持倉消失時由外部呼叫 on_fill / on_position_closed 更新追蹤
"""

from __future__ import annotations

try:
    from trade import trade_api
except (ImportError, ModuleNotFoundError):
    try:
        import trade_api  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        trade_api = None  # type: ignore

try:
    from api import append_system_log as _log_sys, append_trade_log as _log_trade
except Exception:
    def _log_sys(msg, level="info"): pass  # type: ignore
    def _log_trade(sid, action, detail, status=""): pass  # type: ignore


class BrokerClient:
    def __init__(self, api):
        self._api = api
        self._sell_orders: dict[str, str] = {}  # sid -> order_id

    @property
    def pending_sell_sids(self) -> set:
        return set(self._sell_orders)

    # ── 買 ────────────────────────────────────────────────────────────────

    def buy(self, sid: str, qty: int) -> object:
        try:
            trade = trade_api.open(self._api, sid, qty)
        except Exception as e:
            _log_sys(f"永豐 BUY 失敗 {sid} {qty}張: {e}", "error")
            raise
        order_id = _order_id(trade)
        status = trade_api.normalize_status(trade.status.status)
        print(f"[BROKER BUY]  {sid} {qty}張 → {status} [{order_id}]", flush=True)
        _log_trade(sid, "buy", f"{qty}張 → {status} [{order_id}]", status)
        return trade

    # ── 賣 ────────────────────────────────────────────────────────────────

    def sell(self, sid: str, qty: int, day_trade: bool = True, market: bool = False) -> object:
        """賣出前先取消同股現有賣單，避免重複委託。
        market=True：以現價掛限價單（貼近成交，比純 MKT 更穩定）。
        """
        if sid in self._sell_orders:
            self._cancel_sell(sid)

        label = ("市價" if market else "限價") + ("當沖" if day_trade else "普通")
        try:
            if market:
                trade = trade_api.close_at_price(self._api, sid, qty, day_trade=day_trade)
            elif day_trade:
                trade = trade_api.close(self._api, sid, qty)
            else:
                trade = trade_api.close_normal(self._api, sid, qty)
        except Exception as e:
            _log_sys(f"永豐 SELL 失敗 {sid} {qty}張 {label}: {e}", "error")
            raise

        order_id = _order_id(trade)
        self._sell_orders[sid] = order_id
        status = trade_api.normalize_status(trade.status.status)
        print(f"[BROKER SELL] {sid} {qty}張 {label} → {status} [{order_id}]", flush=True)
        _log_trade(sid, "sell", f"{qty}張 {label} → {status} [{order_id}]", status)
        return trade

    # ── 取消 ──────────────────────────────────────────────────────────────

    def cancel_all_orders(self) -> dict:
        """
        取消所有 SENT 買/賣單。
        回傳 {"buy": [(sid, cost), ...], "sell": [sid, ...]}，
        結構與 trade_api.cancel_sent_orders 相同，供 reconcile 還原額度。
        """
        try:
            result = trade_api.cancel_sent_orders(self._api)
        except Exception as e:
            _log_sys(f"永豐 cancel_all_orders 失敗: {e}", "error")
            raise
        n_buy = len(result.get("buy", []))
        n_sell = len(result.get("sell", []))
        if n_buy or n_sell:
            _log_sys(f"取消委託：買 {n_buy} 筆，賣 {n_sell} 筆")
        for sid in result.get("sell", []):
            self._sell_orders.pop(sid, None)

        # 若 _sell_orders 仍有殘留（cancel_sent_orders 未能取消），只記錄警告
        for sid in list(self._sell_orders):
            msg = f"警告：{sid} [{self._sell_orders[sid]}] 未被 cancel 取消，下次繼續嘗試"
            print(f"[BROKER] {msg}", flush=True)
            _log_sys(f"[BROKER] {msg}", "warning")

        return result

    # ── 成交 / 持倉消失回報 ───────────────────────────────────────────────

    def on_fill(self, sid: str, direction: str):
        """SDEAL 成交回報：賣出成交後解除追蹤。"""
        if direction == "sell":
            self._sell_orders.pop(sid, None)

    def on_position_closed(self, sid: str):
        """持倉已從 broker 消失（sync_positions 偵測）：解除追蹤。"""
        self._sell_orders.pop(sid, None)

    def sync_with_positions(self, current_sids: set):
        """移除不再有持倉的 sid（避免長期殘留）。"""
        for sid in list(self._sell_orders):
            if sid not in current_sids:
                self._sell_orders.pop(sid, None)

    # ── 內部 ──────────────────────────────────────────────────────────────

    def _cancel_sell(self, sid: str):
        """取消指定股票的現有賣單（by order_id），失敗也清除追蹤。"""
        order_id = self._sell_orders.get(sid)
        try:
            if order_id:
                self._api.update_status()
                for t in self._api.list_trades():
                    if getattr(getattr(t, "order", None), "id", None) == order_id:
                        if trade_api.normalize_status(t.status.status) == "SENT":
                            result = self._api.cancel_order(t)
                            raw = str(result.status.status)
                            print(f"[BROKER] 取消舊賣單 {sid} [{order_id}] → {raw}", flush=True)
                            _log_sys(f"取消舊賣單 {sid} [{order_id}] → {raw}")
                        break
        except Exception as e:
            msg = f"取消舊賣單 {sid} 失敗: {e}"
            print(f"[BROKER] {msg}", flush=True)
            _log_sys(f"[BROKER] {msg}", "error")
        finally:
            self._sell_orders.pop(sid, None)


def _order_id(trade) -> str:
    return getattr(getattr(trade, "order", None), "id", "") or ""
