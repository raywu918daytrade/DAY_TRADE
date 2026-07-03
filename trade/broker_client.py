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


class BrokerClient:
    def __init__(self, api):
        self._api = api
        self._sell_orders: dict[str, str] = {}  # sid -> order_id

    @property
    def pending_sell_sids(self) -> set:
        return set(self._sell_orders)

    # ── 買 ────────────────────────────────────────────────────────────────

    def buy(self, sid: str, qty: int) -> object:
        trade = trade_api.open(self._api, sid, qty)
        order_id = _order_id(trade)
        status = trade_api.normalize_status(trade.status.status)
        print(f"[BROKER BUY]  {sid} {qty}張 → {status} [{order_id}]", flush=True)
        return trade

    # ── 賣 ────────────────────────────────────────────────────────────────

    def sell(self, sid: str, qty: int, day_trade: bool = True, market: bool = False) -> object:
        """賣出前先取消同股現有賣單，避免重複委託。
        market=True：以現價掛限價單（貼近成交，比純 MKT 更穩定）。
        """
        if sid in self._sell_orders:
            self._cancel_sell(sid)

        if market:
            trade = trade_api.close_at_price(self._api, sid, qty, day_trade=day_trade)
        elif day_trade:
            trade = trade_api.close(self._api, sid, qty)
        else:
            trade = trade_api.close_normal(self._api, sid, qty)

        order_id = _order_id(trade)
        self._sell_orders[sid] = order_id
        status = trade_api.normalize_status(trade.status.status)
        label = ("市價" if market else "限價") + ("當沖" if day_trade else "普通")
        print(f"[BROKER SELL] {sid} {qty}張 {label} → {status} [{order_id}]", flush=True)
        return trade

    # ── 取消 ──────────────────────────────────────────────────────────────

    def cancel_all_orders(self) -> dict:
        """
        取消所有 SENT 買/賣單。
        回傳 {"buy": [(sid, cost), ...], "sell": [sid, ...]}，
        結構與 trade_api.cancel_sent_orders 相同，供 reconcile 還原額度。
        """
        result = trade_api.cancel_sent_orders(self._api)
        for sid in result.get("sell", []):
            self._sell_orders.pop(sid, None)

        # 若 _sell_orders 仍有殘留（cancel_sent_orders 未能取消），只記錄警告
        # 可能原因：sim session 隔離（上一次 session 的單不可見）或 broker 延遲
        for sid in list(self._sell_orders):
            print(
                f"[BROKER] 警告：{sid} [{self._sell_orders[sid]}] 未被 cancel_sent_orders 取消，下次繼續嘗試",
                flush=True,
            )

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
                        break
        except Exception as e:
            print(f"[BROKER] 取消舊賣單 {sid} 失敗: {e}", flush=True)
        finally:
            self._sell_orders.pop(sid, None)


def _order_id(trade) -> str:
    return getattr(getattr(trade, "order", None), "id", "") or ""
