import base64
import os
import tempfile
from pathlib import Path

import shioaji as sj
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")


def login() -> sj.Shioaji:
    api = sj.Shioaji()
    api.login(
        api_key=os.environ["SINOPAC_API_KEY"],
        secret_key=os.environ["SINOPAC_SECRET_KEY"],
    )
    # CA_B64：永豐 .pfx 憑證檔以 base64 編碼存在 .env，避免實體檔案上版控
    # 產生方式：base64 -i 永豐證券.pfx | tr -d '\n'
    pfx_data = base64.b64decode(os.environ["SINOPAC_CA_B64"])
    with tempfile.NamedTemporaryFile(suffix=".pfx", delete=False) as f:
        f.write(pfx_data)
        ca_path = f.name
    api.activate_ca(
        ca_path=ca_path,
        ca_passwd=os.environ["SINOPAC_CA_PASSWD"],
        person_id=os.environ["SINOPAC_PERSON_ID"],
    )
    return api


def test():
    api = sj.Shioaji(simulation=True)  # 是否用虛擬主機登入
    api.login(
        api_key=os.environ["SINOPAC_API_KEY"],
        secret_key=os.environ["SINOPAC_SECRET_KEY"],
    )
    return api


_STATUS_MAP = {
    "OrderStatus.Filled":        "FILLED",
    "OrderStatus.PartFilled":    "PARTIAL",
    "OrderStatus.Submitted":     "SENT",
    "OrderStatus.PendingSubmit": "SENT",
    "OrderStatus.PreSubmitted":  "SENT",
    "OrderStatus.Failed":        "FAILED",
    "OrderStatus.Cancelled":     "FAILED",
    "OrderStatus.Inactive":      "FAILED",
}


def normalize_status(trade_status) -> str:
    """Shioaji OrderStatus → 統一格式 FILLED / PARTIAL / SENT / FAILED"""
    return _STATUS_MAP.get(str(trade_status), "SENT")


def _tick(price: float) -> float:
    if price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1.0
    return 5.0


def close(api: sj.Shioaji, stock_id: str, quantity: int):
    """當沖賣出（現股當沖，order_cond=DayTradingSell → 證交稅 0.15%，非 0.3%）"""
    contract = api.Contracts.Stocks[stock_id]
    price = api.snapshots([contract])[0].close
    raw = price * 0.995
    tick = _tick(raw)
    limit_price = round(int(raw / tick) * tick, 2)  # floor，確保賣得掉

    trade = api.place_order(
        contract,
        api.Order(
            price=limit_price,
            quantity=quantity,
            action=sj.Action.Sell,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
            order_cond=sj.StockOrderCond.Netting,  # 現股當沖賣出（新版 shioaji），稅率 0.15%
        ),
    )
    return trade


def close_normal(api: sj.Shioaji, stock_id: str, quantity: int):
    """普通賣出（昨日持股，yd_quantity > 0）—— 不用 Netting，稅率 0.3%"""
    contract = api.Contracts.Stocks[stock_id]
    price = api.snapshots([contract])[0].close
    raw = price * 0.995
    tick = _tick(raw)
    limit_price = round(int(raw / tick) * tick, 2)
    trade = api.place_order(
        contract,
        api.Order(
            price=limit_price,
            quantity=quantity,
            action=sj.Action.Sell,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
        ),
    )
    return trade


def close_at_price(api: sj.Shioaji, stock_id: str, quantity: int, day_trade: bool = True):
    """以現價掛限價賣出（不打折），用於立即平倉場景（close_now）。
    比 MKT 穩定：台股 MKT 在漲停/跌停時行為不一致；現價限價單集保不會出問題。
    day_trade=True  → 使用 Netting（當沖，稅 0.15%）
    day_trade=False → 普通賣出（昨日持股，稅 0.3%）
    """
    contract = api.Contracts.Stocks[stock_id]
    price = api.snapshots([contract])[0].close
    tick = _tick(price)
    # 現價直接掛（不打折），確保立刻成交
    limit_price = round(round(price / tick) * tick, 2)
    order_kwargs = dict(
        price=limit_price,
        quantity=quantity,
        action=sj.Action.Sell,
        price_type=sj.StockPriceType.LMT,
        order_type=sj.OrderType.ROD,
    )
    if day_trade:
        order_kwargs["order_cond"] = sj.StockOrderCond.Netting
    trade = api.place_order(contract, api.Order(**order_kwargs))
    return trade


def close_market(api: sj.Shioaji, stock_id: str, quantity: int, day_trade: bool = True):
    """市價賣出，保證成交。day_trade=True 用 Netting（當沖），False 用普通賣出（昨日持股）"""
    contract = api.Contracts.Stocks[stock_id]
    order_kwargs = dict(
        price=0,
        quantity=quantity,
        action=sj.Action.Sell,
        price_type=sj.StockPriceType.MKT,
        order_type=sj.OrderType.ROD,
    )
    if day_trade:
        order_kwargs["order_cond"] = sj.StockOrderCond.Netting
    trade = api.place_order(contract, api.Order(**order_kwargs))
    return trade


def open(api: sj.Shioaji, stock_id: str, quantity: int, action: sj.Action = sj.Action.Buy):
    """當沖買進（現股）"""
    contract = api.Contracts.Stocks[stock_id]
    close = api.snapshots([contract])[0].close
    raw = close * 1.01
    tick = _tick(raw)
    limit_price = round(round(raw / tick) * tick, 2)

    trade = api.place_order(
        contract,
        api.Order(
            price=limit_price,
            quantity=quantity,
            action=action,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
        ),
    )
    return trade


def get_positions(api: sj.Shioaji, day_trade_only: bool = False) -> dict[str, dict]:
    """回傳 {stock_id: {"quantity": lots, "avg_price": fill_price, "yd_quantity": 昨日庫存}} 目前持倉。
    day_trade_only=True：只回傳今天新買的（yd_quantity == 0），排除昨日帶過來的長期持股。
    """
    result = {}
    for p in api.list_positions(api.stock_account):
        yd_qty = getattr(p, "yd_quantity", 0) or 0
        if day_trade_only and yd_qty > 0:
            continue  # 昨日帶過來的長期持股，略過
        result[p.code] = {
            "quantity": p.quantity,
            "avg_price": float(p.price),
            "yd_quantity": yd_qty,
        }
    return result


def get_closed_today(api: sj.Shioaji) -> list[dict]:
    """
    從今日委託記錄重建已平倉清單。
    每筆賣出成交各算一回合（FIFO 配對對應買進），而非按股票合併。
    回傳 [{"stock_id", "buy_avg", "sell_avg", "quantity", "pnl_pct", "sell_time"}, ...]
    """
    from datetime import datetime, timezone, timedelta
    from collections import deque
    today = datetime.now(timezone(timedelta(hours=8))).date()  # 台灣時間
    api.update_status()
    trades = sorted(api.list_trades(), key=lambda t: t.status.order_datetime)

    # 按股票建立買進 FIFO queue：[(price, qty), ...]
    buy_queues: dict[str, deque] = {}
    # 賣出事件清單：[(price, qty, time), ...]
    sell_events: dict[str, list] = {}

    for t in trades:
        order_dt = t.status.order_datetime
        order_date = getattr(order_dt, "date", lambda: order_dt)()
        if str(order_date) != str(today):
            continue
        filled = t.status.deal_quantity
        if filled <= 0:
            continue
        code = t.contract.code
        price = float(t.order.price)
        action = str(t.order.action).lower()
        time_str = order_dt.strftime("%H:%M:%S") if hasattr(order_dt, "strftime") else str(order_dt)[11:19]
        if "buy" in action:
            buy_queues.setdefault(code, deque()).append((price, filled))
        else:
            sell_events.setdefault(code, []).append((price, filled, time_str))

    result = []
    for code, sells in sell_events.items():
        q = buy_queues.get(code, deque())
        buy_rem = 0
        buy_price = 0.0
        for sell_price, sell_qty, sell_time in sells:
            matched = 0
            cost = 0.0
            rem = sell_qty
            while rem > 0:
                if buy_rem == 0:
                    if not q:
                        break
                    buy_price, buy_rem = q.popleft()
                take = min(rem, buy_rem)
                cost += buy_price * take
                buy_rem -= take
                matched += take
                rem -= take
            if matched <= 0:
                continue
            buy_avg = cost / matched
            pnl_pct = round((sell_price - buy_avg) / buy_avg * 100, 4) if buy_avg else 0.0
            result.append({
                "stock_id": code,
                "quantity": matched,
                "buy_avg": round(buy_avg, 3),
                "sell_avg": round(sell_price, 3),
                "pnl_pct": pnl_pct,
                "sell_time": sell_time,
            })
        # 未配對的買進剩餘量放回（仍持倉）
        if buy_rem > 0:
            q.appendleft((buy_price, buy_rem))
    return result


def cancel_sent_orders(api: sj.Shioaji) -> dict:
    """
    取消今日所有 SENT（掛單未成交）的買單和賣單。
    - 買單：訊號時效過，取消後若訊號還在本分鐘重新以新價掛
    - 賣單：取消後由 BrokerClient 或 SL/TP 重掛新價
    回傳:
      {"buy": [(sid, cost), ...], "sell": [sid, ...]}
      cost = 委託價 × 委託張數 × 1000（未成交金額，供還原 _used_quota 用）
    """
    from datetime import datetime, timezone, timedelta
    today = datetime.now(timezone(timedelta(hours=8))).date()  # 台灣時間
    api.update_status()
    cancelled: dict = {"buy": [], "sell": []}
    for t in api.list_trades():
        order_dt = t.status.order_datetime
        order_date = getattr(order_dt, "date", lambda: order_dt)()
        if str(order_date) != str(today):
            continue
        if normalize_status(t.status.status) != "SENT":
            continue
        sid = t.contract.code
        direction = "buy" if "buy" in str(t.order.action).lower() else "sell"
        try:
            print(f"[CANCEL] {sid} {direction} order_id={t.order.id} price={t.order.price}", flush=True)
            result = api.cancel_order(t)
            raw = str(result.status.status)
            confirmed = raw in ("OrderStatus.Cancelled", "OrderStatus.Failed", "OrderStatus.Inactive")
            print(f"[CANCEL] {sid} {direction} → {'✓' if confirmed else '✗ 未確認'} {raw}", flush=True)
            if confirmed:
                if direction == "buy":
                    cost = float(t.order.price) * int(t.order.quantity) * 1000
                    cancelled["buy"].append((sid, cost))
                else:
                    cancelled["sell"].append(sid)
        except Exception as e:
            print(f"[CANCEL] {sid} {direction} 取消失敗: {e}", flush=True)
    return cancelled


def list_orders(api: sj.Shioaji) -> list:
    api.update_status()
    trades = sorted(api.list_trades(), key=lambda t: t.status.order_datetime)
    return [
        {
            "order_id": t.order.id,
            "time": t.status.order_datetime,
            "stock_id": t.contract.code,
            "action": t.order.action,
            "price": t.order.price,
            "quantity": t.order.quantity,
            "filled": t.status.deal_quantity,
            "status": t.status.status,
            "msg": t.status.msg,
        }
        for t in trades
    ]


def list_orders_today(api: sj.Shioaji) -> list[dict]:
    """今日委託/成交狀態，供 dashboard 重啟後重建記憶體快取。"""
    from datetime import datetime, timezone, timedelta
    today = datetime.now(timezone(timedelta(hours=8))).date()  # 台灣時間
    api.update_status()
    trades = sorted(api.list_trades(), key=lambda t: t.status.order_datetime)

    result = []
    for t in trades:
        order_dt = t.status.order_datetime
        order_date = getattr(order_dt, "date", lambda: order_dt)()
        if order_date != today:
            continue

        action = str(t.order.action)
        direction = "buy" if "buy" in action.lower() else "sell"
        result.append({
            "order_id": t.order.id,
            "time": order_dt,
            "stock_id": t.contract.code,
            "direction": direction,
            "action": action,
            "price": float(t.order.price),
            "quantity": int(t.order.quantity),
            "filled": int(t.status.deal_quantity),
            "status": normalize_status(t.status.status),
            "raw_status": str(t.status.status),
            "msg": t.status.msg,
        })
    return result


def get_settlements(api: sj.Shioaji) -> list:
    """交割明細（對帳單）"""
    print("交割明細（對帳單）")
    return [
        {
            "date": s.t_date,
            "stock_id": s.code,
            "action": s.action,
            "quantity": s.quantity,
            "price": s.price,
            "amount": s.amount,
        }
        for s in api.settlements(api.stock_account)
    ]


if __name__ == "__main__":
    # api = test()
    api = login()
    # print(open(api, "0050", 1))
    # for i in list_orders(api):
    # print(i)

    # for s in get_settlements(api):
    # print(s)

    api.logout()
