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
        ),
    )
    return trade


def open(api: sj.Shioaji, stock_id: str, quantity: int, action: sj.Action = sj.Action.Buy):
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


def get_positions(api: sj.Shioaji) -> dict[str, dict]:
    """回傳 {stock_id: {"quantity": lots, "avg_price": fill_price}} 目前持倉（永豐實際成交均價）"""
    return {
        p.code: {"quantity": p.quantity, "avg_price": float(p.price)}
        for p in api.list_positions(api.stock_account)
    }


def get_closed_today(api: sj.Shioaji) -> list[dict]:
    """
    從今日委託記錄重建已平倉清單。
    當沖：同一支股票有買進成交 + 賣出成交 → 視為已平倉。
    回傳 [{"stock_id", "buy_avg", "sell_avg", "quantity", "pnl_pct"}, ...]
    """
    from datetime import date
    today = date.today()
    api.update_status()
    trades = api.list_trades()

    buys: dict[str, list] = {}
    sells: dict[str, list] = {}
    for t in trades:
        # 過濾今日
        order_date = getattr(t.status.order_datetime, "date", lambda: t.status.order_datetime)()
        if order_date != today:
            continue
        filled = t.status.deal_quantity
        if filled <= 0:
            continue
        code = t.contract.code
        price = float(t.order.price)
        action = str(t.order.action).lower()
        if "buy" in action:
            buys.setdefault(code, []).append((price, filled))
        else:
            sells.setdefault(code, []).append((price, filled))

    result = []
    for code in sells:
        if code not in buys:
            continue
        buy_lots = sum(q for _, q in buys[code])
        sell_lots = sum(q for _, q in sells[code])
        qty = min(buy_lots, sell_lots)
        buy_avg = sum(p * q for p, q in buys[code]) / sum(q for _, q in buys[code])
        sell_avg = sum(p * q for p, q in sells[code]) / sum(q for _, q in sells[code])
        pnl_pct = round((sell_avg - buy_avg) / buy_avg * 100, 4)
        result.append({
            "stock_id": code,
            "quantity": qty,
            "buy_avg": round(buy_avg, 3),
            "sell_avg": round(sell_avg, 3),
            "pnl_pct": pnl_pct,
        })
    return result


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
