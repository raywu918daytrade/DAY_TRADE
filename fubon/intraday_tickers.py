"""
診斷用：呼叫 fubon/trade_api.py::intraday_tickers()，把 TWSE+TPEx（type=EQUITY）
原始回傳結果存到 db/tickers/tickers.parquet，方便直接看資料長什麼樣子，
再決定 fubon/subscribe_list.py 的過濾規則要怎麼寫。

使用方式：
    python -m fubon.intraday_tickers
"""
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_OUT_PATH = _ROOT / "db/tickers/tickers.parquet"


def fetch_tickers() -> pd.DataFrame:
    from fubon import trade_api

    sdk, _ = trade_api.login()
    try:
        trade_api.init_market_data(sdk)
        rows = []
        for exchange in ("TWSE", "TPEx"):
            for item in trade_api.intraday_tickers(sdk, exchange, type_="EQUITY"):
                if not str(item.get("industry", "")).isdigit():
                    continue
                row = dict(item)
                row["exchange"] = exchange
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        trade_api.logout(sdk)


if __name__ == "__main__":
    df = fetch_tickers()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)
    print(f"儲存完成：{len(df)} 筆 → {_OUT_PATH}")
