"""
即時交易進入點（Render background worker）

流程：
    1. 啟動時載入模型 + 歷史日K
    2. 連上 Fugle WebSocket，訂閱所有股票分K
    3. 每分K收盤後觸發 on_minute：
       - 只處理 SESSION_START ~ SESSION_END 的 bar
       - 即時計算特徵 + 模型推論
       - 印出達門檻的買入訊號
"""

import pandas as pd

from date_trade_model import SESSION_END, SESSION_START, load_model, predict_live
from tay_trade.m1_websocket import M1Collector
from tay_trade.query import load_day

THRESHOLD = 0.55

print("載入模型...")
model = load_model()

print("載入日K...")
_day = load_day()

print(f"就緒，等待盤中訊號（門檻={THRESHOLD}）...")


def on_minute(minute_str: str, df: pd.DataFrame):
    dt = pd.Timestamp(minute_str)
    h, m = dt.hour, dt.minute

    if not (SESSION_START <= (h, m) <= SESSION_END):
        return

    signals = predict_live(minute_str, _day, model=model, threshold=THRESHOLD)

    if signals:
        print(f"\n[{minute_str}] 訊號（{len(signals)} 支）:")
        for s in signals:
            print(f"  {s['stock_id']:8s}  機率={s['proba']:.3f}  價={s['price']:.2f}")
    else:
        print(f"[{minute_str}] 無訊號")


if __name__ == "__main__":
    collector = M1Collector(on_minute=on_minute)
    try:
        collector.start()
    except KeyboardInterrupt:
        collector.stop()
        print("已停止")
