"""
本機執行用：依序跑完 m1_data_loader.update_m1() → day_data_loader.update_day()
→ fubon.tick_api.update_tick_today() → build_m3_m5_rolling.build() →
build_m3_m5_std.build()。

前三支要打 API（Fugle/富邦），最後兩支只讀本機 db/m1/ 做聚合，不吃 API，
所以接在下載完之後、用 --incremental 只重建有更新的月份即可。
tick更新失敗不中斷整支流程（m3/m5 rolling重建只依賴db/m1，跟tick無關，
不應該被tick的失敗連帶卡住）。各支腳本仍可獨立執行，這支只是串起來方便
本機一次跑完。

⚠️ 執行前檢查 finmind 的 backfill_tick 有沒有還在跑同一個月份（兩邊都會寫
db/tick/{year}_{month}.parquet，同時處理同一個月可能互相覆蓋遺失資料，
見 finmind/backfill_tick_history.py 的相關說明）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.m1_data_loader import update_m1
from data.day_data_loader import update_day
from data.build_m3_m5_rolling import build as build_m3_m5_rolling
from data.build_m3_m5_std import build as build_m3_m5_std
from fubon.tick_api import update_tick_today

if __name__ == "__main__":
    print("=== 下載 1 分鐘K ===")
    update_m1()
    print("=== 下載日K ===")
    update_day()
    print("=== 更新今天的tick ===")
    try:
        update_tick_today()
    except Exception as e:
        print(f"⚠️ tick更新失敗，跳過繼續（不影響後面的m3/m5重建）：{e}")
    print("=== 建立 m3/m5 rolling ===")
    build_m3_m5_rolling(incremental=True)
    print("=== 建立 m3/m5 std ===")
    build_m3_m5_std(incremental=True)
    print("全部完成")
