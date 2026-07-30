"""
本機執行用：依序跑完 m1_data_loader.update_m1() → day_data_loader.update_day()
→ build_m3_m5_rolling.build() → build_m3_m5_std.build()。

前兩支要打 API（Fugle/富邦），後兩支只讀本機 db/m1/ 做聚合，不吃 API，
所以接在 m1 下載完之後、用 --incremental 只重建有更新的月份即可。
各支腳本仍可獨立執行，這支只是串起來方便本機一次跑完。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.m1_data_loader import update_m1
from data.day_data_loader import update_day
from data.build_m3_m5_rolling import build as build_m3_m5_rolling
from data.build_m3_m5_std import build as build_m3_m5_std

if __name__ == "__main__":
    print("=== 下載 1 分鐘K ===")
    update_m1()
    print("=== 下載日K ===")
    update_day()
    print("=== 建立 m3/m5 rolling ===")
    build_m3_m5_rolling(incremental=True)
    print("=== 建立 m3/m5 std ===")
    build_m3_m5_std(incremental=True)
    print("全部完成")
