"""
本機執行用：依序跑完 m1_data_loader.update_m1() → day_data_loader.update_day()
→ fubon.tick_api.update_tick_today() → build_volume_profile.build() →
build_poc.build() → build_tick_adjust_factor.build() → build_m3_m5_rolling.build()
→ build_m3_m5_std.build()。

前三支要打 API（Fugle/富邦），後面幾支只讀本機資料做聚合，不吃 API，所以
接在下載完之後、用 --incremental 只重建有更新的月份即可。build_poc 依賴
build_volume_profile 的輸出（db/volume_profile/），所以順序不能對調。
build_tick_adjust_factor 依賴 db/tick 與 db/fugle_day（不依賴 build_volume_profile/
build_poc 的輸出），放在後面純粹是接續 tick 這條線、順序上沒有硬性依賴。
tick更新失敗不中斷整支流程，但 volume_profile/poc/tick_adjust_factor 都是從
tick（或搭配day K）算出來的，tick沒更新就沒新資料可增量，所以一併跳過
（m3/m5 rolling重建只依賴db/m1，跟tick無關，不受影響）。各支腳本仍可獨立
執行，這支只是串起來方便本機一次跑完。

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
from data.build_volume_profile import build as build_volume_profile
from data.build_poc import build as build_poc
from data.build_tick_adjust_factor import build as build_tick_adjust_factor
from fubon.tick_api import update_tick_today

if __name__ == "__main__":
    print("=== 下載 1 分鐘K ===")
    update_m1()
    print("=== 下載日K ===")
    update_day()
    print("=== 更新今天的tick ===")
    tick_ok = True
    try:
        update_tick_today()
    except Exception as e:
        tick_ok = False
        print(f"⚠️ tick更新失敗，跳過繼續（不影響後面的m3/m5重建）：{e}")
    if tick_ok:
        print("=== 建立 Volume Profile ===")
        build_volume_profile(incremental=True)
        print("=== 建立每日 POC ===")
        build_poc(incremental=True)
        print("=== 建立除權息調整係數 ===")
        build_tick_adjust_factor(incremental=True)
    else:
        print("=== 跳過 Volume Profile / POC / 除權息調整係數（tick 未更新）===")
    print("=== 建立 m3/m5 rolling ===")
    build_m3_m5_rolling(incremental=True)
    print("=== 建立 m3/m5 std ===")
    build_m3_m5_std(incremental=True)
    print("全部完成")
