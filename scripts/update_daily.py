"""
本機執行用：依序跑完 m1_data_loader.update_m1() → day_data_loader.update_day()
→ day_data_loader.update_adjustment_day() → fubon.tick_api.update_tick_today() →
build_volume_profile.build() → build_poc.build() → build_tick_adjust_factor.build() →
build_adjustment_factor.build() → build_breakout_retest_day /
build_breakout_retest_trigger → build_m3_m5_rolling.build() →
build_m3_m5_std.build()。

前四支要打 API（Fugle/富邦），後面幾支只讀本機資料做聚合，不吃 API，所以
接在下載完之後、用 --incremental 只重建有更新的月份即可。build_poc 依賴
build_volume_profile 的輸出（db/volume_profile/），所以順序不能對調。
build_breakout_retest_day 依賴日K+POC；build_breakout_retest_trigger 再依賴
M1+tick，tick 更新失敗時跳過 trigger 物化。

update_day() 下載原始日K（db/d1/，2026-08-03 取代原本的 db/fugle_day），
update_adjustment_day() 下載完整還原日K（db/adjustment_day/，只給 pattern 系列
用，見 data/adjustment_query.py 檔頭說明）——**這兩支要依序執行、不能同時跑**，
兩者都用 Fugle雙帳號+富邦三路併發下載，同時跑會互搶同一組 rate limit。
build_tick_adjust_factor.build() 直接從 db/d1 的完整歷史偵測拆股/合股、重算
db/tick_adjust_factor（系統預設基準）；build_adjustment_factor.build() 逐日比對
db/adjustment_day vs db/d1，重算 db/adjustment_factor（pattern 專用完整還原
基準）。兩張 factor 表互不依賴，誰先誰後不影響結果。

⚠️ 執行前檢查 finmind 的 backfill_tick 有沒有還在跑同一個月份（兩邊都會寫
db/tick/{year}_{month}.parquet，同時處理同一個月可能互相覆蓋遺失資料，
見 finmind/backfill_tick_history.py 的相關說明）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.m1_data_loader import update_m1
from data.day_data_loader import update_day, update_adjustment_day
from data.build_m3_m5_rolling import build as build_m3_m5_rolling
from data.build_m3_m5_std import build as build_m3_m5_std
from data.build_volume_profile import build as build_volume_profile
from data.build_poc import build as build_poc
from data.build_tick_adjust_factor import build as build_tick_adjust_factor
from data.build_adjustment_factor import build as build_adjustment_factor
from data.build_breakout_retest_day import build as build_breakout_retest_day
from data.build_breakout_retest_trigger import build as build_breakout_retest_trigger
from fubon.tick_api import update_tick_today

if __name__ == "__main__":
    print("=== 下載 1 分鐘K ===")
    update_m1()
    print("=== 下載日K（原始，db/d1）===")
    update_day()
    print("=== 下載日K（完整還原，db/adjustment_day，pattern專用）===")
    update_adjustment_day()
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
        print("=== 建立除權息調整係數（系統預設，只還原拆股/合股）===")
        build_tick_adjust_factor()
        print("=== 建立除權息調整係數（pattern專用，完整還原）===")
        build_adjustment_factor()
    else:
        print("=== 跳過 Volume Profile / POC / 除權息調整係數（tick 未更新）===")
    print("=== 物化 breakout_retest 日K候選 ===")
    build_breakout_retest_day(force=False)
    if tick_ok:
        print("=== 物化 breakout_retest 盤中觸發快照 ===")
        build_breakout_retest_trigger(force=False)
    else:
        print("=== 跳過 breakout_retest_trigger（tick 未更新）===")
    print("=== 建立 m3/m5 rolling ===")
    build_m3_m5_rolling(incremental=True)
    print("=== 建立 m3/m5 std ===")
    build_m3_m5_std(incremental=True)
    print("全部完成")
