"""
本機執行用：依序跑完 m1_data_loader.update_m1() → day_data_loader.update_day()
→ day_data_loader.update_adjustment_day() → finmind.tick_universe.build_tick_universe()
→ fubon.tick_api.update_tick_today() → build_volume_profile.build() → build_poc.build() →
build_tick_adjust_factor.build() → build_adjustment_factor.build() →
build_m3_m5_rolling.build() → build_m3_m5_std.build()。

前面幾支要打 API（Fugle/富邦），後面幾支只讀本機資料做聚合，不吃 API，所以
接在下載完之後、用 --incremental 只重建有更新的月份即可。build_poc 依賴
build_volume_profile 的輸出（db/volume_profile/），所以順序不能對調。

2026-08-19加 build_tick_universe()：候選股母體（db/tickers/tick_universe.
parquet）原本只靠 fubon/subscribe_list.py 在 live_trader.py 開機/每日06:00
時自我修復重建，本機/GHA各自獨立觸發、也沒有同步到HF——若某個環境（例如
沒有本機既有db/的雲端部署）開機時發現檔案不存在，會觸發完整重建（實測
~6分鐘、打幾百次富邦API查當沖資格），拖慢開機速度。改成併入這支本機/GHA
共用的每日排程統一重建一次，重建結果正常同步到HF（見 push_db_to_hf.py 的
_IGNORE_PATTERNS 已拿掉這份檔案的排除規則），其他環境開機時只要下載現成
檔案，subscribe_list.py 會偵測到 verify_date 不是今天，走既有的「輕量
重新驗證」路徑（不是完整重建），不用改 subscribe_list.py 任何程式碼。

2026-08-14拿掉 build_breakout_retest_day／build_breakout_retest_trigger：
這兩支是給 strategy/breakout_retest_ml 用的候選事件表，但這個策略目前
沒有排進 .env 的 STRATEGY_MODULES（沒有上線交易），每天卻還是要花約
5分鐘物化它的資料，使用者確認不需要就先移除，省下這段時間。兩支腳本
本身沒刪，之後真的要恢復這個策略、需要重新產生資料時，可以手動執行
`python -m data.build_breakout_retest_day`／
`python -m data.build_breakout_retest_trigger`補回來。

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

2026-08-15加：週末（六、日）直接跳過整個pipeline，不執行。原因：m1/d1
的gap判斷本來就有保護（週末查「今天」一律回傳「尚無今日資料」，不會
誤標記flag，見 data/day_data_loader.py::_expected_prior_trading_day()），
但 fubon/tick_api.py::fetch_trades_today() 底層的富邦 intraday.trades
端點不是用日期查詢、而是回傳「最近一個session」的成交明細——實測發現
週六執行時，這支端點把上週五的成交資料又完整回傳一次（1878支全部
「有資料」，不是空的），導致tick階段白白重跑一次上週五早就抓過的資料
（實測多花約32分鐘、上千次API請求），雖然merge+dedupe邏輯正確、沒有
造成資料損毀，但完全是浪費。刻意只擋六日，不做完整假日行事曆（跟
`_expected_prior_trading_day()` 的設計理由一致：國定假日隔天最多只是
沒享受到快路徑，這裡最多只是沒省到週末的浪費，都是安全的失敗方式）。
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.m1_data_loader import update_m1
from data.day_data_loader import update_day, update_adjustment_day
from data.build_m3_m5_rolling import build as build_m3_m5_rolling
from data.build_m3_m5_std import build as build_m3_m5_std
from data.build_volume_profile import build as build_volume_profile
from data.build_poc import build as build_poc
from data.build_tick_adjust_factor import build as build_tick_adjust_factor
from data.build_adjustment_factor import build as build_adjustment_factor
from fubon.tick_api import update_tick_today
from finmind.tick_universe import build_tick_universe, _universe_file_path
from finmind.m1_api import _atomic_to_parquet

_TW = timezone(timedelta(hours=8))

if __name__ == "__main__":
    if datetime.now(_TW).weekday() >= 5:  # 5=Saturday, 6=Sunday
        print("今天是週末（台北時間），非交易日，跳過整個pipeline")
        sys.exit(0)
    print("=== 下載 1 分鐘K ===")
    update_m1()
    print("=== 下載日K（原始，db/d1）===")
    update_day()
    print("=== 下載日K（完整還原，db/adjustment_day，pattern專用）===")
    update_adjustment_day()
    print("=== 建立候選股母體（tick_universe）===")
    try:
        _universe = build_tick_universe()
        _atomic_to_parquet(_universe, _universe_file_path(), index=False, compression="zstd")
        print(f"✓ tick_universe：已寫入 {len(_universe)} 支 → {_universe_file_path()}")
    except Exception as e:
        print(f"⚠️ tick_universe 重建失敗，跳過（不影響其他步驟）：{e}")
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
    print("=== 建立 m3/m5 rolling ===")
    build_m3_m5_rolling(incremental=True)
    print("=== 建立 m3/m5 std ===")
    build_m3_m5_std(incremental=True)
    print("全部完成")
