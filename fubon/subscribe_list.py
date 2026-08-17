"""
富邦當沖候選股清單。

2026-08-01改：候選股母體固定成 db/tickers/tick_universe.parquet（見
finmind/tick_universe.py），不再即時打富邦API抓全市場~2700支、也不再用20日
均量動態排序截斷（母體規模遠低於WebSocket上限1000支，不需要截斷）。訓練
（data/m1_data_loader.py、data/day_data_loader.py）跟推論/交易統一用同一批
固定母體，減少訓練雜訊，也大幅減少每天開盤前的API用量/等待時間。

2026-08-17改版：母體規模上限從400調成800、選股邏輯也整個換掉（改成
db/tickers/stock_universe_2000.parquet全市場股票，篩「均量>1000張且
ATR(14)%>1%」，通過的再依富邦當沖資格分「能多空」「只能做多」兩級，能多空
優先、依均量排序取到上限，見 finmind/tick_universe.py 的說明）。實際跑出來
的數字是「篩選出來多少算多少」，不強制湊滿800，量能+ATR這兩個門檻本身就會
限制住上限，不是排名/當沖資格不夠。

2026-08-19再改（重要）：拿掉原本另外存一份的 db/fubon_subscribe/
subscribe_list.parquet，改成**直接讀寫 db/tickers/tick_universe.parquet
本身**——原本兩個檔案內容高度重疊（同一批股票代號，只差 connection_id/
驗證日期兩欄），使用者要求合併成一份，不要同樣的東西分兩邊存。做法：

    1. 讀 db/tickers/tick_universe.parquet 現有的候選母體（avg_volume/
       atr_pct/day_trade_tier/rank/name 這些欄位維持原樣，不受這裡影響——
       母體要變動只能靠手動重跑 `python -m finmind.tick_universe` 整套
       重新篩選，這裡完全不碰）
    2. canBuyDayTrade 逐支確認（_filter_day_tradable()）——這個還是每天做，
       確認這批候選股裡有沒有股票今天被停資/注意處置、不能當沖（跟
       tick_universe.py 建置母體時的當沖分級是不同時間點的驗證，目的
       不同、頻率也不同：母體是手動重建才變、這裡是每天驗證一次）
    3. 把這次驗證結果寫回同一個檔案的 daytrade_ok（今天能不能當沖）、
       connection_id（WebSocket分組，只有daytrade_ok=True才有值）、
       verify_date（驗證日期）三欄——**不刪除任何一列**，今天不能當沖的
       股票還留在檔案裡（daytrade_ok=False），不會整列消失，明天重新
       驗證通過的話一樣能自動恢復，不用重跑整套均量/ATR篩選。

分連線：切成每組 ≤200 支、最多 5 組（對應富邦 WebSocket rate limit：
200 檔/連線、5 條連線 → 上限 1000 檔），依 rank（均量排序）分組，只對
今天 daytrade_ok=True 的子集分組。

舊的動態選股邏輯（_fubon_normal_tickers()/ranked_candidates()/
all_normal_stocks()）保留在檔案裡沒有刪除，只是目前的日常流程不再呼叫，
之後如果要重新擴大母體/改回動態選股可以直接復用。

這份清單是「唯一」的當沖候選股來源：
    - fubon/marketdata_ws.py 用分組後的 batches 決定 WebSocket 訂閱誰
    - main/premarket.py::refresh_tickers() 直接呼叫 build_and_save_subscribe_list()
      當作 state.day_trade_stocks / state.tickers 的來源
    - pattern/pattern_api.py::_load_daytrade_list() 讀 daytrade_ok=True 的列，
      供前端股票清單欄「當沖候選」選項用

使用方式：
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會自動呼叫
    build_and_save_subscribe_list()，一般不用手動跑。要單獨測試/預覽：
        python -m fubon.subscribe_list

    其他地方要讀已存好的清單（不重算）：
        from fubon.subscribe_list import load_candidates, load_subscribe_batches
        df = load_candidates()          # 今天 daytrade_ok=True 的完整 DataFrame（含 name）
        batches = load_subscribe_batches()  # 分組後的 stock_id list，給 WebSocket 用
"""
import os
import shutil
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fubon.config import MAX_CONNECTIONS, MAX_PER_CONNECTION

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_TW = timezone(timedelta(hours=8))

# fubon_neo SDK 底層沒有設定任何 timeout（翻過套件原始碼確認過），網路/伺服器
# 卡住的話 REST 呼叫可能無限期不回應，2026-07-25 討論加這層保護。
_TICKER_CHECK_TIMEOUT = float(os.environ.get("FUBON_TICKER_CHECK_TIMEOUT", "8"))


def _call_with_timeout(fn, timeout: float, *args):
    """幫沒有內建 timeout 的呼叫包一個逾時保護。用 daemon thread（不是
    concurrent.futures.ThreadPoolExecutor）——ThreadPoolExecutor 預設會在
    process 結束時等所有丟進去的任務 join 完，真的卡住不回應的呼叫會讓
    程式沒辦法乾淨結束；daemon thread 不會擋 process 退出，逾時就直接放棄
    等待，底下那條線程若之後真的回應了，結果被丟棄，不影響呼叫端。
    每次呼叫都開一支新的 thread（不是共用 pool），確保某支股票卡住不會
    連帶拖慢後面要查的股票（各自獨立逾時，不會排隊等前一支放棄）。
    """
    result: dict = {}

    def _target():
        try:
            result["value"] = fn(*args)
        except Exception as e:  # noqa: BLE001
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"逾時 {timeout} 秒未回應")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _fubon_normal_tickers() -> dict[str, str]:
    """股票清單（垃圾代號、債券ETF已經在 fubon/intraday_tickers.py::update_tickers()
    這個唯一過濾源頭濾掉了），回傳 {stock_id: name}。

    不額外用代號碼數濾槓桿/反向/主動型ETF（曾經這樣做過，2026-07-14 發現這是錯的：
    台股ETF代號是「00+3碼」＝5碼才是現行規則，00878/00919這類完全正常、成交量
    很大的高股息ETF也是5碼，用碼數過濾會連這些一起誤殺）。是否進候選股完全交給
    isNormal=true（intraday_tickers.py 呼叫 API 時已經濾過）+ ranked_candidates()
    的成交量排序決定，槓桿/反向ETF成交量高就會排前面，不特別排除。"""
    from fubon.intraday_tickers import update_tickers

    df = update_tickers()
    if df.empty:
        return {}
    return dict(zip(df["stock_id"], df["name"]))


def all_normal_stocks() -> list[str]:
    """完整股票母體（isNormal=true、排除債券ETF，不做均量排序/上限）。
    2026-08-01改：日常流程（data/m1_data_loader.py、data/day_data_loader.py、
    fubon/subscribe_list.py 自己）都已經改成固定讀 tick_universe（見
    build_and_save_subscribe_list()），不再呼叫這支——保留函式本身，之後
    如果要重新擴大母體/改回動態選股可以直接復用，不用重寫。"""
    return list(_fubon_normal_tickers().keys())


def ranked_candidates(names: dict[str, str]) -> list[str]:
    """依 20日均量排序（高→低）的候選股，最多 MAX_SUBSCRIPTIONS 支（.env，
    目前設為 1000，剛好對應 5 條連線 × 200 檔）。2026-08-01改：日常流程不再
    呼叫這支，說明同 all_normal_stocks()。"""
    from data.data_manager import _volume_filter
    from data.query import load_day

    day = load_day()
    return _volume_filter(set(names.keys()), day[["stock_id", "date", "volume"]])


def _filter_day_tradable(stock_ids: list[str]) -> list[str]:
    """用 intraday/ticker/{symbol}（單支查詢，見 fubon/intraday_ticker.py 的
    診斷測試）逐支確認 canDayTrade/canBuyDayTrade 皆為 true，比 isNormal 更直接
    反映「能不能當沖」，2026-07-14 實測連垃圾代碼（industry非數字那批）也會被
    這兩個欄位抓到 canDayTrade=false。

    放在 ranked_candidates() 之後（均量排序+截斷到 MAX_SUBSCRIPTIONS 之後）才做，
    只查最後入選的候選股，不用對全市場 ~2700 支都查一次（300次/分鐘的話要
    9分鐘，候選股通常只有 1000 支內，約 4 分鐘內）。

    每支呼叫都包 _call_with_timeout()（預設8秒，.env 的 FUBON_TICKER_CHECK_TIMEOUT
    可調）——fubon_neo SDK 本身沒有 timeout，網路/伺服器卡住的話單一支股票可能
    無限期不回應，2026-07-25 討論加這層保護，逾時當作查詢失敗、直接跳過那支，
    不會拖住後面的股票（見 _call_with_timeout() 的說明）。

    _FORCE_INCLUDE（例如0050）的股票直接跳過查詢、無條件視為可以當沖
    （2026-07-25討論）：這些股票不管 canBuyDayTrade/canDayTrade 回傳什麼都
    一定要留著（見 _FORCE_INCLUDE 的說明），與其查完API結果又忽略，不如
    根本不查，省一次API呼叫跟0.25秒等待，語意也更明確——不是「查了但結果
    被覆蓋」，是「這幾支從一開始就不受這個檢查限制」。"""
    from fubon import fubon_api as trade_api

    sdk, _ = trade_api.login()
    tradable = list(_FORCE_INCLUDE)
    try:
        trade_api.init_market_data(sdk)
        for sid in stock_ids:
            if sid in _FORCE_INCLUDE:
                continue
            try:
                info = _call_with_timeout(trade_api.intraday_ticker, _TICKER_CHECK_TIMEOUT, sdk, sid)
                # canDayTrade 先註解掉，不要求（2026-07-22：0050 這種正常標的
                # 也會 canDayTrade=False/canBuyDayTrade=True，兩個都要求會把
                # 它濾掉，先只看 canBuyDayTrade；不要刪，之後確認 canDayTrade
                # 的正確用法再決定要不要恢復）。
                if info.get("canBuyDayTrade"):  # and info.get("canDayTrade")
                    tradable.append(sid)
            except TimeoutError as e:
                print(f"  警告：{sid} intraday_ticker {e}，略過", flush=True)
            except Exception as e:
                print(f"  警告：{sid} intraday_ticker 查詢失敗，略過: {e}", flush=True)
            time.sleep(0.25)  # 300次/分鐘上限，留緩衝
    finally:
        trade_api.logout(sdk)
    return tradable


# 大盤代理（0050）— strategy/rally/features.py 的市場代理特徵（idx_ret_1/idx_atr/
# idx_up 等）用這一檔的即時分K廣播給所有候選股，是必要依賴，不是要拿來當沖
# 交易，只是要抓報價。2026-07-22 發現：0050 canDayTrade=False 時會被
# _filter_day_tradable() 濾掉，導致 rally 全部候選股的大盤特徵在 merge 時對不到
# 這一分鐘的 0050 row、整批變 NaN，被 dropna 篩光，rally 當下完全沒有任何推論
# 結果（而且不會報錯，只會安靜地一直印 0 支）。所以這裡無條件強制加回去，不受
# 當沖資格檢查限制。
_FORCE_INCLUDE = ["0050"]


def _assign_connections(df: pd.DataFrame) -> pd.DataFrame:
    """依 rank（均量排序）切成 ≤MAX_PER_CONNECTION 支一組、最多
    MAX_CONNECTIONS 組（富邦 WebSocket 連線本身的 rate limit，定義在
    fubon/config.py），回傳多一欄 connection_id 的 DataFrame。超過
    MAX_CONNECTIONS×MAX_PER_CONNECTION 上限的股票 connection_id 是
    pd.NA（daytrade_ok 還是True，只是這次沒被排進WebSocket訂閱分組）——
    目前母體規模遠低於這個上限（1000支），正常不會發生，保留這段邊界
    處理避免母體之後成長超過上限時默默出錯。"""
    df = df.sort_values("rank").reset_index(drop=True)
    conn = pd.Series(df.index // MAX_PER_CONNECTION, dtype="Int64")
    df = df.assign(connection_id=conn.where(conn < MAX_CONNECTIONS))
    return df


def _try_download_from_hf(path: Path) -> None:
    """一律先試著把 HF Hub 上最新的 tick_universe.parquet 下載蓋過本機
    再繼續（2026-08-19：候選母體現在由 scripts/update_daily.py 統一每天
    重建一次、同步到HF，見該檔案說明），不管本機原本有沒有這個檔案都下載，
    確保用到的是最新版本。下載失敗（沒設定HF_REPO_ID、網路問題、或HF上
    還沒有這個檔案）不當成錯誤，靜默放棄，讓呼叫端 fallback 到本機既有
    檔案或整包重建（見 _load_or_rebuild_universe_pool()）。"""
    repo_id = os.environ.get("HF_REPO_ID", "")
    if not repo_id:
        return
    try:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or None
        downloaded = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", token=token,
            filename="db/tickers/tick_universe.parquet",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        shutil.copy(downloaded, tmp)
        os.replace(tmp, path)
        print(f"  已從HF Hub下載最新 tick_universe.parquet → {path}", flush=True)
    except Exception as e:
        print(f"  HF下載失敗（{e}），改用本機現有檔案或重建", flush=True)


def _load_or_rebuild_universe_pool(path: Path) -> pd.DataFrame:
    """讀 db/tickers/tick_universe.parquet；一律先試著從HF下載最新版本
    （見 _try_download_from_hf()），下載後（或下載失敗、沿用本機原有檔案）
    檔案仍不存在或是空的，才自動呼叫 finmind.tick_universe.build_tick_universe()
    整包重建（見 build_and_save_subscribe_list() 的說明）。"""
    _try_download_from_hf(path)
    if path.exists():
        pool = pd.read_parquet(path)
        if not pool.empty:
            return pool
        print(f"  {path} 是空的，", end="", flush=True)
    else:
        print(f"  找不到 {path}，", end="", flush=True)

    print("自動重新執行 finmind.tick_universe 建立候選母體（會呼叫大量富邦API，需要幾分鐘）...", flush=True)
    from finmind.m1_api import _atomic_to_parquet
    from finmind.tick_universe import build_tick_universe

    pool = build_tick_universe()
    if pool.empty:
        return pool
    _atomic_to_parquet(pool, path, index=False, compression="zstd")
    print(f"候選母體重建完成：{len(pool)} 支 → {path}", flush=True)
    return pool


def build_and_save_subscribe_list(force: bool = False) -> pd.DataFrame:
    """更新候選股清單「今天能不能當沖」的驗證結果，直接寫回
    db/tickers/tick_universe.parquet（見本檔案頭的2026-08-19說明，不再
    另外存一份 db/fubon_subscribe/subscribe_list.parquet）。
    main/premarket.py::refresh_tickers() 開機、每天 06:00 都會呼叫這個，
    不用另外排程。

    只更新 daytrade_ok/connection_id/verify_date 三欄，候選母體本身
    （avg_volume/atr_pct/day_trade_tier/rank/name）完全不動——母體要
    變動只能靠手動重跑 `python -m finmind.tick_universe`。

    force=False（預設，2026-07-25新增）：如果 verify_date 已經是「今天」，
    直接讀取回傳，不重新跑一次 _filter_day_tradable() 那段3~4分鐘、逐支查
    富邦API的流程——同一天內 main/live_trader.py 重啟多次（開機時的
    _startup()、_daily_refresh() 的立即補載都會呼叫這支函式），沒必要每次
    都重新查一次同一天的候選資格。force=True 才強制整個重跑（例如手動用
    `python -m fubon.subscribe_list` 想確認今天最新結果、或懷疑舊資料有
    問題時）。

    回傳值：只回傳今天 daytrade_ok=True 的子集（維持跟舊版
    subscribe_list.parquet 相同的「已經是篩選過的當沖清單」語意，呼叫端
    ——main/premarket.py::refresh_tickers()——不用另外再篩一次）。

    2026-08-19加：候選母體檔案不存在或是空的（例如意外被刪掉、或
    Dropbox同步把檔案改名的那類意外——2026-08-19實際發生過一次），
    自動呼叫 finmind.tick_universe.build_tick_universe() 重新跑一次完整的
    均量+ATR+當沖分級篩選並存檔，不用人工先手動執行
    `python -m finmind.tick_universe` 才能讓系統恢復運作——代價是這種情況
    下開機/背景重試會多花幾分鐘（逐支查富邦API），但比起卡住不動、要等
    人發現才手動介入好。跟「今天已驗證過就不重算」的快取邏輯不衝突：
    剛重建出來的母體還沒有 verify_date，一定會往下走真正驗證那段，正常
    情況（檔案好好的）完全不受影響。"""
    from finmind.tick_universe import _universe_file_path

    path = _universe_file_path()
    pool = _load_or_rebuild_universe_pool(path)
    if pool.empty:
        print("  警告：候選母體重建後仍是空的，清單維持空白", flush=True)
        return pool

    today = datetime.now(_TW).strftime("%Y-%m-%d")
    if not force and "verify_date" in pool.columns and (pool["verify_date"] == today).any():
        print(f"候選清單已是今天（{today}）驗證過，直接沿用，不重新查詢 → {path}", flush=True)
        return pool[pool["daytrade_ok"] == True]  # noqa: E712

    candidate_ids = pool["stock_id"].astype(str).tolist()
    tradable_ids = set(_filter_day_tradable(candidate_ids))

    pool = pool.copy()
    pool["daytrade_ok"] = pool["stock_id"].astype(str).isin(tradable_ids)
    pool["verify_date"] = today
    pool["connection_id"] = pd.array([pd.NA] * len(pool), dtype="Int64")

    ok = _assign_connections(pool[pool["daytrade_ok"]])
    pool = pool.set_index("stock_id")
    pool.loc[ok["stock_id"].values, "connection_id"] = ok["connection_id"].values
    pool = pool.reset_index()

    _atomic_to_parquet_local(pool, path)
    n_conn = ok["connection_id"].nunique(dropna=True)
    print(
        f"儲存完成：{pool['daytrade_ok'].sum()} 支可當沖（母體共 {len(pool)} 支），{n_conn} 條連線 → {path}（{today}）",
        flush=True,
    )
    return pool[pool["daytrade_ok"] == True]  # noqa: E712


def _atomic_to_parquet_local(df: pd.DataFrame, path: Path) -> None:
    """先寫暫存檔再 rename，避免寫入過程被中斷導致 parquet 檔損毀（比照
    finmind/m1_api.py::_atomic_to_parquet() 同樣的做法，這裡獨立一份是
    因為那支的參數簽名跟這裡想要的不完全一樣，不強行共用）。"""
    import tempfile

    os.makedirs(path.parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        tmp_path = f.name
    try:
        df.to_parquet(tmp_path, index=False, compression="zstd")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_candidates() -> pd.DataFrame:
    """讀取今天 daytrade_ok=True 的候選股清單（不重算），欄位：
    stock_id/avg_volume/atr_pct/day_trade_tier/rank/forced_include/name/
    daytrade_ok/connection_id/verify_date。

    找不到檔案、或 verify_date 不是今天，仍照樣回傳篩過的結果（可能是舊
    清單），並印警告讓呼叫端自行判斷。
    """
    from finmind.tick_universe import _universe_file_path

    path = _universe_file_path()
    if not path.exists():
        print(f"找不到 {path}，請先執行 `python -m finmind.tick_universe`", flush=True)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "daytrade_ok" not in df.columns:
        print("警告：候選清單還沒做過當沖資格驗證，請先執行 `python -m fubon.subscribe_list`", flush=True)
        return pd.DataFrame()
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    verify_date = str(df["verify_date"].dropna().iloc[0]) if df["verify_date"].notna().any() else None
    if verify_date != today:
        print(f"警告：候選股清單當沖資格是 {verify_date} 驗證的，非今日（{today}），可能尚未更新", flush=True)
    return df[df["daytrade_ok"] == True]  # noqa: E712


def load_subscribe_batches() -> list[list[str]]:
    """開 WebSocket 連線時用：回傳依 connection_id 分組、依 rank 排序的 batches。"""
    df = load_candidates()
    if df.empty:
        return []
    df = df.dropna(subset=["connection_id"])
    if df.empty:
        return []
    return [
        g.sort_values("rank")["stock_id"].tolist()
        for _, g in df.sort_values("connection_id").groupby("connection_id", sort=False)
    ]


if __name__ == "__main__":
    # 手動跑（單獨測試/預覽）故意強制重跑，不要沿用今天已存的舊結果。
    build_and_save_subscribe_list(force=True)
    for i, b in enumerate(load_subscribe_batches(), 1):
        preview = "、".join(b[:5])
        print(f"  連線 {i}：{len(b)} 支  前5：{preview}")
