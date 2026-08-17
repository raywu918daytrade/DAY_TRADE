"""
K線資料載入器 (Data Loader for Pattern Detection)

支援時間週期：
- "1m"  : 1 分鐘 K 線 (讀 db/m1/ 或 db/m1_live/)
- "3m"  : 3 分鐘標準獨立 K 線 (讀 db/m3_std/ 或由 1m resample)
- "5m"  : 5 分鐘標準獨立 K 線 (讀 db/m5_std/ 或由 1m resample)
- "day" : 日 K 線 (讀 db/adjustment_day/)

價格基準（2026-08-01加，2026-08-03改）：pattern 系列是系統裡唯一需要「完整
還原（含一般除權息）」基準的地方——除息造成的真實跳空會讓型態偵測的轉折點
判斷誤判（實測數字見 data/adjustment_query.py 檔頭說明），跟系統預設的
「只還原拆股/合股」基準（data/query.py）不一樣。這支檔案不再直接戳
db/m1／db/fugle_day 路徑，統一透過 data/raw_query.py（讀原始資料）+
data/adjustment_query.py（套用 pattern 專用完整還原係數）取得資料，
不自己維護一套 factor 換算邏輯。db/m1_live（當天即時資料）不用調整——
「今天」相對於自己必然是同一個基準，raw=adjusted。
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import pyarrow.dataset as ds

from data import raw_query
from data.adjustment_query import load_pattern_day, load_pattern_day_by_stock, _load_adjustment_factor
from data.resample import compute_m3_std, compute_m5_std

_ROOT = Path(__file__).parent.parent


def _apply_pattern_adjust_factor(df: pd.DataFrame, stock_id: Optional[str] = None) -> pd.DataFrame:
    """把 df 的 open/high/low/close 換算成 pattern 專用完整還原後價格。df 需有
    stock_id、date（datetime）欄位；stock_id 參數選填，單一股票查詢時帶入可以
    讓 _load_adjustment_factor() 走 pyarrow filter pushdown，不用載入其他股票的
    factor。"""
    if df.empty:
        return df
    df = df.copy()
    df["_day"] = df["date"].dt.strftime("%Y-%m-%d")
    start = df["_day"].min()
    factor_df = _load_adjustment_factor(stock_id, None, start)
    if factor_df.empty:
        return df.drop(columns=["_day"])
    df = df.merge(
        factor_df[["stock_id", "date", "factor"]].rename(columns={"date": "_day"}),
        on=["stock_id", "_day"],
        how="left",
    )
    df["factor"] = df["factor"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = (df[col] * df["factor"]).round(2).astype("float32")
    return df.drop(columns=["_day", "factor"])


def get_stock_candles(
    stock_id: str,
    timeframe: str = "day",
    date: Optional[str] = None,
    limit: int = 120,
    full_day: bool = False,
) -> pd.DataFrame:
    """載入單一股票指定時間週期的最近 N 根 K 線。

    Args:
        stock_id: 股票代號 (例如 "2330")
        timeframe: "1m", "3m", "5m", "day"
        date: 基準日期 "YYYY-MM-DD" (選填)。不填則抓最新可用的 K 線。
        limit: 最多抓取的 K 線根數，預設 120 根。full_day=True 時忽略。
        full_day: 2026-08-12加（股票清單欄「當日」按鈕用）。True 時改成
            回傳 date（沒帶則視為今天）當天整天的 K 線（開盤到現在/收盤），
            用 date 上下界各自過濾，不再用「取最後 N 根」的方式——後者在
            過去日期成交根數不到 limit 時，會往前多抓到前幾個交易日的
            資料混在一起，跟同一天的 VWAP／型態判斷邏輯（每天重新起算）
            對不上，畫出來的圖會誤以為是連續一直線。只對 1m/3m/5m 有意義，
            day timeframe 本來就一天一根，忽略這個參數。
    """
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except Exception:
            limit = 120
    if timeframe == "day":
        df = load_pattern_day_by_stock(stock_id, date=None)
        if date:
            df = df[df["date"] <= f"{date} 23:59:59"]
        else:
            # 2026-08-17加：db/adjustment_day 是隔夜批次（見
            # backfill_day_history.log，每天早上才補前一交易日），今天盤中
            # 這支股票的日K本來就還沒有——沒帶 date（＝要「最新」）時，補一根
            # 「今天進行中」的合成日K，OHLCV 從今天的1分K（db/m1_live，含
            # 開盤到現在）現算：open=第一筆、high/low=極值、close=最新一筆、
            # volume=加總。收盤後隔天批次補上正式那筆，這根合成的就不會再
            # 被用到（df 裡已經有當天正式資料，下面比對到日期重複就不會疊加）。
            today_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
            has_today = not df.empty and str(df["date"].max().strftime("%Y-%m-%d")) == today_str
            if not has_today:
                df_1m_today = get_stock_candles(stock_id, timeframe="1m", date=None, limit=100000, full_day=True)
                if not df_1m_today.empty:
                    today_row = pd.DataFrame([{
                        "stock_id": stock_id,
                        "date": pd.Timestamp(today_str),
                        "open": float(df_1m_today["open"].iloc[0]),
                        "high": float(df_1m_today["high"].max()),
                        "low": float(df_1m_today["low"].min()),
                        "close": float(df_1m_today["close"].iloc[-1]),
                        # db/adjustment_day 的 volume 單位是股，1分K（db/m1／
                        # db/m1_live）是張——實測 3230 2026-08-14 整天 1分K
                        # 加總 1589（張），對照當天官方日K volume 1,601,769
                        # （股），差了 1000 倍，這裡要乘回來，不然合成的這根
                        # 「今天」volume 會比其他歷史日K小1000倍。
                        "volume": int(df_1m_today["volume"].sum()) * 1000,
                    }])
                    df = pd.concat([df, today_row], ignore_index=True)
        if df.empty:
            return df
        if limit and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    elif timeframe == "1m":
        # 先試 db/m1_live：沒帶 date（呼叫端要「最新」）視為今天，理當去查
        # db/m1_live；帶了明確的過去日期才不用查（db/m1_live 本來就只有
        # 當天資料，查過去日期一定查不到）。2026-08-11 修：原本用「呼叫端
        # 有沒有帶 date」判斷要不要查 db/m1_live，導致沒帶 date 時永遠只吃
        # db/m1（隔夜批次更新，沒有當天資料），股票清單欄選 1分K 盤中查
        # 不到當天資料、圖表空白。
        today_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
        effective_date = date or today_str

        # 2026-08-12改：db/m1（隔夜批次）跟 db/m1_live（當天即時）改成
        # 「聯集去重複」，不是原本的「live 有資料就直接回傳、不然才查
        # db/m1」的互斥 fallback。原本的寫法有兩個問題：
        # 1. 查「最近N根」（沒帶 date，不是 full_day）這種橫跨好幾天的
        #    範圍時，只會查到其中一個來源——db/m1_live 只有今天、db/m1
        #    隔夜批次通常還沒有今天，導致「最近N根」永遠看不到今天盤中
        #    的資料（batch 還沒跑完那段時間）。
        # 2. db/m1_live 存在但這支股票剛好沒被收集到（例如全市場~1900檔
        #    裡不在即時收集清單內的冷門股）時，會整個跳過、退回 db/m1，
        #    但 db/m1 隔夜批次本來就不會有今天資料，等於「今天」這段
        #    永遠是空的，不是「找不到今天的才退回去」而已。
        # 兩個來源都查、依 date 去重（同一個時間點兩邊都有時保留 live，
        # 因為它比隔夜批次更新——concat 時 live 放在後面、
        # drop_duplicates(keep='last') 自然會留下它），不管哪邊缺資料，
        # 只要另一邊有就補得起來。
        df_live = None
        df_batch = None

        # db/m1_live/{date}.parquet 不是只有「今天」才有——即時收集程式
        # 每天存一個檔案，過去幾天的檔案還留在硬碟上沒有被清掉（實測
        # db/m1_live/ 底下同時有 2026-08-04、2026-08-11、2026-08-12…等
        # 好幾天的檔案）。2026-08-13 發現：db/m1（隔夜批次）某些股票在
        # 某些日子完全沒被收進去（例如 3481 在 2026-08-12 這天，db/m1
        # 裡其他1857支股票都有這天的資料，唯獨3481沒有——不確定是批次
        # 收集清單跟即時收集清單本來就不完全一樣，還是那天批次漏跑，
        # 但不管原因是什麼，只要 db/m1_live 那天的檔案還在、剛好有這支
        # 股票，就該拿來補，不用管是不是「今天」——檔案存不存在自己就是
        # 最準的判斷依據，不用另外用日期比對去限制。
        live_path = _ROOT / f"db/m1_live/{effective_date}.parquet"
        if live_path.exists():
            df_live = pd.read_parquet(live_path)
            df_live = df_live[df_live["stock_id"] == stock_id]
            if not df_live.empty:
                df_live["date"] = pd.to_datetime(df_live["date"])
            else:
                df_live = None

        dataset_dir = _ROOT / "db/m1"
        if dataset_dir.exists():
            try:
                # db/m1 是全市場5年多歷史、80個月檔、共2.1GB（2026-08-12
                # 實測）——不能整個目錄丟給 ds.dataset() 只靠 row-level
                # filter pushdown 篩 stock_id，那樣每次都要打開全部80個
                # 檔案，單一支股票的查詢就要5秒以上，股票清單欄點1分K會
                # 卡住甚至看起來像沒反應。改成先用檔名（YYYY_MM.parquet）
                # 篩出真正需要的月份檔案，只開那幾個檔案，跟
                # get_all_stocks_candles()「3m/5m」分支、
                # get_stock_candles()「3m/5m」分支已經在用的做法一致。
                all_files = sorted(f for f in dataset_dir.iterdir() if f.suffix == ".parquet")
                if full_day or date:
                    cutoff_file = effective_date[:7].replace("-", "_")
                    files = [f for f in all_files if f.stem <= cutoff_file][-2:]
                else:
                    # 沒帶日期＝要「最新」，只需要最近1~2個月的檔案就夠涵蓋
                    # limit 根（1分K一天約270根，就算 limit 開到上限5000根
                    # 也只要約20個交易日，2個月綽綽有餘）。
                    files = all_files[-2:]
                if not files:
                    files = all_files[-1:]
                if files:
                    dataset = ds.dataset([str(f) for f in files], format="parquet")
                    filt = ds.field("stock_id") == stock_id
                    if full_day:
                        # db/m1 是跨多天的批次資料，只有上界會把前幾個交易日也
                        # 一起抓進來（成交根數不到 limit 時尤其明顯）——full_day
                        # 要「單一天」，上下界都要設。
                        filt = filt & (ds.field("date") >= f"{effective_date} 00:00:00") & (ds.field("date") <= f"{effective_date} 23:59:59")
                    elif date:
                        filt = filt & (ds.field("date") <= f"{date} 23:59:59")
                    table = dataset.to_table(filter=filt)
                    if table.num_rows > 0:
                        df_batch = table.to_pandas()
                        df_batch["date"] = pd.to_datetime(df_batch["date"], format="mixed")
            except FileNotFoundError:
                # 2026-08-12發現：db/m1 這次改成「一定會查」（不再只在
                # live 缺資料時才 fallback，見上面的說明），撞見背景 flush
                # 工作正在改寫 db/m1/*.parquet（先寫 .tmp 檔再 rename）的
                # 機率變高很多——pyarrow 先列出檔名、實際讀檔那一刻檔案
                # 剛好被搬走，會噴 FileNotFoundError。這只是「這次剛好沒
                # 讀到批次資料」，不是真的沒有資料，忽略掉讓 df_batch 維持
                # None，照樣用 df_live（如果有）繼續組資料，不要讓整個
                # request 500。
                print(f"[get_stock_candles] db/m1 讀取撞到背景寫檔時間差（{stock_id}），略過這次批次資料", flush=True)

        # 组合順序固定「batch 在前、live 在後」，不管上面兩段哪個先查到——
        # drop_duplicates(keep='last') 才會在兩邊都有同一個時間點時留下
        # live（比隔夜批次更新）。
        frames = [f for f in (df_batch, df_live) if f is not None]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df.drop_duplicates(subset=["date"], keep="last", inplace=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = _apply_pattern_adjust_factor(df, stock_id=stock_id)
        if not full_day and limit and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    elif timeframe in ("3m", "5m"):
        # 目錄命名是 db/m3_std、db/m5_std（m 開頭），不是 db/3m_std／db/5m_std
        # ——原本寫成 f"db/{timeframe}_std" 順序反了，永遠找不到目錄
        # （2026-08-11 發現：股票清單欄選 5m 只看得到今天，因為找不到這份
        # 預先算好、涵蓋多年歷史的資料，只好整個 fallback 到別的資料源）。
        today_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
        effective_date = date or today_str
        std_dir = _ROOT / f"db/m{timeframe[:-1]}_std"
        if std_dir.exists():
            try:
                dataset = ds.dataset(str(std_dir), format="parquet")
                filt = ds.field("stock_id") == stock_id
                if full_day:
                    # 跟 1m 的 full_day 分支同理：db/m3_std／db/m5_std 也是跨
                    # 多天的批次資料，上下界都要設，才不會把前幾個交易日的
                    # 資料也混進同一張圖。
                    filt = filt & (ds.field("date") >= pd.Timestamp(f"{effective_date} 00:00:00")) & (ds.field("date") <= pd.Timestamp(f"{effective_date} 23:59:59"))
                elif date:
                    # db/m3_std／db/m5_std 的 date 欄位是原生 timestamp 型別
                    # （不像 db/m1 是字串），用字串比較會直接噴
                    # ArrowNotImplementedError（'less_equal' has no kernel
                    # matching input types (timestamp[ns], string)）——用
                    # pd.Timestamp() 轉成 pyarrow 看得懂的型別再比較。
                    # 2026-08-11 發現：股票清單欄選日期+5分週期查不到資料。
                    filt = filt & (ds.field("date") <= pd.Timestamp(f"{date} 23:59:59"))
                table = dataset.to_table(filter=filt)
                if table.num_rows > 0:
                    df = table.to_pandas()
                    df["date"] = pd.to_datetime(df["date"], format="mixed")
                    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
                    df = df.sort_values("date").reset_index(drop=True)
                    df = _apply_pattern_adjust_factor(df, stock_id=stock_id)
                    if not full_day and limit and len(df) > limit:
                        df = df.iloc[-limit:].reset_index(drop=True)
                    return df
            except FileNotFoundError:
                # 跟上面 1m 分支同一個背景寫檔時間差問題（見那邊的說明），
                # 略過這次，往下走 1m resample fallback，不要讓 request 500。
                print(f"[get_stock_candles] {std_dir} 讀取撞到背景寫檔時間差（{stock_id}），改走 1m resample fallback", flush=True)

        # Fallback: 從 1m 現算 resample（today 的 std_dir 通常還沒有批次
        # 預算好的資料，會走到這裡——full_day 原樣往下傳，1m 那層才是真正
        # 負責篩單一天範圍的地方，這裡不用重複篩）。
        df_1m = get_stock_candles(stock_id, timeframe="1m", date=date, limit=limit * 5, full_day=full_day)
        if df_1m.empty:
            return pd.DataFrame()
        if timeframe == "3m":
            df_res = compute_m3_std(df_1m)
        else:
            df_res = compute_m5_std(df_1m)
        if not full_day and limit and len(df_res) > limit:
            df_res = df_res.iloc[-limit:].reset_index(drop=True)
        return df_res

    return pd.DataFrame()


def get_all_stocks_candles(
    timeframe: str = "day",
    date: Optional[str] = None,
    limit: int = 120,
    start_date: Optional[str] = None,
    stock_ids: Optional[set] = None,
) -> Dict[str, pd.DataFrame]:
    """批次載入全市場所有股票在指定時間週期的最近 N 根 K 線。

    stock_ids：選填，只載入這個集合裡的股票（用 pyarrow filter pushdown
    在讀檔當下就篩掉，不是讀完全部股票再丟棄）。2026-08-11加：
    scan_patterns() 實際上只會拿 TICK_UNIVERSE_SET（~400檔）裡的股票去跑
    型態偵測（見 pattern_api.py 的說明），但這支函式原本沒地方讓呼叫端先
    講清楚「只需要這些股票」，導致 intraday（3m/5m）掃描要多讀 5~7 倍的
    股票資料（db/m*_std 是全市場~2900檔），是「全部型態掃描會逾時/斷線」
    的主因之一。

    Returns:
        Dict[stock_id, pd.DataFrame]
    """
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except Exception:
            limit = 120
    stock_filter = ds.field("stock_id").isin(stock_ids) if stock_ids else None
    if timeframe == "day":
        ref_date = date if (date and isinstance(date, str)) else "today"
        cutoff = start_date or (pd.Timestamp(ref_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        df = load_pattern_day(start_date=cutoff)
        if df.empty:
            return {}
        if stock_ids:
            df = df[df["stock_id"].isin(stock_ids)]
        if date and isinstance(date, str):
            df = df[df["date"] <= f"{date} 23:59:59"]

    elif timeframe == "1m":
        # 先看當日即時
        if date:
            live_path = _ROOT / f"db/m1_live/{date}.parquet"
            if live_path.exists():
                df = pd.read_parquet(live_path)
                df["date"] = pd.to_datetime(df["date"])
                if stock_ids:
                    df = df[df["stock_id"].isin(stock_ids)]
            else:
                dir_path = _ROOT / "db/m1"
                cutoff = date[:7].replace("-", "_")
                files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet" and f.stem >= cutoff)
                if not files:
                    files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")[-1:]
                if not files:
                    return {}
                df = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=stock_filter).to_pandas()
                df["date"] = pd.to_datetime(df["date"], format="mixed")
                df = df[df["date"] <= f"{date} 23:59:59"]
                df = _apply_pattern_adjust_factor(df)
        else:
            dir_path = _ROOT / "db/m1"
            files = sorted(f for f in dir_path.iterdir() if f.suffix == ".parquet")[-1:]
            if not files:
                return {}
            df = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=stock_filter).to_pandas()
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df = _apply_pattern_adjust_factor(df)

    elif timeframe in ("3m", "5m"):
        # 目錄命名是 db/m3_std、db/m5_std（m 開頭），不是 db/3m_std／db/5m_std
        # ——原本寫成 f"db/{timeframe}_std" 順序反了，永遠找不到目錄
        # （2026-08-11 發現：股票清單欄選 5m 只看得到今天，因為找不到這份
        # 預先算好、涵蓋多年歷史的資料，只好整個 fallback 到別的資料源）。
        std_dir = _ROOT / f"db/m{timeframe[:-1]}_std"
        if std_dir.exists():
            # 要有「date 所在月份」以前的檔案才撈得到往回推 limit 根的歷史
            # ——原本寫成 f.stem >= cutoff_file 方向反了：只挑 date 當月
            # 之後（含）的檔案，但下面又只留 date 之前的資料列，date 剛好
            # 落在該月很早期（或非交易日，例如週末）時，選到的檔案裡完全
            # 沒有任何一列通得過 <= date 的篩選，整批股票變成空清單。
            # 2026-08-11 發現：型態掃描選日期+3分/5分週期掃描不到任何股票。
            # 只取最近4個月（不是撈整個多年歷史），避免全部股票一次性讀進
            # 記憶體——型態掃描的K線根數上限是500（見#pattern-limit），
            # 4個月的5分K絕對夠用。
            cutoff_file = date[:7].replace("-", "_") if date else ""
            if cutoff_file:
                files = sorted(f for f in std_dir.iterdir() if f.suffix == ".parquet" and f.stem <= cutoff_file)[-4:]
            else:
                files = sorted(f for f in std_dir.iterdir() if f.suffix == ".parquet")
            if not files:
                files = sorted(f for f in std_dir.iterdir() if f.suffix == ".parquet")[-1:]
            if files:
                df = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=stock_filter).to_pandas()
                df["date"] = pd.to_datetime(df["date"], format="mixed")
                if date:
                    df = df[df["date"] <= f"{date} 23:59:59"]
                df = _apply_pattern_adjust_factor(df)
            else:
                m1_candles = get_all_stocks_candles("1m", date=date, limit=limit * 5, stock_ids=stock_ids)
                res = {}
                for sid, m1_df in m1_candles.items():
                    res_df = compute_m3_std(m1_df) if timeframe == "3m" else compute_m5_std(m1_df)
                    if not res_df.empty:
                        res[sid] = res_df.iloc[-limit:].reset_index(drop=True)
                return res
        else:
            m1_candles = get_all_stocks_candles("1m", date=date, limit=limit * 5, stock_ids=stock_ids)
            res = {}
            for sid, m1_df in m1_candles.items():
                res_df = compute_m3_std(m1_df) if timeframe == "3m" else compute_m5_std(m1_df)
                if not res_df.empty:
                    res[sid] = res_df.iloc[-limit:].reset_index(drop=True)
            return res
    else:
        return {}

    # 按 stock_id 分組裁切至 limit 根
    result = {}
    for sid, group in df.groupby("stock_id"):
        group = group.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        if limit and len(group) > limit:
            group = group.iloc[-limit:]
        result[str(sid)] = group.reset_index(drop=True)

    return result


def get_latest_candle_timestamp(timeframe: str = "day", date: Optional[str] = None) -> str:
    """取得特定 timeframe 及 date 設定下最新一根 K 線的時間戳記，作為 Cache Key 版本識別。

    day：改用 db/adjustment_day/ 底下所有月檔的最新 mtime，疊加 2330 今天最新
    一根1分K的時間戳（2026-08-17改）。原本跟 intraday 共用「採樣 2330 最新
    一根K線時間戳」的邏輯，只能偵測「多了新的一天」，偵測不到「舊資料被
    回補/修正」——例如 data.day_data_loader.update_adjustment_day() 整批重建
    歷史時，2330 當天最新一根K線的日期前後不變，快取版本號不會跳，其他股票
    被回補/改掉的舊資料就會一直卡在重建前的舊快取，直到 process 重啟或手動
    /cache/clear（2026-08-17 實測 3230 卡到這個問題）。改成看檔案 mtime，
    任何一支月檔被重寫（不管是新增一天還是回補舊資料）都會讓版本號跳動，
    整批快取失效重算；成本也比原本查 2330 K線（開 parquet row group）更低，
    只是 stat 檔案。
    另外疊加 2330 今天最新1分K時間戳：get_stock_candles() 的 day 分支
    （見該函式說明）沒帶 date 時會現算一根「今天進行中」的合成日K，OHLCV
    隨盤中每分鐘變化——只看 adjustment_day mtime（隔夜批次才會動）偵測不到
    這個變化，today 這根會整天卡在第一次被查到的樣子不再更新，所以要跟
    intraday 共用「有沒有新一分鐘」這個訊號。收盤後 2330 不再有新1分鐘，
    版本自然穩定，直到下一個交易日。
    intraday（1m/3m/5m）維持原本邏輯不動：db/m1 等檔案數量龐大，逐檔 stat
    反而較貴，而且沒有歷史回補這個問題，用「有沒有新一分鐘」這一個訊號就夠。
    """
    if timeframe == "day":
        day_dir = _ROOT / "db/adjustment_day"
        try:
            mtime = max(f.stat().st_mtime for f in day_dir.glob("*.parquet"))
        except (FileNotFoundError, ValueError):
            mtime = "no-file"
        intraday_tick = ""
        try:
            sample = get_stock_candles("2330", timeframe="1m", date=None, limit=1)
            if not sample.empty:
                intraday_tick = str(sample["date"].iloc[-1])
        except Exception:
            pass
        return f"{mtime}|{intraday_tick}"
    try:
        sample = get_stock_candles("2330", timeframe=timeframe, date=date, limit=1)
        if not sample.empty:
            return str(sample["date"].iloc[-1])
    except Exception:
        pass
    return date or "latest"


def get_stocks_10d_avg_vol_lots(date: Optional[str] = None) -> Dict[str, float]:
    """從 db/adjustment_day/ 計算全市場各股票近 10 個交易日的平均成交量 (單位: 張 = 1000 股)。

    Returns:
        Dict[stock_id, float] (例如 {"2330": 28450.5})
    """
    ref_date = date if (date and isinstance(date, str)) else "today"
    cutoff = (pd.Timestamp(ref_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    df = load_pattern_day(start_date=cutoff)
    if df.empty:
        return {}
    if date and isinstance(date, str):
        df = df[df["date"] <= f"{date} 23:59:59"]

    avg_vols = {}
    for sid, group in df.groupby("stock_id"):
        recent10 = group.drop_duplicates(subset=["date"], keep="last").sort_values("date").tail(10)
        if not recent10.empty:
            avg_shares = float(recent10["volume"].mean())
            avg_vols[str(sid)] = round(avg_shares / 1000.0, 1)

    return avg_vols
