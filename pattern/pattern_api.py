"""
Pattern Recognition API Endpoints (FastAPI Router)

端點：
- GET  /api/pattern/stocks            股票清單（tick=~400當沖候選、full=~1900全市場，含中文名稱）
- GET  /api/pattern/types             取得可用的技術型態選單清單 (含中文名稱)
- GET  /api/pattern/scan             過濾篩選特定型態與時區的符合股票清單（同步版本，支援快取）
- GET  /api/pattern/scan/submit      非同步版本：立刻回傳 job_id，掃描完成後透過 SSE
                                       （type: pattern_scan_done）推播結果，見 _run_scan_job()
                                       的說明——2026-08-11加，避免掃描（純Python迴圈跑型態
                                       偵測，長時間佔用GIL）卡住 /stream 導致前端SSE斷線。
- GET  /api/pattern/{stock_id}/detail  取得單一股票的 K 線與型態擬合細節（含轉折點與趨勢線座標，支援快取）
- POST /api/pattern/cache/clear      手動清空快取
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from pattern.abcd_bear.detector import AbcdBearDetector
from pattern.abcd_bull.detector import AbcdBullDetector
from pattern.breakdown_retest.detector import BreakdownRetestDetector
from pattern.breakout_retest.detector import BreakoutRetestDetector
from pattern.cup_handle.detector import CupHandleDetector
from pattern.data_loader import get_all_stocks_candles, get_latest_candle_timestamp, get_stock_candles, get_stocks_10d_avg_vol_lots
from pattern.head_shoulders_bottom.detector import HeadShouldersBottomDetector
from pattern.head_shoulders_top.detector import HeadShouldersTopDetector
from pattern.m_top.detector import MTopDetector
from pattern.macd_hist_bear.detector import MacdHistBearDetector
from pattern.macd_hist_bull.detector import MacdHistBullDetector
from pattern.triangle.detector import TriangleDetector
from pattern.w_bottom.detector import WBottomDetector
from data.adjustment_query import load_pattern_volume_profile
from data.build_poc import compute_poc_dataframe

router = APIRouter(prefix="/api/pattern", tags=["技術型態"])

# 註冊所有可用型態檢測器
DETECTORS = {
    "triangle": TriangleDetector(),
    "abcd_bull": AbcdBullDetector(),
    "abcd_bear": AbcdBearDetector(),
    "w_bottom": WBottomDetector(),
    "m_top": MTopDetector(),
    "head_shoulders_bottom": HeadShouldersBottomDetector(),
    "head_shoulders_top": HeadShouldersTopDetector(),
    "cup_handle": CupHandleDetector(),
    "breakout_retest": BreakoutRetestDetector(),
    "breakdown_retest": BreakdownRetestDetector(),
    "macd_hist_bull": MacdHistBullDetector(),
    "macd_hist_bear": MacdHistBearDetector(),
}

# 從 db/tickers/tick_universe.parquet 一次載入股票集合與名稱對照
try:
    _uni_df = pd.read_parquet(
        Path(__file__).parent.parent / "db/tickers/tick_universe.parquet",
        columns=["stock_id", "name"],
    )
    TICK_UNIVERSE_SET = set(_uni_df["stock_id"].astype(str))
    STOCK_NAME_MAP = {
        str(sid): str(name).strip()
        for sid, name in zip(_uni_df["stock_id"], _uni_df["name"])
        if pd.notna(name) and str(name).strip()
    }
except Exception:
    TICK_UNIVERSE_SET = set()
    STOCK_NAME_MAP = {}

# 全市場股票清單（db/tickers/stock_universe_2000.parquet，~1900檔，含中文
# 名稱），供前端「股票清單」欄的 universe=full 選項用（見 GET /stocks）。
# 跟上面 tick_universe 那份是不同來源、不同數量的股票池，分開載入。
try:
    _full_uni_df = pd.read_parquet(
        Path(__file__).parent.parent / "db/tickers/stock_universe_2000.parquet",
        columns=["stock_id", "name"],
    )
    FULL_UNIVERSE_LIST = [
        {"stock_id": str(sid), "name": str(name).strip()}
        for sid, name in zip(_full_uni_df["stock_id"], _full_uni_df["name"])
        if pd.notna(name) and str(name).strip()
    ]
except Exception:
    FULL_UNIVERSE_LIST = []

# 欄位名故意跟 FULL_UNIVERSE_LIST 一致用 stock_id（不是 id）：前端
# navMonitoring() 鍵盤上下切換清單統一用 item.stock_id 找目前選到的股票
# （見 dashboard.html 其他清單，例如 _patternList/_conList 都是這個欄位名），
# 股票清單欄的清單要跟這個慣例一致，不能自己另外取名。
TICK_UNIVERSE_LIST = [{"stock_id": sid, "name": name} for sid, name in STOCK_NAME_MAP.items()]

# 記憶體快取 (In-Memory Cache)
_SCAN_CACHE: Dict[tuple, Dict[str, Any]] = {}
_DETAIL_CACHE: Dict[tuple, Dict[str, Any]] = {}


@router.post("/cache/clear", summary="手動清空 Pattern 快取")
def clear_pattern_cache() -> Dict[str, Any]:
    """清空記憶體中的 Pattern 掃描與詳情快取。"""
    scan_count = len(_SCAN_CACHE)
    detail_count = len(_DETAIL_CACHE)
    _SCAN_CACHE.clear()
    _DETAIL_CACHE.clear()
    return {
        "ok": True,
        "message": f"已清空快取 (scan 快取: {scan_count} 筆, detail 快取: {detail_count} 筆)",
    }


@router.get("/stocks", summary="股票清單（供前端股票清單欄選擇用）")
def get_stock_list(
    universe: str = Query("tick", description="tick=當沖候選(~400檔)、full=全市場(~1900檔)"),
) -> Dict[str, Any]:
    """回傳指定股票池的代號＋中文名稱清單，供前端「股票清單」欄渲染。"""
    stocks = FULL_UNIVERSE_LIST if universe == "full" else TICK_UNIVERSE_LIST
    return {"universe": universe, "stocks": stocks}


@router.get("/types", summary="取得可用的技術型態選單清單")
def get_pattern_types() -> Dict[str, Any]:
    """回傳所有已註冊的技術型態 ID 與中文名稱，供前端選單使用。"""
    return {
        "patterns": [
            {
                "id": key,
                "name": detector.display_name,
            }
            for key, detector in DETECTORS.items()
        ]
    }


@router.get("/scan", summary="過濾篩選符合特定型態與時區的股票清單")
def scan_patterns(
    pattern_type: str = Query("triangle", description="型態種類: 可帶單一型態(triangle)、逗號分隔多型態(triangle,w_bottom)、或全型態(all)。可用型態: triangle, abcd_bull, abcd_bear, w_bottom, m_top, head_shoulders_bottom, head_shoulders_top, cup_handle, breakout_retest, breakdown_retest, macd_hist_bull, macd_hist_bear, all"),
    timeframe: str = Query("day", description="時間週期: 1m, 3m, 5m, day"),
    date: Optional[str] = Query(None, description="基準日期 (YYYY-MM-DD)，預設為最新交易日"),
    min_score: float = Query(60.0, description="最小信心度分數 (0~100)"),
    min_vol_lots: Optional[float] = Query(1000.0, description="日 K 10 日均量過濾門檻 (張)，預設 1000 張；設為 None 或 0 表示不限制"),
    limit: int = Query(120, description="K 線視窗根數，預設 120 根"),
) -> Dict[str, Any]:
    """掃描 tick_universe（約 400 檔）股票，找出符合特定/多個/全型態與時區條件的清單（支援記憶體快取與 10 日均量過濾）。"""
    # 處理直接在 Python 內部調用函式時可能傳入 Query 物件的情況
    if hasattr(pattern_type, "default"):
        pattern_type = pattern_type.default
    if hasattr(timeframe, "default"):
        timeframe = timeframe.default
    if hasattr(min_score, "default"):
        min_score = min_score.default
    if hasattr(min_vol_lots, "default"):
        min_vol_lots = min_vol_lots.default
    if hasattr(limit, "default"):
        limit = limit.default

    # 1. 解析型態參數 (支援逗號分隔與 "all")
    raw_types = [t.strip() for t in pattern_type.split(",") if t.strip()]
    
    if "all" in raw_types:
        selected_types = list(DETECTORS.keys())
    else:
        selected_types = []
        invalid_types = []
        for p in raw_types:
            if p in DETECTORS:
                if p not in selected_types:
                    selected_types.append(p)
            else:
                invalid_types.append(p)
        
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"尚未支援或無效的型態: {invalid_types}。可用型態: {list(DETECTORS.keys())} 或 all",
            )

    if not selected_types:
        raise HTTPException(
            status_code=400,
            detail=f"未指定有效的型態。可用型態: {list(DETECTORS.keys())} 或 all",
        )

    # 正規化型態鍵值以穩定命中快取
    normalized_pattern_key = ",".join(sorted(selected_types))

    # 2. 取得最新 K 線時間戳以構造智慧快取 Key
    latest_ts = get_latest_candle_timestamp(timeframe=timeframe, date=date)
    cache_key = (normalized_pattern_key, timeframe, date or "latest", min_score, min_vol_lots, limit, latest_ts)

    if cache_key in _SCAN_CACHE:
        return _SCAN_CACHE[cache_key]

    # 3. 只讀 tick_universe 母體（~400檔）的 K 線，不要整個市場（~2900檔）
    # 都讀進來才在下面的迴圈丟掉——這支函式本來就只會對 TICK_UNIVERSE_SET
    # 裡的股票跑型態偵測，intraday（3m/5m）資料量大，先用 stock_ids
    # 篩掉不需要的股票，實測全型態掃描才不會因為讀太多用不到的資料逾時/
    # 斷線（2026-08-11 使用者提出：已經用成交量篩過候選池了，應該先篩
    # 再抓K線，不要抓完K線才篩）。
    all_candles = get_all_stocks_candles(timeframe=timeframe, date=date, limit=limit, stock_ids=TICK_UNIVERSE_SET)
    avg_vol_map = get_stocks_10d_avg_vol_lots(date=date)

    active_detectors = [(pt, DETECTORS[pt]) for pt in selected_types]

    matches = []
    for stock_id in TICK_UNIVERSE_SET:
        df_candles = all_candles.get(stock_id)
        if df_candles is None or df_candles.empty or len(df_candles) < 20:
            continue

        str_stock_id = str(stock_id)

        # 10 日均量 (張) 過濾
        avg_vol_lots = avg_vol_map.get(str_stock_id, 0.0)
        if min_vol_lots and min_vol_lots > 0 and avg_vol_lots < min_vol_lots:
            continue

        # 一支股票可同時匹配多個 Detector
        for pt_key, detector in active_detectors:
            try:
                res = detector.detect(df_candles, stock_id=stock_id, timeframe=timeframe)
                if res and res.score >= min_score:
                    stock_name = STOCK_NAME_MAP.get(str_stock_id, str_stock_id)
                    res_dict = res.to_dict()
                    res_dict["pattern_name"] = detector.display_name
                    res_dict["stock_name"] = stock_name
                    res_dict["name"] = stock_name
                    res_dict["in_tick_universe"] = True
                    res_dict["details"]["avg_vol_10d_lots"] = avg_vol_lots
                    matches.append(res_dict)
            except Exception:
                continue

    # 按信心分數遞減排序
    matches.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "pattern_type": pattern_type,
        "pattern_types": selected_types,
        "timeframe": timeframe,
        "date": date or (matches[0]["date"] if matches else None),
        "min_vol_lots": min_vol_lots,
        "total_matches": len(matches),
        "results": matches,
    }

    _SCAN_CACHE[cache_key] = result
    return result


# ── 非同步版本（2026-08-11加）─────────────────────────────────────────────
# 型態偵測是純 Python 巢狀迴圈（見各 detector 的 detect()），跑起來會連續
# 占用 GIL 45~70秒以上，即使 FastAPI 把同步路由丟到執行緒池跑，同一個
# process 裡的 asyncio event loop 執行緒一樣要搶 GIL，會導致 /stream
# （SSE）長時間排不到執行時間，前端看起來像斷線（見檔頭說明）。這裡改成
# 「submit 立刻回傳 job_id，掃描在背景 async task 跑、每處理一批股票就
# await asyncio.sleep(0) 讓出 event loop，完成後用既有的 SSE _broadcast()
# 推播結果」，不做輪詢。scan_patterns()（同步版本）保留不動，供直接
# import 呼叫的場合使用，行為完全不變。
_scan_jobs: Dict[str, Dict[str, Any]] = {}
_SCAN_YIELD_EVERY = 20  # 每處理這麼多檔股票就讓出 event loop 一次


async def _scan_stocks_async(
    all_candles: Dict[str, pd.DataFrame],
    avg_vol_map: Dict[str, float],
    active_detectors: list,
    timeframe: str,
    min_score: float,
    min_vol_lots: Optional[float],
) -> list:
    """邏輯跟 scan_patterns() 內的迴圈完全一樣，差別只在定期
    await asyncio.sleep(0)，讓任何一段連續佔用 GIL 的時間都壓在很短
    （幾百毫秒等級），SSE heartbeat／健康檢查才擠得進去。"""
    matches = []
    for i, stock_id in enumerate(TICK_UNIVERSE_SET):
        if i % _SCAN_YIELD_EVERY == 0:
            await asyncio.sleep(0)
        df_candles = all_candles.get(stock_id)
        if df_candles is None or df_candles.empty or len(df_candles) < 20:
            continue

        str_stock_id = str(stock_id)
        avg_vol_lots = avg_vol_map.get(str_stock_id, 0.0)
        if min_vol_lots and min_vol_lots > 0 and avg_vol_lots < min_vol_lots:
            continue

        for pt_key, detector in active_detectors:
            try:
                res = detector.detect(df_candles, stock_id=stock_id, timeframe=timeframe)
                if res and res.score >= min_score:
                    stock_name = STOCK_NAME_MAP.get(str_stock_id, str_stock_id)
                    res_dict = res.to_dict()
                    res_dict["pattern_name"] = detector.display_name
                    res_dict["stock_name"] = stock_name
                    res_dict["name"] = stock_name
                    res_dict["in_tick_universe"] = True
                    res_dict["details"]["avg_vol_10d_lots"] = avg_vol_lots
                    matches.append(res_dict)
            except Exception:
                continue

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


async def _run_scan_job(
    job_id: str,
    pattern_type: str,
    timeframe: str,
    date: Optional[str],
    min_score: float,
    min_vol_lots: Optional[float],
    limit: int,
) -> None:
    from api import _broadcast

    try:
        raw_types = [t.strip() for t in pattern_type.split(",") if t.strip()]
        if "all" in raw_types:
            selected_types = list(DETECTORS.keys())
        else:
            selected_types = []
            for p in raw_types:
                if p in DETECTORS and p not in selected_types:
                    selected_types.append(p)

        if not selected_types:
            _scan_jobs[job_id] = {"status": "error", "error": "未指定有效的型態"}
            _broadcast({"type": "pattern_scan_done", "job_id": job_id, "error": "未指定有效的型態"})
            return

        # 讀K線資料（get_all_stocks_candles/get_stocks_10d_avg_vol_lots）本來
        # 以為只是「一次性成本、不是這次問題的主因」，但實測（2026-08-11）
        # 光是這段就會擋住 event loop 快30秒，跟型態偵測迴圈一樣嚴重——
        # 丟進預設執行緒池跑（loop.run_in_executor），這些底層是
        # pyarrow/pandas 的 I/O／C-level 運算，會釋放 GIL，才能真的讓
        # event loop 在這段期間繼續處理 SSE／健康檢查。
        loop = asyncio.get_event_loop()
        latest_ts = await loop.run_in_executor(
            None, lambda: get_latest_candle_timestamp(timeframe=timeframe, date=date)
        )
        normalized_pattern_key = ",".join(sorted(selected_types))
        cache_key = (normalized_pattern_key, timeframe, date or "latest", min_score, min_vol_lots, limit, latest_ts)

        if cache_key in _SCAN_CACHE:
            result = _SCAN_CACHE[cache_key]
        else:
            all_candles = await loop.run_in_executor(
                None,
                lambda: get_all_stocks_candles(
                    timeframe=timeframe, date=date, limit=limit, stock_ids=TICK_UNIVERSE_SET
                ),
            )
            avg_vol_map = await loop.run_in_executor(None, lambda: get_stocks_10d_avg_vol_lots(date=date))
            active_detectors = [(pt, DETECTORS[pt]) for pt in selected_types]
            matches = await _scan_stocks_async(
                all_candles, avg_vol_map, active_detectors, timeframe, min_score, min_vol_lots
            )
            result = {
                "pattern_type": pattern_type,
                "pattern_types": selected_types,
                "timeframe": timeframe,
                "date": date or (matches[0]["date"] if matches else None),
                "min_vol_lots": min_vol_lots,
                "total_matches": len(matches),
                "results": matches,
            }
            _SCAN_CACHE[cache_key] = result

        _scan_jobs[job_id] = {"status": "done"}
        _broadcast({"type": "pattern_scan_done", "job_id": job_id, "data": result})
    except Exception as e:
        _scan_jobs[job_id] = {"status": "error", "error": str(e)}
        _broadcast({"type": "pattern_scan_done", "job_id": job_id, "error": str(e)})


@router.get("/scan/submit", summary="非同步版本：立刻回傳job_id，掃描完成後透過SSE推播結果")
async def submit_scan(
    pattern_type: str = Query("triangle", description="同 /scan 的說明"),
    timeframe: str = Query("day", description="時間週期: 1m, 3m, 5m, day"),
    date: Optional[str] = Query(None, description="基準日期 (YYYY-MM-DD)，預設為最新交易日"),
    min_score: float = Query(60.0, description="最小信心度分數 (0~100)"),
    min_vol_lots: Optional[float] = Query(1000.0, description="日 K 10 日均量過濾門檻 (張)"),
    limit: int = Query(120, description="K 線視窗根數，預設 120 根"),
) -> Dict[str, Any]:
    """立刻回傳 job_id，不等掃描完成——一定要宣告成 async def 才能在這裡
    呼叫 asyncio.create_task()（同步 def 路由會被 FastAPI 丟到背景執行緒
    跑，那個執行緒沒有 running event loop，呼叫 create_task 會直接出錯）。
    """
    job_id = str(uuid.uuid4())
    _scan_jobs[job_id] = {"status": "pending"}
    asyncio.create_task(
        _run_scan_job(job_id, pattern_type, timeframe, date, min_score, min_vol_lots, limit)
    )
    return {"job_id": job_id}


@router.get("/{stock_id}/detail", summary="單一股票 K 線與型態繪圖細節")
def get_pattern_detail(
    stock_id: str,
    pattern_type: str = Query("triangle", description="型態種類: triangle, w_bottom, m_top, abcd_bull, abcd_bear, head_shoulders_bottom, cup_handle, macd_hist_bull, macd_hist_bear"),
    timeframe: str = Query("day", description="時間週期: 1m, 3m, 5m, day"),
    date: Optional[str] = Query(None, description="基準日期 (YYYY-MM-DD)"),
    limit: int = Query(120, description="K 線視窗根數，預設 120 根"),
) -> Dict[str, Any]:
    """回傳 K 線數據（時間已轉為前端所需的 UTC timestamp）以及型態關鍵轉折點與趨勢線線段資訊（支援快取）。"""
    # 處理直接在 Python 內部調用函式時可能傳入 Query 物件的情況
    if hasattr(pattern_type, "default"):
        pattern_type = pattern_type.default
    if hasattr(timeframe, "default"):
        timeframe = timeframe.default
    if hasattr(date, "default"):
        date = date.default
    if hasattr(limit, "default"):
        limit = limit.default

    if pattern_type not in DETECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"尚未支援或無效的型態: {pattern_type}。可用型態: {list(DETECTORS.keys())}",
        )

    # 取得最新 K 線時間戳以構造智慧快取 Key
    latest_ts = get_latest_candle_timestamp(timeframe=timeframe, date=date)
    cache_key = (stock_id, pattern_type, timeframe, date or "latest", limit, latest_ts)

    if cache_key in _DETAIL_CACHE:
        return _DETAIL_CACHE[cache_key]

    df_candles = get_stock_candles(stock_id=stock_id, timeframe=timeframe, date=date, limit=limit)
    if df_candles.empty:
        raise HTTPException(status_code=404, detail=f"查無 {stock_id} 在 timeframe={timeframe} 的 K 線資料")

    detector = DETECTORS[pattern_type]
    pattern_res = detector.detect(df_candles, stock_id=stock_id, timeframe=timeframe)

    # 轉換 K 線給前端圖表使用 (依照 CLAUDE.md 使用 tw_naive_to_epoch)
    from api import tw_naive_to_epoch

    candles_output = []
    for _, row in df_candles.iterrows():
        dt = row["date"]
        ts = tw_naive_to_epoch(dt)
        candles_output.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        })

    pattern_output = None
    if pattern_res:
        pattern_dict = pattern_res.to_dict()
        pattern_dict["pattern_name"] = detector.display_name
        # 把 lines 和 pivots 裡的時間也轉成 epoch 秒數方便前端畫圖
        for p in pattern_dict.get("pivots", []):
            try:
                p["time"] = tw_naive_to_epoch(pd.Timestamp(p["date"]))
            except Exception:
                p["time"] = None

        for l in pattern_dict.get("lines", []):
            try:
                t1 = tw_naive_to_epoch(pd.Timestamp(l["start_date"]))
                t2 = tw_naive_to_epoch(pd.Timestamp(l["end_date"]))
                l["start_time"] = t1
                l["end_time"] = t2
            except Exception:
                l["start_time"] = None
                l["end_time"] = None

        pattern_output = pattern_dict

    # 載入「整個可視K線範圍」的 Volume Profile & POC 籌碼數據（2026-08-01改：
    # 原本只查K線視窗最後一天，跟使用者實際打開的視窗範圍（例如120天）完全
    # 脫鉤，數值看起來不合理、橫條圖也擠成一團貼在單日POC旁邊。現在改成把
    # df_candles 涵蓋的整段日期範圍內、每個價位的量都加總起來，再用跟
    # build_poc.py 同一套演算法重新算這個範圍專屬的 POC/VAH/VAL——不能直接
    # 沿用 db/poc_day 的每日precompute值，那是單日的，範圍一拉大就對不上）
    volume_profile_output = []
    poc_data_output = None

    try:
        if not df_candles.empty:
            range_start = df_candles["date"].iloc[0].strftime("%Y-%m-%d")
            range_end = df_candles["date"].iloc[-1].strftime("%Y-%m-%d")
            vp_df = load_pattern_volume_profile(stock_id=stock_id, start_date=range_start)
            if not vp_df.empty:
                vp_df = vp_df[(vp_df["date"] >= range_start) & (vp_df["date"] <= range_end)]

            if not vp_df.empty:
                agg = (
                    vp_df.groupby("price", as_index=False)[["volume", "buy_volume", "sell_volume", "neutral_volume"]]
                    .sum()
                    .sort_values("price")
                )
                for _, r in agg.iterrows():
                    volume_profile_output.append({
                        "price": float(r["price"]),
                        "volume": int(r["volume"]),
                        "buy_volume": int(r["buy_volume"]),
                        "sell_volume": int(r["sell_volume"]),
                        "neutral_volume": int(r["neutral_volume"]),
                    })

                agg_for_poc = agg.assign(stock_id=stock_id, date="range")
                poc_df = compute_poc_dataframe(agg_for_poc)
                if not poc_df.empty:
                    r_poc = poc_df.iloc[0]
                    poc_data_output = {
                        "poc": float(r_poc["poc"]),
                        "pocs": str(r_poc["pocs"]),
                        "poc_volume": int(r_poc["poc_volume"]),
                        "poc_count": int(r_poc["poc_count"]),
                        "profile_type": str(r_poc["profile_type"]),
                        "vah": float(r_poc["vah"]),
                        "val": float(r_poc["val"]),
                        "total_volume": int(r_poc["total_volume"]),
                    }
    except Exception:
        pass

    result = {
        "stock_id": stock_id,
        "stock_name": STOCK_NAME_MAP.get(str(stock_id), str(stock_id)),
        "in_tick_universe": str(stock_id) in TICK_UNIVERSE_SET,
        "timeframe": timeframe,
        "pattern_type": pattern_type,
        "pattern_name": detector.display_name,
        "candles": candles_output,
        "pattern": pattern_output,
        "volume_profile": volume_profile_output,
        "poc_data": poc_data_output,
    }

    _DETAIL_CACHE[cache_key] = result
    return result
