"""
Pattern Recognition API Endpoints (FastAPI Router)

端點：
- GET  /api/pattern/types             取得可用的技術型態選單清單 (含中文名稱)
- GET  /api/pattern/scan             過濾篩選特定型態與時區的符合股票清單（支援快取）
- GET  /api/pattern/{stock_id}/detail  取得單一股票的 K 線與型態擬合細節（含轉折點與趨勢線座標，支援快取）
- POST /api/pattern/cache/clear      手動清空快取
"""

from typing import Any, Dict, List, Optional
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
from pattern.triangle.detector import TriangleDetector
from pattern.w_bottom.detector import WBottomDetector

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
}

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
    pattern_type: str = Query("triangle", description="型態種類: 可帶單一型態(triangle)、逗號分隔多型態(triangle,w_bottom)、或全型態(all)。可用型態: triangle, abcd_bull, abcd_bear, w_bottom, m_top, head_shoulders_bottom, head_shoulders_top, cup_handle, breakout_retest, breakdown_retest, all"),
    timeframe: str = Query("day", description="時間週期: 1m, 3m, 5m, day"),
    date: Optional[str] = Query(None, description="基準日期 (YYYY-MM-DD)，預設為最新交易日"),
    min_score: float = Query(60.0, description="最小信心度分數 (0~100)"),
    min_vol_lots: Optional[float] = Query(1000.0, description="日 K 10 日均量過濾門檻 (張)，預設 1000 張；設為 None 或 0 表示不限制"),
    limit: int = Query(120, description="K 線視窗根數，預設 120 根"),
) -> Dict[str, Any]:
    """掃描市場上所有股票，找出符合特定/多個/全型態與時區條件的股票清單（支援極速記憶體快取與 10 日均量過濾）。"""
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

    # 3. 僅讀取一次全市場 K 線 (最佳化 I/O 效能)
    all_candles = get_all_stocks_candles(timeframe=timeframe, date=date, limit=limit)
    avg_vol_map = get_stocks_10d_avg_vol_lots(date=date)

    active_detectors = [(pt, DETECTORS[pt]) for pt in selected_types]

    matches = []
    for stock_id, df_candles in all_candles.items():
        if df_candles.empty or len(df_candles) < 20:
            continue

        # 10 日均量 (張) 過濾
        avg_vol_lots = avg_vol_map.get(str(stock_id), 0.0)
        if min_vol_lots and min_vol_lots > 0 and avg_vol_lots < min_vol_lots:
            continue

        # 一支股票可同時匹配多個 Detector
        for pt_key, detector in active_detectors:
            try:
                res = detector.detect(df_candles, stock_id=stock_id, timeframe=timeframe)
                if res and res.score >= min_score:
                    res_dict = res.to_dict()
                    res_dict["pattern_name"] = detector.display_name
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


@router.get("/{stock_id}/detail", summary="單一股票 K 線與型態繪圖細節")
def get_pattern_detail(
    stock_id: str,
    pattern_type: str = Query("triangle", description="型態種類: triangle, w_bottom, m_top, abcd_bull, abcd_bear, head_shoulders_bottom, cup_handle"),
    timeframe: str = Query("day", description="時間週期: 1m, 3m, 5m, day"),
    date: Optional[str] = Query(None, description="基準日期 (YYYY-MM-DD)"),
    limit: int = Query(120, description="K 線視窗根數，預設 120 根"),
) -> Dict[str, Any]:
    """回傳 K 線數據（時間已轉為前端所需的 UTC timestamp）以及型態關鍵轉折點與趨勢線線段資訊（支援快取）。"""
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

    result = {
        "stock_id": stock_id,
        "timeframe": timeframe,
        "pattern_type": pattern_type,
        "pattern_name": detector.display_name,
        "candles": candles_output,
        "pattern": pattern_output,
    }

    _DETAIL_CACHE[cache_key] = result
    return result
