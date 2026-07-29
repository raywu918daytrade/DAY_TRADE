"""
技術型態識別系統基礎類別與資料結構 (Base classes and Data Structures)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class PivotPoint:
    """K線波段轉折點 (Peak 或 Trough)"""
    index: int                  # DataFrame 中的整數索引
    date: str                   # 轉折點時間
    price: float                # 轉折點價格
    type: str                   # "peak" (波段高點) 或 "trough" (波段低點)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "date": str(self.date),
            "price": round(float(self.price), 2),
            "type": str(self.type),
        }


@dataclass
class TrendLine:
    """畫型態使用的趨勢線/支撐壓力線"""
    start_index: int
    end_index: int
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    slope: float
    intercept: float
    r_squared: float
    line_type: str              # "resistance" (抗力/上軌) 或 "support" (支撐/下軌)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_index": int(self.start_index),
            "end_index": int(self.end_index),
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "start_price": round(float(self.start_price), 2),
            "end_price": round(float(self.end_price), 2),
            "slope": round(float(self.slope), 6),
            "intercept": round(float(self.intercept), 2),
            "r_squared": round(float(self.r_squared), 4),
            "line_type": str(self.line_type),
        }


@dataclass
class PatternResult:
    """型態識別結果"""
    stock_id: str
    pattern_type: str           # e.g. "triangle", "w_bottom", "m_top", "abcd_bull", "abcd_bear"
    sub_type: str               # e.g. "symmetrical", "ascending", "descending"
    timeframe: str              # e.g. "1m", "3m", "5m", "day"
    score: float                # 信心度分數 (0 ~ 100)
    date: str                   # 偵測時的最新 K 線日期/時間
    pivots: List[PivotPoint] = field(default_factory=list)
    lines: List[TrendLine] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_id": str(self.stock_id),
            "pattern_type": str(self.pattern_type),
            "sub_type": str(self.sub_type),
            "timeframe": str(self.timeframe),
            "score": round(float(self.score), 2),
            "date": str(self.date),
            "pivots": [p.to_dict() for p in self.pivots],
            "lines": [l.to_dict() for l in self.lines],
            "details": self.details,
        }


class BasePatternDetector(ABC):
    """型態檢測器基類"""

    def __init__(self, name: str, display_name: str = ""):
        self.name = name
        self.display_name = display_name or name

    @abstractmethod
    def detect(self, df: pd.DataFrame, stock_id: str, timeframe: str) -> Optional[PatternResult]:
        """對單一股票的 K 線 DataFrame 進行型態偵測"""
        pass
