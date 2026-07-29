# 技術型態識別系統 (Pattern Recognition System)

本模組提供 K 線技術型態（如三角收斂、W底、M頭、頭肩頂、頭肩底、杯柄、ABCD 等 8 種型態）的自動識別演算、跨時間週期資料載入、FastAPI 端點與記憶體快取機制。

---

## 1. 目錄與架構 (Module Structure)

```
pattern/
├── README.md                 # 本說明文件
├── __init__.py               # 模組對外匯入點
├── base.py                   # 統一資料結構 (PivotPoint, TrendLine, PatternResult) 與 Detector 基類
├── data_loader.py            # 跨週期 K 線載入 (1m, 3m, 5m, day) 與 10日均量計算
├── pattern_api.py            # FastAPI APIRouter 進入點 (掛載至 api.py /api/pattern)
├── triangle/                 # 三角收斂型態檢測器
│   ├── __init__.py
│   └── detector.py           # 精準包絡線 (Envelope) 三角收斂演算法
├── abcd_bull/                # ABCD 上漲型態檢測器
│   ├── __init__.py
│   └── detector.py           # 斐波那契 ABCD 上漲型態演算法
├── abcd_bear/                # ABCD 下跌型態檢測器
│   ├── __init__.py
│   └── detector.py           # 斐波那契 ABCD 下跌型態演算法
├── w_bottom/                 # W 底型態檢測器
│   ├── __init__.py
│   └── detector.py           # 雙底對齊、時間與振幅對稱演算法
├── m_top/                    # M 頭型態檢測器
│   ├── __init__.py
│   └── detector.py           # 雙頭對齊、時間與振幅對稱演算法
├── head_shoulders_bottom/     # 頭肩底型態檢測器
│   ├── __init__.py
│   └── detector.py           # 七點結構與肩部對齊演算法
├── head_shoulders_top/        # 頭肩頂型態檢測器
│   ├── __init__.py
│   └── detector.py           # 七點結構與頭部最高演算法
└── cup_handle/               # 杯柄型態檢測器
    ├── __init__.py
    └── detector.py           # U型底圓弧判定與柄部回撤演算法
```

---

## 2. 三角收斂演算法 (`pattern/triangle/detector.py`)

採用 **ZigZag 波段轉折點 + 外軌包絡線 (Envelope Fitting) + 波動度收窄驗證 + 近現性優先 (Recency Weighting)** 幾何演算法：

### 核心識別邏輯：
1. **波段轉折點 (Pivot Finding)**：
   - 提取 Local Highs (Peaks) 與 Local Lows (Troughs)。
   - **強制交替驗證**：轉折點必須呈現 $Peak_1 \to Trough_1 \to Peak_2 \to Trough_2$ 波浪交替（總點數 $\ge 4$）。
2. **外軌包絡線 (Envelope Fitting)**：
   - **阻力線 (Upper Envelope)**：貼合波段高點 (Peaks) 外側，不被 K 線實體與影線穿透。
   - **支撐線 (Lower Envelope)**：貼合波段低點 (Troughs) 外側，不被 K 線實體與影線穿透。
3. **三種三角子型態 (Sub-types)**：
   - **對稱三角 (Symmetrical)**：上軌斜率 $< 0$ 且 下軌斜率 $> 0$。
   - **上升三角 (Ascending)**：上軌斜率 $\approx 0$ 且 下軌斜率 $> 0$。
   - **下降三角 (Descending)**：上軌斜率 $< 0$ 且 下軌斜率 $\approx 0$。
4. **波動度收窄驗證 (Volatility Squeeze)**：
   - 檢驗型態尾段的震盪幅度與首段的振幅比例 (`squeeze_ratio`)，要求尾段幅度必須顯著縮小 15%~50% 以上。
5. **品質與近現性評分 (Score 0~100)**：
   - 轉折點數量 (最高 25 分) + 波幅收窄強度 (最高 25 分) + 收斂進度與交點距離 (最高 25 分) + **近現性加權 (最高 25 分)**。

---

## 3. ABCD 上漲型態演算法 (`pattern/abcd_bull/detector.py`)

採用 **4 點轉折 (A-B-C-D) + 斐波那契比例 (Fibonacci Ratios) + 時間對稱性 + 近現性優先** 幾何演算法：

### 核心幾何與時間對稱性條件：
1. **$A$ 點 (Trough 起漲低點)**：$AB$ 上漲浪開端。
2. **$B$ 點 (Peak 第一波高點)**：$P_B > P_A$。
3. **$C$ 點 (Trough 修正拉回低點)**：**硬性要求 $P_C > P_A$ 且 $P_C < P_B$**（低點墊高，修正浪 $C$ 絕對不能跌破起漲點 $A$）。
4. **$D$ 點 (Peak 第二波目標/現價高點)**：**硬性要求 $P_D > P_B$ 且 $P_D > P_C$**（第二波攻擊浪 $CD$ 必須強勢突破前高 $B$）。
5. **時間對稱性 (Time Symmetry)**：**硬性要求 $CD$ 浪與 $AB$ 浪 K 線根數比例 $T_{CD} / T_{AB}$ 介於 $0.4 \sim 2.5$ 之間**。

### 斐波那契黃金比例驗證：
- **$BC$ 拉回比例**：$(P_B - P_C) / (P_B - P_A)$ 介於 $0.35 \sim 0.85$ 之間（最佳黃金比例為 **0.618**）。
- **$CD$ 展幅比例**：$(P_D - P_C) / (P_B - P_A)$ 介於 $0.75 \sim 1.8$ 之間（通常 $AB \approx CD$ 或 $CD = 1.272 \times AB$）。

### 品質與近現性評分 (Score 0~100)：
- 結合 $BC$ 黃金比例 0.618 (25 分) + $CD$ 展幅比 (25 分) + 時間對稱性 (15 分) + 最新突破力道 (10 分) + **$D$ 點近現性加權 (最高 25 分)**。

---

## 4. ABCD 下跌型態演算法 (`pattern/abcd_bear/detector.py`)

採用 **4 點轉折 (A-B-C-D) + 斐波那契比例 (Fibonacci Ratios) + 時間對稱性 + 近現性優先** 幾何演算法：

### 核心幾何與時間對稱性條件：
1. **$A$ 點 (Peak 起跌高點)**：$AB$ 下降浪開端。
2. **$B$ 點 (Trough 第一波低點)**：$P_B < P_A$。
3. **$C$ 點 (Peak 修正反彈高點)**：**硬性要求 $P_C < P_A$ 且 $P_C > P_B$**（高點降低，反彈浪 $C$ 絕對不能高於起跌點 $A$）。
4. **$D$ 點 (Trough 第二波目標/現價低點)**：**硬性要求 $P_D < P_B$ 且 $P_D < P_C$**（第二波跌勢浪 $CD$ 必須強勢跌破前低 $B$）。
5. **時間對稱性 (Time Symmetry)**：**硬性要求 $CD$ 浪與 $AB$ 浪 K 線根數比例 $T_{CD} / T_{AB}$ 介於 $0.4 \sim 2.5$ 之間**。

### 斐波那契黃金比例驗證：
- **$BC$ 反彈比例**：$(P_C - P_B) / (P_A - P_B)$ 介於 $0.35 \sim 0.85$ 之間（最佳黃金比例為 **0.618**）。
- **$CD$ 跌幅比例**：$(P_C - P_D) / (P_A - P_B)$ 限制在 $0.75 \sim 1.8$ 之間。

### 品質與近現性評分 (Score 0~100)：
- 結合 $BC$ 黃金比例 (25 分) + $CD$ 跌幅比 (25 分) + 時間對稱性 (15 分) + 最新跌破力道 (10 分) + **$D$ 點近現性加權 (最高 25 分)**。

---

## 5. W 底型態演算法 (`pattern/w_bottom/detector.py`)

採用 **四段全對稱 (Four-Segment Symmetry) + 近現性優先** 幾何演算法，確保型態在起跌、反彈、回測、突破四個階段的比例均衡：

### 核心幾何條件：
1. **四段線段定義**：
   - $S_1$ (起跌): $P_0 \to L_1$
   - $S_2$ (反彈): $L_1 \to H$
   - $S_3$ (回測): $H \to L_2$
   - $S_4$ (突破): $L_2 \to D$
2. **時間對稱性 (Time Symmetry)**：硬性要求四段所花的 K 線根數 $T_1, T_2, T_3, T_4$ 彼此比例最大差距不超過 4 倍。
3. **振幅對稱性 (Amplitude Symmetry)**：硬性要求四段的垂直價差 $A_1, A_2, A_3, A_4$ 彼此比例最大差距不超過 4 倍。
4. **雙底價格相近度**：$|P_{L1} - P_{L2}| / \min(P_{L1}, P_{L2}) \le 5\%$。
5. **頸線深度**：$12\%$ 為基準，介於 $3\% \sim 30\%$ 之間。

### 品質與近現性評分 (Score 0~100)：
- 時間全對稱 (15 分) + 振幅全對稱 (15 分) + 雙底對齊度 (20 分) + 頸線深度 (10 分) + 突破力道 (15 分) + **第二底 $L_2$ 近現性加權 (最高 25 分)**。

---

## 6. M 頭型態演算法 (`pattern/m_top/detector.py`)

採用 **四段全對稱 (Four-Segment Symmetry) + 近現性優先** 幾何演算法，確保型態在起漲、修正、反彈、跌破四個階段的比例均衡：

### 核心幾何條件：
1. **四段線段定義**：
   - $S_1$ (起漲): $L_0 \to H_1$
   - $S_2$ (修正): $H_1 \to L$
   - $S_3$ (反彈): $L \to H_2$
   - $S_4$ (跌破): $H_2 \to D$
2. **時間對稱性 (Time Symmetry)**：硬性要求四段所花的 K 線根數 $T_1, T_2, T_3, T_4$ 彼此比例最大差距不超過 4 倍。
3. **振幅對稱性 (Amplitude Symmetry)**：硬性要求四段的垂直價差 $A_1, A_2, A_3, A_4$ 彼此比例最大差距不超過 4 倍。
4. **雙頭價格相近度**：$|P_{H1} - P_{H2}| / \max(P_{H1}, P_{H2}) \le 5\%$。
5. **頸線深度**：$12\%$ 為基準，介於 $3\% \sim 30\%$ 之間。

### 品質與近現性評分 (Score 0~100)：
- 時間全對稱 (15 分) + 振幅全對稱 (15 分) + 雙頭對齊度 (20 分) + 頸線深度 (10 分) + 跌破力道 (15 分) + **第二頭 $H_2$ 近現性加權 (最高 25 分)**。

---

## 7. 頭肩底型態演算法 (`pattern/head_shoulders_bottom/detector.py`)

採用 **七點結構 (7-Point Structure) + 肩部價格對齊 + 近現性優先** 幾何演算法：

### 核心幾何條件：
1. **結構點定義**：$P_0 \to LS$ (左肩) $\to N_1$ (頸線1) $\to H$ (頭部) $\to N_2$ (頸線2) $\to RS$ (右肩) $\to D$ (結束點)。
2. **頭部最低原則**：頭部 $H$ 必須是整段型態中的最低價格。
3. **肩部價格相近度**：左肩 $LS$ 與右肩 $RS$ 的價格差距 $\le 10\%$。
4. **頸線平整度**：$N_1$ 與 $N_2$ 的價格差距 $\le 5\%$。
5. **時間對稱性**：要求各段時間比例最大差距不超過 4.5 倍。

### 品質與近現性評分 (Score 0~100)：
- 肩部對齊度 (20 分) + 頸線平整度 (10 分) + 頭部深度 (15 分) + 時間振幅對稱性 (15 分) + 突破力道 (15 分) + **右肩 $RS$ 近現性加權 (最高 25 分)**。

---

## 8. 頭肩頂型態演算法 (`pattern/head_shoulders_top/detector.py`)

採用 **七點結構 (7-Point Structure) + 頭部最高原則 + 近現性優先** 幾何演算法：

### 核心幾何條件：
1. **結構點定義**：$L_0 \to LS$ (左肩) $\to N_1$ (頸線1) $\to H$ (頭部) $\to N_2$ (頸線2) $\to RS$ (右肩) $\to D$ (跌破點)。
2. **頭部最高原則**：頭部 $H$ 必須是整段型態中的最高價格。
3. **肩部價格相近度**：左肩 $LS$ 與右肩 $RS$ 的價格差距 $\le 10\%$。
4. **頸線平整度**：$N_1$ 與 $N_2$ 的價格差距 $\le 5\%$（頸線用 `line_type="support"` 支撐線標示）。
5. **時間對稱性**：要求各段時間比例最大差距不超過 4.5 倍。

### 品質與近現性評分 (Score 0~100)：
- 肩部對齊度 (20 分) + 頸線平整度 (10 分) + 頭部高度 (15 分) + 時間振幅對稱性 (15 分) + 跌破力道 (15 分) + **右肩 $RS$ 近現性加權 (最高 25 分)**。

---

## 9. 杯柄型態演算法 (`pattern/cup_handle/detector.py`)

採用 **圓弧底判定 (U-Shape Detection) + 柄部回撤 (Handle Retracement) + 近現性優先** 演算法：

### 核心幾何條件：
1. **結構定義**：杯身起點 $P_1 \to$ 杯底 $T \to$ 杯緣 $P_2 \to$ 柄部底 $T_{handle} \to$ 結束點 $D$。
2. **U 型底圓潤度**：硬性要求杯身區間內，價格處於底部 $20\%$ 區間的 K 線根數佔比 $\ge 20\%$（區分尖銳 V 型底）。
3. **柄部回撤深度**：柄部回測深度介於杯身深度的 $5\% \sim 50\%$ 之間。
4. **柄部長度限制**：柄部整理時間必須顯著短於杯身長度（$< 80\%$）。
5. **杯緣平整度**：$P_1$ 與 $P_2$ 價格差距 $\le 8\%$。

### 品質與近現性評分 (Score 0~100)：
- U型底圓潤度 (25 分) + 柄部回撤優質度 (20 分) + 杯緣對稱性 (10 分) + 突破力道 (20 分) + **柄部 $T_{handle}$ 近現性加權 (最高 25 分)**。

---

## 10. API 端點規範 (`pattern/pattern_api.py`)

系統提供選股與繪圖端點，掛載於主 FastAPI (`api.py`) 的 `/api/pattern` 前綴下：

### 1. `GET /api/pattern/types`
- **用途**：取得可用的 8 種技術型態選單清單（含中文名稱）。
- **回傳內容**：
  ```json
  {
    "patterns": [
      { "id": "triangle", "name": "三角收斂" },
      { "id": "abcd_bull", "name": "ABCD 上漲" },
      { "id": "abcd_bear", "name": "ABCD 下跌" },
      { "id": "w_bottom", "name": "W底" },
      { "id": "m_top", "name": "M頭" },
      { "id": "head_shoulders_bottom", "name": "頭肩底" },
      { "id": "head_shoulders_top", "name": "頭肩頂" },
      { "id": "cup_handle", "name": "杯柄型態" }
    ]
  }
  ```

### 2. `GET /api/pattern/scan`
- **用途**：全市場型態選股過濾。
- **查詢參數**：
  - `pattern_type` (str): 型態種類，可選 `triangle`, `abcd_bull`, `abcd_bear`, `w_bottom`, `m_top`, `head_shoulders_bottom`, `head_shoulders_top`, `cup_handle`。
  - `timeframe` (str): 時間週期，可選 `1m`, `3m`, `5m`, `day`（預設 `day`）。
  - `date` (str, 選填): 基準日期 `YYYY-MM-DD`（預設最新交易日）。
  - `min_score` (float): 最低信心分數門檻，預設 `60.0`。
  - `min_vol_lots` (float, 選填): **日 K 10日均量過濾門檻 (張)**，預設 `1000.0` 張（設為 0 不限制）。
  - `limit` (int): K 線視窗根數，預設 `120` 根。
- **回傳內容**：符合條件的股票代號清單，包含 `pattern_type` (英文 ID)、`pattern_name` (中文名稱)、信心分數、10日均量張數 (`avg_vol_10d_lots`)、突破狀態 (`inside` / `breakout_up` / `breakout_down`)。

### 3. `GET /api/pattern/{stock_id}/detail`
- **用途**：取得單一股票的 K 線歷史數據與型態繪圖座標。
- **查詢參數**：`pattern_type`, `timeframe`, `date`, `limit`
- **回傳內容**：
  - `pattern_type` & `pattern_name`: 型態英文 ID 與中文名稱（例如 `"head_shoulders_top"` 與 `"頭肩頂"`）。
  - `candles`: K 線陣列，`time` 欄位已依 `CLAUDE.md` 規範轉換為台北本地時間 UTC Timestamp 秒數。
  - `pattern`:
    - `pattern_name`: 型態中文名稱。
    - `pivots`: 波段高低點轉折陣列 `[{ time, price, type: "peak"/"trough" }]`
    - `lines`: 上下軌趨勢線線段座標 `[{ start_time, start_price, end_time, end_price, slope, line_type: "resistance"/"support" }]`

### 4. `POST /api/pattern/cache/clear`
- **用途**：手動清空記憶體中的 Pattern 掃描與詳情快取。

---

## 11. 智慧快取機制 (In-Memory Caching)

為了避免全市場 2,700+ 支股票重複掃描運算，採用**雙軌智慧記憶體快取**：

- **快取 Key 結構**：
  `(pattern_type, timeframe, date, min_score, min_vol_lots, limit, latest_ts)`
- **自動失效與更新**：
  - 快取 Key 自動綁定 `get_latest_candle_timestamp()`（最新 K 線時間戳）。
  - **盤中時間**：每分鐘寫入新 1 分 K 線時，時間戳更新，快取自動失效並重算最新型態。
  - **盤後時間**：日 K 或盤後資料不變，快取持續生效，二次查詢時間由 1.9 秒降至 **< 50ms**（加速約 40 倍）。
- **參數隔離**：不同型態、週期或均量門檻的查詢條件會生成獨立 Key，互不干擾。
