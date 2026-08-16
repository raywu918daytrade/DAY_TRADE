# ib_margin_intraday_dir

D-1 已公布的法人買超／融資券增減（相對自己近 20 日），能不能分開 **今日 09:05→10:00**（以及 09:05→收）。小量統計，不做模型。

## 定義摘要

| 項目 | 規則 |
|--|--|
| 母體 | **交集**：D-1 有 ib **且** 有 margin **且** D 有 m5 09:05+10:00。約 900 檔這一層，不是全市場。`--tick_only` 再 ∩ 當沖 400 |
| 特徵日 | 一律 **D-1**（當天法人/資券收盤後才齊，不能用來預測當天盤中） |
| 法人 | `(買-賣)/當日量` → 自己 20 日 PR。外資／投信／自營（self+hedging+foreign dealer）分欄 |
| 融資券 | 餘額增減％、券資比 → 同樣 20 日 PR |
| 主 label | `close@10:00 / close@09:05 - 1` |
| 次 label | 當日 `close / close@09:05 - 1` |
| 分桶 | lt20 / 20to50 / 50to80 / ge80；高低桶 Mann-Whitney + Spearman |
| 對照 | 同股打亂 `foreign_pr` 日期 |
| 融券×股漲 | D-1 融券增，且收漲 **≥ 0.5×自己 ATR**（濾掉不漲不跌）；label 另報 |ret|<0.3% 的平% |
| VWAP+SR | `verify.py` **沒有**過濾穿越。要接到 dashboard 那套收盤變號，用 `sr_overlay.py` |

## 跑法

```bash
python -m strategy_test.ib_margin_intraday_dir.verify \
    --start_date 2025-01-01 --end_date 2026-08-14

# dashboard「VWAP+壓力支撐」穿越 × 同一套 D-1 特徵（訊號價→10:00／收）
python -m strategy_test.ib_margin_intraday_dir.sr_overlay \
    --start_date 2025-01-01 --end_date 2026-08-14
```
