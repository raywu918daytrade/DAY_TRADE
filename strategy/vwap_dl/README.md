# vwap_dl — ResNet + GRU 混合模型（VWAP z-score + 大盤 VWAP 特徵）

## 模型架構
- **ResNet**：看近 10 分鐘原始 OHLCV（5 channels × 10 步）
- **GRU**：從 9:00 到當下逐分鐘看 18 維特徵（含 VWAP z-score + 大盤 VWAP 特徵）
- **Concat → MLP → 3 分類**（回歸=0 / 無訊號=1 / 延續=2）

## 特徵演進
| 日期 | 版本 | GRU 維度 | 說明 |
|:----:|:----:|:--------:|------|
| 2026-07-27 | v1 | 14 | OHLCV(5) + 技術指標(6) + VWAP z-score(3) |
| 2026-07-28 | v2 | **18** | + market_z_score_m5 / market_vwap_alignment_score / market_vwap_spread_1_5 / velocity_ratio_to_market |

## 訓練設定
- 資料範圍：2024-01-01 ~ 2026-07-25
- 測試集：最後 30 天（4,273 筆）
- 訓練集：92,446 筆
- patience=8 / max_rounds=30 / lr=5e-4 / batch_size=256
- class_weight balanced

## evaluate
```
測試集: 4,273 筆（最後 30 天）
Accuracy: 0.4908
AUC (one-vs-rest): 0.6810

混淆矩陣（列=實際，欄=預測，順序 回歸/無訊號/延續）:
[[ 981  908  343]
 [ 467 1003  350]
 [  52   56  113]]

分類報告:
              precision    recall  f1-score   support

          回歸       0.65      0.44      0.53      2232
         無訊號       0.51      0.55      0.53      1820
          延續       0.14      0.51      0.22       221

    accuracy                           0.49      4273
   macro avg       0.43      0.50      0.43      4273
weighted avg       0.57      0.49      0.51      4273
```

## confidence
```
測試集: 4,273 筆

── argmax（無門檻），判斷覆蓋率(非無訊號)=0.5397 ──
Accuracy: 0.4908
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[ 981  908  343]
 [ 467 1003  350]
 [  52   56  113]]
              precision    recall  f1-score   support

          回歸       0.65      0.44      0.53      2232
         無訊號       0.51      0.55      0.53      1820
          延續       0.14      0.51      0.22       221

    accuracy                           0.49      4273
   macro avg       0.43      0.50      0.43      4273
weighted avg       0.57      0.49      0.51      4273


── 信心度門檻=0.40，判斷覆蓋率(非無訊號)=0.4784 ──
Accuracy: 0.4863
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[ 896 1067  269]
 [ 416 1081  323]
 [  39   81  101]]
              precision    recall  f1-score   support

          回歸       0.66      0.40      0.50      2232
         無訊號       0.48      0.59      0.53      1820
          延續       0.15      0.46      0.22       221

    accuracy                           0.49      4273
   macro avg       0.43      0.48      0.42      4273
weighted avg       0.56      0.49      0.50      4273


── 信心度門檻=0.50，判斷覆蓋率(非無訊號)=0.2425 ──
Accuracy: 0.4961
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[ 538 1578  116]
 [ 139 1518  163]
 [  16  141   64]]
              precision    recall  f1-score   support

          回歸       0.78      0.24      0.37      2232
         無訊號       0.47      0.83      0.60      1820
          延續       0.19      0.29      0.23       221

    accuracy                           0.50      4273
   macro avg       0.48      0.45      0.40      4273
weighted avg       0.61      0.50      0.46      4273


── 信心度門檻=0.60，判斷覆蓋率(非無訊號)=0.1194 ──
Accuracy: 0.4793
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[ 300 1885   47]
 [  42 1707   71]
 [   9  171   41]]
              precision    recall  f1-score   support

          回歸       0.85      0.13      0.23      2232
         無訊號       0.45      0.94      0.61      1820
          延續       0.26      0.19      0.22       221

    accuracy                           0.48      4273
   macro avg       0.52      0.42      0.35      4273
weighted avg       0.65      0.48      0.39      4273


── 信心度門檻=0.70，判斷覆蓋率(非無訊號)=0.0578 ──
Accuracy: 0.4669
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[ 173 2039   20]
 [   7 1797   16]
 [   6  190   25]]
              precision    recall  f1-score   support

          回歸       0.93      0.08      0.14      2232
         無訊號       0.45      0.99      0.61      1820
          延續       0.41      0.11      0.18       221

    accuracy                           0.47      4273
   macro avg       0.60      0.39      0.31      4273
weighted avg       0.70      0.47      0.35      4273


── 信心度門檻=0.80，判斷覆蓋率(非無訊號)=0.0211 ──
Accuracy: 0.4442
AUC (OvR): 0.6810
  AUC(回歸): 0.6542
  AUC(無訊號): 0.6343
  AUC(延續): 0.7543
[[  69 2161    2]
 [   0 1815    5]
 [   0  207   14]]
              precision    recall  f1-score   support

          回歸       1.00      0.03      0.06      2232
         無訊號       0.43      1.00      0.60      1820
          延續       0.67      0.06      0.12       221

    accuracy                           0.44      4273
   macro avg       0.70      0.36      0.26      4273
weighted avg       0.74      0.44      0.29      4273