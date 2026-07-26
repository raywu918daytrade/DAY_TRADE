"""
vwap_dl — VWAP 偏離策略的 CNN + GRU 深度學習版本。

基於 vwap_ml 相同的 VWAP z-score 候選觸發邏輯（三個時間框 m1/m3/m5，任一
|z|>=STD_MULT 即觸發），但改用 CNN + GRU 網路從原始報價視窗中自行學習
特徵，取代 LightGBM 手工特徵。

三分類標籤沿用 vwap_ml：0=回歸VWAP／1=無訊號／2=延續突破。
"""
