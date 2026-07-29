有# 破底翻 Feature Interaction 分析報告
**資料期間**: 2026-06-01 ~ 2026-06-15
**總樣本數**: 2,469 筆

## Break Minutes × Recovery Minutes
![Break Minutes × Recovery Minutes](interaction_heatmaps/break_minutes_x_recovery_minutes_heatmap.png)

### Win Rate Pivot Table
| _bin_a        |   (0.999, 2.0] |   (2.0, 4.0] |   (4.0, 11.0] |   (11.0, 34.0] |   (34.0, 260.0] |
|:--------------|---------------:|-------------:|--------------:|---------------:|----------------:|
| (-0.001, 1.0] |           53.2 |         43.2 |          49.6 |           70   |            12   |
| (1.0, 4.0]    |           51   |         68.1 |          67.6 |           69.3 |            15.5 |
| (4.0, 7.0]    |           70.7 |         83.1 |          83.8 |           75.6 |            11.1 |
| (7.0, 13.0]   |           70.8 |         74.6 |          69.3 |           71.2 |            11.3 |
| (13.0, 19.0]  |           72.1 |         75.6 |          72.4 |           73.7 |            12.1 |

## Rolling Return × Recovery Minutes
![Rolling Return × Recovery Minutes](interaction_heatmaps/rolling_return_after_x_recovery_minutes_heatmap.png)

### Win Rate Pivot Table
| _bin_a           |   (0.999, 2.0] |   (2.0, 4.0] |   (4.0, 11.0] |   (11.0, 34.0] |   (34.0, 260.0] |
|:-----------------|---------------:|-------------:|--------------:|---------------:|----------------:|
| (-4.089, -0.174] |           47.2 |         55.6 |          65.3 |           66   |             5.2 |
| (-0.174, 0.0]    |           68.5 |         76.3 |          67.7 |           73.4 |            14.6 |
| (0.0, 0.309]     |           49   |         54.7 |          63.9 |           87.5 |            10.5 |
| (0.309, 5.385]   |           68.4 |         70.1 |          65   |           70   |            33.3 |

## Opening Range × Break Minutes
![Opening Range × Break Minutes](interaction_heatmaps/opening_range_x_break_minutes_heatmap.png)

### Win Rate Pivot Table
| _bin_a         |   (-0.001, 1.0] |   (1.0, 4.0] |   (4.0, 7.0] |   (7.0, 13.0] |   (13.0, 19.0] |
|:---------------|----------------:|-------------:|-------------:|--------------:|---------------:|
| (-0.001, 0.07] |            52.5 |         60   |         59.3 |          51.3 |           62.7 |
| (0.07, 0.302]  |            42.8 |         48.3 |         65.9 |          65.3 |           76.8 |
| (0.302, 0.85]  |            54.3 |         52.6 |         65.3 |          64.8 |           63.3 |
| (0.85, 2.5]    |            43   |         44.3 |         70.1 |          54.4 |           53.5 |
| (2.5, 660.0]   |            36.3 |         45.6 |         59.7 |          67.9 |           68.7 |

## Opening Volume × Rolling Return
![Opening Volume × Rolling Return](interaction_heatmaps/opening_volume_x_rolling_return_after_heatmap.png)

### Win Rate Pivot Table
| _bin_a            |   (-4.089, -0.174] |   (-0.174, 0.0] |   (0.0, 0.309] |   (0.309, 5.385] |
|:------------------|-------------------:|----------------:|---------------:|-----------------:|
| (0.999, 8.0]      |               34.4 |            51.7 |           58.8 |             65.7 |
| (8.0, 33.0]       |               42.4 |            56.8 |           50   |             74   |
| (33.0, 118.0]     |               43.9 |            59.8 |           48.5 |             72.2 |
| (118.0, 543.2]    |               41.7 |            58.7 |           57.9 |             59.8 |
| (543.2, 138001.0] |               53   |            57.7 |           54.9 |             69.8 |

## Break % × Recovery Minutes
![Break % × Recovery Minutes](interaction_heatmaps/break_pct_x_recovery_minutes_heatmap.png)

### Win Rate Pivot Table
| _bin_a                |   (0.999, 2.0] |   (2.0, 4.0] |   (4.0, 11.0] |   (11.0, 34.0] |   (34.0, 260.0] |
|:----------------------|---------------:|-------------:|--------------:|---------------:|----------------:|
| (-0.0558, -0.00537]   |           71.2 |         70.9 |          83.3 |           85.2 |            27.1 |
| (-0.00537, -0.00365]  |           62.6 |         71   |          64.6 |           67   |             9.5 |
| (-0.00365, -0.0024]   |           65.9 |         68.9 |          70.7 |           66.1 |             2.2 |
| (-0.0024, -0.00157]   |           57.6 |         58.1 |          64   |           77.1 |             0   |
| (-0.00157, -0.000166] |           60.8 |         67.4 |          52.6 |           64   |             0   |

## Opening Range × Rolling Volume
![Opening Range × Rolling Volume](interaction_heatmaps/opening_range_x_rolling_volume_after_heatmap.png)

### Win Rate Pivot Table
| _bin_a         |   (-0.001, 1.0] |   (1.0, 7.0] |   (7.0, 23.0] |   (23.0, 109.4] |   (109.4, 26770.0] |
|:---------------|----------------:|-------------:|--------------:|----------------:|-------------------:|
| (-0.001, 0.07] |            48.2 |         63.9 |          66.7 |            68.4 |               63.9 |
| (0.07, 0.302]  |            55   |         59.3 |          54.1 |            57.4 |               63.9 |
| (0.302, 0.85]  |            61.6 |         54.8 |          57.8 |            65.2 |               60.5 |
| (0.85, 2.5]    |            51.1 |         49.3 |          58.1 |            47.1 |               55   |
| (2.5, 660.0]   |            52   |         47.2 |          50.6 |            52.6 |               55.8 |

## Top 20 Feature Pairs (by Win Rate)
| feature_a            | feature_b            |   win_rate |   count |
|:---------------------|:---------------------|-----------:|--------:|
| mfe_pct              | mae_pct              |    100     |      54 |
| rolling_return_after | mae_pct              |    100     |     197 |
| opening_volume       | mae_pct              |    100     |     105 |
| recovery_minutes     | mae_pct              |    100     |     106 |
| break_pct            | mae_pct              |    100     |      93 |
| opening_range        | mae_pct              |    100     |      94 |
| break_minutes        | mae_pct              |    100     |      82 |
| rolling_volume_after | mae_pct              |    100     |     131 |
| recovery_minutes     | mfe_pct              |     98.529 |      68 |
| opening_range        | mfe_pct              |     98.148 |      54 |
| opening_volume       | mfe_pct              |     96.875 |      64 |
| rolling_volume_after | mfe_pct              |     96.721 |      61 |
| rolling_return_after | mfe_pct              |     96.429 |      84 |
| break_minutes        | mfe_pct              |     95.726 |     117 |
| break_pct            | mfe_pct              |     90.698 |      86 |
| break_minutes        | rolling_return_after |     86.458 |      96 |
| break_pct            | recovery_minutes     |     85.185 |     108 |
| break_minutes        | recovery_minutes     |     83.75  |      80 |
| break_minutes        | opening_volume       |     79.808 |     104 |
| recovery_minutes     | opening_range        |     78.226 |     124 |

