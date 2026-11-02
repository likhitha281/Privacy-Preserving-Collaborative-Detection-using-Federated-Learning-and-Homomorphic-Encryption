# Split report (ieee, 3 banks, label_skew, alpha=0.5, seed=42)

Rows: 590,540. Frauds: 20,663 (3.50%). Time split: train 70%, validation 10%, global test 20% (latest transactions by TransactionDT).

| split | rows | frauds | fraud rate |
|---|---|---|---|
| train | 413,378 | 14,538 | 3.52% |
| val | 59,054 | 2,061 | 3.49% |
| test | 118,108 | 4,064 | 3.44% |

## Class balance per bank

| bank | split | rows | frauds | fraud rate | cards |
|---|---|---|---|---|---|
| 0 | train | 208,462 | 6,280 | 3.01% | 8,820 |
| 0 | val | 29,404 | 772 | 2.63% | 3,640 |
| 0 | test | 58,555 | 1,754 | 3.00% | 4,930 |
| 1 | train | 42,389 | 1,105 | 2.61% | 1,457 |
| 1 | val | 6,450 | 166 | 2.57% | 638 |
| 1 | test | 11,531 | 332 | 2.88% | 869 |
| 2 | train | 162,527 | 7,153 | 4.40% | 1,965 |
| 2 | val | 23,200 | 1,123 | 4.84% | 1,037 |
| 2 | test | 48,022 | 1,978 | 4.12% | 1,281 |

## Feature views (columns)

- txn: 95
- beh: 10
- idn: 109
