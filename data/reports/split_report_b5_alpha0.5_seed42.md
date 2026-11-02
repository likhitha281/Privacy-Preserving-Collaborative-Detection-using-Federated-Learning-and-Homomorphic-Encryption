# Split report (ieee, 5 banks, label_skew, alpha=0.5, seed=42)

Rows: 590,540. Frauds: 20,663 (3.50%). Time split: train 70%, validation 10%, global test 20% (latest transactions by TransactionDT).

| split | rows | frauds | fraud rate |
|---|---|---|---|
| train | 413,378 | 14,538 | 3.52% |
| val | 59,054 | 2,061 | 3.49% |
| test | 118,108 | 4,064 | 3.44% |

## Class balance per bank

| bank | split | rows | frauds | fraud rate | cards |
|---|---|---|---|---|---|
| 0 | train | 120,421 | 4,921 | 4.09% | 970 |
| 0 | val | 17,367 | 627 | 3.61% | 560 |
| 0 | test | 34,450 | 1,309 | 3.80% | 686 |
| 1 | train | 38,607 | 763 | 1.98% | 3,068 |
| 1 | val | 5,524 | 125 | 2.26% | 1,234 |
| 1 | test | 11,061 | 174 | 1.57% | 1,678 |
| 2 | train | 124,196 | 5,487 | 4.42% | 1,043 |
| 2 | val | 17,460 | 787 | 4.51% | 617 |
| 2 | test | 33,861 | 1,667 | 4.92% | 765 |
| 3 | train | 64,303 | 2,566 | 3.99% | 1,412 |
| 3 | val | 9,529 | 415 | 4.36% | 683 |
| 3 | test | 20,218 | 741 | 3.67% | 856 |
| 4 | train | 65,851 | 801 | 1.22% | 5,749 |
| 4 | val | 9,174 | 107 | 1.17% | 2,221 |
| 4 | test | 18,518 | 173 | 0.93% | 3,095 |

## Feature views (columns)

- txn: 95
- beh: 10
- idn: 109
