# LightGBM vs XGBoost

## feature set medium, 1/4 training examples, no gpu

|         | LGBM | XGB   |
|---------|------|-------|
| train   | 68.1 | 569.5 |
| predict | 11.1 | 4.3   |

## feature set medium, 1/4 training examples, with gpu

|         | LGBM | XGB   |
|---------|------|-------|
| train   | 51.8 | 196.6 |
| predict | 12.0 | 4.2   |

## feature set large, 1/4 training examples, with gpu

|         | LGBM  | XGB   |
|---------|-------|-------|
| train   | 100.8 | 495.9 |
| predict | 18.5  | 9.3   |