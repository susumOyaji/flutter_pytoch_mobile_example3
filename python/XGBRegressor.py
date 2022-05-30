# 乱数シードの固定
from optuna.integration import lightgbm as lgb
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import random as rn

np.random.seed(123)
rn.seed(123)

# ライブラリのインポート


# iris（アヤメ）データインポート
iris = load_iris()

# 訓練用データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=123)

#LGB用のデータに変形
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test)

# 訓練方法の定義
params = {
    'objective': 'multiclass ',
    'num_class': 3,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'deterministic': True,  # 再現性確保用のパラメータ
    'force_row_wise': True  # 再現性確保用のパラメータ
}

booster = lgb.LightGBMTuner(
    params=params,
    train_set=lgb_train,
    valid_sets=lgb_eval,
    optuna_seed=123,  # 再現性確保用のパラメータ
)

# 訓練の実施
booster.run()

# 訓練で得た最良のパラメータを表示
booster.best_params

# 訓練で得た最良のモデル（Boosterオブジェクト）を取得する
best_booster = booster.get_best_booster()

# テストデータに対して予測を実施
pred = best_booster.predict(x_test)
pred = np.argmax(pred, axis=1)

# 混同行列（Confusion Matrix）の表示
cm = confusion_matrix(y_test, pred)
print(cm)
