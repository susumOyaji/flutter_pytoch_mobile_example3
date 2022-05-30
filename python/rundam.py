from sklearn.ensemble import RandomForestRegressor # ランダムフォレスト回帰用
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト用
from sklearn.tree import DecisionTreeClassifier             # 決定木用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import visualize_tree


from sklearn.datasets import make_blobs # ダミーデータの生成用
X, y = make_blobs(n_samples=500, centers=4, random_state=8, cluster_std=2.4)
# n_samples:サンプル数 centers:中心点の数 random_state:seed値 cluster_std: ばらつき度合い

plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='jet')


clf = DecisionTreeClassifier(max_depth=2, random_state=0)  # インスタンス作成 max_depth:木の深さ
#visualize_tree(clf, X, y)    # 描画実行


# インスタンス作成　n_estimators:作る決定木の数の指定
clf = RandomForestClassifier(n_estimators=100, random_state=0)
#visualize_tree(clf, X, y, boundaries=False)


'''
ランダムフォレストで回帰をやってみる
ランダムフォレストは回帰もできる。
sinを使って、大きな波の中で小さな波が動いてるようなデータを用意
'''
x = 10 * np.random.rand(100)


def sin_model(x, sigma=0.2):
    """大きな波＋小さな波＋ノイズからなるダミーデータ。"""
    noise = sigma * np.random.randn(len(x))

    return np.sin(5 * x) + np.sin(0.5 * x) + noise


# xからyを計算
y = sin_model(x)

# Plotしてみる。
plt.figure(figsize=(16, 8))
plt.errorbar(x, y, 0.1, fmt='o')






'''sklearnで実行'''
# 確認用に0〜10の1000個のデータを用意
xfit = np.linspace(0, 10, 1000)  # 0〜10まで1000個

# ランダムフォレスト実行
rfr = RandomForestRegressor(100)  # インスタンスの生成　木の数を100個に指定
rfr.fit(x[:, None], y)            # 学習実行
yfit = rfr.predict(xfit[:, None])  # 予測実行

# 結果比較用に実際の値を取得。
ytrue = sin_model(xfit, 0)  # xfitを波発生関数に食わせて、その結果を取得

# 結果確認
plt.figure(figsize=(16, 8))
plt.errorbar(x, y, 0.1, fmt='o')
plt.plot(xfit, yfit, '-r')                # 予測値のplot
plt.plot(xfit, ytrue, '-k', alpha=0.5)  # 正解値のplot
