import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# シグモイド関数
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# データを用意する------------------------------------------
# ロジスティック回帰のサンプルデータセットを生成する
n = 20                                                          # クラス毎のデータ数
df = pd.DataFrame()                                             # データフレーム初期化
for i in range(2):                                              # データ作成ループ
    x = pd.Series(np.random.uniform(0 + 2 * i, 3 + 2 * i, n))   # ランダムなx値を作成
    p = pd.Series(np.full(n, i))                                # 確率（発生するかしないか）を0か1で表現
    temp_df = pd.DataFrame(np.c_[x, p])                         # クラス毎のデータフレームを作成
    df = pd.concat([df, temp_df])                               # 作成されたクラス毎のデータを逐次結合
df.index = np.arange(0, len(df), 1)                             # index(行ラベル)を初期化
# プロット用分類(色分けするため)
class_0 = df[df[1] == 0]                                        # p=0データのみ抽出
class_1 = df[df[1] == 1]                                        # p=1データのみ抽出


# ----------------------------------------------------------
# トレーニングデータを.fitで使えるフォーマットにする
X = df[0].to_numpy().reshape(-1, 1)
Y = df[1].to_numpy()

# 正則化パラメータCを複数振ってそれぞれのモデルを生成するループ
param_C = [0.1, 1, 1000]                                        # 正則化パラメータリスト
p = [0] * len(param_C)                                          # 初期化確率pリスト
for j in range(len(param_C)):
    clf = LogisticRegression(C=param_C[j], solver='lbfgs')      # ロジスティック回帰モデルを定義
    clf.fit(X, Y)                                               # フィッティング

    # 学習済モデルを使って予測
    x_reg = np.arange(0, 6, 0.1)                                # 回帰式のx軸を作成
    y_reg = clf.predict(x_reg.reshape(-1, 1))                   # 予測
    w0 = clf.intercept_                                         # 偏回帰定数（切片w0）
    wi = clf.coef_                                              # 偏回帰係数ベクトル
    p[j] = sigmoid(w0 + wi[0] * x_reg)                          # 確率pを計算してリストに格納



# ここからグラフ描画----------------------------------------
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフオブジェクトを定義する。
fig = plt.figure()
ax1 = plt.subplot(111)

# グラフの上下左右に目盛線を付ける。
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax1.set_xlabel('x')
ax1.set_ylabel('p')

# データプロットする。
n = 0
for k in range(len(param_C)):
    ax1.plot(x_reg, p[k], label='Possibility (C=' + str(param_C[k]) + ')')
ax1.scatter(class_0[0], class_0[1], label='class=0', edgecolors='black')
ax1.scatter(class_1[0], class_1[1], label='class=1', edgecolors='black')
plt.legend()

# グラフを表示する。
plt.show()
plt.close()
# ----------------------------------------------------------