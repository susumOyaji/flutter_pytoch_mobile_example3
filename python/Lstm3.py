# 警告を無視する
from keras.layers import RepeatVector
import warnings
warnings.filterwarnings('ignore')

# ライブラリを読み込む
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
import random
import datetime

# tensorflowのライブラリを読み込む
import tensorflow as tf

# 乱数シードを固定する
tf.random.set_seed(1234)

# kerasのライブラリを読み込む
from keras.models import Sequential
from keras.layers import Dense, LSTM

# scikit-learnの正規を行うライブラリを読み込む
from sklearn.preprocessing import MinMaxScaler

# scikit-learnで決定係数とRMSEの計算を行うライブラリを読み込む
from sklearn.metrics import r2_score, mean_squared_error

# グラフ表示のライブラリとグラフ表示で日本語を表示するためのライブラリを読み込む
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import japanize_matplotlib
from matplotlib import dates as mdates


# トヨタ(証券コード：7203)の株価を取得する
df = web.DataReader('7203.JP', data_source='stooq', start='2020-01-01', end='2021-12-31')

# 取得した株価を日付で昇順に変換する
df = df[::-1]

# csvファイルとして保存する
df.to_csv('toyota_7203.csv', sep=',')

# csvファイルを読み込む
df = pd.read_csv('toyota_7203.csv', sep=',')

# 取得した株価データから予測に使用する列を抽出する
#data = df.filter(['Open', 'High', 'Low', 'Close', 'Volume'])
# data = df.filter(['Open', 'High', 'Close'])
data = df.filter(['Open', 'High', 'Volume'])
#data = df.filter(['Open', 'High', 'Close', 'Volume'])

# 日付を抽出する
data_days = df.iloc[:, 0]
data_days = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in data_days]

# 開始値を抽出する
data_open = df.iloc[:, 1]

# 抽出した株価データをdatasetに代入する
dataset = data.values

# 取得した株価データの8割を訓練データとする
# math.ceil : 小数点以下を切り上げ
# training_data_lenは194となる
training_data_len = math.ceil(len(dataset) * .8)

# 最小値:5839->0、最大値:8014->1となるように正規化する
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 正規化したデータから訓練で使用する行数分のデータを抽出する
# 0番から193番までが訓練で使用するデータとなる
train_data = scaled_data[0:training_data_len, :]

# 訓練データと正解データを保存する配列を用意する
# x_train:訓練データ
# y_train:正解データ
x_train = []
y_train = []

# 訓練データとして60日分のデータをx_train[]に追加する
# 正解データとして61日目のデータをy_train[]に追加する
for i in range(60, len(train_data)):
    xset = []
    for j in range(train_data.shape[1]):
        a = train_data[i - 60:i, j]
        xset.append(a)
    x_train.append(xset)
    y_train.append(train_data[i, 0])

# 訓練データと教師データをNumpy配列に変換する
x_train, y_train = np.array(x_train), np.array(y_train)

# 訓練データのNumpy配列について、奥行を訓練データの数、行を60日分のデータ、列を抽出した株価データの種類数、の3次元に変換する
x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# LSTMモデルを定義する
n_hidden = 60       # 隠れ層の数
units1 = 25         # 第1層の出力数
units2 = 1          # 第2層の出力数

model = Sequential()
model.add(LSTM(n_hidden, activation='tanh', return_sequences=True,
          input_shape=(x_train_3D.shape[1], 60)))
model.add(LSTM(n_hidden, return_sequences=False))
model.add(Dense(units1))
model.add(Dense(units2))


'''
#LSTMモデルを定義する
_hidden = 60       # 隠れ層の数
units1 = 25         # 第1層の出力数
units2 = 1          # 第2層の出力数
m = Sequential()
# 入力データ数が24なので、input_shapeの値が(24,1)です。
m.add(LSTM(100, activation='relu', input_shape=(24, 1)))

# 予測範囲は12ステップなので、RepeatVectoorに12を指定する必要があります。
m.add(RepeatVector(12))

m.add(LSTM(100, activation='relu', return_sequences=True))
m.add(TimeDistributed(Dense(1)))


m.compile(optimizer='adam', loss='mse')
'''


from keras.layers import TimeDistributed
# LSTMモデルを定義する
a_n_hidden = 60       # 隠れ層の数
a_units1 = 25         # 第1層の出力数
a_units2 = 1          # 第2層の出力数

a_model = Sequential()
# 入力データ数が60なので、input_shapeの値が(60,1)です。
a_model.add(LSTM(n_hidden, activation='relu', return_sequences=True,input_shape=(x_train_3D.shape[1], 60)))

# 予測範囲は12ステップなので、RepeatVectoorに12を指定する必要があります。
a_model.add(RepeatVector(12))

a_model.add(LSTM(100, activation='relu', return_sequences=True))
a_model.add(TimeDistributed(Dense(1)))
a_model.add(Dense(units2))








# 定義したLSTMモデルをコンパイルする
# 最適化手法:adam
# 損失関数:最小2乗誤差
model.compile(optimizer='adam', loss='mean_squared_error')

# コンパイルしたモデルの学習を行う
batch_size = 1      # バッチサイズ
epochs = 10         # 訓練の回数
model.fit(x_train_3D, y_train, batch_size=batch_size, epochs=epochs)

# 検証データを用意する
# 194番から241番までをテストデータとする
# 最初の194番をテストするためには、134番～193番の終値の株価データが必要となる
# 検証データの最初の番は、訓練データの最後から60を引いた134番となる
# 検証データの総番数は108となる
test_data = scaled_data[training_data_len - 60:, :]

# x_test:検証データ
# y_test:正解データ
x_test = []
y_test = []
# y_test = scaled_data[training_data_len:, :]

# 検証データをセットする
for i in range(60, len(test_data)):
    xset = []
    for j in range(test_data.shape[1]):
        a = test_data[i - 60:i, j]
        xset.append(a)
    x_test.append(xset)
    y_test.append(test_data[i, 0])

# 検証データをNumpy配列に変換する
x_test = np.array(x_test)

# 検証データのNumpy配列について、奥行を訓練データの数、行を60日分のデータ、列を抽出した株価データの種類数、の3次元に変換する
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# モデルに検証データを代入して予測を行う
predictions = model.predict(x_test)

# モデルの精度を評価する
# 決定係数とRMSEを計算する
# 決定係数は1.0に、RMSEは0.0に近いほど、モデルの精度は高い
r2_score = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'r2_score: {r2_score:.4f}')
print(f'rmse: {rmse:.4f}')

# 作業用のnumpy配列を用意する
temp_column = np.zeros(dataset.shape[1] - 1)

# 正規化する前にdatasetと同じnumpy形式に変換する
def padding_array(val):
    xset = []
    for x in val:
        a = np.insert(temp_column, 0, x)
        xset.append(a)

    xset = np.array(xset)
    return xset

# 予測データは正規化されているので、元の株価に戻す
predictions = scaler.inverse_transform(padding_array(predictions))

# 実際の価格、予測の価格をグラフで表示する

# 訓練期間と検証期間を抽出する
data_days_train = data_days[:training_data_len]
data_days_valid = data_days[training_data_len:]

# グラフを表示する領域をfigとする
fig = plt.figure(figsize=(12, 6))

# グラフ間の余白を設定する
fig.subplots_adjust(wspace=0.6, hspace=0.2)

# GridSpecでfigを縦10、横15に分割する
gs = gridspec.GridSpec(9, 14)

# 分割した領域のどこを使用するかを設定する
# gs[a1:a2, b1:b2]は、縦の開始位置(a1)から終了位置(a2)、横の開始位置(b1)から終了位置(b2)
ax1 = plt.subplot(gs[0:8, 0:8])
ax2 = plt.subplot(gs[0:5, 9:14])

# 1番目のグラフを設定する
ax1.set_title('開始値の履歴と予測結果', fontsize=16)
ax1.set_xlabel('日付', fontsize=12)
ax1.set_ylabel('開始値 円', fontsize=12)
ax1.plot(data_days, data_open)
ax1.plot(data_days_valid, predictions[:, 0])
ax1.legend(['実際の価格', '予測の価格'], loc='lower right')
ax1.grid()

# 1番目のx軸ラベルの表示ルールを設定する
data_0_mdates = mdates.MonthLocator([1, 4, 7, 10])
data_0_mdates_fmt = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_major_locator(data_0_mdates)
ax1.xaxis.set_major_formatter(data_0_mdates_fmt)

# 2番目のグラフを設定する
ax2.set_title('予測の価格と実際の価格の散布図表示', fontsize=16)
ax2.set_xlabel('予測の価格', fontsize=12)
ax2.set_ylabel('実際の価格', fontsize=12)
ax2.scatter(data_open[training_data_len:], predictions[:, 0], label=f'r2_score: {r2_score:.4f} \n rmse: {rmse:.4f}')
ax2.plot(data_open[training_data_len:], data_open[training_data_len:], 'k-')
ax2.legend()
ax2.grid()

fig.savefig('img.png')
plt.show()