'''
アップデート必要なパッケージのリスト(これは便利)
pip list -o

pipのパッケージを一括アップデートする方法
pip3 install --upgrade ((pip3 freeze) -replace '==.+','')
'''
# kerasとその他ライブラリをインポート
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
#from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.models import Model
from keras.layers import Input, LSTM


# SVGの表示に必要なライブラリをインポート
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import os
os.environ["PATH"] += os.pathsep +'C:/Program Files/Graphviz/bin/'

import pandas as pd # Pandas（DataFrameデータ構造やデータ解析ツールを提供するPythonライブラリ）のインポート
from pandas import DataFrame # pandasモジュールのクラスDataFrameをインポート
import numpy as np # numpyモジュールをnpで使用できるようにインポート
import matplotlib.pyplot as plt #matplotlibをpltで使えるようにインポート
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from datetime import datetime as dt # datetime型
from sklearn.preprocessing import StandardScaler # 標準化用にscikit-learnの前処理ライブラリを読み込み
import matplotlib
import sys, os
#os.getcwd() #カレントディレクトリの表示


#df_NikkeiAll = pd.read_csv("^N225_20100711-20200711.csv")  # 日経平均のデータの読み込み
'''データの前処理'''
'''使うデータを読み込む。'''
'''学習データ'''

from pandas_datareader import data as pdr
code = '6976'#'6758'
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
# (2021, 1, 1)  # 教師データ(今までのデータ)
start_train = datetime.date.today() + relativedelta(days=-700)
# 昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_train = datetime.date.today() + relativedelta(days=-1)

data = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データを読み込む。
Nikkei_df = pdr.get_data_yahoo('^N225', start_train, end_train) #Nikkei 教師データを読み込む。

#データの前処理
#欠損データがあるので、欠損値NaNを除外する
#df_NikkeiAll_drop = df_NikkeiAll.dropna()

#df_NikkeiAll_drop.head() # 先頭の5行を表形式で表示
data.head()

#datetime64[ns]型に変換した列をpandas.DataFrameに新たな列として追加
#df_NikkeiAll_drop['datetime'] = pd.to_datetime(df_NikkeiAll_drop['Date'], format='%Y-%m-%d')


# datetimeの列のデータのみを取り出し
data = data.reset_index(drop=False)
CodeDate = data['Date'].values

# Closeの列のデータのみを取り出し
data['NikkiClose'] = Nikkei_df['Close'].values

# カラムの並べ替え
df = data[['Date', 'High', 'Low', 'Open', 'Close']]

# Closeの列のデータのみを取り出し
CodeData = df['Close'].values


#リシェイプ
CodeData = CodeData.reshape(-1,1) # float64
CodeDate = CodeDate.reshape(-1,1) # datetime64[ns]


# 読み込んだ日経平均をプロット
k = 700 # 表示する数
i = CodeData.shape[0]-k
j = CodeData.shape[0]
xdata = CodeDate[i:j]
ydata = CodeData[i:j]



'''
ax.plot(xdata, ydata)

# 軸目盛の設定
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval= 100, tz=None))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# ラベルを45deg回転
labels = ax.get_xticklabels()
plt.setp(labels, rotation=70, fontsize=10)

#ラベル名
plt.xlabel('Date')
plt.ylabel('SONY')

plt.grid()
plt.show()
'''


# 特徴量の尺度を揃える、特徴データを標準化して配列に入れる
scaler = StandardScaler()

# 特徴データを標準化（平均0、分散1になるように変換）
CodeData_norm = scaler.fit_transform(CodeData)

# maxlen 日分の日経平均株価を入力として、1 日後の日経平均株価を予測
# maxlen 日分のN225(Close) と、maxlen+1 日のN225(Close)のデータセットを作成
maxlen = 10
x_data = []
y_data_price = []
#y_data_updown = []
datelabel = []
for i in range(len(CodeData_norm) - maxlen): # i を データ数-maxlen　まで繰り返し
    x_data.append(CodeData_norm[i:i + maxlen]) # NikkeiData_norm[i]からNikkeiData_norm[i+maxlen]までのデータをx_dataに追加
    y_data_price.append(CodeData_norm[i + maxlen]) # NikkeiData_norm[i+maxlen]のデータをy_dataに追加
    datelabel.append(CodeDate[i + maxlen])
x_data = np.asarray(x_data).reshape((-1, maxlen, 1))
y_data_price = np.asarray(y_data_price)


# 訓練データサイズ 
train_size = int(len(x_data) * 0.8) # 全データのうち、80% のサイズを取得

'''# 訓練用データ'''
x_train = x_data[:train_size] # 全データのうち、80% を訓練用データに格納
y_train_price = y_data_price[:train_size]  # 全データのうち、80% を訓練用データに格納




'''# 検証用データ'''
x_test = x_data[train_size:]  # 全データのうち、20% を検証用データに格納
y_test_price = y_data_price[train_size:]  # 全データのうち、20% を検証用データに格納



'''LSTM を Keras を使って構築'''
length_of_sequence = maxlen
in_out_neurons = 1 # 入力と出力層の数
hidden_neurons = 300 # 隠れ層の数（えいや）

# (*,10,1)のTensorを持った入力を、300個のLSTM中間層に投げ、それを１個の出力層に集約し、linear活性化関数を掛け合わせ
model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
#optimizer = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer='adam')

#    hidden_neurons: 隠れ層・・・数が多い程，学習モデルの複雑さが増加
#    batch_input_shape: LSTMに入力するデータの形を指定([バッチサイズ，step数，特徴の次元数]）
#    Dense: ニューロンの数を調節する、今回は、N225のy軸の値が出力なので、ノード数1にする
#    linear: 線形の活性化関数を用いる
#    compile: 誤差関数：最小2乗誤差、最適化手法：Adamを用いるように定義


#fit(): 訓練データと、教師データ、バッチサイズ、エポックサイズ、バリデーションデータとして訓練データの何%を使用するか指定
#収束判定コールバックを指定することで、収束したら自動的にループを止めることが出来る

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=2)

hist = model.fit(x_train, y_train_price,
          batch_size=256,
          epochs=5,
          validation_split=0.1,
          callbacks=[early_stopping]
          )

#今回は，学習データの10%をvalidationに用いて，5 epochで学習
#1行目のearly_stoppingをcallbacksで定義することで、validationの誤差値(val_loss)の変化が収束したと判定された場合に自動で学習を終了する
#modeをautoにすることで収束の判定を自動で行う
#patienceは判定値からpatienceの値の分だけのepochは学習して、変化がなければ終了するように判定する
#patience=0だとval_lossが上昇した瞬間学習が終了することになる
'''
batch_sizeというのは、訓練データをいくつかに分けて学習するミニバッチという手法を利用する際に、
1個のデータのサイズ（データ数）です。ここでは256個ずつに分けます。
epochというのは、繰り返しの回数を表します。ここでは5回とします。
'''




# loss(訓練データに対する判定結果)、val_loss(テストデータに対する判定結果)をプロットする
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
'''
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.show()
'''

#グラフ用に日にちデータを作成
datelabel = np.asarray(datelabel) #リストをNumpy配列に変換
datelabel = datelabel[train_size:] #学習データサイズ(全データの80%)分の日にちデータを抽出
datelabel.shape

#学習データを用いた予測
predicted = model.predict(x_test) # x_testデータを用いて、10日間のデータを基に次の日の予測を行った結果を格納

# 標準化を戻す
predicted_N =    scaler.inverse_transform(predicted)
y_test_price_N = scaler.inverse_transform(y_test_price)

'''
plt.plot(datelabel, predicted_N, marker='.', label='predicted')
plt.plot(datelabel, y_test_price_N, marker='.', label='y_test_price')
#plt.plot(range(len(predicted)), predicted, marker='.', label='predicted')
#plt.plot(range(len(y_test_price)), y_test_price, marker='.', label='y_test_price')
plt.legend(loc='best')
plt.grid()
plt.xlabel('date')
plt.ylabel('SONY')
#plt.show()
'''



'''未来予測'''
a = y_test_price.shape[0] - 10 # 488 - 10  = 478
tempdata = y_test_price[a:] #
x_tempdata = np.asarray(tempdata).reshape((-1, maxlen, 1)) # モデルに入力できるように、(1, maxlen ,1) 形式に変換

predicted_temp = model.predict(x_tempdata) # x_tempdataデータを用いて、10日間のデータを基に次の日の予測を行った結果を格納
a = np.append(tempdata, predicted_temp) # 標準化した１０日分データと、予測結果をリストに追加

for i in range(10):
    b = a[i+1:] # 直近１０日間のデータを抽出
    x_tempdata = np.asarray(b).reshape((-1, maxlen, 1)) # モデルに入力できるように、(1, maxlen ,1) 形式に変換
    predicted_temp = model.predict(x_tempdata) # x_testデータを用いて、10日間のデータを基に次の日の予測を行った結果を格納
    a = np.append(a,predicted_temp).reshape((-1, 1))

# 標準化を戻す
predicted_futureN = scaler.inverse_transform(a)

'''
#未来予測結果をプロット
plt.plot(range(len(predicted_futureN)), predicted_futureN, marker='.', label='future predicted')
plt.plot(range(len(predicted_futureN[:10])), predicted_futureN[:10], marker='.', label='real data',color="0.5")
#plt.plot(range(len(y_test_price)), y_test_price, marker='.', label='y_test_price')
plt.legend(loc='best')
plt.grid()
plt.xlabel('date')
plt.ylabel('SONY')
#plt.show()
'''

# グラフを描画するためのライブラリを読み込む
'''
t = np.linspace(-np.pi, np.pi, 1000)

x1 = np.sin(2*t)
x2 = np.cos(2*t)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

axL.plot(t, x1, linewidth=2)
axL.set_title('sin')
axL.set_xlabel('t')
axL.set_ylabel('x')
axL.set_xlim(-np.pi, np.pi)
axL.grid(True)

axR.plot(t, x2, linewidth=2)
axR.set_title('cos')
axR.set_xlabel('t')
axR.set_ylabel('x')
axR.set_xlim(-np.pi, np.pi)
axR.grid(True)

fig.show()
'''




# 取得したデータから終値を抽出する
#X1 = JPY_USD['Close']
#X2 = N225['Close']
#X3 = IXIC['Close']
#X4 = DJI['Close']
#X5 = Stock_train_df['Close']

'''
# 抽出した終値でグラフを作成する
fig = plt.figure()
axes = fig.subplots(1, 3)

axes[0].plot(range(epochs), loss, marker='.', label='loss(training data)', color='g')
axes[0].plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data)', color='b')
#axes[1, 1].plot(datelabel, predicted_N, marker='.', label='predicted')
#axes[1, 1].plot(datelabel, y_test_price_N, marker='.', label='y_test_price')
#axes[1, 2].plot(range(len(predicted_futureN)), predicted_futureN, marker='.', label='future predicted', color='k')
#axes[1, 2].plot(range(len(predicted_futureN[:10])), predicted_futureN[:10], marker='.', label='real data', color='m')
#axes[0, 2].plot(X5, color='r')

# タイトルを設定する

axes[0].set_title('loss')
#axes[0, 1].set_title('日経平均(N225)')
#axes[1, 0].set_title('NASDAQ総合(IXIC)')
#axes[1, 1].set_title('ダウ平均(DJI)')
#axes[0, 2].set_title('SONY')

# グラフを表示する
#plt.show()

'''
#描画するデータの読み込み
fig = plt.figure(figsize=(15, 7), dpi=100, facecolor='None', edgecolor='black' )
ax = fig.add_subplot(2, 1, 1)
# 図全体のタイトル
fig.suptitle(
    "Long Short-Term Memory (Deep Larning) of Artificial Intelligence[AI]", fontsize=20)
plt.title("Test Graph", {"fontsize": 20})


ax1 = plt.subplot(2,2,1)   # 2x2の1番目
ax1.plot(xdata, ydata)# 1番目に描画
ax1.legend(loc='best')
ax1.grid()
ax1.set_xlabel('Date')   # 1番目にxラベルを追加
ax1.set_ylabel(f'{code}')   # 1番目にyラベルを追加

ax2 = plt.subplot(2,2,2)   # 2x2の1番目
ax2.plot(range(epochs), loss, marker='.', label='loss(training data)')# 1番目に描画
ax2.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data)')   # 1番目に追加描画
ax2.legend(loc='best')
ax2.grid()
ax2.set_xlabel('epoch')   # 1番目にxラベルを追加
ax2.set_ylabel('loss')   # 1番目にyラベルを追加

ax3 = plt.subplot(2,2,3)   # 2x2の3番目
ax3.plot(datelabel, predicted_N, marker='.', label='predicted')# 1番目に描画
ax3.plot(datelabel, y_test_price_N, marker='.', label='y_test_price')# 1番目に追加描画
ax3.legend(loc='best')
ax3.grid()
ax3.set_xlabel('Date')
ax3.set_ylabel(f'{code}')

ax4 = plt.subplot(2,2,4)   # 2x2の4番目
ax4.plot(range(len(predicted_futureN)), predicted_futureN, marker='.', label='future predicted')# 1番目に描画
ax4.plot(range(len(predicted_futureN[:10])), predicted_futureN[:10], marker='.', label='real data',color="0.5")# 1番目に追加描画
ax4.legend(loc='best')
ax4.grid()
ax4.set_xlabel('Date')   # 1番目にxラベルを追加
ax4.set_ylabel(f'{code}')   # 1番目にyラベルを追加


# グラフを表示する
plt.show()

'''
from sklearn.metrics import accuracy_score

# 作成したモデルより検証データを用いて予測を行う
pred = model.predict(X_val_np_array)
pred[:10]

# 実際の結果から予測値の正解率を計算する
print('accuracy = ', accuracy_score(y_true=y_val_new, y_pred=pred))
'''