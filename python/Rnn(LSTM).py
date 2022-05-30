'''機械学習ノートブックメモ  Example_RNN'''

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
#from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import sys
import numpy as np
import plotly
import plotly.graph_objs as go
import pandas as pd
import math
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

#plotly.offline.init_notebook_mode(connected=False)
warnings.filterwarnings('ignore')
sns.set_style("whitegrid", {'grid.linestyle': '--'})
'''
関数
罪()
正弦波を生成する

toy_problem()
正弦波にノイズを含める

make_dataset
RNNの学習のためのデータセットを生成する
'''

#はじめに，sin波を生成する．
def sin(x, T=100):
    return np.sin(2.0*np.pi * x / T)



# sin波にノイズを付与する
def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

f = toy_problem()



def make_dataset(raw_data, n_prev=100, maxlen=25):
    data, target = [], []
    
    for i in range(len(raw_data) - maxlen):
        data.append(raw_data[i : i + maxlen])
        target.append(raw_data[i + maxlen])
        
    reshaped_data = np.array(data).reshape(len(data), maxlen, 1)
    reshaped_target = np.array(target).reshape(len(target), 1)
    
    return reshaped_data, reshaped_target



#g -> 学習データ，h -> 学習ラベル
#g, h = make_dataset(f)

'''追記'''
# 300周期 (601サンプル)にデータに拡張
ex_function = toy_problem(T=300)

# 50サンプルごとに分割
g, h = make_dataset(ex_function, maxlen=50)




'''モデル構築'''

# 1つの学習データのStep数(今回は25)
length_of_sequence = g.shape[1] 
in_out_neurons = 1
n_hidden = 300# 隠れ層 -> 数が多い程，学習モデルの複雑さが増加

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))#LSTMに入力するデータの形を指定([バッチサイズ，step数，特徴の次元数]を指定する）
model.add(Dense(in_out_neurons))#ニューロンの数を調節しているだけ．今回は，時間tにおけるsin波のy軸の値が出力なので，ノード数1にする．
model.add(Activation("linear"))
#optimizer = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer='adam')


'''学習
生成した学習データ，定義したモデルを用いて，学習を行う．
今回は,学習データの10%をvalidationに用いて,100 epochで学習させた.
1行目のearly_stoppingをcallbacksで定義することで,validationの誤差値(val_loss)の変化が収束したと判定された場合に自動で学習を終了する,modeをautoにすることで,収束の判定を自動で行う．
patienceは,判定値からpatienceの値の分だけのepochは学習して,変化がなければ終了するように判定する,なので,patience=0だと,val_lossが上昇した瞬間,学習が終了することになる．
'''

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
model.fit(g, h,
          batch_size=300,
          epochs=100,#100 epochで学習
          validation_split=0.1,#,学習データの10%をvalidationに用いた
          callbacks=[early_stopping]
          )


'''予測
学習データで予測
学習データを予測して,sin波が再現できるか確認.
'''
#予測
predicted = model.predict(g)


'''
これで,predictedにt=25以降のsin波を予測させることができる.
実際にplotしてみる.
'''

plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.legend()
plt.show()
#予測sin波の方は，ほとんどノイズの影響を受けずに予測することができている．






# 1つの学習データの時間の長さ -> 25
future_test = g[0] 
time_length = future_test.shape[0]

# 未来の予測データを保存していく変数
future_result = np.empty((1))

# 未来予想
for step2 in range(400):

    test_data = np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)

    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)


# sin波をプロット
plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.plot(range(0+len(f), len(future_result)+len(f)), future_result, color="g", label="future_predict")
plt.legend()
plt.show()









'''





'''








#ノイズシンウェーブを生成
#ノイズ入りサイン波の生成

function = toy_problem(T=100)
#LSTM のデータシェーピング
#LSTM用にデータ整形

data, label = make_dataset(function, maxlen=25)
print(data.shape)

#(176, 25, 1)
'''
length_of_sequence : LSTMの入力の長さ
in_out_neurons : 時系列データの単位時間における特徴量の次元数 (今回は単位時間あたり1つのスカラー量なので 1)
n_hidden : 隠れ層の次元数
'''





'''モデル構築'''
# 1つの学習データのStep数(今回は25)
length_of_sequence = data.shape[1]
in_out_neurons = 1
n_hidden = 300


'''モデル定義'''
model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation('linear'))
#optimizer = Adam(lr=1e-3)
model.compile(loss="mean_squared_error", optimizer='adam')

'''学習'''
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
model.fit(data, label,
         batch_size=100, epochs=200,
         validation_split=0.1, callbacks=[early_stopping]
         )



'''予測|トレーニング データ'''
predicted = model.predict(data)

'''予測|将来のデータ'''
future_test = data[-1].T
time_length = future_test.shape[1]
future_result = np.empty((0))
for step in range(400):
    test_data= np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)
    
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)
    
    future_result = np.append(future_result, batch_predict)

'''シーボーンとのプロット'''
fig = plt.figure(figsize=(10,5),dpi=200)
sns.lineplot(
    color="#086039",
    data=function,
    label="Raw Data",
    marker="o"
)

sns.lineplot(
    color="#f44262",
    x=np.arange(25, len(predicted)+25),
    y=predicted.reshape(-1),
    label="Predicted Training Data",
    marker="o"
)

sns.lineplot(
    color="#a2fc23",
    y= future_result.reshape(-1),
    x = np.arange(0+len(function), len(future_result)+len(function)),
    label="Predicted Future Data",
    marker="o"
)
#<matplotlib.axes._subplots.AxesSubplot at 0x7f8cf0a36320>

#プロットでプロット
data_raw = go.Scatter(
    y = function, 
    x  = np.arange(0, len(function)),
    name = 'Raw Data',
    mode = 'lines',
    line = dict(
        color = 'red'
    )
)

predicted_graph = go.Scatter(
    y = predicted.reshape(-1),
    x = np.arange(25, len(predicted)+25),
    name  = 'Predicted Training Data',
    mode = 'lines',
    line = dict(
        color = 'blue'
    )
)

predicted_future = go.Scatter(
    y = future_result.reshape(-1),
    x = np.arange(0+len(function), len(future_result)+len(function)),
    name = 'Predicted Future Data',
    mode = 'lines',
    line = dict(
        color = 'green'
    )
)

data = [data_raw, predicted_graph, predicted_future]
#plotly.offline.iplot(data)


'''
0
100
200
300
400
500
600
-1.5
-1
-0.5
0
0.5
1
1.5
2
2.5
サンプルの長さを展開
サンプルの長さを長くしてみる

ノイズシンウェーブを生成
ノイズ入りサイン波の生成
'''

ex_function = toy_problem(T=300)

#LSTM のデータシェーピング
#LSTM用にデータ整形

ex_data, ex_label = make_dataset(ex_function, maxlen=50)
print(ex_data.shape)
#(551, 50, 1)
'''
ex_length_of_sequence : LSTMの入力の長さ
ex_in_out_neurons : 時系列データの単位時間における特徴量の次元数 (今回は単位時間あたり1つのスカラー量なので 1)
ex_n_hidden : 隠れ層の次元数
'''

ex_length_of_sequence  = ex_data.shape[1]
ex_in_out_neurons = 1
ex_n_hidden = 300

'''モデル定義'''
ex_model = Sequential()
ex_model.add(LSTM(ex_n_hidden, batch_input_shape=(None, ex_length_of_sequence, ex_in_out_neurons), return_sequences=False))
ex_model.add(Dense(ex_in_out_neurons))
ex_model.add(Activation('linear'))
#ex_optimizer = Adam(lr=1e-3)
ex_model.compile(loss="mean_squared_error", optimizer='adam')

'''学習'''
ex_early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
ex_model.fit(
    ex_data, ex_label,
    batch_size=100, epochs=200,
    validation_split=0.1, callbacks=[ex_early_stopping]
)



'''予測|トレーニング データ'''
ex_predicted = ex_model.predict(ex_data)

'''予測|将来のデータ'''
ex_future_test = ex_data[-1].T
ex_time_length = ex_future_test.shape[1]
ex_future_result = np.empty((0))
for step in  range(400):
    ex_test_data = np.reshape(ex_future_test, (1, ex_time_length, 1))
    ex_batch_predict = ex_model.predict(ex_test_data)
    
    ex_future_test = np.delete(ex_future_test, 0, axis=1)
    ex_future_test = np.append(ex_future_test, ex_batch_predict, axis=1)
    
    ex_future_result = np.append(ex_future_result, ex_batch_predict)

'''シーボーンとのプロット'''
fig = plt.figure(figsize=(10, 5), dpi=200)
sns.lineplot(
    color="#086039",
    data=ex_function,
    label="Raw Data",
    marker="o"
)

sns.lineplot(
    color="#f44262",
    x=np.arange(50, len(ex_predicted)+50),
    y=ex_predicted.reshape(-1),
    label="Predicted Training Data",
    marker="o"
)

sns.lineplot(
    color="#5293fa",
    x= np.arange(0+len(ex_function), len(ex_future_result) + len(ex_function)),
    y = ex_future_result.reshape(-1),
    label="Predicted Future Data",
    marker="o"
)
#<matplotlib.axes._subplots.AxesSubplot at 0x7f8cda898518>

'''プロットでプロット'''
data_raw = go.Scatter(
    x = np.arange(0, len(ex_function)),
    y = ex_function,
    name = 'Raw Data',
    mode = 'lines',
    line = dict(
        color = 'red'
    )
)

predicted_graph = go.Scatter(
    x = np.arange(50, len(ex_predicted)+25),
    y =  ex_predicted.reshape(-1),
    name = 'Predicted Training Data',
    line = dict(
    color='blue'
    )
)

predicted_future = go.Scatter(
    x = np.arange(0 + len(ex_function), len(ex_future_result)+len(ex_function)),
    y = ex_future_result.reshape(-1),
    name = 'Predicted Future Data',
    mode = 'lines',
    line = dict(
        color = 'green'
    )
)

#描画するデータの読み込み
fig = plt.figure(figsize=(15,10),dpi=100)
ax = fig.add_subplot(2,1,1)
fig.suptitle("Long Short-Term Memory (Deep Larning) of Artificial Intelligence[AI]", fontsize=20) # 図全体のタイトル
plt.title("Test Graph", {"fontsize": 20})
#未来予測結果をプロット
ax4 = plt.subplot(2,2,4)   # 2x2の4番目
ax4.plot(len(ex_function), len(ex_future_result)+len(ex_function), marker='.', label='future predicted')# 1番目に描画
#ax4.plot(range(len(predicted_futureN[:10])), predicted_futureN[:10], marker='.', label='real data',color="0.5")# 1番目に追加描画
ax4.legend(loc='best')
ax4.grid()
ax4.set_xlabel('Date')   # 1番目にxラベルを追加
#ax4.set_ylabel(f'{code}')   # 1番目にyラベルを追加
# グラフを表示する
plt.show()


data =  [data_raw, predicted_graph, predicted_future]
#plotly.offline.iplot(data)