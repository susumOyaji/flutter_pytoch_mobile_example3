import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import numpy as np



'''
LSTM(Long short-term memory) を用いて、時系列データの予測を行います。
PythonのKerasを使います。
'''

'''簡単な例'''
#まずは、簡単な例で試します。 3つの入力から、2つの値を出力(予測)することにします。
# 学習用データ。xが入力、yが出力(答え)です。
x = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])#[50,60,70],[60,70,80]
y = np.array([[40, 50], [50, 60], [60, 70], [70, 80]])#[80,90],[90,100]


# 行列のフォーマット変更。
# LSTMは、入力フォーマットを[サンプルの数, 入力のステップ数(この場合は3), features]とする必要があるためです。
x = x.reshape((x.shape[0], x.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

#次にネットワークを構築します。
m = Sequential()
m.add(LSTM(100, activation='relu', input_shape=(3, 1)))
m.add(RepeatVector(2))
m.add(LSTM(100, activation='relu', return_sequences=True))
m.add(TimeDistributed(Dense(1)))
m.compile(optimizer='adam', loss='mse')

'''
少し解説します。
1行目:Sequentialは、あるレイヤの全ノードと、次レイヤの全ノードをつなぐDNNのモデルです。
2行目:DNNの第一レイヤとして、LSTMを追加します。第一引数は出力の次元数です。ここでは100としています。activationは活性化関数で、ここではReLUを使うように設定しています。input_shapeは、入力データのフォーマットです。
3行目:RepeatVectorにより、入力を繰り返します。ここでの繰り返し回数は、予測範囲(今回は2データ)となります。
4行目:再びLSTM。ただし、ここではreturn_sequences = Trueを指定します。
5行目:TimeDistributedを指定し、かつ、Dense(1)で、出力の次元数を「１」に指定します。
6行目:最後にcompileメソッドで、学習時の最適化手法や、損失関数を指定します。
ここでは最適化手法としてAdamを、損失関数としてMSE(Mean Squared Error 平均二乗誤差)を指定します。
'''

#fitメソッドで学習を行います。 学習。時間が少しかかる可能性があります。
##m.fit(x, y, epochs=1000, verbose=0)

#学習済みのモデルに、[50, 60, 70]という入力を与えて、結果がどうなるかを見てみます。 理想では[80, 90]となればOKです。
##x_input = np.array([50, 60, 70])
##x_input = x_input.reshape((1, 3, 1))
##yhat = m.predict(x_input)
##print(yhat)

'''
結果は以下です。
[[[82.211136]
  [93.43616]]]
少しズレがありますが、まあまあですね。
'''




'''AirPassengers.csvで実験'''
#次に、AirPassengers.csvのデータで試してみます。 このデータは色々なところで勉強用に使われている。
#まずはデータを読み込みます。
df = pd.read_csv('AirPassengers.csv', index_col='Month', dtype={1: 'float'})
ts = df['#Passengers']

#学習用データxと、回答データyを用意します。 AirPassengersのデータは、「毎月の乗客数」です。
# 10年分くらいのデータがあるので、その一部を学習用に使います。 
# 2年間のデータ(24データ)を用いて、次に一年(12データ)を予測するように学習します。
x = []  # train
y = []  # test (answer)
for i in range(0, 72):
    tmpX = []
    for j in range(0, 24):  # 2年間のデータ(24データ)
        tmpX.append(ts[i+j])
    x.append(tmpX)

    tmpY = []
    for j in range(0, 12):  # 1年間のデータ(12データ)
        tmpY.append(ts[24+i+j])
    y.append(tmpY)

#学習用データxと、回答データyができたので、numpy配列にして、LSTM用にreshapeします。
x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))


#ネットワークを組み、学習します。
m = Sequential()
# 入力データ数が24なので、input_shapeの値が(24,1)です。
m.add(LSTM(100, activation='relu', input_shape=(24, 1)))
# 予測範囲は12ステップなので、RepeatVectoorに12を指定する必要があります。
m.add(RepeatVector(12))
m.add(LSTM(100, activation='relu', return_sequences=True))
m.add(TimeDistributed(Dense(1)))
m.compile(optimizer='adam', loss='mse')
m.fit(x, y, epochs=1000, verbose=0)

#いざ、予測してみます。
# データ60番～83番から、次の一年(84番～95番)を予測
input = np.array(ts[60:84])
input = input.reshape((1, 24, 1))
yhat = m.predict(input)

# 可視化用に、予測結果yhatを、配列predictに格納
predict = []
for i in range(0, 12):
    predict.append(yhat[0][i])

# 比較するために実データをプロット
plt.plot(ts)

# 予測したデータをプロット
xdata = np.arange(84, 96, 1)
plt.plot(xdata, predict, 'r')
# 良い感じに予測できています。


# さらに次の一年を予測してみます。
input = np.array(ts[72:96])
input = input.reshape((1, 24, 1))
yhat = m.predict(input)

predict = []
for i in range(0, 12):
    predict.append(yhat[0][i])

plt.plot(ts)
xdata = np.arange(96, 108, 1)
plt.plot(xdata, predict, 'r')
plt.show()
#まあまあではないでしょうか？ チューニングすれば、より正確な予測ができるのではないかと思います。
