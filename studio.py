import numpy as np
import matplotlib.pyplot as plt
import torch
#import torch.nn as nn
from random import uniform
from datetime import datetime, timedelta

'''日付シリアル値からyyyy/MM/dd HH:MM:ss 形式に変換
    Args:
        serialVal (float): シリアル値(Ex: 44434.3412172569)
    Returns:
        str: 時刻  yyyy/MM/dd HH:MM:ss 形式
'''
sDateTime = (datetime(1899, 12, 30) + timedelta(1600)
             ).strftime('%Y/%m/%d %H:%M:%S')
    
print('Time',sDateTime)



def make_data(num_div, cycles, offset=0):
    step = 2 * np.pi / num_div
    res0 = [np.sin(step * i + offset) for i in range(num_div * cycles + 1)]
    res1 = [np.sin(step * i + offset) + uniform(-0.02, 0.02) for i in range(num_div * cycles + 1)]
    return res0, res1


'''
サイン波を格納するデータを生成する関数
このmake_data関数は、num_div/cycles/offsetの3つのパラメーターを取ります。
num_divは1周期を何分割するかを、
cyclesは周期の数を指定するものです。
最後のoffsetはX軸（時刻）に対するオフセットを指定するものです（本稿の最後でこれを使ってみます）。
range関数の引数で最後に1を足しているのは、その方が周期最後のデータがうまくはまったからで、それ以上の意味はありません。

関数内部では2つのリストを生成していますが（リスト内包表記部分）、1つはノイズを含まないキレイなサイン波となるような値を、もう1つはノイズを含めたサイン波となる値を格納します。正解ラベルの値については、ノイズを含まないサイン波を基にしたいので、少々汚いコードになっていますが、このようにしています。





実際にこの関数を呼び出して、データを作成して、グラフをプロットしてみましょう。
'''

num_div = 100
cycles = 2
sample_data, sample_data_w_noise = make_data(num_div, cycles, 25)
#print(sample_data)
#plt.plot(sample_data_w_noise)
#plt.grid()
#plt.show()


#make_data関数で作成する値を基に、訓練データと正解ラベルを生成する関数のコードを以下に示します。
def make_train_data(num_div, cycles, num_batch, offset=0):
    x, x_w_noise = make_data(num_div, cycles, offset)
    data, labels = [], []
    count = len(x) - num_batch
    data = [x_w_noise[idx:idx+num_batch] for idx in range(count)]
    labels = [x[idx+num_batch] for idx in range(count)]
    num_items = len(data)
    train_data = torch.tensor(data, dtype=torch.float)
    train_data = train_data.reshape(num_items, num_batch, -1)
    train_labels = torch.tensor(labels, dtype=torch.float)
    train_labels = train_labels.reshape(num_items, -1)
    return train_data, train_labels


#学習で使用する訓練データと正解ラベルを生成して、その形状を確認 
X_train, y_train = make_train_data(num_div, cycles, 25)
print(X_train.shape, y_train.shape)


'''RNNを使用するニューラルネットワークの定義'''
#PyTorchにはRNN機能を提供するクラスが幾つか用意されています。
# 今回はその中でも基本的なRNNクラス（torch.nn.RNNクラス）を使用します。
# このクラスと、既におなじみのLinearクラスを組み合わせることにしましょう。

#実際のクラス定義のコードは次の通りです。
#RNNを使用したNetクラスの定義
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x, hidden):
        output, h = self.rnn(x, hidden)
        output = self.fc(output[:, -1])
        return output, h   

'''
__init__メソッドでは、インスタンス変数self.rnnにRNNクラスのインスタンスを代入しています。
RNNクラスのインスタンス生成時には「torch.nn.RNN(input_size, hidden_size)」のように
「入力のサイズ」と「隠れ状態のサイズ」の2つの引数を指定しています（これらは__init__メソッドのパラメーターに渡されるようにしてあります）。
'''        

class Predictor(torch.nn.Module):#「Predictor」はただの名前だから好きなもので良い.
    # __init__ 内でモデルの持つ層を宣言します。 このモデルは2つの層を持っており、nn.LSTM と nn.Linear を持っている。
    #  nn.LSTM は LSTM の層を、nn.Linear は全結合層を表す。
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = torch.nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = False)
        self.output_layer = torch.nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0):
        output, h = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output,h







'''学習'''
#今回は、学習のコードは簡略化したものとします（いわゆるミニバッチのようなことはせずに一度のforループで学習をしてみます）。
#まずは、訓練データと正解ラベル、Netクラスのインスタンスを用意します。訓練データは先ほども試しに作りましたが、見通しをよくするように、ここで作り直しておきましょう。
#訓練データと正解ラベルは前述した通り、make_train_data関数を呼び出すだけです。

num_div = 100
cycles = 2
num_batch = 25
X_train, y_train = make_train_data(num_div, cycles, num_batch)
'''訓練データと正解ラベルの生成'''
#次に上で定義したNetクラスのインスタンスを生成して、損失関数と最適化アルゴリズムを選択します。

input_size = 1 #一度に1個のデータを入力していくので、input_sizeは1です。
hidden_size = 32 #（このサイズに合わせて、この後の隠れ状態の初期化を行います）。
net = Net(input_size, hidden_size)
#net = Predictor(input_size,hidden_size,1)
#Netクラスのインスタンス生成、損失関数と最適化アルゴリズムの選択
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

'''
先ほども述べたように、このRNNには一度に1個のデータを入力していくので、input_sizeは1です。
隠れ状態のサイズは適当に32としてあります（このサイズに合わせて、この後の隠れ状態の初期化を行います）。
損失関数には、MSELoss関数を使います。これは出力層からの出力が1つだけのためだからです。
最適化アルゴリズムもこれまでの回と同様です。
'''







'''次に学習を行うコードを示します。'''

num_layers = 1
EPOCHS = 100

losses = []

for epoch in range(EPOCHS):
    print('epoch:', epoch)
    optimizer.zero_grad()
    hidden = torch.zeros(num_layers, num_batch, hidden_size)
    output, hidden = net(X_train, hidden)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    print(f'loss: {loss.item() / len(X_train):.6f}')
    losses.append(loss.item() / len(X_train))

'''
RNNには訓練データに加えて、隠れ状態の初期値を渡す必要もありました。
上のコードではforループの中で「hidden = torch.zeros(num_layers, num_batch, hidden_size)」として、
ループのたびにこれをリセットするようにしています。
これは「RNN層の数（ここでは1）xバッチサイズ（ひとかたまりの時系列データの数。ここでは25個）xRNNクラスのインスタンス生成時に指定した隠れ状態のサイズ（ここでは32）」という形状である必要があります。
そこで、ここではnu_layers、num_batch、hidden_sizeを指定して、そのようなサイズでゼロ初期化したテンソルを用意しています。

後は訓練データとこれを一緒にNetクラスのインスタンスに渡すだけです。
その後のコードについては特に説明の必要はないでしょう（netインスタンスの呼び出しでは「output, hidden = net(……)」のようにして、隠れ状態の現在の状態を受け取っていますが、
上のコードではループのたびにこれをリセットしているので、実際にはこれは使っていません）。
変数lossesには損失をX_trainの要素数で割った値を蓄積しておき、後でこれを表示してみましょう。
'''    

plt.plot(losses)
plt.show() 


'''
これは非常にシンプルな例だったので、学習にかかる時間も少なく、比較的早いうちにそれなりの損失になっているようです。
ただ、問題はほんとうに推測ができているかです。
そこで、最後の学習結果であるoutputに格納されている値を使って、グラフをプロットしてみましょう（上述しましたが、outputには「25個の時系列データから推測される次の値」が含まれています）。

これをプロットするコードを以下に示します。
'''
output = output.reshape(len(output)).detach()
plt.plot(sample_data)
plt.plot(range(24, 200), output)
plt.grid()
plt.show() 
#推測結果からグラフをプロット

'''
上のコードでは、outputは推測値のみを要素とするテンソルを要素とするテンソル（2次元）になっているので、reshapeメソッドでこれを1次元のテンソルに展開しています。
また、detachメソッドで勾配計算に必要だった要素を分離しています（これを行わないとグラフをプロットできません）。
また、最初に作成していたsample_data（ノイズなしのサイン波のデータ）も同時にプロットするようにしました。
そのため、推測値についてはグラフ上での描画開始位置を細工しています。

オレンジ色のラインがノイズなしのサイン波です。
青いラインが推測値を使って描画したサイン波です
（サイン波が途中から始まっているのは、X_trainの最初の25個については、推測値がないためです。興味のある方は変数num_divを増やすなどして試してみてください）。
最初の部分があまりよくないのですが、そこを除けば、まあまあサイン波といってもよいでしょう。
'''