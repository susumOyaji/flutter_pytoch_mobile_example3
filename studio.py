import numpy as np
import matplotlib.pyplot as plt
import torch
from random import uniform

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
sample_data, sample_data_w_noise = make_data(num_div, cycles)

plt.plot(sample_data_w_noise)
plt.grid()
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

input_size = 1
hidden_size = 32
net = Net(input_size, hidden_size)

#Netクラスのインスタンス生成、損失関数と最適化アルゴリズムの選択
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)


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

plt.plot(losses)
plt.show()    