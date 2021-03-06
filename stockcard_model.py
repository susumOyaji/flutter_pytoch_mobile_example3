
#以下に、validationも書き加えたソースコードの全体を載せます。
# 恐らくこれをそのままコピペすれば動くはずなので、ぜひ参考にしてください。

import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import datetime
from pandas_datareader import data
import matplotlib.pyplot as plt
import random
import pandas as pd


start = datetime.date(2021, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY
df = data.DataReader(f'{code}.T', 'yahoo', start, end)








class Predictor(nn.Module):
    # __init__ 内でモデルの持つ層を宣言します。 このモデルは2つの層を持っており、nn.LSTM と nn.Linear を持っている。
    #  nn.LSTM は LSTM の層を、nn.Linear は全結合層を表す。
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output




def mkDataSet(data_size, data_length=50, freq=60., noise=0.00):
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []#トレーニングデータ（t=1,2,...,size-1の値)
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t



def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def main():
    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 5
    batch_size = 100

    train_x, train_t = mkDataSet(training_size)

    '''
    random.seed(0)
    # 乱数の係数
    random_factor = 0.05
    # サイクルあたりのステップ数
    steps_per_cycle = 80
    # 生成するサイクル数
    number_of_cycles = 50

    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    print(df)
    df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * random_factor))
    print(df["sin_t"])
    df[["sin_t"]].head(steps_per_cycle * 2).plot()
    '''
    a =np.arange(train_x,10000)
    plt.plot(train_x,train_t)
    plt.show()

    test_x, test_t = mkDataSet(test_size)

    model = Predictor(1, hidden_size, 1)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            #running_loss += loss.data[0]
            running_loss += loss.data.item()
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
