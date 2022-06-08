import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np







from pandas_datareader import data as pdr
from dateutil.relativedelta import relativedelta
import datetime
code = '6976'#'6758'
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
# (2021, 1, 1)  # 教師データ(今までのデータ)
start_train = datetime.date.today() + relativedelta(days=-700)
# 昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_train = datetime.date.today() + relativedelta(days=-1)


class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()
        self.input_dim = inputDim
        self.hidden_dim = hiddenDim
        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True
                            )
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs,hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        #print(output)
        #インデックスは 0 から開始
        #負のインデックスは後ろ側から逆順に数える
        #スライスにコロン : を使用 - start:stop:step
        output = self.output_layer(output[: , -1 ,: ])#負のインデックスは後ろ側から逆順に数える
        #print(output)

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
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t


def stockDataSet(code):
    train_x = []
    train_t = []
    future_num = 1 #何日先を予測するか

    df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データを読み込む。
    #カラム名の取得
    cols = ['High','Low','Open','Close','Volume','Adj Close']
    X_data = df.iloc[:-future_num][cols].values
    print(df,X_data)

    

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
    epochs_num = 10#1000
    hidden_size = 5
    batch_size = 100

    train_x, train_t = mkDataSet(training_size)
    test_x, test_t = mkDataSet(test_size)

    #test_x, test_t = stockDataSet(code)

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