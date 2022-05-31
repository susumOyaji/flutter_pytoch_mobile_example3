import torch
import torch.nn as nn
import torch.nn.functional as F
import theano

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr









code = '6976'  # '6976'#6758
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
# (2021, 1, 1)  # 教師データ(今までのデータ)
start_train = datetime.date.today() + relativedelta(days=-700)
# 昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_train = datetime.date.today() + relativedelta(days=-1)
start_test = datetime.date.today()

adjclosed = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)["Adj Close"] # 教師データを読み込む。
Dow_df = pdr.get_data_yahoo('^DJI', start_train, end_train)  # 試験データのcsvファイルを読み込む。
Nikkei_df = pdr.get_data_yahoo('^N225', start_train, end_train)  # 試験データのcsvファイルを読み込む。




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


'''学習'''
'''教師データの数値の配列(train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくるというそれだけです。'''
'''###教師データをつくる'''
# 過去の株価と上がり下がり(結果)を学習する
# まずは一番面倒な株価の調整後終値(Adj Clouse)から教師データを作るまでのコードを用意します。
# これは終値のリストを渡すと train_X と train_y が返るようにすれば良いでしょう。


def train_data(adjclosed):  # arr = test_X
    train_X = []  # 教師データ
    train_y = []  # 上げ下げの結果の配列

    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-30, -15):
        s = i + 14  # 14 日間の変化を素性にする
        feature = adjclosed.iloc[i:s]  # i~s行目を取り出す
        if feature[-1] < adjclosed[s]:  # その翌日、株価は上がったか？
            train_y.append(1)  # YES なら 1 を
        else:
            train_y.append(0)  # NO なら 0 を
        train_X.append(feature.values)

    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(train_X), np.array(train_y)










class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4,1)

    def forward(self, X):
        #shape -> [1,2,2], shape[0] is batch_size
        X = X.view(X.shape[0], -1)
        return torch.sigmoid(self.fc(X))



num_classes = 10
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ネットワークのモジュール化
class Net2(nn.Module):
    def __init__(self, input):
        super(Net2, self).__init__()
        
        # ネットワークを定義
        self.linear1 = nn.Linear(4, 10)
        self.linear2 = nn.Linear(10, 8)
        self.linear3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    # 順伝搬を定義
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x  








#関数prepare_dataではLSTMに入力するデータを30日分ずつまとめる役割を果たしています。
def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去の30日分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats



#モデルの定義です。PytorchのライブラリからLSTMを引っ張ってきました。
class LSTMClassifier(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(LSTMClassifier, self).__init__()
        self.input_dim = lstm_input_dim
        self.hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, 
                            hidden_size=lstm_hidden_dim,
                            num_layers=1, #default
                            #dropout=0.2,
                            batch_first=True
                            )
        self.dense = nn.Linear(lstm_hidden_dim, target_dim)

    def forward(self, X_input):
        _, lstm_out = self.lstm(X_input)
        # LSTMの最終出力のみを利用する。
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)



#「nn.Module」はこのclassがnn.Moduleというclassを継承していることを意味する.
# なぜ継承するかというとnn.ModuleがNetworkを操作する上でパラメータ操作などの重要な機能を持つためである.
#「def __init__(self)」は初期化関数の定義で,コンストラクタなどと呼ばれる.初めてclassを呼び出したときにこの中身のものが実行される.
#「super(Predictor, self).__init__()」は継承したnn.Moduleの初期化関数を起動している.superの引数の「Predictor」はもちろん自身が定義したclassの名前である.
#「def forward(self, x)」には実際の処理を書いていく.
class Predictor(nn.Module):#「Predictor」はただの名前だから好きなもので良い.
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

#「self.xxxxx」という変数はずっと保持される.
# なので,convolutionやfully connectなどの学習に必要なパラメータを保持したいものなどをここに書いていく.









if __name__ == "__main__":
    model = LSTMClassifier()
    print(model)
    print(model.eval())

    example = torch.rand(1,2,2) #tensor of size input_shape
    print(example)
    traced_script_module = torch.jit.trace(model, example)
    #traced_script_module.save("example/assets/models/custom_model.pt")
    traced_script_module.save("testmodels/stackcard_model.pt")


    from torch.autograd import Variable
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import torch.optim as optim

    iris = datasets.load_iris()
    y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
    y[np.arange(len(iris.target)), iris.target] = 1
    X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)
    x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
    y = Variable(torch.from_numpy(y_train).float())
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(3000):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # test
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))


    # utility function to predict for an unknown data
    def predict(X):
        X = Variable(torch.from_numpy(np.array(X)).float())
        outputs = net(X)
        return np.argmax(outputs.data.numpy())
      