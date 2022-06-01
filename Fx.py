'''PyTorchのLSTMで時系列データ予測'''


'''LSTMモデルの実装'''
'''
定数の定義と必要なパッケージのインポート
先にこのモデルの学習とテストで利用する定数を定義しておきます。

future_numでは、価格が上がるか下がるかを予測する未来の10分足数です。ここでは10分足データの144足分のため、1日先の価格が上がるか下がるか、の予測となります。
feature_numは入力データの特徴量の数で、ボリューム、Open, High, Low, Closeの5項目を利用します。
batch_sizeはLSTMの学習時に一度に投入するデータポイント数です。
time_stepsは、LSTMが予測で利用する過去のデータポイントの数です。今回は過去の50個分のデータを見て、144個先のClose値が現在に比べて上がるのか下がるのかを予測するモデルとしています。
moving_average_numで500と指定しています。これは、LSTMに投入するデータは過去500足分の移動平均に対する現在の値の比率とするためです。
n_epochsはLSTMのトレーニングで何epoch数分実施するかです。
val_idx_from、test_idx_fromはそれぞれデータの何行目以降を評価用、テスト用として分割するかの位置です。
lstm_hidden_dim, target_dimはLSTMの隠れ層の出力サイズと最終出力サイズです。
'''

future_num = 144 #何足先を予測するか
feature_num = 5 #volume, open, high, low, closeの5項目
batch_size = 128
time_steps = 50 # lstmのtimesteps
moving_average_num = 500 # 移動平均を取るCandle数
n_epocs = 30 
#データをtrain, testに分割するIndex
val_idx_from = 80000
test_idx_from = 100000

lstm_hidden_dim = 16
target_dim = 1

'''
LSTMのモデルに必要なパッケージをインポートしておきます。
deviceはGPUの利用可否に応じでcudaまたはcpuがセットされます。
足りないパッケージなどあればpip等でインストールしておいてください。
'''

import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

'''データ準備'''
#LSTMで学習できるようにデータを準備していきます。
#Oanda APIで取得したCSVデータを読み込みます。
#教師データとして、144足先のClose値と現在のClose値を比較し、上がって入れば1、下がっていれば0をセットします。
#数量や価格はそのまま利用するのではなく、直近500足データの移動平均に対する率とします。
#約3.5日分の移動平均に対して何%上下しているかを予測のためのインプットとします。
#データを分割し、PyTorchで利用できるようにtorchのtensorに変換しておきます。

# 1. CSVファイルの読み込み
df = pd.read_csv('./data/USD_JPY_201601-201908_M10.csv', index_col='Datetime')

# 2. 教師データの作成
future_price = df.iloc[future_num:]['Close'].values
curr_price = df.iloc[:-future_num]['Close'].values
y_data_tmp = future_price - curr_price
y_data = np.zeros_like(y_data_tmp)
y_data[y_data_tmp > 0] = 1
y_data = y_data[moving_average_num:]
# 3. 価格の正規化
cols = df.columns
for col in cols:
    df['Roll_' + col] = df[col].rolling(window=500, min_periods=500).mean()
    df[col] = df[col] / df['Roll_' + col] - 1

#最初の500足分は移動平均データがないため除く。後半の144足分は予測データがないため除く
X_data = df.iloc[moving_average_num:-future_num][cols].values

# 4. データの分割、TorchのTensorに変換
#学習用データ
X_train = torch.tensor(X_data[:val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:val_idx_from], dtype=torch.float, device=device)
#評価用データ
X_val   = torch.tensor(X_data[val_idx_from:test_idx_from], dtype=torch.float, device=device)
y_val   = y_data[val_idx_from:test_idx_from]
#テスト用データ
X_test  = torch.tensor(X_data[test_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[test_idx_from:]



'''LSTMモデル定義'''
#時系列データを処理するためのLSTMのクラスを定義します。
#このクラスでは、（バッチ数、時系列データ数、特徴量数）のデータを受けて、LSTMを通し、LSTMの最終出力をLinear層に渡し、
#Linear層の出力をsigmoidでバイナリの予測として出力する、というモデルにしています。

#今回はLSTMで二値分類のため、LSTMの時系列の出力は利用せず、最終出力のみを利用します。
#各データポイント毎に50個分の時系列データをLSTMに渡して、LSTMは50個分の時系列の結果を返しますが、途中の結果は利用せずに最終出力結果のみを利用します。


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

#次に一つヘルパーファンクションを定義しておきます。
#このファンクションは重要で、データポイントのindexのバッチ数分の配列を受けたら、
#その各index毎に過去50個分の過去データを2つめの次元に追加してそれを一つの固まりとしてLSTMに投入できるようにします。

#バッチ毎の処理数が128、特徴量の数(ボリューム、Open, High, Low, Close）が5のため、
#このファンクションの入力データ（X_data）の次元は（128, 5)となります。

#この各データポイントに対して、過去50個分（time_steps数）のデータを合成してfeatsとして返します。
#そのため、戻り値の次元は(128, 50, 5）となります。2次元目に合成されたデータが過去50個分の時系列データとなります。

def prep_feature_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去のN足分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats



'''LSTM学習の実施'''
#ここまで準備が整ったら、実際に学習を実施してみましょう。
#LSTMのインスタンスを生成し、損失関数と最適化関数を設定します。
#loss functionは二値分類（上がるか下がるか）なので、素直にbinary classification entropy loss（BCELoss）を利用、
#optmizerはAdamを利用します。

# Prepare for training
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
loss_function = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=1e-4)



'''学習を実行していきます。'''
#時系列処理とはいえ、全件を1件づつ回していくと時間がかかるので、ミニバッチを作るためにIndexをランダムに入れ替えます。
#対象のミニバッチデータのそれぞれに、時系列データの50個分の過去データを付与します。
#PyTorchのモデルを使って学習させます。
#epoch毎に評価用データを使って予測、結果を確認します。
#各評価用データの結果を比較し、ベストのモデルを保存します。
#最後にベストのモデルでテスト用のデータを評価します。

train_size = X_train.size(0)
best_acc_score = 0
for epoch in range(n_epocs):
    # 1. まずはtrainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
    perm_idx = np.random.permutation(np.arange(time_steps, train_size))
    # 2. batch size毎にperm_idxの対象のindexを取得
    for t_i in range(0, len(perm_idx), batch_size):
        batch_idx = perm_idx[t_i:(t_i + batch_size)]
        # 3. LSTM入力用の時系列データの準備
        feats = prep_feature_data(batch_idx, time_steps, X_train, feature_num, device)
        y_target = y_train[batch_idx]
        # 4. pytorch LSTMの学習実施
        model.zero_grad()
        train_scores = model(feats) # batch size x time steps x feature_num
        loss = loss_function(train_scores, y_target.view(-1, 1))
        loss.backward()
        optimizer.step()

    # 5. validationデータの評価
    print('EPOCH: ', str(epoch), ' loss :', loss.item())
    with torch.no_grad():
        feats_val = prep_feature_data(np.arange(time_steps, X_val.size(0)), time_steps, X_val, feature_num, device)
        val_scores = model(feats_val)
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        bi_scores = np.round(tmp_scores)
        acc_score = accuracy_score(y_val[time_steps:], bi_scores)
        roc_score = roc_auc_score(y_val[time_steps:], tmp_scores)
        print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score)

    # 6. validationの評価が良ければモデルを保存
    if acc_score > best_acc_score:
        best_acc_score = acc_score
        torch.save(model.state_dict(),'./models/pytorch_v1.mdl')
        print('best score updated, Pytorch model was saved!!', )

# 7. bestモデルで予測する。
model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))
with torch.no_grad():
    feats_test = prep_feature_data(np.arange(time_steps, X_test.size(0)), time_steps, X_test, feature_num, device)
    val_scores = model(feats_test)
    tmp_scores = val_scores.view(-1).to('cpu').numpy()
    bi_scores = np.round(tmp_scores)
    acc_score = accuracy_score(y_test[time_steps:], bi_scores)
    roc_score = roc_auc_score(y_test[time_steps:], tmp_scores)
    print('Test ACC Score :', acc_score, ' ROC AUC Score :', roc_score)



'''    
実行結果は次のようになりました。

EPOCH:  0  loss : 0.697539210319519
Val ACC Score : 0.4637593984962406  ROC AUC Score : 0.4977486521773986
EPOCH:  1  loss : 0.6920570135116577
Val ACC Score : 0.4664160401002506  ROC AUC Score : 0.5264374821400171
EPOCH:  2  loss : 0.6927231550216675
Val ACC Score : 0.4641102756892231  ROC AUC Score : 0.5345851548226157

~~~~~~~~~~~~~~ 途中省略 ~~~~~~~~~~~~~~~~~~~~~~
Val ACC Score : 0.5016040100250626  ROC AUC Score : 0.5261788558490879
best score updated, Pytorch model was saved!!
EPOCH:  24  loss : 0.6927350163459778
Val ACC Score : 0.4756390977443609  ROC AUC Score : 0.524350051179761
EPOCH:  25  loss : 0.6947165131568909
Val ACC Score : 0.49177944862155387  ROC AUC Score : 0.5257184865046414
EPOCH:  26  loss : 0.6922991871833801
Val ACC Score : 0.48235588972431076  ROC AUC Score : 0.525200819748971
EPOCH:  27  loss : 0.6945008635520935
Val ACC Score : 0.47904761904761906  ROC AUC Score : 0.5195235473772817
EPOCH:  28  loss : 0.6937764883041382
Val ACC Score : 0.48230576441102757  ROC AUC Score : 0.522760487446614
EPOCH:  29  loss : 0.6925824284553528
Val ACC Score : 0.48987468671679196  ROC AUC Score : 0.5201632287277098
Test ACC Score : 0.4947853134910954  ROC AUC Score : 0.5020565230922682
'''

'''LSTMのモデルでバックテスト実施'''
#LSTMのモデルを使ってバックテストを実施してみます。
#準備として必要なパッケージのインポート、csvデータの読み込み等を行います。

from backtesting import Strategy
from backtesting import Backtest
df = pd.read_csv('./data/USD_JPY_201601-201908_M10.csv', index_col='Datetime')
df.index = pd.to_datetime(df.index)


'''続いて、バックテストで利用するためのLSTMのStrategyクラスを定義します。'''
#1. initではLSTMの学習済みモデルを読み込んでおきます。
#2. PyTorchのLSTMに投入するためにデータを整えます。
#3. LSTMのPyTorchのモデルで予測します。
#4. 予測結果が1であれば買い、0であれば売りの指示を出します。今回はst(stop loss）、tp（take profit）も指定してみました。

class myLSTMStrategy(Strategy):
    def init(self):
        # 1. LSTMの学習済みモデルの読み込み
        self.model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
        # load model
        self.model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))

    def next(self): 
        # 過去500ステップ分のデータが貯まるまではスキップ
        # 1日に1回のみ取引するため、hour & minuteが0の時のみ処理するようにする。
        if len(self.data) >= moving_average_num + time_steps and len(self.data) % future_num == 0:
            # 2. 推測用データの用意
            x_array = self.prepare_data()
            x_tensor = torch.tensor(x_array, dtype=torch.float, device=device)
            # 3. 予測の実行
            with torch.no_grad():
                y_pred = self.predict(x_tensor.view(1, time_steps, feature_num))

            # 4. 予測が買い(1)であればbuy()、それ以外はsell()
            if y_pred == 1:
                self.buy(sl=self.data.Close[-1]*0.99, 
                         tp=self.data.Close[-1]*1.01)
            else:
                self.sell(sl=self.data.Close[-1]*1.01, 
                         tp=self.data.Close[-1]*0.99)

    def prepare_data(self):
        # いったんPandasのデータフレームに変換
        tmp_df = pd.concat([
                    self.data.Volume.to_series(), 
                    self.data.Open.to_series(), 
                    self.data.High.to_series(), 
                    self.data.Low.to_series(), 
                    self.data.Close.to_series(), 
                    ], axis=1)

        # 500足の移動平均に対する割合とする。
        cols = tmp_df.columns
        for col in cols:
            tmp_df['Roll_' + col] = tmp_df[col].rolling(window=moving_average_num, min_periods=moving_average_num).mean()
            tmp_df[col] = tmp_df[col] / tmp_df['Roll_' + col] - 1

        #最後のtime_steps分のみの値を返す
        return tmp_df.tail(time_steps)[cols].values

    def predict(self, x_array):
        y_score = self.model(x_array) 
        return np.round(y_score.view(-1).to('cpu').numpy())[0]



'''バックテストを実行します。'''

bt = Backtest(df[100000:], myLSTMStrategy, cash=100000, commission=.00004)
bt.run()

'''
実行結果はどうだったでしょうか。

Start                     2018-09-06 13:10:00
End                       2019-08-01 03:50:00
Duration                    328 days 14:40:00
Exposure [%]                          97.6289
Equity Final [$]                       103012
Equity Peak [$]                        104203
Return [%]                            3.01237
Buy & Hold Return [%]                 1.76702
Max. Drawdown [%]                    -3.79017
Avg. Drawdown [%]                   -0.179441
Max. Drawdown Duration       77 days 13:50:00
Avg. Drawdown Duration        1 days 13:34:00
# Trades                                  228
Win Rate [%]                          50.8772
Best Trade [%]                       0.993839
Worst Trade [%]                      -1.00702
Avg. Trade [%]                        0.01088
Max. Trade Duration           3 days 01:00:00
Avg. Trade Duration           1 days 09:47:00
Expectancy [%]                        0.25287
SQN                                    0.4725
Sharpe Ratio                        0.0330574
Sortino Ratio                       0.0501238
Calmar Ratio                       0.00287057
_strategy                      myLSTMStrategy
dtype: object
228回の取引でReturnが3.01%、10万円が10万3,012円となっていました。
'''