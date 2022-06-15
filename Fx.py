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

future_num = 1#144 
#価格が上がるか下がるかを予測する未来の10分足数です。
#ここでは10分足データの144足分のため、1日先の価格が上がるか下がるか、の予測となります。

feature_num = 5 
#入力データの特徴量の数で、Volume、Open, High, Low, Close, Adj Closeの6項目を利用します。

batch_size = 128
#LSTMが予測で利用する過去のデータポイントの数です。

time_steps = 30#50 
#LSTMが予測で利用する過去のデータポイントの数です。
# 今回は過去の50個分のデータを見て、144個先のClose値が現在に比べて上がるのか下がるのかを予測するモデルとしています。

moving_average_num = 500 
#500と指定しています。
# これは、LSTMに投入するデータは過去500足分の移動平均に対する現在の値の比率とするためです。

n_epocs = 5#30 
#LSTMのトレーニングで何epoch数分実施するかです。

val_idx_from = 800#80000  # 評価用
test_idx_from = 1000#100000  # テスト用
#それぞれデータの何行目以降を評価用、テスト用として分割するかの位置です。

lstm_hidden_dim = 16
target_dim = 1
#LSTMの隠れ層の出力サイズと最終出力サイズです。




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
#deviceはGPUの利用可否に応じでcudaまたはcpuがセットされます。



import datetime
start = datetime.date(2016, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY
stock = []
'''データ準備'''
#LSTMで学習できるようにデータを準備していきます。
#Oanda APIで取得したCSVデータを読み込みます。

''' 1. CSVファイルの読み込み '''
df = pd.read_csv('USD_JPY_201601-201908_M10.csv', index_col='Datetime')

from pandas_datareader import data as pdr
#df = pdr.get_data_yahoo(f'{code}.T',  start, end )  # 株価データの取得
print(df)






''' 2. 教師データの作成 '''
#教師データとして、144足先のClose値と現在のClose値を比較し、上がって入れば1、下がっていれば0をセットします。
#数量や価格はそのまま利用するのではなく、直近500足データの移動平均に対する率とします。
#約3.5日分の移動平均に対して何%上下しているかを予測のためのインプットとします。
#データを分割し、PyTorchで利用できるようにtorchのtensorに変換しておきます。
future_price = df.iloc[future_num:]['Close'].values # 144足先のClose値
curr_price = df.iloc[:-future_num]['Close'].values # 現在のClose値
y_data_tmp = future_price - curr_price
y_data = np.zeros_like(y_data_tmp)#元の配列と同じ形状の配列を生成する
y_data[y_data_tmp > 0] = 1
y_data = y_data[moving_average_num:]

''' 3. 価格の正規化 '''
cols = df.columns
for col in cols:
    df['Roll_' + col] = df[col].rolling(window=500, min_periods=500).mean()
    df[col] = df[col] / df['Roll_' + col] - 1

#最初の500足分は移動平均データがないため除く。後半の144足分は予測データがないため除く
X_data = df.iloc[moving_average_num:-future_num][cols].values

''' 4. データの分割、TorchのTensorに変換 '''
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
    #_init__メソッドで定義した層が実際にどのようにつながっているか（ニューラルネットワークがどのように計算を連ねていくか）はforwardメソッドで定めています。
    # forwardメソッドは入力x（あやめの特徴を示す4つのデータ）を受け取り、それをself.lstmで処理して、
    # その結果をにtorch.sigmoid関数に通した結果を今度はself.fc2メソッドで処理し、その結果を戻り値（ニューラルネットワークの計算結果）としています
    # （ここで「self.fc1とself.fc2はLinearクラスのインスタンスなのに、メソッドのように呼び出している」ことに気付くかもしれません。
    # が、PyTorchではこのような書き方ができるようになっています。
    # forwardはデータxをtensor型で受け取る
    def forward(self, X_input):
        _, lstm_out = self.lstm(X_input)# _, Return値を無視
        #print(X_input)
        # LSTMの最終出力のみを利用する。
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)
    #0 要素のテンソルを形状 [0, -1] に再形成できないのは、指定されていない寸法サイズ -1 は任意の値にすることができ、あいまいであるためです



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, x):
        #forwardはデータxをtensor型で受け取る
        # LSTMのinputは(batch_size, seq_len, input_size)にする
        # LSTMのoutputは(batch_size, seq_len, hidden_layer_size)となる
        # hidden stateとcell stateにはNoneを渡して0ベクトルを渡す
        lstm_out, (hn, cn) = self.lstm(x, None)
        # Linearのinputは(N,∗,in_features)にする
        # lstm_out(batch_size, seq_len, hidden_layer_size)のseq_len方向の最後の値をLinearに入力する
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction








#次に一つヘルパーファンクションを定義しておきます。
#このファンクションは重要で、データポイントのindexのバッチ数分の配列を受けたら、
#その各index毎に過去50個分の過去データを2つめの次元に追加してそれを一つの固まりとしてLSTMに投入できるようにします。

#バッチ毎の処理数が128、特徴量の数(ボリューム、Open, High, Low, Close, Adj Close）が6のため、
#このファンクションの入力データ（X_data）の次元は（128, 6)となります。

#この各データポイントに対して、過去50個分（time_steps数）のデータを合成してfeatsとして返します。
#そのため、戻り値の次元は(128, 50, 6）となります。2次元目に合成されたデータが過去50個分の時系列データとなります。

def prep_feature_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去のN足分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats


def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去の30日分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps, b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats

'''LSTM学習の実施'''
#ここまで準備が整ったら、実際に学習を実施してみましょう。
#LSTMのインスタンスを生成し、損失関数と最適化関数を設定します。
#loss functionは二値分類（上がるか下がるか）なので、素直にbinary classification entropy loss（BCELoss）を利用、
#optmizerはAdamを利用します。

# Prepare for training
#feature_num = 5 #volume, open, high, low, closeの5項目
#lstm_hidden_dim = 16
#target_dim = 1
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
#model = LSTM(feature_num, lstm_hidden_dim, target_dim).to(device)
loss_function = nn.BCELoss() #二値分類（上がるか下がるか）なので、素直にbinary classification entropy loss（BCELoss）を利用
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
        train_scores = model(feats) # batch size x time steps x feature_num （バッチ数、時系列データ数、特徴量数）
        print(train_scores)
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
        torch.save(model.state_dict(),'assets/models/pytorch_v1.mdl')
        print('best score updated, Pytorch model was saved!!', )

# 7. bestモデルで予測する。
model.load_state_dict(torch.load('assets/models/pytorch_v1.mdl'))
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

