'''Pytorchで日経平均の予測'''

'''データ集め'''
'''
データはSBI証券からダウンロードできるHYPER SBIを用いてデータ集めを行いました。
精度６７％のディープラーニング株価予測モデル_1では上場企業423社の20年分の株価を学習していました。
なので僕はまずは、日経平均のみで次の日、上がるかを学習しました。データは20年分になります。

概要
データを20年分集めてtrain用、test用で分けて学習させました。
test用でのaccuracyが62.5%を記録しました。しかしながら、lossが収束しなかったり、学習の途中で混合行列から計算されるf1の値が0になるなど、問題もあります。

最終的にAccuracy60.5%に落ち着きました。
'''

#定数の定義
#main.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import torch.optim as optim
import datetime
from pandas_datareader import data as pdr
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


start = datetime.date(2016, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY

future_num = 1 #何日先を予測するか
feature_num = 6 #5#'始値', '高値','安値','終値','5日平均','25日平均','75日平均'の7項目
batch_size = 128 #LSTMの学習時に一度に投入するデータポイント数です。

#LSTMが予測で利用する過去のデータポイントの数です。
#今回は過去の50個分のデータを見て、144個先のClose値が現在に比べて上がるのか下がるのかを予測するモデルとしています。
time_steps = 60#30 #lstmのtimesteps
moving_average_num = 30 #移動平均を取る日数
n_epocs = 10

lstm_hidden_dim = 16
target_dim = 1


'''データの読み込み'''
#データの集め方はSBI公式サイト：よくあるご質問Q&Aに乗っていました。ここで指標を選択して日経平均のデータを20年分ダウンロードしました。

#main.py
#path = "./data/nikkei_heikin.csv"
path = "USD_JPY_201601-201908_M10.csv"
model_name = "assets/models/pytorch_v1.mdl"#"./models/nikkei.mdl"

#data load
#flist = glob.glob(path)
#or file in flist:

#df = pd.read_csv(path, header=0, encoding='cp932')
'''
Datetime  Volume     Open     High      Low    Close
0       2016-01-03 22: 00: 00     162  120.195  120.235  120.194  120.227
1       2016-01-03 22: 10: 00     208  120.226  120.253  120.209  120.236
2       2016-01-03 22: 20: 00     333  120.235  120.283  120.233  120.274
3       2016-01-03 22: 30: 00     359  120.274  120.304  120.268  120.286
4       2016-01-03 22: 40: 00     242  120.288  120.330  120.277  120.313
...                     ...     ...      ...      ...      ...      ...
133481  2019-08-01 03: 10: 00      62  109.152  109.187  109.152  109.182
133482  2019-08-01 03: 20: 00      70  109.180  109.212  109.172  109.204
133483  2019-08-01 03: 30: 00      52  109.207  109.210  109.185  109.195
133484  2019-08-01 03: 40: 00      73  109.197  109.232  109.192  109.216
133485  2019-08-01 03: 50: 00      71  109.218  109.252  109.218  109.239

[133486 rows x 6 columns]
'''

    #dt = pd.read_csv(file, header=0, encoding='cp932')

df = pdr.get_data_yahoo(f'{code}.T',  start, end )  # 株価データの取得
'''
High      Low     Open    Close      Volume     Adj Close
Date
2016-01-04   3066.0   2940.0   2958.0   2957.0  14332100.0   2843.607910
2016-01-05   2999.0   2930.0   2980.0   2962.5   8021400.0   2848.896973
2016-01-06   3014.0   2862.0   3013.0   2897.5  13992200.0   2786.389404
2016-01-07   2834.0   2751.0   2797.5   2796.5  20924500.0   2689.262695
2016-01-08   2895.0   2745.5   2755.0   2824.5  13972900.0   2716.188721
...             ...      ...      ...      ...         ...           ...
2022-05-30  12025.0  11660.0  11700.0  11995.0   8379800.0  11995.000000
2022-05-31  12180.0  11955.0  12145.0  12115.0   5440100.0  12115.000000
2022-06-01  12420.0  12085.0  12110.0  12370.0   4416500.0  12370.000000
2022-06-02  12205.0  11975.0  12130.0  11975.0   3888200.0  11975.000000
2022-06-03  12235.0  12080.0  12195.0  12200.0   3214800.0  12200.000000

[1588 rows x 6 columns]
'''

#print(df)
#データをtrain, testに分割するIndex
#val_idx_from = 3500
#test_idx_from = 4000

future_price = df.iloc[future_num:]['Close'].values
curr_price = df.iloc[:-future_num]['Close'].values

#future_num日後との比較した価格を正解ラベルとして扱う
y_data_tmp = future_price / curr_price
#正解ラベル用のlistを用意
y_data = np.zeros_like(y_data_tmp)

#予測するfuture_num日後が前日以上なら正解

for i in range(len(y_data_tmp)):
    if y_data_tmp[i] > 1.0:
        y_data[i] = 1

#価格の正規化をした際にmoving_average_num分空白のデータができるためその部分を削っておく
y_data = y_data[moving_average_num:]

#価格の正規化
#カラム名の取得
cols = ['High','Low','Open','Close','Volume','Adj Close']
#cols = ['High','Low','Open','Close','Volume']

#出来高のデータに缺損があったため抜いた
#for col in cols:
#    df[col] = df[col].rolling(window=25, min_periods=25).mean()
#    df[col] = df[col] / df[col] - 1


X_data = df.iloc[moving_average_num:-future_num][cols].values
#print(df,X_data)
#データをtrain, testに分割するIndex
val_idx_from = int(len(X_data) * 0.8) # 全データのうち、80% のサイズを取得
test_idx_from = 4000




#データの分割、TorchのTensorに変換
#学習用データ
X_train = torch.tensor(X_data[:val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:val_idx_from], dtype=torch.float, device=device)
#print(X_data,y_train)
#評価用データ
X_val   = torch.tensor(X_data[val_idx_from:], dtype=torch.float, device=device)
y_val   = y_data[val_idx_from:]



#テスト用データ
X_test  = torch.tensor(X_data[val_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[val_idx_from:]
#元のデータ数は約4500ありました。トレーニングデータ数を3500、バリデーションを500残りをテストデータとします。

'''モデル定義'''
'''
まず、__init__の中で層を宣言しています。
Pytorch では基本的に __init__ 内でモデルの持つ層を宣言します。
このモデルは2つの層を持っており、nn.LSTM と nn.Linear を持っていますね。
nn.LSTM は LSTM の層を、nn.Linear は全結合層を表します。
'''
#main.py
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
        # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1)) #全結合層
        #linear_out = self.dense(lstm_out[: , -1 ,: ])
        #print(linear_out)
        return torch.sigmoid(linear_out)
#モデルの定義です。PytorchのライブラリからLSTMを引っ張ってきました。
'''
そのうち欲しいのは、lstm_out の一番最後の値だけなので、output[:, -1, :] で時系列の最後の値（ここではベクトル）を取り出します。
これを Linear(self.output_layer) にぶち込むことで、サイズ1のベクトル（要するに [0.123] みたいなやつ）を得ます。
'''








class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, batch_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, batch_size)

        #self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size),
        #                    torch.zeros(1, self.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        lstm_out = self.lstm(input_seq.view(seq_len, batch_size, 1)) #lstmのデフォルトの入力サイズは(シーケンスサイズ、バッチサイズ、特徴量次元数)
        predictions = self.linear(self.hidden_cell[0].view(batch_size, -1))
        return torch.sigmoid(lstm_out)
        #return predictions[:, 0]











#main.py
def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去の30日分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats
#関数prepare_dataではLSTMに入力するデータを30日分ずつまとめる役割を果たしています。

'''学習と評価'''
#main.py
#学習

'''
LSTM学習の実施

ここまで準備が整ったら、実際に学習を実施してみましょう。
LSTMのインスタンスを生成し、損失関数と最適化関数を設定します。
loss functionは二値分類（上がるか下がるか）なので、素直にbinary classification entropy loss（BCELoss）を利用、
optmizerはAdamを利用します。
'''
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
#model = LSTM(feature_num, lstm_hidden_dim, target_dim).to(device)
loss_function = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=1e-4)



'''

'''


train_size = X_train.size(0)
best_acc_score = 0

for epoch in range(n_epocs):
    # trainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
    perm_idx = np.random.permutation(np.arange(time_steps, train_size))
    for t_i in range(0, len(perm_idx), batch_size):
        batch_idx = perm_idx[t_i:(t_i + batch_size)]
        # LSTM入力用の時系列データの準備
        feats = prepare_data(batch_idx, time_steps, X_train, feature_num, device)
        y_target = y_train[batch_idx]
        model.zero_grad()
        train_scores = model(feats) # batch size x time steps x feature_num
        loss = loss_function(train_scores, y_target.view(-1, 1))
        loss.backward()
        optimizer.step()

    # validationデータの評価
    print('EPOCH: ', str(epoch), ' loss :', loss.item())
    with torch.no_grad():
        feats_val = prepare_data(np.arange(time_steps, X_val.size(0)), time_steps, X_val, feature_num, device)
        val_scores = model(feats_val)
        #print(val_scores)
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        bi_scores = np.round(tmp_scores)
        acc_score = accuracy_score(y_val[time_steps:], bi_scores)
        roc_score = roc_auc_score(y_val[time_steps:], tmp_scores)
        f1_scores = f1_score(y_val[time_steps:], bi_scores)
        print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score, 'f1 Score :', f1_scores)

    # validationの評価が良ければモデルを保存
    if acc_score > best_acc_score:
        best_acc_score = acc_score
        torch.save(model.state_dict(),model_name)
        print('best score updated, Pytorch model was saved!!', )

''' bestモデルで予測する。'''
model.load_state_dict(torch.load(model_name))

with torch.no_grad():
    feats_test = prepare_data(np.arange(time_steps, X_test.size(0)), time_steps, X_test, feature_num, device)
    val_scores = model(feats_test)
    print(val_scores)
    tmp_scores = val_scores.view(-1).to('cpu').numpy()   
    bi_scores = np.round(tmp_scores)
    acc_score = accuracy_score(y_test[time_steps:], bi_scores)
    roc_score = roc_auc_score(y_test[time_steps:], tmp_scores)
    f1_scores = f1_score(y_test[time_steps:], bi_scores)
    print('Test ACC Score :', acc_score, ' ROC AUC Score :', roc_score, 'f1 Score :', f1_scores)
#学習ではAccuracy,　ROC Score,　f1 Scoreを出力しています。Accuracyの最高が更新されれば重みが保存されます。

'''結果'''
'''
EPOCH:  0  loss : 0.7389694452285767
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5448111497752646 f1 Score : 0.653295128939828
best score updated, Pytorch model was saved!!
EPOCH:  1  loss : 0.6844338178634644
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5550601710888793 f1 Score : 0.653295128939828
EPOCH:  2  loss : 0.7206816673278809
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5678012179208352 f1 Score : 0.653295128939828
EPOCH:  3  loss : 0.7066923975944519
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5815934464259822 f1 Score : 0.653295128939828
EPOCH:  4  loss : 0.7148252129554749
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.6025717703349283 f1 Score : 0.653295128939828
EPOCH:  5  loss : 0.6946689486503601
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.6224264172828766 f1 Score : 0.653295128939828
EPOCH:  6  loss : 0.7018400430679321
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.639100333478324 f1 Score : 0.653295128939828
EPOCH:  7  loss : 0.7006129026412964
.
.
.
.
EPOCH:  43  loss : 0.7038401961326599
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6018921270117442 f1 Score : 0.0
EPOCH:  44  loss : 0.6951379179954529
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6018921270117443 f1 Score : 0.0
EPOCH:  45  loss : 0.6788191795349121
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6018921270117443 f1 Score : 0.0
EPOCH:  46  loss : 0.6547065377235413
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6018558793678411 f1 Score : 0.0
EPOCH:  47  loss : 0.6936472654342651
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6016746411483254 f1 Score : 0.0
EPOCH:  48  loss : 0.719009280204773
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6016202696824707 f1 Score : 0.0
EPOCH:  49  loss : 0.6854437589645386
Val ACC Score : 0.5148936170212766  ROC AUC Score : 0.6014934029288096 f1 Score : 0.0
Test ACC Score : 0.6252100840336134  ROC AUC Score : 0.6860275646683414 f1 Score : 0.6915629322268327
追記
エポック数を50から500にした場合、ロスもまともな数値になったので載せておきます。

EPOCH:  0  loss : 0.7389694452285767
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5448111497752646 f1 Score : 0.653295128939828
best score updated, Pytorch model was saved!!
EPOCH:  1  loss : 0.6844338178634644
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5550601710888793 f1 Score : 0.653295128939828
EPOCH:  2  loss : 0.7206816673278809
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5678012179208352 f1 Score : 0.653295128939828
EPOCH:  3  loss : 0.7066923975944519
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.5815934464259822 f1 Score : 0.653295128939828
EPOCH:  4  loss : 0.7148252129554749
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.6025717703349283 f1 Score : 0.653295128939828
EPOCH:  5  loss : 0.6946689486503601
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.6224264172828766 f1 Score : 0.653295128939828
EPOCH:  6  loss : 0.7018400430679321
Val ACC Score : 0.4851063829787234  ROC AUC Score : 0.639100333478324 f1 Score : 0.653295128939828
EPOCH:  7  loss : 0.7006129026412964
.
.
.
.
EPOCH:  493  loss : 0.694491982460022
Val ACC Score : 0.6042553191489362  ROC AUC Score : 0.638157894736842 f1 Score : 0.5079365079365079
EPOCH:  494  loss : 0.6935185194015503
Val ACC Score : 0.6063829787234043  ROC AUC Score : 0.638157894736842 f1 Score : 0.5144356955380578
EPOCH:  495  loss : 0.6262539029121399
Val ACC Score : 0.5957446808510638  ROC AUC Score : 0.6370704654197477 f1 Score : 0.48087431693989063
EPOCH:  496  loss : 0.5570085644721985
Val ACC Score : 0.6085106382978723  ROC AUC Score : 0.6387559808612441 f1 Score : 0.5422885572139303
EPOCH:  497  loss : 0.6102970838546753
Val ACC Score : 0.6  ROC AUC Score : 0.6374691895026823 f1 Score : 0.4973262032085562
EPOCH:  498  loss : 0.6443783640861511
Val ACC Score : 0.6042553191489362  ROC AUC Score : 0.6395534290271132 f1 Score : 0.5633802816901409
EPOCH:  499  loss : 0.663628876209259
Val ACC Score : 0.6085106382978723  ROC AUC Score : 0.6380310279831811 f1 Score : 0.518324607329843
Test ACC Score : 0.6050420168067226  ROC AUC Score : 0.644737139882771 f1 Score : 0.660894660894661
'''

'''結果の考察'''
#学習途中を見ればAccuracyが更新されていない時が多々あります。そしてf1 Scoreが0になっています。
#Lossが収束していないなど問題が多いように思えます。しかしながらtestデータの結果を見ればAccuracy62.5%を記録しています。

#エポック数を増加させることによってROC Scoreやf1 Scoreがまともな数値になっています。ただAccuracy自体は60.5％になってしまいました。

#データ数や特徴量を増やしたり、自分でLSTMを実装したりすることで精度を上げていきたいと思います。

#次回は精度６７％のディープラーニング株価予測モデル_1で行われていたように3%増加する銘柄を予想していきます。

