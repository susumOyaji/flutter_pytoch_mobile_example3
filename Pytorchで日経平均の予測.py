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
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

future_num = 1 #何日先を予測するか
feature_num = 7 #'始値', '高値','安値','終値','5日平均','25日平均','75日平均'の7項目
batch_size = 128

time_steps = 30 #lstmのtimesteps
moving_average_num = 30 #移動平均を取る日数
n_epocs = 50

lstm_hidden_dim = 16
target_dim = 1


'''データの読み込み'''
#データの集め方はSBI公式サイト：よくあるご質問Q&Aに乗っていました。ここで指標を選択して日経平均のデータを20年分ダウンロードしました。

#main.py
path = "./data/nikkei_heikin.csv"

model_name = "./models/nikkei.mdl"

#data load
flist = glob.glob(path)
for file in flist:
    df = pd.read_csv(file, header=0, encoding='cp932')
    dt = pd.read_csv(file, header=0, encoding='cp932')


#データをtrain, testに分割するIndex
val_idx_from = 3500
test_idx_from = 4000

future_price = df.iloc[future_num:]['終値'].values
curr_price = df.iloc[:-future_num]['終値'].values

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
cols = ['始値', '高値','安値','終値','5日平均','25日平均','75日平均']
#出来高のデータに缺損があったため抜いた

for col in cols:
    dt[col] = df[col].rolling(window=25, min_periods=25).mean()
    df[col] = df[col] / dt[col] - 1


X_data = df.iloc[moving_average_num:-future_num][cols].values

#データの分割、TorchのTensorに変換
#学習用データ
X_train = torch.tensor(X_data[:val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:val_idx_from], dtype=torch.float, device=device)
#評価用データ
X_val   = torch.tensor(X_data[val_idx_from:test_idx_from], dtype=torch.float, device=device)
y_val   = y_data[val_idx_from:test_idx_from]
#テスト用データ
X_test  = torch.tensor(X_data[test_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[test_idx_from:]
#元のデータ数は約4500ありました。トレーニングデータ数を3500、バリデーションを500残りをテストデータとします。

'''モデル定義'''
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
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)
#モデルの定義です。PytorchのライブラリからLSTMを引っ張ってきました。

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
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
loss_function = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=1e-4)


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

# bestモデルで予測する。
model.load_state_dict(torch.load(model_name))

with torch.no_grad():
    feats_test = prepare_data(np.arange(time_steps, X_test.size(0)), time_steps, X_test, feature_num, device)
    val_scores = model(feats_test)
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

