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


def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去の30日分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats



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


    
      