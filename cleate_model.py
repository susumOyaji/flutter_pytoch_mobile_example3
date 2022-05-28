""" ライブラリ読込 """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


""" GPU設定 """
# GPUが利用できる場合はGPUを利用し、利用不可の場合はCPUを利用するための記述
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" (1) データの準備 """
# データロード
iris = datasets.load_iris()
# 説明変数 "Sepal Length", "Sepal Width", "Petal Length", "Petal Width"
X = iris.data.astype(np.float32)
# 目的変数 "Species"
Y = iris.target.astype(np.int64)
# 学習データ＆テストデータ分割
X_train, X_test,  Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=3)
# テンソル化
X_train, X_test, Y_train, Y_test = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(Y_train).to(device), torch.from_numpy(Y_test).to(device)


""" (2)モデル情報を記述 """
# nnクラスの関数(初期設定)を下記に記述
class IrisModel(nn.Module):
    
    # ユニット・層の数・活性化関数等ニューラルネットワークの模型となるものを下記に記述
    def __init__(self):
        super(IrisModel, self).__init__()
        self.model_info = nn.ModuleList([
             nn.Linear(4,6),   # 入力層
             nn.Sigmoid(),     # 活性化関数(シグモイド)
             nn.Linear(6,3),   # 出力層
            ])
    
    # 順方向の計算処理の流れを下記に記述
    def forward(self,x):
        for i in range(len(self.model_info)):
            x = self.model_info[i](x)
        return x


""" (3)モデルとパラメータ探索アルゴリズムの設定 """
model     = IrisModel().to(device)                # モデル
optimizer = optim.SGD(model.parameters(),lr=0.05) # パラメータ探索アルゴリズム
criterion = nn.CrossEntropyLoss()                 # 損失関数


""" (4) モデル学習 """
data_size  = len(X_train)           # データのサイズ
mini_batch = int(data_size * 3/4)   # ミニバッチサイズ
repeat = 1500                       # エポック数

for epoch in range(repeat):
    
    # permutation(渡した引数の数値をシャッフル)
    dx = np.random.permutation(data_size)
    
    for num in range(0,data_size,mini_batch):
        # 説明変数(ミニバッチサイズ)
        ex_var = X_train[dx[num:(num + mini_batch) if (num + mini_batch) < data_size else data_size]]
        # 目的変数(ミニバッチサイズ)
        target = Y_train[dx[num:(num + mini_batch) if (num + mini_batch) < data_size else data_size]] 
        # モデルのforward関数を用いた準伝播の予測→出力値算出
        output = model(ex_var)  
        # 上記出力値(output)と教師データ(target)を損失関数に渡し、損失関数を計算
        loss = criterion(output, target)
        # 勾配を初期化
        optimizer.zero_grad()
        # 損失関数の値から勾配を求め誤差逆伝播による学習実行
        loss.backward()
        # 学習結果に基づきパラメータを更新
        optimizer.step()


""" (5)モデルの結果を出力 """
# torch.save(model.state_dict(), "iris.model")    # モデル保存する場合
# model.load_state_dict(torch.load("iris.model")) # モデルを呼び出す場合


""" (6) モデルの性能評価 """
model.eval()
with torch.no_grad():
    pred_model  = model(X_test)
    pred_result = torch.argmax(pred_model,1) #予測値
    print("正解率: " + str(round(((Y_test == pred_result).sum()/len(pred_result)).item(),3))+"[%]")