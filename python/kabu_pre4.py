from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # x(入力)のユニット数は4 
        self.fc1 = nn.Linear(4, 10)
        # 隠れ層1のユニット数は10
        self.fc2 = nn.Linear(10, 10)
        # 隠れ層2のユニット数は10
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

#データ集めがめんどくさいので、今回はsklearnにあるirisのデータセットを使います。
iris = datasets.load_iris()

#データを「訓練データ」と「評価データ」に分けておきます。この辺の処理は他の機械学習手法でもお馴染みですね。
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25 ,random_state=0)


#次に、pytorchで扱えるように、tensor(テンソル)と呼ばれる行列のようなものに変換します。(実際には行列とは少し異なりますが、今回の主題ではないのでスルーします)
x = torch.tensor(X_train,dtype = torch.float32)
y = torch.tensor(y_train,dtype = torch.long)



# 学習モデルのインスタンスを作成
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 損失関数の定義
criterion = nn.CrossEntropyLoss()

sum_loss = 0.0 
epoch = 5000#エポック数(学習回数)は5000回
for i in range(1,epoch):
    # 勾配の初期化
    optimizer.zero_grad()
    # 説明変数xをネットワークにかける
    output = model(x)
    
    # 損失関数の計算
    loss = criterion(output, y)
    # 勾配の計算 
    loss.backward()
    # パラメタの更新
    optimizer.step()

    sum_loss += loss.item()
    if i % 1000 == 0:
        print("loss : {0}".format(sum_loss/i))

outputs = model(torch.tensor(X_test, dtype = torch.float))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
accuracy = 100 * np.sum(predicted.numpy() == y_test) / len(y_predicted)
print('accuracy: {:.1f}%'.format(accuracy))