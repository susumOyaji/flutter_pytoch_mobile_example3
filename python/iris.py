from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = load_iris()


for idx, item in enumerate(zip(iris.data, iris.target)):
    if idx == 5:
        break
    #print('data:', item[0], ', target:', item[1])




from sklearn.model_selection import train_test_split
#print('length of iris.data:', len(iris.data))  # iris.dataのデータ数
#print('length of iris.target:', len(iris.target))  # iris.targetのデータ数

# iris.dataとiris.targetに含まれるデータをシャッフルして分割
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
#print('length of X_train:', len(X_train))
#print('length of y_train:', len(y_train))
#print('length of X_test:', len(X_test))
#print('length of y_test:', len(y_test))



for idx, item in enumerate(zip(X_train, y_train)):
    if idx == 5:
        break
    #print('data:', item[0], ', target:', item[1])



import torch

X_train = torch.from_numpy(X_train).float()
y_train = torch.tensor([[float(x)] for x in y_train])
X_test = torch.from_numpy(X_test).float()
y_test = torch.tensor([[float(x)] for x in y_test])
print('X_train[0:3]\n',X_train[0:3])

'''


'''
from torch import nn

INPUT_FEATURES = 4
HIDDEN = 5
OUTPUT_FEATURES = 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN)# 入力層（入力：4、出力：5）
        self.fc2 = nn.Linear(HIDDEN, OUTPUT_FEATURES)# 隠れ層（入力：5、出力：1）

    def forward(self, x):
        x = self.fc1(x)# 入力データ→入力層
        x = torch.sigmoid(x)# 入力層→活性化関数
        x = self.fc2(x)# 活性化関数→隠れ層→出力
        return x



'''学習に向かう前'''
net = Net()  # ニューラルネットワークのインスタンスを生成

outputs = net(X_train[0:3])  # 訓練データの先頭から3個の要素を入力
print(outputs)
for idx in range(3):
    print('output:', outputs[idx], ', label:', y_train[idx])




'''学習（訓練）と精度の検証'''
net = Net()  # ニューラルネットワークのインスタンスを生成

criterion = nn.MSELoss()  # 損失関数
optimizer = torch.optim.SGD(net.parameters(), lr=0.003)  # 最適化アルゴリズム

EPOCHS = 2000  # 上と同じことを2000回繰り返す
for epoch in range(EPOCHS):
    optimizer.zero_grad()  # 重みとバイアスの更新で内部的に使用するデータをリセット
    outputs = net(X_train)  # 手順1：ニューラルネットワークにデータを入力
    loss = criterion(outputs, y_train)  # 手順2：正解ラベルとの比較
    loss.backward()  # 手順3-1：誤差逆伝播
    optimizer.step()  # 手順3-2：重みとバイアスの更新
    
    if epoch % 100 == 99:  # 100回繰り返すたびに損失を表示
        print(f'epoch: {epoch+1:4}, loss: {loss.data}')

print('training finished')



for idx, item in enumerate(zip(outputs, y_train)):
    if idx == 5:
        break
    print(item[0].data, '<--->', item[1])

