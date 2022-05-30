# サポートベクターマシーンのimport
from sklearn import svm
# train_test_splitのimport
from sklearn.model_selection import train_test_split
# accuracy_scoreのimport
from sklearn.metrics import accuracy_score
# Pandasのimport
import pandas as pd
from pandas_datareader import data as pdr
import datetime
from sklearn import datasets
#import pandas_datareader.data as web
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # 決定木（分類）




#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start = datetime.date(2021, 1, 1)
end = datetime.date.today()

X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'

 
# 株価データの読み込み

#stock_data = pdr.DataReader(f'{code}.T', 'yahoo', start, end)
stock_data = pdr.get_data_yahoo(f'{code}.T', start, end)["Adj Close"]  # 株価データの 調整後終値を抽出取得
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
#prices = np.array(stock, dtype='f8')  # 浮動小数点数による NumPy 配列にしておく
#df.to_csv('data/stocks_price_data/kabu_pre9_data.csv')  # csv書き出し
   




#stock_data = pd.read_csv("stock_price.csv", encoding="shift-jis")


# データをDataFrameに変換
#df = pd.DataFrame(stock_data)
print(stock_data)
 
# 要素数の設定
count_s = len(stock_data)
 
'''
それぞれ、loc, atは行や列の名前で要素へアクセスし、iがついているilocやiatは名前ではなくindex（NumPyの配列のように数字）でアクセスします。
loc/ilocは複数要素をスライスすることができ、at/iatは一つの要素を取り出すことができます。
'''

# 株価の上昇率を算出、おおよそ-1.0～1.0の範囲に収まるように調整
modified_data = []
for i in range(1,count_s):
   #print(stock_data.loc['Date'])
   print((stock_data.iloc[i,5]))
   #modified_data.append((stock_data.iloc[i,5]- stock_data.iloc[i-1,5])/(stock_data.iloc[i-1,5])*20)
   modified_data.append(float(stock_data.iloc[1,['Close']] - stock_data.iloc[i-1,['Close']])/float(stock_data.iloc[i-1,['Close']])*20)
# 要素数の設定
count_m = len(modified_data)
 
# 過去４日分の上昇率のデータを格納するリスト
successive_data = []
 
# 正解値を格納するリスト　価格上昇: 1 価格低下:0
answers = []
 
#  連続の上昇率のデータを格納していく
for i in range(4, count_m):
    successive_data.append([modified_data[i-4],modified_data[i-3],modified_data[i-2],modified_data[i-1]])
    # 上昇率が0以上なら1、そうでないなら0を格納
    if modified_data[i] > 0:
        answers.append(1)
    else:
        answers.append(0)

# データの分割（データの80%を訓練用に、20％をテスト用に分割する）
X_train, X_test, y_train, y_test =train_test_split(successive_data, answers, train_size=0.8,test_size=0.2,random_state=1)
         



# 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
for i in np.arange(-30, -15):
    s = i + 14  # 14 日間の変化を素性にする
    feature = stock_data.iloc[i:s]  # i~s行目を取り出す
    if feature[-1] < stock_data[s]:  # その翌日、株価は上がったか？
        y_train.append(1)  # YES なら 1 を
    else:
        y_train.append(0)  # NO なら 0 を
    X_train.append(feature.values)

# データの分割（データの80%を訓練用に、20％をテスト用に分割する）
X_train, X_test, y_train, y_test =train_test_split(feature, y_train, train_size=0.8,test_size=0.2,random_state=1)
 


 

# サポートベクターマシーン
clf = svm.LinearSVC()
# サポートベクターマシーンによる訓練
clf.fit(X_train , y_train)
 
# 学習後のモデルによるテスト
# トレーニングデータを用いた予測
y_train_pred = clf.predict(X_train)
# テストデータを用いた予測
y_val_pred = clf.predict(X_test)
 
# 正解率の計算
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_val_pred)
 
# 正解率を表示
print("トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
print("テストデータに対する正解率：" + str(test_score * 100) + "%")
 
#トレーニングデータに対する正解率：58.333333333333336%
#テストデータに対する正解率：40.816326530612244%



#%%
# グリッドサーチのimport
from sklearn.model_selection import GridSearchCV
 
# グリッドサーチするパラメータを設定
parameters = {'C':[1, 3, 5],'loss':('hinge', 'squared_hinge')}
 
# グリッドサーチを実行
clf = GridSearchCV(svm.LinearSVC(), parameters)
clf.fit(X_train, y_train) 
 
# グリッドサーチ結果(最適パラメータ)を取得
GS_C, GS_loss = clf.best_params_.values()
print ("最適パラメータ：{}".format(clf.best_params_))
 
#最適パラメータ：{'C': 3, 'loss': 'hinge'}
 
# 最適パラメーターを指定して再度学習
clf = svm.LinearSVC(loss=GS_loss, C=GS_C)
clf.fit(X_train , y_train)
 
# 再学習後のモデルによるテスト
# トレーニングデータを用いた予測
y_train_pred = clf.predict(X_train)
# テストデータを用いた予測
y_val_pred = clf.predict(X_test)
 
# 正解率の計算
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_val_pred)
# 正解率を表示
print("トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
print("テストデータに対する正解率：" + str(test_score * 100) + "%")
 
#トレーニングデータに対する正解率：60.416666666666664%
#テストデータに対する正解率：42.857142857142854%

#%%
from sklearn.linear_model import LogisticRegression
# 分類器を作成（ロジスティック回帰）
clf = LogisticRegression(max_iter=10000)

# 学習
clf.fit(X_train, y_train)

# 予測
y_pred = clf.predict(X_test)
print('ロジスティック回帰:=',y_pred)  # 予測結果 (test_y)

# 正解率を表示
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
#0.7488372093023256

# 適合率を表示
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred))
#0.4158878504672897

# F値を表示
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))
#0.39732142857142855