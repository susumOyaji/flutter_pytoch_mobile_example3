#@title デフォルトのタイトル テキスト
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.datasets import make_moons # サンプルのデータセット
from sklearn.model_selection import train_test_split # データを学習用とテスト用に分割する関数
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier # 決定木（分類）
from sklearn.linear_model import LinearRegression
import datetime
import sys
import pandas_datareader.data as web








#%%
start = datetime.date(2021, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY
stock = []
adj_closed = []

def main():
    anser = get_csv('6758')

    #これで train_X (教師データの配列＝学習データ) と train_y (それに対する 1 か 0 かのラベル＝結果) が返ってきます。
    X_train,y_train = train_data(anser)  # adjclosed = test_X
    print(X_train)

    #データの分割を行う（学習用データ 0.8 評価用データ 0.2）
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    # 過去 15日間のデータでテストをする
    i = -15
    s = i + 14
    test_X = anser.iloc[i:-1].values  # '''テストデータの数値の配列 (test_X)'''
    X = np.array(test_X).reshape(-1, 14)
    print(X)

    logis(X_train, y_train, y_test, X)  # X= x_test(評価データ),y_test=(結果データ)
   


#%%
'''ロジスティック回帰による学習と評価'''
from sklearn.linear_model import LogisticRegression


def logis(X_train, y_train, y_test,X_test):
    # 分類器を作成（ロジスティック回帰）
    clf = LogisticRegression(max_iter=10000)

    # 学習
    clf.fit(X_train, y_train)
    #これであとは clf.predict() 関数にテストデータ(X_test)を渡すことで予測結果が返ってくるようになります。
    #1 が返ってくれば株価は「上がる」
    #0 が返ってくれば株価は「下がる」


    # 予測
    y_pred = clf.predict(X_test)

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



def get_csv(code):
    df = web.DataReader(f'{code}.T', 'yahoo', start, end)
    adj_closed = web.DataReader(f'{code}.T', 'yahoo', start, end)["Adj Close"]  # 株価データの 調整後終値を抽出取得
    closed = web.DataReader(f'{code}.T', 'yahoo', start, end)["Close"]  # 株価データの取得
    prices = np.array(stock, dtype='f8') # 浮動小数点数による NumPy 配列にしておく
    df.to_csv('data/stocks_price_data/kabu_pre7_data.csv')#csv書き出し
    return adj_closed


#%%
'''学習'''
'''教師データの数値の配列(train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくるというそれだけです。'''
'''教師データをつくる'''
# 過去の株価と上がり下がり(結果)を学習する
# まずは一番面倒な株価の調整後終値(Adj Clouse)から教師データを作るまでのコードを用意します。
# これは終値のリストを渡すと train_X と train_y が返るようにすれば良いでしょう。

def train_data(arr):#arr = test_X
    train_X = []#教師データ
    train_y = []#上げ下げの結果の配列

    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-30, -15):
        s = i + 14 # 14 日間の変化を素性にする
        feature = arr.iloc[i:s]  # i~s行目を取り出す
        if feature[-1] < arr[s]: # その翌日、株価は上がったか？
            train_y.append(1) # YES なら 1 を
        else:
            train_y.append(0) # NO なら 0 を
        train_X.append(feature.values)
    
    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(train_X), np.array(train_y)

anser = get_csv('6758')
#これで train_X (教師データの配列＝学習データ) と train_y (それに対する 1 か 0 かのラベル＝結果) が返ってきます。
res = train_data(anser)#adjclosed = test_X
print('(train_X) = ',res[0])
print('(train_y) = ',res[1])

#%%
'''リターンインデックスを算出する'''
###生の株価データそのままですと、会社ごとに価格帯も全然ちがいますから教師データ(train_X)としてはちょっと使いづらいです。
# 正規化しても良いのですが、ここは資産価値の変化をあらわすリターンインデックスに注目しましょう。
# 算出方法は pandas で求まります。

returns = pd.Series(adj_closed).pct_change() # 騰落率を求める
ret_index = (1 + returns).cumprod() # 累積積を求める
ret_index[0] = 1 # 最初の値を 1.0 にする 

'''リターンインデックスの変化を決定木に学習させる'''
#ここからがキモです。
#こうして求まったリターンインデックスから教師データ(train_X)を抽出し分類器に学習させます。
train_X, train_y = train_data(ret_index)

# 決定木のインスタンスを生成
clf = DecisionTreeClassifier(max_depth=2, random_state=0)

# 学習させる
clf.fit(train_X, train_y)
#これであとは clf.predict() 関数にテストデータを渡すことで予測結果が返ってくるようになります。
#1 が返ってくれば株価は「上がる」
#0 が返ってくれば株価は「下がる」
#と予測されたことになります。

#%%
'''うまく学習したかどうか分類器を試す'''
#さっそく試してみましょう。
#まずはテストとして、教師データ(train_X)とまったく同じデータをテストデータとして流してみます。

test_y = []
# 過去 30 日間のデータでテストをする
for i in np.arange(-30, -15):
    s = i + 14
    # リターンインデックスの(train_X)と全く同じ期間をテストとして分類させてみる
    test_X = ret_index.iloc[i:s].values#'''テストデータの数値の配列 (test_X)'''
    X = np.array(test_X).reshape(-1, 14)

    # 結果を格納して返す
    result = clf.predict(X)
    test_y.append(result[0])#予測結果 (test_y)

#print(test_X)
#print(result)


print(train_y)  # 期待すべき答え
#=> [1 1 1 0 1 1 0 0 0 1 0 1 0 0 0]

print(np.array(test_y))  # 分類器が出した予測
#=> [1 1 1 0 1 1 0 0 0 1 0 1 0 0 0]
#おや、まったく同じ。すなわち全問正解のようですね。




#%%
'''実際に予想する'''

#start = datetime.date(2021, 1, 1)
#end = datetime.date.today()
#code = '6758'  # SONY
#stock = []

#df = web.DataReader(f'{code}.T', 'yahoo', start, end)
#adjclosed = web.DataReader(f'{code}.T', 'yahoo', start, end)["Adj Close"]  # 株価データの 調整後終値を抽出取得
#3/4 の終値を予測する
#3/8 までの 90 営業日のデータをもとに予測するとこうなりました。

'''リターンインデックスを算出する'''
#生の株価データそのままですと、会社ごとに価格帯も全然ちがいますから教師データ(train_X)としてはちょっと使いづらいです。
# 正規化しても良いのですが、ここは資産価値の変化をあらわすリターンインデックスに注目しましょう。
# 算出方法は pandas で求まります。

#returns = pd.Series(adjclosed).pct_change() # 騰落率を求める
#ret_index = (1 + returns).cumprod() # 累積積を求める
#ret_index[0] = 1 # 最初の値を 1.0 にする 

'''リターンインデックスの変化を決定木に学習させる'''
#ここからがキモです。
#こうして求まったリターンインデックスから教師データ(train_X)を抽出し分類器に学習させます。
#train_X, train_y = train_data(ret_index)

# 決定木のインスタンスを生成
clf = DecisionTreeClassifier(max_depth=2, random_state=0)

# 学習させる
#clf.fit(train_X, train_y)
clf.fit(res[0], res[1])
#これであとは clf.predict() 関数にテストデータを渡すことで予測結果が返ってくるようになります。
#1 が返ってくれば株価は「上がる」
#0 が返ってくれば株価は「下がる」
#と予測されたことになります。
#まずはテストとして、教師データとまったく同じデータをテストデータとして流してみます。

test_y = []
# 過去 15日間のデータでテストをする
#for i in np.arange(-15, 1):
i = -15
s = i + 14
test_X = adjclosed.iloc[i:-1].values#'''テストデータの数値の配列 (test_X)'''
X = np.array(test_X).reshape(-1, 14)
print(X)

# 結果を格納して返す
result = clf.predict(X)
test_y.append(result[0])#予測結果 (test_y)

print(np.array(test_y))  # 分類器が出した予測


print(test_X)


if result[0] < 1: # その翌日、株価は上がったか？
    res = "下がる"
else:
    res = "上がる"

print("予測：",res)