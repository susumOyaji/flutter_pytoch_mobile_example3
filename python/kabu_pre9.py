#@title デフォルトのタイトル テキスト
#from msilib.schema import Feature
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.datasets import make_moons  # サンプルのデータセット
from sklearn.model_selection import train_test_split  # データを学習用とテスト用に分割する関数
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier  # 決定木（分類）
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import datetime
import sys
from pandas_datareader import data as pdr
#import yfinance as yfin
#yfin.pdr_override()

#import pandas_datareader.data as web


#%%
'''予測株価のデータ取得'''
#today = datetime.date.today().strftime("%Y-%m-%d")

#次に、2014年から今日までの6年間のデータを取得しましょう。期日を決めて行きます。
#start_date = "2020-01-01"

#終了日はプログラムの実行日にしたいので、日時と文字列を相互に変換するメソッドstrftime関数を使います。
#様々なフォーマットの日付や時間を操作することが可能です。
#end_date = datetime.date.today().strftime("%Y-%m-%d")


#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start = datetime.date(2021, 1, 1)
end = datetime.date.today()

X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []


#%%
def main():
    anser = finance_csv('6758')  # Fainance Data 取得

    #これで train_X (教師データの配列＝学習データ) と train_y (それに対する 1 か 0 かのラベル＝結果) が返ってきます。
    X_train,y_train = train_data(anser)  # adjclosed = test_X


#%%
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # 説明変数をdataXに格納
    dataX = X_train
    print(dataX)

    # 目的変数をdataYに格納
    dataY = y_train
    #print(dataY)

    # この段階で説明変数を標準化しておく
    #sc = StandardScaler()
    #dataX_std = pd.DataFrame(sc.fit_transform(dataX), columns=dataX.columns, index=dataX.index)
    
   
    
    # データの分割を行う（学習用データ 0.8 評価用データ 0.2）
    #X_train, X_test, y_train, y_test = train_test_split(dataX_std, dataY, train_size=0.8, test_size=0.2, stratify=dataY)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)





    '''決定木のインスタンスを生成'''
    #clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        
    '''学習させる'''
    # train_X(教師データの配列＝学習データ) と train_y(それに対する 1 か 0 かのラベル＝結果)
    #clf.fit(X_train, y_train)

    #clf.fit(train_X, train_y)
    #これであとは clf.predict() 関数にテストデータを渡すことで予測結果が返ってくるようになります。

    #%%
    '''実際に予想する'''
    # 過去 30 日間のデータでテストデータを作成する
    i = -15
    s = i + 14

    test_X = anser.iloc[i:s].values  # '''テストデータの数値の配列 (test_X)'''i~s
    X = np.array(test_X).reshape(-1, 14)
    #print(X)


    decis(X,y_test)  # 決定木
    logis(X)
    svc(X)
    kne(X)
    random(X)
    




    '''
    #print("test_X:テストデータの数値=", X)

    #clf.predict() 関数にテストデータXを渡すことで予測結果が返ってくる
    results = clf.predict(X)

    y_test.append(results)  # 予測結果 (test_y)

    print("test_y:予測結果=", y_test)

    if results < 1:  # その翌日、株価は上がったか？
        res = "whereabouts=下落"
    else:
        res = "Soaring=高騰"

    print("予測：", res)
    '''
   


#%%
def finance_csv(code):
    #df = web.DataReader(f'{code}.T', 'yahoo', start, end)
    adjclosed = pdr.get_data_yahoo(f'{code}.T', start, end)["Adj Close"]  # 株価データの 調整後終値を抽出取得
    #closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
    #prices = np.array(stock, dtype='f8')  # 浮動小数点数による NumPy 配列にしておく
    #df.to_csv('data/stocks_price_data/kabu_pre9_data.csv')  # csv書き出し
    return adjclosed

#%%
'''学習'''
'''教師データの数値の配列(train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくるというそれだけです。'''
'''###教師データをつくる'''
# 過去の株価と上がり下がり(結果)を学習する
# まずは一番面倒な株価の調整後終値(Adj Clouse)から教師データを作るまでのコードを用意します。
# これは終値のリストを渡すと train_X と train_y が返るようにすれば良いでしょう。


def train_data(adjclosed):  # arr = test_X
    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-30, -15):
        s = i + 14  # 14 日間の変化を素性にする
        feature = adjclosed.iloc[i:s]  # i~s行目を取り出す
        if feature[-1] < adjclosed[s]:  # その翌日、株価は上がったか？
            y_train.append(1)  # YES なら 1 を
        else:
            y_train.append(0)  # NO なら 0 を
        X_train.append(feature.values)
    
    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(X_train), np.array(y_train)














'''
アルゴリズム	                            概要
ロジスティック回帰(LogisticRegression)	    0〜1の確率で返される2択の予測結果を分類に使用する手法
サポートベクターマシン(SVC)	                クラスを最大に分ける境界線を引いて、分類する手法
K近傍法(KNeighborsClassifier)	           予測対象データの近傍にあるデータ群の多数決で分類する手法
ランダムフォレスト(RandomForestClassifier)	決定木(Yes/Noの分岐条件)を複数作って多数決で分類する手法
デシジョンツリー
'''

'''ロジスティック回帰による学習と評価'''
def logis(X_test):
    # 分類器を作成（ロジスティック回帰）
    clf = LogisticRegression(max_iter=10000)

    # 学習
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)
    print('ロジスティック回帰:=',y_pred)  # 予測結果 (test_y)

    # 正解率を表示
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #0.7488372093023256

    # 適合率を表示
    from sklearn.metrics import precision_score
    #print(precision_score(y_test, y_pred))
    #0.4158878504672897

    # F値を表示
    from sklearn.metrics import f1_score
    #print(f1_score(y_test, y_pred))
    #0.39732142857142855




'''サポートベクターマシン（SVM）による学習と評価'''
def svc(X_test):
    # 分類器を作成（サポートベクターマシン）
    clf = SVC(kernel='rbf', gamma=0.1, probability=True)
        # 学習
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)
    print('サポートベクターマシン:=',y_pred)
    # 正解率を表示
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #0.7581395348837209

    # 適合率を表示
    from sklearn.metrics import precision_score
    #print(precision_score(y_test, y_pred))
    #0.42168674698795183

    # F値を表示
    from sklearn.metrics import f1_score
    #print(f1_score(y_test, y_pred))
    #0.35000000000000003


'''K近傍法（KNN）による学習と評価'''
def kne(X_test):
    from sklearn.neighbors import KNeighborsClassifier

    # 分類器を作成（K近傍法）
    clf = KNeighborsClassifier(n_neighbors=9)

    # 学習
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)
    print('K近傍法:=',y_pred)
    # 正解率を表示
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #0.68

    # 適合率を表示
    from sklearn.metrics import precision_score
    #print(precision_score(y_test, y_pred))
    #0.31543624161073824

    # F値を表示
    from sklearn.metrics import f1_score
    #print(f1_score(y_test, y_pred))
    #0.3533834586466166



'''ランダムフォレストによる学習と評価'''
from sklearn.ensemble import RandomForestClassifier
def random(X_test):
    # 分類器を作成（ランダムフォレスト）
    clf = RandomForestClassifier(
    random_state=100,
    n_estimators=50,
    min_samples_split=100
    )

    # 学習
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)
    print('ランダムフォレスト:=',y_pred)
    # 正解率を表示
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #0.7851162790697674

    # 適合率を表示
    from sklearn.metrics import precision_score
    #print(precision_score(y_test, y_pred))
    #0.5121951219512195

    # F値を表示
    from sklearn.metrics import f1_score
    #print(f1_score(y_test, y_pred))
    #0.35294117647058826



def decis(X_test,y_test):
    # 決定木の学習を行う
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None)

    # 学習
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)
    print('決定木:=',y_pred)
    # 正解率を表示
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #0.7851162790697674

    # 適合率を表示
    from sklearn.metrics import precision_score
    #print(precision_score(y_test, y_pred))
    #0.5121951219512195

    # F値を表示
    from sklearn.metrics import f1_score
    #print(f1_score(y_test, y_pred))
    #0.35294117647058826






if __name__ == '__main__':
    main()
   


