#必要な機能のimport。
from socket import herror
import pandas as pd#　                              データフレームを扱うための機能。
from urllib.request import urlretrieve#　           ネット上からファイルをダウンロードし保存するのに使う機能。
from sklearn.linear_model import LinearRegression#　今回使うアルゴリズム
import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
from sklearn.tree import DecisionTreeClassifier # 決定木（分類）
import numpy as np
import csv
from sklearn import svm



'''線形回帰モデル'''
'''主に利用するメソッドは以下の通りです。'''
'''
fitメソッド：線形モデルの重みを学習
predictメソッド：線形モデルから目的変数を予測
scoreメソッド：決定係数（線形モデルがどの程度目的変数を説明できるか）を出力
'''
from sklearn.datasets import load_boston
boston = load_boston() # データセットの読み込み

import pandas as pd
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) # 説明変数(boston.data)をDataFrameに保存
boston_df['MEDV'] = boston.target # 目的変数(boston.target)もDataFrameに追加


boston_df.head()

import matplotlib.pyplot as plt
#%matplotlib inline
plt.scatter(boston_df['RM'], boston_df['MEDV']) # 平均部屋数と住宅価格の散布図をプロット

plt.title('Scatter Plot of RM vs MEDV')    # 図のタイトル
plt.xlabel('Average number of rooms [RM]') # x軸のラベル
plt.ylabel('Prices in $1000\'s [MEDV]')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()                                 # 図の表示


boston_df[['RM','MEDV']].corr()

'''
線形回帰モデルの構築
fitメソッドで重みを学習することで、線形回帰モデルを構築します。
学習の際には、説明変数Xと目的変数YにはNumpyの配列を利用するため、
values属性で説明変数と目的変数の列からNumpyの配列を取り出しています。
'''

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

X = boston_df[['RM']].values         # 説明変数（Numpyの配列）
Y = boston_df['MEDV'].values         # 目的変数（Numpyの配列）

lr.fit(X, Y)                         # 線形モデルの重みを学習

print('coefficient = ', lr.coef_[0]) # 説明変数の係数を出力
print('intercept = ', lr.intercept_) # 切片を出力

coefficient =  9.10210898118
intercept =  -34.6706207764

plt.scatter(X, Y, color = 'blue')         # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red') # 回帰直線をプロット

plt.title('Regression Line')               # 図のタイトル
plt.xlabel('Average number of rooms [RM]') # x軸のラベル
plt.ylabel('Prices in $1000\'s [MEDV]')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()                                 # 図の表示

'''
線形回帰モデルの性能評価
学習により得られた線形モデルの性能を評価するには、学習には用いていないデータでモデルを検証することが必要です。
構築したモデルを今後利用する（例：売上予測モデルの予測結果を使ってビジネス計画を策定する・なんらかの施策を打っていく）ことを考慮すると、
モデル構築時には得られない将来のデータに対して精度よく予測できることが重要であるためです。
そのためには、まず手元のデータを学習データと検証データに分けます。
そして、学習データでモデルを構築し、検証データを将来のデータと見立て、これに対するモデルの性能（汎化性能と呼ぶ）を評価します。

以下のコードでは、model_selectionのtrain_test_split（公式ドキュメント：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html ）を利用して、
データを学習用と検証用に7:3の割合で分割し、学習データを用いて線形モデルを構築しています。
'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0) # データを学習用と検証用に分割

lr = LinearRegression()
lr.fit(X_train, Y_train) # 線形モデルの重みを学習

'''
線形回帰モデルの性能評価には、主に以下の方法・指標を利用します。

残差プロット：残差（目的変数の真値と予測値の差分）を可視化
平均二乗誤差：残差平方和をデータ数で正規化した値
決定係数：相関係数の二乗

残差プロットは、残差（目的変数の真値と予測値の差分）の分布を可視化したものです。
線形モデルが目的変数を完璧に予測できる場合は残差は0となるので、予測精度の良い線形モデルの残差プロットは、0を中心にランダムにばらついたものになります。
残差プロットに何かパターンが見られる場合は、線形モデルで説明しきれない情報があることが示唆されます。以下のコードは、残差プロットを描画します。
'''


Y_pred = lr.predict(X_test) # 検証データを用いて目的変数を予測

plt.scatter(Y_pred, Y_pred - Y_test, color = 'blue')      # 残差をプロット 
plt.hlines(y = 0, xmin = -10, xmax = 50, color = 'black') # x軸に沿った直線をプロット
plt.title('Residual Plot')                                # 図のタイトル
plt.xlabel('Predicted Values')                            # x軸のラベル
plt.ylabel('Residuals')                                   # y軸のラベル
plt.grid()                                                # グリッド線を表示

plt.show()                                               # 図の表示

#残差プロットを見てみると、残差は0を中心に分布していますが、線形モデルで説明しきれないパターンもあるように見えます。

#平均二乗誤差は、残差の平方和をデータ数で正規化したものであり、モデルの性能を数値化するのに役立ちます。
#もちろん、誤差が小さいほどモデルの性能は良いといえます。平均二乗誤差は、metricsのmean_squared_errorを利用することで算出できます。


from sklearn.metrics import mean_squared_error

Y_train_pred = lr.predict(X_train) # 学習データに対する目的変数を予測
print('MSE train data: ', mean_squared_error(Y_train, Y_train_pred)) # 学習データを用いたときの平均二乗誤差を出力
print('MSE test data: ', mean_squared_error(Y_test, Y_pred))         # 検証データを用いたときの平均二乗誤差を出力

'''
学習データ、検証データそれぞれを用いたときの平均二乗誤差を比較すると、検証データを用いたときの誤差の方が大きいことがわかります。
このことから、構築した線形モデルは学習データにフィットしすぎている（過学習と呼ぶ）ことが示唆されます。
'''

#MSE train data:  42.1576508631
#MSE test data:  47.0330474798

#決定係数も、線形モデルの予測誤差を反映した指標であり、値が大きいほど線形モデルがデータにフィットしているといえます。
# 決定係数は、metricsのr2_scoreを利用することで算出できます。また、LinearRegressionモデルのscoreメソッドでも算出できます。


from sklearn.metrics import r2_score

print('r^2 train data: ', r2_score(Y_train, Y_train_pred))
print('r^2 test data: ', r2_score(Y_test, Y_pred))

#学習データ、検証データそれぞれを用いたときの決定係数を比較すると、検証データを用いたときの決定係数の方が小さいことがわかります。
# このことからも、構築した線形モデルには過学習が起こっている可能性があることがわかります。

#r^2 train data:  0.502649763004
#r^2 test data:  0.435143648321
#ここまでは単回帰のコード例を示してきましたが、重回帰の場合も簡単に試すことができます。
#例えば、住宅価格（目的変数）と、平均部屋数および低所得者の割合（説明変数）の関係を表現する線形回帰モデルは、以下のようなコードで構築することができます。


lr = LinearRegression()

X = boston_df[['RM', 'LSTAT']].values         # 説明変数（Numpyの配列）
Y = boston_df['MEDV'].values         # 目的変数（Numpyの配列）

lr.fit(X, Y)                         # 線形モデルの重みを学習


'''
おわりに
この記事では、scikit-learnライブラリで線形回帰をする方法について簡単に触れました。
目的変数をより精度よく表現する線形モデルを構築するためには、特徴量（説明変数）選択や正則化を行うことを検討する必要がありますが、その点についても今後まとめてみようと思います。
'''













X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'


'''教師データの数値の配列 (train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる'''
'''次回の株価予測AIを作る'''
#次回の株価を直近10日分の株価の推移を基に予測しましょう。株価予測AIの条件設定の部分を以下のように変更して、学習させます。
'''条件設定。'''
interval = 50 #直近何日分の株価を基に予測するか。(test_x)
future = 1 #何日先の株価を予測するかは”future”の値を変更する。(test_y)

#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)#教師データ(今までのデータ)
end_train = datetime.date(2021,12,31)


#datetime.date.today() + relativedelta(days=-1)
start_test = datetime.date(2022, 1, 1)#試験データ
#start_test = datetime.date.today() + relativedelta(days= - (interval+future))#試験データ
end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）





'''使うデータを読み込む。'''
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)# 試験データのcsvファイルを読み込む。


print(Stock_test_df,len(Stock_test_df))

length = len(Stock_test_df)

#if interval < length:
    #interval = length

#Stock_df = pd.read_csv("7974_2018.csv", encoding = "ANSI")#　　 教師データのcsvファイルを読み込む。
#StockNew_df = pd.read_csv("7974_2019.csv", encoding = "ANSI")#　試験データのcsvファイルを読み込む。

'''説明変数としてのinterval回分の株価の推移と、それに対応する目的変数としてのfuture回後の株価の配列を作るための関数の作成。'''
#interval = 直近何回分の株価を基に予測するか。
#future =  何回後の株価を予測するか。
def make_data(data):
    x = []#　説明変数
    y = []#　目的変数
    temps=list(data["Adj Close"])#株価（終値調整値）の配列を作成。
    for i in range(len(temps) - future):#i番目の株価について、(50-1)=len49
        if i < interval:continue#iがinterval(50)分までは予測できないから、iがinterval(50)より小さければ何もしない。
        y.append([temps[i + future - 1]])#i + future(1)番目の株価をyに付け足す。
        xa = []#i - interval番目からi-1番目の株価を格納するxaを作成。
        for p in range(interval):#i - interval番目からi-1番目の株価を
            d = i - interval + p
            xa.append(temps[d])#xaにため込んでひとまとめ（[　]で囲まれた、interval回分の株価データ群をイメージ）にして、
        x.append(xa)#xにxaを付けたしていく。
    return(x, y)#完成したx,yを返す。



def train_data(arr):  # arr = test_X
    train_X = []  # 教師データ
    train_y = []  # 上げ下げの結果の配列

    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-interval, -15):
        s = i + 14  # 14 日間の変化を素性にする
        feature = arr.iloc[i:s]  # i(-50)~s(-36)行目を取り出す
        if feature[-1] < arr[s]:  # その翌日、株価は上がったか？
            train_y.append(1)  # YES なら 1 を
        else:
            train_y.append(0)  # NO なら 0 を
        train_X.append(feature.values)

    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(train_X), np.array(train_y)

'''教師データと試験データの作成'''
#train_x,train_y = make_data(Stock_train_df)#　 教師データ作成。
#test_x,test_y = make_data(Stock_test_df)#　試験データ作成。

#train_x, train_y = train_data(Stock_train_df["Adj Close"])  # 　 教師データ作成。
#test_x, test_y = train_data(Stock_test_df["Adj Close"])  # 　試験データ作成。
#print("予測結果(test_y)=",test_y)


'''学習と予測'''
#アルゴリズムの選択。
lr = LinearRegression(normalize = True)#   アルゴリズムの選択。

#from sklearn.svm import SVC
#lr = SVC(verbose=True, random_state=2525)

# 正則化パラメータCを複数振ってそれぞれのモデルを生成するループ
#param_C = [0.1, 1, 1000]                                        # 正則化パラメータリスト
#lr = LogisticRegression(C=param_C[j], solver='lbfgs') 

# 決定木のインスタンスを生成
#lr = DecisionTreeClassifier(max_depth=2, random_state=0)

'''教師データと試験データの作成'''
learning = False

if learning == True:
    train_x,train_y = make_data(Stock_train_df)#　 教師データ作成。
    test_x,test_y = make_data(Stock_test_df)#　試験データ作成。
else:
    train_x, train_y = train_data(Stock_train_df["Adj Close"])  # 　 教師データ作成。
    test_x, test_y = train_data(Stock_test_df["Adj Close"])  # 　試験データ作成。

print("テストデータ(test_x)=", test_x)


'''教師データの数値の配列 (train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる'''
'''学習'''
lr.fit(train_x, train_y)



'''予測'''
#テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる
test_y = lr.predict(test_x)#テストデータの数値の配列 (test_X)
print("翌日予測(test_y=)",test_y)

yesterdayprice = Stock_test_df["Adj Close"][interval+3]

# 翌日、株価は上がったか？
if test_y[0] < yesterdayprice:
#if pre_y[0] < 1:
    res = "高騰"
else:
    res = "下落"

print("予測：", res)




#グラフ表示用
today = "{0:%Y/%m/%d}".format(end_test)
yesterday = end_test#datetime.date.today() + relativedelta(days=-1)
print(end_test)

 
header = pd.DataFrame([[f'{yesterday}',f'{yesterdayprice}',f'{res}']],columns={'date','price','予測'})
header.to_csv('data/stocks_price_data/stock_data_history.csv')  # 書き出しCSVファイルで株価データを保存



print(lr.score(test_x, test_y))

import matplotlib.pyplot as plt
import japanize_matplotlib

# pyplot.plot(x,y)
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(list(range(1, interval + 1)),
            list(Stock_test_df["Adj Close"])[0: interval], c='k', label="過去の実測値（教師データ=学習データ）")
plt.plot(list(range(interval, interval + future + 1)),
         list(Stock_test_df["Adj Close"])[interval - 1: interval  + future], c='b', label="現在の実測値")
plt.plot([interval, interval + future],
         [list(Stock_test_df["Adj Close"])[interval  - 1], test_y[0]], c='r', label=f'予測値= {today} {res}')

#x軸の目盛りをつける位置を設定。plt.xticks([目盛りをつけたい場所])
plt.xticks(list(range(2, interval + future + 1, 2)))
plt.xlabel("（学習日数）")  #x軸のラベルを設定。
plt.ylabel("株価（円）")  #y軸のラベルを設定。
plt.legend(bbox_to_anchor=(0.5, 1.0),  # 凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))
           loc='best',  # loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad=0,  # borderaxespad = anchorとlocの距離
           )  # prop={"family":フォント名}
#plt.show()  # グラフを描画





from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#訓練データ,テストデータに対する予測
X = np.array(train_y).reshape(-1,1)
print(X)

#ValueError: X には 10 個のフィーチャがありますが、LinearRegression は 11 個のフィーチャを入力として期待しています
y_pred_train = lr.predict(X)
y_pred_test = lr.predict(test_x)

#最初の１０サンプルだけ表示してみる
print(y_pred_train[:10])
print(y_pred_test[:10])


#訓練データ
print('accuracy：', accuracy_score(y_true=y_train, y_pred=y_pred_train))
print('precision：', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall：', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1 score：', f1_score(y_true=y_train, y_pred=y_pred_train))
print('confusion matrix = \n', confusion_matrix(y_true=y_train, y_pred=y_pred_train))
"""output
accuracy =  0.9894366197183099
precision =  0.9829545454545454
recall =  1.0
f1 score =  0.9914040114613181
confusion matrix = 
 [[108   3]
 [  0 173]]
"""

#テストデータ
print('accuracy：', accuracy_score(y_true=y_test, y_pred=y_pred_test))
print('precision：', precision_score(y_true=y_test, y_pred=y_pred_test))
print('recall：', recall_score(y_true=y_test, y_pred=y_pred_test))
print('f1 score：', f1_score(y_true=y_test, y_pred=y_pred_test))
print('confusion matrix = \n', confusion_matrix(y_true=y_test, y_pred=y_pred_test))
"""output
accuracy =  0.9754385964912281
precision =  0.9783783783783784
recall =  0.9836956521739131
f1 score =  0.981029810298103
confusion matrix = 
 [[ 97   4]
 [  3 181]]
"""


'''実測値'''
#plt.plot(list(range(1, interval + 1)),list(StockNew_df["Adj Close"])[0 : interval], c = 'k', label="実測値（説明変数）")

'''実測値'''
#plt.plot(list(range(interval, interval + future + 1)),list(StockNew_df["Adj Close"])[interval - 1 : interval + future],c = 'b', label="実測値")

'''予測値'''
#plt.plot([interval,interval + future],[list(StockNew_df["Adj Close"])[9], pre_y[0]],c = 'r', label="予測値")

#plt.xticks(list(range(2, interval + future + 1, 2)))#　　　　      x軸の目盛りをつける位置を設定。plt.xticks([目盛りをつけたい場所])
#plt.xlabel("（回）", fontname="MS Gothic")#　　　　　　　　　　      x軸のラベルを設定。
#plt.ylabel("株価（円）", fontname="MS Gothic")#　　　　　　　　　　　 y軸のラベルを設定。
#plt.legend(bbox_to_anchor = (1.0, 1.0),#                          凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))
#           loc = 'upper right',#                                    loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
#           borderaxespad = 0,#                                     borderaxespad = anchorとlocの距離
#           prop={"family":"MS Gothic"})#　　　　　　　　　　　　　　　prop={"family":フォント名}
#plt.show()#　　　　　　　　　　　　　　　　　　　　                    グラフを描画


'''
plt.figure(figsize= (10, 6), dpi = 100)
plt.plot(list(range(1, interval + 1)),
        list(StockNew_df["Adj Close"])[0 : interval], c = 'k', label="実測値（説明変数=学習データ）")
#plt.plot(list(range(1, len(StockNew_df["Adj Close"]) + 1)),list(StockNew_df["Adj Close"]), c='b', label="実測値")
plt.plot(list(range(interval, interval + future + 1)),
        list(StockNew_df["Adj Close"])[interval - 1 : interval + future],c = 'b', label="実測値=テストデータ")
plt.plot(list(range(interval + future + 1, len(pre_y) + interval + future + 1)), pre_y, c='r', label="予測値=結果")
plt.xlabel("（学習日数）")  #x軸のラベルを設定。
plt.ylabel("株価（円）")  #y軸のラベルを設定。
plt.legend(bbox_to_anchor=(0.5, 1.0),  # 凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))
           loc='upper left',  # loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad=0,  # borderaxespad = anchorとlocの距離
)  #prop={"family":フォント名}
#plt.show()
'''
'''
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6), dpi = 100)
plt.plot(list(range(1,len(StockNew_df["Adj Close"]) + 1)),list(StockNew_df["Adj Close"]), c = 'b', label="実測値")#
plt.plot(list(range(interval + future + 1, len(pre_y) + interval + future + 1)),pre_y,c = 'r', label="予測値")
plt.xlabel("（回）", fontname="MS Gothic")#x軸のラベルを設定。
plt.ylabel("株価（円）", fontname="MS Gothic")#y軸のラベルを設定。
plt.legend(bbox_to_anchor = (0.5, 1.0),#                          凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))
           loc = 'upper left',#                                    loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad = 0,#                                     borderaxespad = anchorとlocの距離
           prop={"family":"MS Gothic"})#prop={"family":フォント名}
#plt.show()#グラフを描画
'''

