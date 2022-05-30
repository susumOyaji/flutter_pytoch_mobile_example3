
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression#　今回使うアルゴリズム
from sklearn.tree import DecisionTreeClassifier # 決定木（分類）
import pandas as pd#　                              データフレームを扱うための機能。
import datetime
from dateutil.relativedelta import relativedelta

'''
目的変数と説明変数
予測において

予測したいもの⇒目的変数
予測の手がかりとするもの⇒説明変数(機械学習の文やでは特徴量という)
と言います。例えば、奥さんの機嫌を知りたいなら、奥さんの機嫌が「目的変数」で、説明変数には「季節」や「給料」、「帰宅時間」、「1日当たりの接触時間」といったものがあるのかもしれません。例えば、帰宅時間が早いほど奥さんの機嫌が悪くなるのであれば、帰宅時間を「説明変数」にして奥さんの機嫌が悪くさせないには何時に帰れば良いか分かるのかもしれません。

この「説明変数」は一つだけではなくて、複数使うこともできるようです。

単回帰モデルでは「1つの目的変数を1つの説明変数によって予測」します。

重回帰モデルでは「1つの目的変数を複数の説明変数によって予測」します。

先ほどの過学習の部分でも書いた通り、過去のデータに過剰にフィットさせると、実際には役に立たなかった、ということになりかねないため、適切な説明変数を使うことが予測では大切だそうです。

単回帰モデルについての理解
単回帰モデルでは、最小二乗法というものによって求められる回帰直線を使って予測を行います。

最小二乗法によって求められる回帰直線は2つの値によって表される散布図を近似する直線を表します。2つの値に相関関係がみられる場合、散布図にプロットされる点の集まりは直線っぽくなります。ですから、単回帰モデルは説明変数と目的変数にある程度の相関がある場合に使えそうです。相関がないようなら短回帰モデルを使ってもまともな結果は求められなさそうです。


単回帰分析をするときに説明変数をreshapeする理由
あるデータtrainがあって、そのうちのdayというカラム(列)を説明変数、yというカラム(列)を目的変数に選択して短回帰モデルを作成するとき

LinearRegression.fit(day.values,y)
1
LinearRegression.fit(day.values,y)
とすると

ValueError: Expected 2D array, got 1D array instead:
1
ValueError: Expected 2D array, got 1D array instead:
というエラーを吐かれます。で、僕しばらくこの2D(2次元)を2列の配列だと勘違いしていていました。

元のデータ”day.values”は[1,2,3,…]と定義されていて1次元配列です。

要求されているのは二次元配列なので、reshape()を使うことで実現できます。

day.values.reshape(-1,1)とすると作成される配列は[[1,2,3,…]]となります。


[[1,2,3,…]]も実際のところは元のデータ”day.values”と同じようにn行1列のデータに変わりないとは思いますが、fit()に渡す説明変数が1つの場合はreshape(-1,1)としてn行1列の2次元配列にしなければならないようでした。
reshape(-1,1)の意味

reshape(-1,n)とすると、列数がnとなるるような配列に変形します。

reshape(n,-1)とすると、行数がnとなるような配列に変形します。


'''


X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'


'''教師データの数値の配列 (train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる'''
'''次回の株価予測AIを作る'''
#次回の株価を直近10日分の株価の推移を基に予測しましょう。株価予測AIの条件設定の部分を以下のように変更して、学習させます。
'''条件設定。'''
interval = 10 #直近何日分の株価を基に予測するか。(test_x)
future = 1 #何日先の株価を予測するかは”future”の値を変更する。(test_y)

import datetime
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)#教師データ(今までのデータ)
#end_train = datetime.date(2021,12,31)
end_train= datetime.date.today() + relativedelta(days=-1)#昨日分(today-1日)まで取得できる（当日分は変動しているため）



from pandas_datareader import data as pdr
#datetime.date.today() + relativedelta(days=-1)
#start_test = datetime.date(2022, 1, 1)#試験データ
start_test = datetime.date.today() + relativedelta(days= -1)#試験データ
#end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_test = datetime.date.today()# + relativedelta(days= -1)




'''使うデータを読み込む。'''
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)["Adj Close"]  # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)["Adj Close"]# 試験データのcsvファイルを読み込む。






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



'''
教師データをつくる
まずは一番面倒な株価の調整後終値から教師データを作るまでのコードを用意します。
これは終値のリストを渡すと train_X と train_y が返るようにすれば良いでしょう。
'''
def train_data(arr):  # arr = test_X
    train_X = []  # 教師データ
    train_y = []  # 上げ下げの結果の配列

    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-30, -15):
        s = i + 14  # 14 日間の変化を素性にする
        feature = arr.iloc[i:s]  # i(-50)~s(-36)行目を取り出す
        if feature[-1] < arr[s]:  # その翌日、株価は上がったか？
            train_y.append(1)  # YES なら 1 を
        else:
            train_y.append(0)  # NO なら 0 を
        train_X.append(feature.values)

    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(train_X), np.array(train_y)
    #これで train_X (教師データの配列) と train_y (それに対する 1 か 0 かのラベル) を返します。





'''
reshape(-1,1)の意味
reshape(-1,n)とすると、列数がnとなるるような配列に変形します。
reshape(n,-1)とすると、行数がnとなるような配列に変形します。

X = test_X.reshape(1,-1)
result = lr.predict(X)
print(result)
'''




'''
# 決定木の学習を行う
# 決定木のインスタンスを生成
tree = DecisionTreeClassifier(criterion='gini', max_depth=None)

# 学習させる
tree.fit(train_X, train_y)


# 決定木のインスタンスを生成
clf = tree.DecisionTreeClassifier()
# 学習させる
clf.fit(train_X, train_y)


pred = tree.predict(test_x)
'''



'''学習と予測'''
#アルゴリズムの選択。
lr = LinearRegression(normalize = True)#   アルゴリズムの選択。


'''
1.X_train: 訓練データ
2.X_test: テストデータ
3.Y_train: 訓練データの正解ラベル
4.Y_test: テストデータの正解ラベル
'''
'''教師データと試験データの作成'''
'''引数stratifyに均等に分割させたいデータ（多くの場合は正解ラベル）を指定すると、そのデータの値の比率が一致するように分割される。'''
#X_train, y_train, = train_test_split(Stock_train_df, test_size=0.2, random_state=0,shuffle=False)
#print(",X_train= ",X_train, "y_train= ",y_train)


'''教師データと試験データの作成'''
train_x, train_y = train_data(Stock_train_df)  # 　 教師データ作成。
test_x, test_y = train_data(Stock_test_df)  # 　試験データ作成。


'''引数stratifyに均等に分割させたいデータ（多くの場合は正解ラベル）を指定すると、そのデータの値の比率が一致するように分割される。'''
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=0,shuffle=False)
print(",train_x= ",train_x, "train_y= ",train_y)






'''学習'''
'''教師データの数値の配列 (train_X) と結果の配列 (train_y) を学習させる'''
lr.fit(X_train, y_train)



'''予測'''
'''テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる'''
print("(test_x=) ",test_x[0])
test_y = lr.predict(test_x)#テストデータの数値の配列 (test_X)



print("翌日予測(test_y=)",test_y)
#pred = tree.predict(test_x)
x_pred =test_y


from sklearn.metrics import accuracy_score
print(test_x)
print(x_pred)
print (accuracy_score(test_x, x_pred))
print (accuracy_score(test_x, x_pred,normalize=False))






'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import random

if __name__ == '__main__':

    # データセットを読み込む
    iris = load_iris()
    x = iris.data
    y = iris.target

    # 読み込んだデータセットをシャッフルする
    p = list(zip(x, y))
    random.shuffle(p)
    x, y = zip(*p)

    # 学習データの件数を指定する
    train_size = 100
    test_size = len(x) - train_size

    # データセットを学習データとテストデータに分割する
    train_x = x[:train_size]
    train_y = y[:train_size]
    test_x = x[train_size:]
    test_y = y[train_size:]

    # 決定木の学習を行う
    tree = DecisionTreeClassifier(criterion='gini', max_depth=None)
    tree.fit(train_x, train_y)

    # 学習させたモデルを使ってテストデータに対する予測を出力する
    count = 0
    pred = tree.predict(test_x)
    for i in range(test_size):
        print('[{0}] correct:{1}, predict:{2}'.format(i, test_y[i], pred[i]))
        if pred[i] == test_y[i]:
            count += 1

    # 予測結果から正答率を算出する
    score = float(count) / test_size
    print('{0} / {1} = {2}'.format(count, test_size, score))
'''





'''
train_test_split
概要
scikit-learnのtrain_test_split()関数を使うと、与えたデータをいろいろな方法で訓練データとテストデータに切り分けてくれる。

scikit-learnのtrain_test_split()関数を使うと、NumPy配列ndarrayやリストなどを二分割できる。
機械学習においてデータを訓練用（学習用）とテスト用に分割してホールドアウト検証を行う際に用いる。
'''

'''
train_test_splitでの戻り値は以下の通りです。

1.X_train: 訓練データ
2.X_test: テストデータ
3.Y_train: 訓練データの正解ラベル
4.Y_test: テストデータの正解ラベル
'''


'''train_test_split()の基本的な使い方'''
#train_test_split()にNumPy配列ndarrayを渡すと、二分割されたndarrayが要素として格納されたリストが返される。



a = np.arange(10)#numpy.ndarray


print(a)
# [0 1 2 3 4 5 6 7 8 9]

print(train_test_split(a))
# [array([3, 9, 6, 1, 5, 0, 7]), array([2, 8, 4])]

print(type(train_test_split(a)))
# <class 'list'>

print(len(train_test_split(a)))
# 2


'''以下のように、アンパックでそれぞれ2つの変数に代入することが多い。'''
a_train, a_test = train_test_split(a)
print(a_train)
# [3 4 0 5 7 8 2]

print(a_test)
# [6 1 9]
#例はnumpy.ndarrayだが、list（Python組み込みのリスト）やpandas.DataFrame, Series、疎行列scipy.sparseにも対応している。
# pandas.DataFrame, Seriesの例は最後に示す。


'''割合、個数を指定: 引数test_size, train_size'''
#引数test_sizeでテスト用（返されるリストの2つめの要素）の割合または個数を指定できる。
#デフォルトはtest_size=0.25で25%がテスト用、残りの75%が訓練用となる。小数点以下は切り上げとなり、上の例では10 * 0.25 = 2.5 -> 3となっている。
#test_sizeには0.0 ~ 1.0の割合か、個数を指定する。

#割合で指定した例。
a_train, a_test = train_test_split(a, test_size=0.6)#引数test_size

print(a_train)
# [9 1 2 6]

print(a_test)
# [5 7 4 3 0 8]


#個数で指定した例。
a_train, a_test = train_test_split(a, test_size=6)

print(a_train)
# [4 2 1 0]

print(a_test)
# [7 6 3 9 8 5]


#引数train_sizeで訓練用の割合・個数を指定することもできる。test_sizeと同様に、0.0 ~ 1.0の割合か、個数を指定する。
a_train, a_test = train_test_split(a, train_size=0.6)

print(a_train)
# [2 9 6 0 4 3]

print(a_test)
# [7 8 5 1]

a_train, a_test = train_test_split(a, train_size=6)

print(a_train)
# [9 3 0 8 7 1]

print(a_test)
# [5 6 4 2]



#これまでの例のように引数test_size, train_sizeのいずれかのみを指定した場合、他方の数はその残りになるが、それぞれを別途指定することも可能。
a_train, a_test = train_test_split(a, test_size=0.3, train_size=0.4)

print(a_train)
# [3 0 4 9]

print(a_test)
# [7 2 8]

a_train, a_test = train_test_split(a, test_size=3, train_size=4)

print(a_train)
# [9 7 0 4]

print(a_test)
# [3 8 5]


'''シャッフルするかを指定: 引数shuffle'''
#これまでの例のように、デフォルトでは要素がシャッフルされて分割される。引数shuffle=Falseとするとシャッフルされずに先頭から順番に分割される。
a_train, a_test = train_test_split(a, shuffle=False)

#print(a_train)
# [0 1 2 3 4 5 6]

#print(a_test)
# [7 8 9]


'''乱数シードを指定: 引数random_state'''
#シャッフルされる場合、デフォルトでは実行するたびにランダムに分割される。引数random_stateを指定して乱数シードを固定すると常に同じように分割される。
a_train, a_test = train_test_split(a, random_state=0)

#print(a_train)
# [9 1 6 7 3 0 5]

#print(a_test)
# [2 8 4]

'''機械学習のモデルの性能を比較するような場合、どのように分割されるかによって結果が異なってしまうため、乱数シードを固定して常に同じように分割されるようにする必要がある。'''


X = np.arange(20).reshape(2, 10).T
Z = np.arange(20).reshape(2, 10).T
'''層化抽出: 引数stratify(相似化)'''
#例えば教師あり学習では特徴行列（説明変数）と正解ラベル（目的変数）の2つのデータを用いる。
#二値分類（2クラス分類）では正解ラベルは、例えば以下のように0, 1のいずれかになる。
y = np.array([0] * 5 + [1] * 5)
#print(y)
# [0 0 0 0 0 1 1 1 1 1]




'''複数データの同時分割
#train_test_split()は複数データを同時に分割することもできる。
#以下の例では、二つの配列を引数として与えている。その結果は、与えた配列ごとに訓練データ、テストデータの順でタプルとして返される。

#教師あり学習のためにデータを分割する場合、訓練用とテスト用の正解ラベルの比率は元のデータの正解ラベルの比率と一致していることが望ましいが、
# 例えば以下のようにテスト用に0の要素が含まれていないといったことが起こり得る。
X_train, X_test, y_train, y_test,Z_train, Z_test = train_test_split(X, y, Z,test_size=0.2, random_state=0, shuffle=False)

print(y_train)
# [0 1 0 0 0 0 1 1]

print(y_test)
# [1 1]

print(X_train)
# [0 1 0 0 0 0 1 1]

print(X_test)
# [1 1]

print(Z_train)
# [0 1 0 0 0 0 1 1]

print(Z_test)
# [1 1]
'''

'''引数stratifyに均等に分割させたいデータ（多くの場合は正解ラベル）を指定すると、そのデータの値の比率が一致するように分割される。'''
#1.X_train: 訓練データ
#2.X_test: テストデータ
#3.Y_train: 訓練データの正解ラベル
#4.Y_test: テストデータの正解ラベル
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.8,test_size=0.2, random_state=100,stratify=y)

print(" 訓練データ",X_train)
# [1 1 0 0 0 1 1 0]

print("テストデータ",X_test)
# [1 0]

print("訓練データの正解ラベル",y_train)
# [1 1 0 0 0 1 1 0]

print("テストデータの正解ラベル",y_test)
# [1 0]



#サンプル数が少ないとイメージしにくいので、次の具体例も参照されたい。

#具体的な例（アイリスデータセット）
#具体的な例として、アイリスデータセットを分割する。

#150件のデータがSepal Length（がく片の長さ）、Sepal Width（がく片の幅）、Petal Length（花びらの長さ）、Petal Width（花びらの幅）の4つの特徴量を持っており、
# Setosa, Versicolor, Virginicaの3品種に分類されている。
#load_iris()でデータを取得する。正解ラベルyには0, 1, 2の3種類が均等に含まれている。

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()

X = data['data']
y = data['target']

print(X.shape)
# (150, 4)

print(X[:5])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

print(y.shape)
# (150,)

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

#train_test_split()で以下のように分割できる。test_sizeやtrain_sizeは設定していないので、デフォルトの通り、訓練用75%とテスト用25%に分割される。サイズが大きいので形状shapeのみ示している。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train.shape)
# (112, 4)

print(X_test.shape)
# (38, 4)

print(y_train.shape)
# (112,)

print(y_test.shape)
# (38,)


#テスト用の正解ラベルy_testを確認すると、各ラベルの数が不均等になっている。
print(y_test)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1]

print((y_test == 0).sum())
# 13

print((y_test == 1).sum())
# 16

print((y_test == 2).sum())
# 9

#引数stratifyを指定すると、分割後の各ラベルの比率が元のデータの比率（この例では3種類が均等）と一致するように分割できる。以下の例のように分割後の個数によっては完全に一致しない（割り切れない）こともあるが、できる限り元の比率に近くなっていることが分かる。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

print(y_test)
# [0 0 0 0 1 1 1 0 1 2 2 2 1 2 1 0 0 2 0 1 2 1 1 0 2 0 0 1 2 1 0 1 2 2 0 1 2
#  2]

print((y_test == 0).sum())
# 13

print((y_test == 1).sum())
# 13

print((y_test == 2).sum())
# 12


#pandas.DataFrame, Seriesの場合
#pandas.DataFrame, Seriesはそれぞれ二次元・一次元の配列と同様に分割できる。

#ここでもアイリスデータセットを例とする。

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()

X_df = pd.DataFrame(data['data'], columns=data['feature_names'])
y_s = pd.Series(data['target'])

print(X_df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2
# 1                  4.9               3.0                1.4               0.2
# 2                  4.7               3.2                1.3               0.2
# 3                  4.6               3.1                1.5               0.2
# 4                  5.0               3.6                1.4               0.2
# ..                 ...               ...                ...               ...
# 145                6.7               3.0                5.2               2.3
# 146                6.3               2.5                5.0               1.9
# 147                6.5               3.0                5.2               2.0
# 148                6.2               3.4                5.4               2.3
# 149                5.9               3.0                5.1               1.8
# 
# [150 rows x 4 columns]

print(type(X_df))
# <class 'pandas.core.frame.DataFrame'>

print(X_df.shape)
# (150, 4)

print(y_s)
# 0      0
# 1      0
# 2      0
# 3      0
# 4      0
#       ..
# 145    2
# 146    2
# 147    2
# 148    2
# 149    2
# Length: 150, dtype: int64

print(type(y_s))
# <class 'pandas.core.series.Series'>

print(y_s.shape)
# (150,)


#引数などの使い方はnumpy.ndarrayの場合とまったく同じ。分割されたpandas.DataFrame, Seriesが返される。
X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
    X_df, y_s, test_size=0.2, random_state=0, stratify=y_s
)

print(type(X_train_df))
# <class 'pandas.core.frame.DataFrame'>

print(X_train_df.shape)
# (120, 4)

print(type(X_test_df))
# <class 'pandas.core.frame.DataFrame'>

print(X_test_df.shape)
# (30, 4)

print(type(y_train_s))
# <class 'pandas.core.series.Series'>

print(y_train_s.shape)
# (120,)

print(type(y_test_s))
# <class 'pandas.core.series.Series'>

print(y_test_s.shape)
# (30,)


#引数stratifyの効果を確認する。訓練用もテスト用も元のデータの比率に合わせて分割されていることが分かる。
print(y_train_s.value_counts())
# 2    40
# 1    40
# 0    40
# dtype: int64

print(y_test_s.value_counts())
# 2    10
# 1    10
# 0    10
# dtype: int64
