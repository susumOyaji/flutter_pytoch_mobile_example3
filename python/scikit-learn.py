import datetime
from pandas_datareader import data as pdr


# common
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# SVM
from sklearn.svm import LinearSVC
# 決定木
from sklearn.tree import  DecisionTreeClassifier
# k-NN
from sklearn.neighbors import  KNeighborsClassifier



# データ分割
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split






#必要なライブラリとデータセットをインポート
import pandas as pd
import sklearn
from sklearn.datasets import load_iris, load_boston #分析対象データ


#分析対象データ
#from sklearn.datasets import load_iris

'''data分割'''
'''条件設定。'''
X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'

interval = 50 #直近何日分の株価を基に予測するか。(test_x)
future = 2 #何日先の株価を予測するかは”future”の値を変更する。(test_y)

#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)#教師データ(今までのデータ)
end_train = datetime.date(2021,12,31)


#datetime.date.today() + relativedelta(days=-1)
start_test = datetime.date(2022, 1, 1)#試験データ
#start_test = datetime.date.today() + relativedelta(days= - (interval+future))#試験データ
end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）





'''使うデータを読み込む。'''
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
#iris = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データのcsvファイルを読み込む。
#Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)# 試験データのcsvファイルを読み込む。

#irisデータセットの読み込み
iris = load_iris()
print(iris)

#特徴量をデータフレームに格納して、最初の五行を表示
iris_features = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_features.head()

#特徴量の基本統計量を確認
iris_features.describe()

#特徴量の欠損値の数を確認
iris_features.isnull().sum()
















X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, stratify = iris.target, random_state=0)

# initial value
best_score = 0
best_method = ""

models = []
models.append(DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0))
models.append(LogisticRegression())
models.append(KNeighborsClassifier(n_neighbors=6))
models.append(LinearSVC())
# clfはclassificationの略語

for model in models:
    model.fit(X_train,y_train)
    print('---------------------------------')
    print('モデル：',model.__class__.__name__)
    print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
    print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))

    if model.score(X_test, y_test) > best_score:
        best_method = model.__class__.__name__
        best_score = model.score(X_test, y_test)

print('---------------------------------')
print('best_method:',best_method)
print('best_score:',best_score)