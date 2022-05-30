'''
scikit-learn基本の予測モデル
Python
scikit-learn
はじめに
pythonは分析ライブラリが豊富で、ライブラリを読み込むだけでお手軽に、様々なモデルを利用することができます。特にscikit-learnという機械学習ライブラリは数多くのモデルを統一的なインタフェースで提供しており、分析のはじめの一歩としてスタンダード化しています。
この記事では各モデルの詳細は語らず、とりあえずコピペで動かせるひな形を五月雨で掲載します。

環境
Google Colab
scikit-learn==0.20.2
手順
データ準備
アヤメのデータを使用します。データを読み込み訓練用とテスト用に分割します。
'''
import pandas as pd

# データセット
from sklearn import datasets

# 訓練データとテストデータに分割
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, stratify = iris.target, random_state=0)
#ちなみにデータは次のような形式で入っています。目的変数は0と1の二値で、説明変数はすべて連続数値です。

import pandas as pd

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
df.head()

'''重回帰'''
# データ分割（訓練データとテストデータ）のための関数
from sklearn.model_selection import train_test_split
# 重回帰モデリングのためのクラス
from sklearn.linear_model import LinearRegression

 # 目的変数にpriceを指定、説明変数にそれ以外を指定
X = iris.data #auto.drop('price', axis=1)
y = iris.target #auto['price']

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 重回帰クラスの初期化と学習
model = LinearRegression()
model.fit(X_train,y_train)

print('決定係数(train):{:.3f}'.format(model.score(X_train,y_train)))
print('決定係数(test):{:.3f}'.format(model.score(X_test,y_test)))

# 回帰係数と切片
print('\n回帰係数\n{}'.format(pd.Series(model.coef_, index=X.columns)))
print('切片: {:.3f}'.format(model.intercept_))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 説明変数と目的変数の
X = iris.data #adult[['age', 'fnlwgt','education-num', 'capital-gain', 'capital-loss']]
y = iris.target #adult['fin_flg']

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=10)

# ロジスティック回帰クラスの初期化と学習
model = LogisticRegression()
model.fit(X_train,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))
print('正解率(test):{:.3f}'.format(np.exp(model.coef_)))
print('オッズ比:',np.exp(model.coef_))


'''標準化'''
# 標準化のためのクラス
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = iris.data #adult[['age','fnlwgt','education-num','capital-gain','capital-loss']]
y = iris.target #adult['fin_flg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model = LogisticRegression()
model.fit(X_train_std,y_train)
print('正解率(train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test_std, y_test)))


'''Redge回帰とLasso回帰'''
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# 訓練データとテストデータへ分割
X = iris.data #auto.drop('price', axis=1)
y = iris.target #auto['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# モデルの構築と評価
linear = LinearRegression()
ridge = Ridge(random_state=0)
lasso = Lasso(random_state=0)

for model in [linear, ridge, lasso]:
    model.fit(X_train,y_train)
    print('{}(train):{:.6f}'.format(model.__class__.__name__ , model.score(X_train,y_train)))
    print('{}(test):{:.6f}'.format(model.__class__.__name__ , model.score(X_test,y_test)))


'''決定木'''
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# データ分割
X = mush_dummy.drop('flg', axis=1)
y = mush_dummy['flg']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 決定木クラスの初期化と学習
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
model.fit(X_train,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))


'''ツリー可視化'''
#!pip install pydotplus

from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())  


'''k-NN'''
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=0)

training_accuracy = []
test_accuracy =[]
N = 100
for n_neighbors in range(1,N):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train,y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

plt.plot(range(1,N), training_accuracy, label='Training')
plt.plot(range(1,N), test_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()


'''サポートベクターマシン'''
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=0)

model = LinearSVC()
model.fit(X_train,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))



'''標準化''''
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model = LinearSVC()
model.fit(X_train_std,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test_std, y_test)))


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
from sklearn.cross_validation import train_test_split

#分析対象データ
from sklearn.datasets import load_iris

# data分割
iris = load_iris()
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
print('best_method：',best_method)
print('best_score：',best_score)


