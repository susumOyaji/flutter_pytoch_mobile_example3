from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_dataset = load_iris()
train_X, test_X, train_y, test_y = train_test_split(iris_dataset["data"], iris_dataset["target"], test_size=0.2)



# モデルを作成
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris_dataset.data, iris_dataset.target)

# 作成したモデルを用いて予測を実行
predicted = clf.predict(iris_dataset.data)

print(predicted)

# 識別率を確認
print(sum(predicted == iris_dataset.target) / len(iris_dataset.target))
tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris_dataset.feature_names,
                         class_names=iris_dataset.target_names,
                         filled=True, rounded=True)



print(pd.DataFrame(train_X,columns = iris_dataset.feature_names))
print(pd.DataFrame(test_X,columns = iris_dataset.feature_names))

#sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)

knn = KNeighborsClassifier()
knn.fit(train_X, train_y)  # fit関数を用いてトレーニングデータの読み込み
X_temp_test = np.array([[4.4, 2.9, 1.4, 0.2]])  # 学習データの中にあったSetosaのデータ
prediction = knn.predict(X_temp_test)  # クラスに予想させる
print(iris_dataset["target_names"][prediction])  # 予想結果の表示


x_pred = knn.predict(test_X)  # テストデータを学習モデルにセット
x_judge = x_pred == test_X  # 予測したデータと答えが一致しているか
true_count = len(np.where(x_judge == True)[0])  # 正解した数
print(x_judge)
#print(true_count/len(x_judge))  # 正答率


'''これを使うことで正解率や正解数を見ることが出来ます。'''
#y_true	正解ラベル train_y, test_y
#y_pred	予測した結果

from sklearn.metrics import accuracy_score
#accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

print(test_y)
print(x_pred)
print(accuracy_score(test_y, x_pred))
print(accuracy_score(test_y, x_pred, normalize=False))
#一か所だけ2の所を1と予測しています。なので、正解率としては96.7%、正解数は29個となっています。

#この関数を使うことでPrecision、Recall、F値、support(正解ラベルのデータの数)が分かります。
from sklearn.metrics import classification_report
print(classification_report(test_y, x_pred,
      target_names=iris_dataset["target_names"]))






if __name__ == '__main__':
    main()      


'''
#引用のデータと同じになるようにPandasのデータフレームにデータを格納
data = pd.DataFrame(iris_dataset.data,columns=iris_dataset.feature_names)
spesies_data = pd.DataFrame(iris_dataset.target)
#分類の数字を文字に変換し格納
for i in range(3):
    spesies_data = spesies_data.replace(i,iris_dataset.target_names[i])
data["spesies"] = spesies_data
sns.pairplot(data,hue="spesies")
plt.show()
'''


'''
ディープラーニングは様々な機械学習の手法の中のあくまで一技術です。

機械学習とは「機械に大量のデータからパターンやルールを発見させ、それをさまざまな物事に利用することで判別や予測をする技術」のことです。
両技術の違いについては以下のようになります。

機械学習
機械学習はデータの中のどの要素が結果に影響を及ぼしているのか（特徴量という）を人間が判断、調整することで予測や認識の精度をあげています。

ディープラーニング
一方、ディープラーニングはデータの中に存在しているパターンやルールの発見、特徴量の設定、学習なども機械が自動的に行うことが特徴です。
人間が判断する必要がないのが画期的です。

ディープラーニングで人間が見つけられない特徴を学習できるようになったおかげで、人の認識・判断では限界があった画像認識・翻訳・自動運転といった技術が飛躍的に上がったのです。


学習
正解がわかっているデータ（入力という）を元に、そのデータのルールやパターンを学習し、分析モデルとして出力します。

認識・予測
そのあとに正解がまだわかっていないデータを新たにインプットして、学習時に決められたルールやパターン（出力）を元に認識・予測をする。

この教師あり学習の代表的な手法として回帰や分類が挙げられます。また、基本的にニューラルネットワークやディープラーニングはこの教師あり学習を発展させたものになります。


機械学習の仕組みとしては4つあります。

決定木・ランダムフォレスト
サポートベクターマシン
ニアレストネイバー法（最近傍法）
ニューラルネットワーク
 

以下で仕組みについて解説していきます。

決定木・ランダムフォレスト
ランダムフォレストは決定木を複数用い、多数決を用いて予測するアルゴリズムです。決定木は処理が高速であることや、スケーリングしやすいという特徴もあり、複雑なモデルを複数組み合わせることで誤差を減らすことが可能です。

ランダムフォレストに関しては、こちらで詳しく解説しています。

サポートベクターマシン
サポートベクターマシンは分類したいデータを与える事によって、そのデータについて教えてくれます。データを分類するアルゴリズムの中でも強力です。

スマートフォンの顔認証や画像認識にも利用されています。サポートベクターマシンのアルゴリズムに関してはこちらから飛ぶことができます。

ニアレストネイバー法（最近傍法）
https://www.youtube.com/watch?v=7HEQy4BoBiQ

K近傍法は事前に学習するフェーズが無く、アルゴリズムがシンプルです。データが手元にあればすぐに解析を始めることが可能です。

流れとしては、既存のデータをプロットしておき、未知データとの距離が近い順に指定されたk個を取り出す。その中で多数決を取り、データを取得するという流れである。

ニューラルネットワーク
ニューラルネットワークとは、人間の脳内にある神経細胞とその繋がりを構造的にモデルで表したものです。
入力層と出力層、隠れ層の三層から構成されており、隠れ層が多数存在するモデルをディープラーニングと言います。

複数の入力、出力が可能でありニューラルネットワークで予測や判断、分類を可能にしています。種類としては代表的なものにDNN、RNN、CNNがあります。DNNはニューラルネットワークを複数層重ねたモデルで、RNNは時系列を得意としており、翻訳ソフトに使われています。CNNは画像処理でよく利用されているモデルです。Facebookの自動画像タギングに使われています。




'''