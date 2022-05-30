'''
!pip install japanize-matplotlib
import japanize_matplotlib

import numpy as np
import matplotlib.pyplot as plt

'''

#データフレーム型、ボストン住宅価格データを使うための機能をインポートする。
import numpy as np


import pandas as pd
from sklearn.datasets import load_boston



# データセット読み込み
boston = load_boston()

# データセットを説明変数と目的変数に分ける。

x = pd.DataFrame(boston.data, columns=boston.feature_names)#　説明変数をデータセット型で読み込む。
y = boston.target#　　　　　　　　　　　　　　　　　　　　　　    目的変数をarray型で読み込む。


'''
機械学習させる前に、説明変数と目的変数のデータを、学習用の教師データと、性能検証用の試験データにランダムに分けます。
なお、ここでの学習データと試験データの分け方がランダムであることから、学習後のできるAIの予測結果や精度は実行のたびに変わります。
よって、この記事での実行結果はその中の一例となります。
'''
from sklearn.model_selection import train_test_split#データを教師データと試験データにランダムに分ける機能を使えるようにする。
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)#データを教師データと試験データにランダムに分ける。



'''
線形回帰アルゴリズムを使う
線形回帰で予測AIを作る
以下のコードで、scikit-learnライブラリのLinearRegressionアルゴリズムで、教師データを学習させて住宅価格予測AIを作り、試験データで住宅価格予測を行います。
'''
from sklearn.linear_model import LinearRegression#　アルゴリズムLinearRegressionを使えるようにする。

#学習と予測
lr = LinearRegression(normalize = True)#   　　　　  アルゴリズムの選択。
lr.fit(x_train, y_train)#　　　　　　　　             学習。
y_pred_lr = lr.predict(x_test)#                      予測。


print("決定係数 : " + str(lr.score(x_test, y_test)))  # 決定係数の算出・表示。

'''
予測価格が実際の価格をうまく予測できているかを、以下のコードでグラフ化して確認しましょう。
'''
import matplotlib.pyplot as plt#　グラフ描画機能を使えるようにする。

#予測結果をグラフで表示
plt.figure(figsize = (6, 6), dpi = 100)
plt.scatter(y_test, y_pred_lr, c = "b", label="予測価格")#　　　　 散布図で予測価格を表示。
plt.plot([0, 55],[0, 55], c = "r",label="理想的な予測分布線")#     正しく予測できたときにはこの線上にデータ点が集まる。
plt.xlim(0, 55)#                                                  x軸範囲。
plt.ylim(0, 55)#                                                  y軸範囲。
plt.xlabel("実際の価格（1,000 $）", fontname="MS Gothic")#　　　　  x軸のラベルを設定。
plt.ylabel("予測された価格（1,000 $）", fontname="MS Gothic")#　　  y軸のラベルを設定。
plt.legend(bbox_to_anchor = (1.0, 1.0),#                         凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))。
           loc = 'upper left',#                                   loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad = 0,#                                    borderaxespad = anchorとlocの距離。
           prop={"family":"MS Gothic"})#　　　　　　　　　　　　　  prop={"family":フォント名}。
#plt.show()#　　　　　　　　　　　　　　　　　　　　　　　　　　　　    グラフ表示。

'''
正しく予測できている場合、予測価格は赤線で示された直線状に集まります。

大した工夫をしていない割に、（特に35,000＄以下では、）それほど大きく外れていないように感じます。

客観的に精度を検証するため、回帰クラスのアルゴリズムを用いたAIの精度を示す指標の一つ決定係数を以下のコードで調べます。
'''

print("決定係数 : " + str(lr.score(x_test,y_test)))# 決定係数の算出・表示。
#決定係数 : 0.7695923600631362
#約80％の精度で予測できていたようです。



'''
線形回帰について簡単に学ぶ
線形回帰では説明変数である複数の特徴量から、どのようにして目的変数の値を予測しているのでしょうか？
ここでは、以下のコードで、今回求められた特徴量ごとの重み（偏回帰係数とも）の値を確認してみましょう。
'''

for i in range(len(boston.feature_names)):
    print(boston.feature_names[i].ljust(7) + " : " + str(lr.coef_[i]))#　特徴量名 : 偏回帰係数を表示

'''
特徴量ごとの重みがこのようになっている、住宅価格の予測の式が作られたようです。
なお、この重みの大小でどの特徴量が重要かを比較することはできません。
一方で重みの符号（正か負か）を見ることで、特徴量が大きくなるほど価格が高くなるのか安くなるのかという見方はある程度できるでしょう。
特徴量ごとの重み、およびｙ切片の数学的な算出方法は、ここで紹介するには難しいので、気になる人は、機械学習の専門書などで自習してください。
'''





'''ランダムフォレストで予測AIを作る'''
#ランダムフォレストアルゴリズムは、特徴量に関する条件分岐を基に目的変数を予測する決定木と呼ばれるものを大量に設計して、
#その決定木ごとの予測の平均値を、求めたい予測値とする方法です。

from sklearn.ensemble import RandomForestRegressor#　 アルゴリズムRandomForestRegressorを使えるようにする。

rfr = RandomForestRegressor(n_jobs= -1)#               アルゴリズムの選択。（n_jobs = -1でCPUの最大コア数-1個使う）
rfr.fit(x_train, y_train)#　　　　　　　　　　　　　　   学習。
y_pred_rfr = rfr.predict(x_test)#                      予測。


import matplotlib.pyplot as plt#　グラフ描画機能を使えるようにする。

plt.figure(figsize = (6, 6), dpi = 100)
plt.scatter(y_test, y_pred_rfr, c = "b", label="予測価格")#　　　    散布図で予測価格を表示。
plt.plot([0, 55],[0, 55], c = "r",label="理想的な予測分布線")#       正しく予測できたときにはこの線上にデータ点が集まる。
plt.xlim(0, 55)#                                                   x軸範囲。
plt.ylim(0, 55)#                                                   y軸範囲。
plt.xlabel("実際の価格（1,000 $）", fontname="Hiragino sans")  # 　　　　   x軸のラベルを設定。
plt.ylabel("予測された価格（1,000 $）", fontname="MS Gothic")#　　   y軸のラベルを設定。
plt.legend(bbox_to_anchor = (1.05, 1.0),#                          凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))。
           loc = 'upper left',#                                    loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad = 0,#                                     borderaxespad = anchorとlocの距離。
           prop={"family":"MS Gothic"})#　　　　　　　　　　　　　   prop={"family":フォント名}。
#plt.show()#　　　　　　　　　　　　　　　　　　　　　　　　　　　　    グラフ表示。

print("決定係数 : " + str(rfr.score(x_test,y_test)))# 決定係数の算出・表示。



from sklearn import tree

est = rfr.estimators_[1]#　作ったランダムフォレストモデルの中の決定木を1つを選択。[　]内の値を変えれば違う決定木が見れる。
tree.plot_tree(est, 
               filled = True, 
               feature_names = boston.feature_names)
plt.figure(figsize = (10, 10), dpi = 100)
plt.show()


'''特徴量ごとの重要度を確認する'''
#ランダムフォレストでは、どの特徴量で条件分岐させると誤差を効率よく減らせるかを示す指標として、重要度を計算することが出来ます。
#より簡単に言うと、重要度の高い特徴量による分岐が、正確な回帰あるいは分類ができるかどうかに影響を与えるということです。逆に重要度の低い特徴量は、良くも悪くも、それほど結果に影響を与えないことになります。
#重要度は以下のようにして出力可能です。
for i in range(len(boston.feature_names)):
    print(boston.feature_names[i].ljust(7) + " : " + str(rfr.feature_importances_[i]))#　特徴量名 : 特徴量重要度を表示。

print("")#　一行空ける
print("重要度の合計 : " + str(np.sum(rfr.feature_importances_)))#　特徴量重要度の総和を算出。

'''
RM：1戸あたりの平均部屋数と、LSTAT : 低所得者の割合の重要度が非常に高いことが分かりました。（特徴量の日本語版が見たければ、記事のサムネイル画像を参照してください。）
重要度の合計は1.0であることも確認しておきましょう。
機械学習では、特徴量の種類が多すぎると、基準が複雑になりすぎて、
よいAIが作れなくなったり、学習・予測に時間がかかるため、時にはいくつかの特徴量を学習データから取り除く必要があります。
今回の方法は、そのような場合に、取り除くべき特徴量を客観的に見極める方法のひとつとすることができます。
'''