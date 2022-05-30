##準備----------------------------------------------------------------------------------------------------
#今回必要なモジュールやパッケージ（言わば便利な機能）を使えるようにする。
import numpy as np#　　　　　　　　　　　　　  numpy: 行列を扱えるようにする。"np"として扱う
import matplotlib.pyplot as plt#　 　　　　　 matplotlib.pyplot : グラフ描画できるようにする。"plt"として扱う。
from sklearn.svm import LinearSVC# 　　　　　 LinearSVC: 今回使用するアルゴリズム。scikit-learnライブラリに含まれる。
from sklearn.metrics import accuracy_score#  accuracy_score: 正解率を調べることができるようになる。

'''教師データを設定'''
#教師データ イモの[色の紫具合、長さ]のデータ10個分
learn_data = np.array([[0.1,0.18],
                       [0.01,0.21],
                       [0.2,0.34],
                       [0.15,0.21],
                       [0.11,0.24],
                       [0.84,0.85],
                       [0.82,0.92],
                       [0.71,0.78],
                       [0.74,0.88],
                       [0.89,0.72]])

#各教師データに対する答えのラベル（0:ジャガイモ、1:サツマイモ）
learn_label = np.array([0,0,0,0,0,1,1,1,1,1])

'''学習'''
#アルゴリズムの指定。今回はLinearSVC。今回は、これ以降"clf"と見たら「LinerSVCで何かするんだなぁ」と思ってください。
clf = LinearSVC()

#教師データとその答えのラベルから、学習させる。
clf.fit(learn_data,learn_label)

##出来上がった分類器を使ってみる----------------------------------------------------------------------------
#分類してほしいデータ
test_data = np.array([[0.11,0.3],
                      [0.21,0.2],
                      [0.78,0.82],
                      [0.87,0.88]])

#分類してほしいデータの答え
test_label_answer = np.array([0,0,1,1])

#学習してできた分類器による予測結果
test_predict = clf.predict(test_data)

print(test_data, "の予測結果:", test_predict)#                        予測結果の表示

'''正解率を計算'''
#accuracy_score(正解ラベル,予測結果)
print("正解率 = ", accuracy_score(test_label_answer,test_predict))#   正解率の表示

##機械が設定した「ジャガイモ」と「サツマイモ」の分類線の可視化--------------------------------------------------
#分類線の式を求める。
w = clf.coef_[0]#                                                    傾きの計算に必要な値を取得
a = -w[0] / w[1]#                                                    傾きを計算
xx = np.linspace(0,1)
yy = a * xx - clf.intercept_[0] / w[1]#                              一次関数y = ax + bの形

#グラフ作成

plt.plot(xx, yy, 'k-', label="dividing line")# 　                     分類線を引く。線で結んだグラフはplt.plot(x軸のデータ,y軸のデータ,'色や線の見た目',lavel = グラフの名前)
plt.scatter(learn_data[:,0],learn_data[:,1],label="learn data")#      散布図として学習データをプロット。散布図のグラフはplt.scatter(x軸のデータ,y軸のデータ,lavel = グラフの名前)
plt.scatter(test_data[:,0],test_data[:,1],label="test data")#      　 散布図として学習データをプロット。
plt.xlabel("Degree of purple")#　　　　　　　　　　                     x軸のラベルを設定。色の紫具合。plt.xlabel(x軸のラベル名)
plt.ylabel("length")#　　　　　　　　　　　　　　　                      y軸のラベルを設定。長さ。plt.xlabel(y軸のラベル名)
plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00])#　　　　                    x軸の目盛りをつける位置を設定。plt.xticks([目盛りをつけたい場所])
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])#　　　　                    y軸の目盛りをつける位置を設定。plt.yticks([目盛りをつけたい場所])
plt.legend(bbox_to_anchor = (1.2, 1.0),#                              凡例を設定。plt.legend(bbox_to_anchor=(凡例を置く場所のx座標, y座標))
           loc = 'upper left',#                                       loc = 凡例のどの角を、先ほど指定した場所(anchor)に置くか。
           borderaxespad = 0,#                                        borderaxespad = anchorとlocの距離
           fontsize = 18)#                                            fontsize = 文字の大きさ。)
plt.show()#　　　　　　　　　　　　　　　　　　　　                       グラフを描画　　　　                       グラフを描画