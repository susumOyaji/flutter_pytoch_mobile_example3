##事前にxに説明変数をyに目的変数を設定しておく。
##必要な機能のimport。
import pandas as pd#　　　　　　　　　　　　　　　　　　    データフレームを扱うための機能。
from urllib.request import urlretrieve#　　　　　　　　   ネット上からファイルをダウンロードし保存するのに使う機能。
from sklearn.utils import all_estimators#　　            scitic-learnの全アルゴリズム。
from sklearn.model_selection import KFold#               K分割クロスバリデーション機能。
from sklearn.model_selection import cross_val_score#     クロスバリデーションのスコアを出力する機能。
import warnings#                                         警告関連の操作。
from pandas_datareader import data as pdr
import datetime
from sklearn.model_selection import train_test_split
import numpy as np



X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'


#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)#教師データ(今までのデータ)
end_train = datetime.date(2021,12,31)

##インターネットから使うデータをダウンロード。
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データのcsvファイルを読み込む。
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"#　データがあるネット上のアドレス。
savepath = "winequality-white.csv"#　保存時のファイル名。
urlretrieve(url, savepath)#　        ネット上の"url"からファイルをダウンロードし、"savepath"の名前で保存。

##使うデータを読み込み、説明変数（成分）と目的変数（quality : 評価値）に分ける。
##データの分割を行う（学習用データ 0.8 評価用データ 0.2）
#X_train, y_test = train_test_split(Stock_train_df, test_size=0.2, random_state=0)

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






wine_df = pd.read_csv("winequality-white.csv",sep = ";", encoding = "utf-8")#　csvファイルを読み込む。今回は";"区切りのcsvであるためsep=";"とした。
print(wine_df)
x = wine_df.drop("quality",axis = 1)#　説明変数データ。".drop(項目名)"で指定の項目列データのみを削除している。
y = wine_df["quality"]#　　　　　　　　 目的変数データ。"[項目名]"で指定の項目だけを抽出している。



x, y = train_data(Stock_train_df["Adj Close"])  # 　 教師データ作成。
#print(x,y)
##アルゴリズム総当たり
allAlgorithms = all_estimators(type_filter = "classifier")#　"classifier"タイプの全てのアルゴリズムを取得する。
#print(allAlgorithms)

kfold_cv = KFold(n_splits = 5, shuffle = True)#　K分割クロスバリデーション用オブジェクト。5分割。


warnings.filterwarnings('ignore')#　警告は無視する。


for(name, algorithm) in allAlgorithms:#　全てのアルゴリズムで、5通りのデータの分け方で学習した場合の精度を出力。
    try:
        clf = algorithm()
        
        if hasattr(clf,"score"):
            scores = cross_val_score(clf, x, y, cv = kfold_cv)
        print(name, "の正解率")
        print(scores)

    except Exception:#　　　　　　　　　　エラーを出したアルゴリズムは無視する。
        pass


































'''
##事前にxに説明変数をyに目的変数を設定しておく。

from sklearn.utils import all_estimators#　　            scitic-learnの全アルゴリズム。
from sklearn.model_selection import KFold#               K分割クロスバリデーション機能。
from sklearn.model_selection import cross_val_score#     クロスバリデーションのスコアを出力する機能。
import warnings#                                         警告関連の操作。


allAlgorithms = all_estimators(type_filter = 絞り込みたいアルゴリズムのタイプ)#　scikit-learnの全てのアルゴリズムを取得する（絞り込み可能）。

warnings.filterwarnings('ignore')#　警告は無視。


kfold_cv = KFold(n_splits = K（Kは整数）, shuffle = True)#K分割クロスバリデーション用オブジェクト。


for(name, algorithm) in allAlgorithms:#　全てのアルゴリズムで、K通りのデータの分け方で学習した場合の精度を出力。
    try:
        clf = algorithm()
        
        if hasattr(clf,"score"):#clfが"score"属性を持つか
            scores = cross_val_score(clf, x, y, cv = kfold_cv)
        print(name, "の正解率")
        print(scores)

    except Exception:#　エラーを出したアルゴリズムは無視する。
        pass
'''