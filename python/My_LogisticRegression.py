import os
import pandas as pd
from optuna.integration import lightgbm as lgb


X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'
code_dow = '^DJI'
code_nikkei = '^N225'



import datetime
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2022, 1, 1)#教師データ(今までのデータ)
#end_train = datetime.date(2021,12,31)
end_train= datetime.date.today()# + relativedelta(days=-1)#昨日分(today-1日)まで取得できる（当日分は変動しているため）

#datetime.date.today() + relativedelta(days=-1)
start_test = datetime.date(2022, 1, 1)#試験データ
#start_test = datetime.date.today() + relativedelta(days= - (interval+future))#試験データ
end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）



'''データの前処理'''
'''使うデータを読み込む。'''
from pandas_datareader import data as pdr
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train) # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)# 試験データのcsvファイルを読み込む。

Dow_df = pdr.get_data_yahoo(code_dow, start_train, end_train)# 試験データのcsvファイルを読み込む。
Nikkei_df = pdr.get_data_yahoo(code_nikkei, start_train, end_train)# 試験データのcsvファイルを読み込む。


# 取得開始日と取得終了日を設定する
#start = datetime.date(2015, 1, 1)
#end = datetime.date(2020, 12, 31)

Stock_train_df.to_csv('./Stock_train_df.csv')
# 円ドル為替の時系列データを取得してCSVに保存する
JPY_USD = pdr.get_data_yahoo('JPY=X', start_train, end_train) # 教師データのcsvファイルを読み込む。#web.DataReader('JPY=X', 'yahoo', start, end)
JPY_USD.to_csv('./JPY_USD.csv')

# 日経平均株価を取得してCSVに保存する
# ^(チルダ)を付け忘れないこと
N225 = pdr.get_data_yahoo('^N225', start_train, end_train)#web.DataReader('^N225', 'yahoo', start_train, end_train)
N225.to_csv('./N225.csv')

# NASDAQ総合を取得してCSVに保存する
# ^(チルダ)を付け忘れないこと
IXIC = pdr.get_data_yahoo('^IXIC', start_train, end_train)#web.DataReader('^IXIC', 'yahoo', start_train, end_train)
IXIC.to_csv('./IXIC.csv')

# ダウ平均を取得してCSVに保存する
# ^(チルダ)を付け忘れないこと
DJI = pdr.get_data_yahoo('^DJI', start_train, end_train)#web.DataReader('^DJI', 'yahoo', start_train, end_train)
DJI.to_csv('./DJI.csv')




#保存
Stock_train_df.to_csv("./data/stocks_price_data/stock_train_df.csv")
Dow_df.to_csv("./data/stocks_price_data/dow_df.csv")
Nikkei_df.to_csv("./data/stocks_price_data/nikkei_df.csv")
print(Dow_df,Nikkei_df)




'''
Pythonでfbprophetを用いて、株価予測をしてみた③（特徴量の追加編）
'''
# Webから情報を取得するためのライブラリを読み込む
import pandas_datareader.data as web

# 時刻を扱うためのライブラリを読み込む
import datetime

# グラフを描画するためのライブラリを読み込む
import matplotlib.pyplot as plt

# グラフで日本語を扱うためのライブラリを読み込む
import japanize_matplotlib

# データフレームを扱うためのライブラリを読み込む
import pandas as pd



# 取得したデータから終値を抽出する
X1 = JPY_USD['Close']
X2 = N225['Close']
X3 = IXIC['Close']
X4 = DJI['Close']
X5 = Stock_train_df['Close']

# 抽出した終値でグラフを作成する
fig = plt.figure()
axes = fig.subplots(2, 3)

axes[0, 0].plot(X1, color='g')
axes[0, 1].plot(X2, color='b')
axes[1, 0].plot(X3, color='k')
axes[1, 1].plot(X4, color='m')
axes[0, 2].plot(X5, color='r')

# タイトルを設定する
axes[0, 0].set_title('円ドル為替')
axes[0, 1].set_title('日経平均(N225)')
axes[1, 0].set_title('NASDAQ総合(IXIC)')
axes[1, 1].set_title('ダウ平均(DJI)')
axes[0, 2].set_title('SONY')

# グラフを表示する
plt.show()

'''
ax1 = plt.subplot(2,2,1)   # 4x4の1番目
ax4 = plt.subplot(2,2,4)   # 4x4の4番目
ax1.plot([1,2,3,4])   # 1番目に描画
ax1.plot([2,3,2,3])   # 1番目に追加描画
ax1.set_xlabel('foo')   # 1番目にxラベルを追加
ax1.set_ylabel('bar')   # 1番目にyラベルを追加
ax4.plot([5,10,20,10])   # 4番目に描画
ax4.set_title('baz')   # 4番目にタイトルを追加
'''

# グラフを表示する
plt.show()

'''


'''

# イテレータで全ての組み合わせリストを作成するためのライブラリを読み込む
import itertools

# データフレームを扱うためのライブラリを読み込む
import pandas as pd

# 組み合わせ対象をリストで定義する
lst = ['JPYUSD_f', 'DJI_f', 'IXIC_f', 'N225_f']

# 組み合わせた結果を保存するリストを用意する
result = []

# 全ての組み合わせを作成し、resultに結合する
for n in range(1, len(lst)+1):
    for conb in itertools.combinations(lst, n):
        result.append(list(conb))

# 全ての組み合わせを表示する
print(result)


# 全ての組み合わせリストの要素を1と0に変換する関数を作成する
def convert_onezero(result_lst):
    # 各変数の初期値を0に設定する
    JPYUSD_f = 0
    DJI_f = 0
    IXIC_f = 0
    N225_f = 0

    # result_lstから1行ごとにデータを読み込んで、
    # そのデータの中に、組み合わせ対象が含まれていたら、1を代入する
    for i in range(len(result_lst)):
        if 'JPYUSD_f' in result_lst[i]:
            JPYUSD_f = 1

        if 'DJI_f' in result_lst[i]:
            DJI_f = 1

        if 'IXIC_f' in result_lst[i]:
            IXIC_f = 1

        if 'N225_f' in result_lst[i]:
            N225_f = 1

    lst_tmp = [JPYUSD_f, DJI_f, IXIC_f, N225_f]
    return lst_tmp

# 1と0に変換した結果を代入するデータフレームを用意する
df_result = pd.DataFrame(columns=lst)

# 1と0に変換し、df_resultに結合する
for l in result:
    # 1と0に変換する
    flag = convert_onezero(l)

    # 変換した結果をリストからデータフレームに変換する
    flag = pd.Series(flag, index=lst)

    # df_resultに変換した結果を結合する
    df_result = df_result.append(flag, ignore_index=True)

# 変換した結果を表示する
print(df_result)








'''
Pythonでfbprophetを用いて、株価予測をしてみた①
'''

import numpy as np
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







# 分割日sdayを設定する
#split_day = pd.to_datetime('2020-11-10')
#end_day = pd.to_datetime('2020-12-30')
#train_index = Stock_train_df['ds'] < split_day test_index = df_1['ds'] >= split_day

# データ分割
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
#X_train, y_train, = train_test_split(Stock_train_df, test_size=0.2, random_state=0,shuffle=False)



# 訓練データと検証データを作成する
#x_train = df_1[train_index]
#x_test = df_1[test_index]

# ytest:予測期間中の正解データを抽出する
#ytest = x_test['y'].values
train_x, train_y = train_data(Stock_train_df["Adj Close"])  # 　 教師データ作成。
test_x, test_y = train_data(Stock_test_df["Adj Close"])  # 　試験データ作成。

# R2を計算するためのライブラリを読み込む
from sklearn.metrics import r2_score

# RMSEを計算するためのライブラリを読み込む
from sklearn.metrics import mean_squared_error






from sklearn.linear_model import LinearRegression#　今回使うアルゴリズム
'''学習と予測'''
#アルゴリズムの選択。
lr = LinearRegression(normalize = True)#   アルゴリズムの選択。

# 時系列モデルを学習する
lr.fit(train_x,train_y)

# 予測する
forecast = lr.predict(test_x)


# forecastから予測部分yhatのみ抽出する
# 2020-11-10から2020-12-30の期間で株取引が行われた日数は36日となる
ypred = forecast#[-36:][['yhat']].values


# R2値,RMSEを計算する
score_r2 = r2_score(test_y, ypred)
score_rmse = np.sqrt(mean_squared_error(test_y, ypred))

# R2,RMSEの計算結果を表示する
print(f'R2 score:{score_r2:.4f}')
print(f'RMSE score:{score_rmse:.4f}')

# 要素ごとのグラフを描画する
# トレンド、週周期、年周期を描画する
fig = lr.plot_components(forecast)

# 訓練データ・検証データ全体のグラフ化
fig, ax = plt.subplots(figsize=(10, 6))

# 予測結果のグラフ表示(prophetの関数)
lr.plot(forecast, ax=ax)

# タイトル設定など
ax.set_title('終値の予測')
ax.set_xlabel('日付')
ax.set_ylabel('終値')

# 時系列グラフを描画する
fig, ax = plt.subplots(figsize=(8, 4))


# 設定する日付は2020-11-10 ～ 2020-12-30となる
dates_test = Stock_test_df['Date']


# グラフを描画する
ax.plot(dates_test, test_y, label='正解データ', c='k')
ax.plot(dates_test, ypred, label='予測結果', c='r')


import matplotlib.dates as mdates
from matplotlib.dates import date2num
# 日付目盛間隔を表示する
# 木曜日ごとに日付を表示する
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付表記を90度回転する
ax.tick_params(axis='x', rotation=90)

# 方眼表示、凡例、タイトルを設定する
ax.grid()
ax.legend()
ax.set_title('終値の予測')

# x座標：2020年11月19日、y座標：7800にscore_r2とscore_rmseを表示する
xdata = date2num(datetime(2020, 11, 19))
ax.text(xdata, 7800, f'socre_R2:{score_r2:.4f}\nscore_RMSE:{score_rmse:.4f}', size=15)
# ax.text(xdata, 6600, f'socre_RMSE:{score_rmse:.4f}', size=15)

# 画面出力
plt.show()