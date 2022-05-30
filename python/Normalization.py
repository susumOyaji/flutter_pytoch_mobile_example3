''''
株価データの正規化
2019-05-07 21:22:25
テーマ：ブログ

株価データの正規化について
正規化とは 
データ等々を一定のルール（規則）に基づいて変形し、利用しやすくすること。別の言い方をするならば、正規形でないものを正規形（比較・演算などの操作のために望ましい性質を持った一定の形）に変形することをいう。
株価データを機械が学習する場合 生の株価データではうまく予測できないことが多いそうです
ここで、学習する株価を正規化して扱うことがあるそうなので、今回は正規化の話
'''

# 株価を扱うためのライブラリのインポート
import sqlite3
from sqlite3 import connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import mpl_finance as mpf
import mplfinance.original_flavor as mpf
import matplotlib.dates as mdates
 

# 機械学習のためのライブラリのインポート
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr

#株価データを読み込む


#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)  # 教師データ(今までのデータ)
#end_train = datetime.date(2021,12,31)
# 昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_train = datetime.date.today()


#datetime.date.today() + relativedelta(days=-1)
#start_test = datetime.date(2022, 1, 1)#試験データ
start_test = datetime.date.today() + relativedelta(days=-1)  # 試験データ
#end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）
end_test = datetime.date.today()  # + relativedelta(days= -1)
X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'

'''使うデータを読み込む。'''
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)#,[Low]     Open    Close      Volume     Adj Close  # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)["Adj Close"]  # 試験データのcsvファイルを読み込む。
print(Stock_train_df)


Stock_train_df.to_csv('data/stocks_price_data/normalize.csv')#csv書き出し
df21 = Stock_train_df.reset_index()
print(df21)

'''
           Date     High      Low     Open    Close      Volume     Adj Close
0    2017-01-04   3333.0   3274.0   3305.0   3333.0   5914000.0   3225.737793
1    2017-01-05   3337.0   3269.0   3335.0   3296.0   7201700.0   3189.929199
2    2017-01-06   3374.0   3311.0   3328.0   3316.0   8873600.0   3209.284912
3    2017-01-10   3439.0   3344.0   3345.0   3394.0  11461200.0   3284.775391
4    2017-01-11   3550.0   3462.0   3464.0   3510.0  13620000.0   3397.041504
...         ...      ...      ...      ...      ...         ...           ...
1302 2022-04-04  12830.0  12590.0  12640.0  12650.0   3031900.0  12650.000000
1303 2022-04-05  12820.0  12595.0  12780.0  12715.0   2796500.0  12715.000000
1304 2022-04-06  12540.0  12390.0  12530.0  12435.0   3164400.0  12435.000000
1305 2022-04-07  12210.0  11970.0  12030.0  12045.0   4166200.0  12045.000000
1306 2022-04-08  12260.0  12005.0  12240.0  12195.0   3531100.0  12195.000000

[1307 rows x 7 columns]
'''

#データの前処理と正規化

#ニューラルネットワークへ学習させるためにデータの前処理を行いましょう。
#正規化するために時間に関する行がジャマなので削除します。

# time（時間）を削除
del df21['Date']
print(df21) 
# データセットのサイズを確認
print(df21.head())


'''
     High     Low    Open   Close      Volume    Adj Close
0  3333.0  3274.0  3305.0  3333.0   5914000.0  3225.737793
1  3337.0  3269.0  3335.0  3296.0   7201700.0  3189.929199
2  3374.0  3311.0  3328.0  3316.0   8873600.0  3209.284912
3  3439.0  3344.0  3345.0  3394.0  11461200.0  3284.775391
4  3550.0  3462.0  3464.0  3510.0  13620000.0  3397.041504
'''

df22 = df21.set_index('High')
print(df22)





#次に「正規化」を行います。今回はScikit-learnに組み込まれているMinMaxScaler()を使います。
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df21)
df_norm = scaler.transform(df21)

print(df_norm)
'''
[[-0.94932319 -0.93435071 -0.96270897 -0.92262221 -0.2152921 ]
 [-0.90698424 -0.88182736 -0.93552265 -0.9505264   0.10121599]
 [-0.93794779 -0.93068401 -0.95701911 -0.95176521 -0.54764184]
 ...
 [-0.17279029 -0.19338198 -0.1595817  -0.17866007 -0.93781049]
 [-0.16539636 -0.18678567 -0.15502985 -0.16694044 -0.97221036]
 [-0.15288357 -0.17579183 -0.14137433 -0.17363737 -0.96446633]]
 '''

#データは正規化されていますので、予測される値も正規化されます。実際にトレードの指標として使うには、元のレートのデータへ戻す必要があります。
#元のレートのデータへ戻すには、Scikit-learnのscaler.inverse_transform()を使うことで、正規化した値から元のレートへ戻せます。
# 正規化から通常の値へ戻す
test_inv = scaler.inverse_transform(df_norm)

'''確認'''

# 正規化の前のテストデータ
print(df21.values[0])
#[7.57699786e+02 8.11030008e+02 7.05479452e+02 8.09920119e+02 2.62250222e+07]

# 正規化後のテストデータ
print(df_norm[0])
#[-0.94932319 -0.93435071 -0.96270897 -0.92262221 -0.2152921 ]

# 正規化から戻したデータ
print(test_inv[0])
#[7.57699786e+02 8.11030008e+02 7.05479452e+02 8.09920119e+02 2.62250222e+07]