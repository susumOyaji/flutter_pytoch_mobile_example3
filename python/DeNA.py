'''
今回の予測は特徴量として昨日の高値、安値、終値、取引量のデータを使って、明日の始値を予測します。
ニューラルネットワークの構築に必要なライブラリを読み込みます。
'''

# 株価を扱うためのライブラリのインポート
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import datetime
import sqlite3
import pandas as pd
import numpy as np

# 機械学習のためのライブラリのインポート
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn.linear_model import LinearRegression  # 　今回使うアルゴリズム



#データを読み込み
'''


'''
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2022, 1, 1)  # 教師データ(今までのデータ)
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
Stock_train_df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)  # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)[
    "Adj Close"]  # 試験データのcsvファイルを読み込む。

'''
         id        Date         Open         High          Low        Close         Volume 
0      8083  2005/02/17   757.699786   811.030008   705.479452   809.920119    26225022.24  
1      8084  2005/02/18   832.140119   906.580453   753.260230   759.919564    36745274.16  
...     ...         ...          ...          ...          ...          ...            ...  
3368  13101  2018/11/14  2123.000000  2159.000000  2117.000000  2143.000000     2209600.00 
3369  13102  2018/11/15  2136.000000  2171.000000  2125.000000  2164.000000     1066200.00   
3370  13103  2018/11/16  2158.000000  2191.000000  2149.000000  2152.000000     1323600.00    

[3371 rows x 7 columns]

データの前処理と正規化
ニューラルネットワークへ学習させるためにデータの前処理を行いましょう。




まずは、データの始値を1日ずらします。
今回の予測は特徴量として昨日の高値、安値、終値、取引量のデータを使って、明日の始値を予測します。
つまり、始値を一日ずらすことで、前日の特徴量に対して当日の始値が教師データとして出来上がるわけです。
説明よりも実際にやってみると解りやすいです。

一番最後は2018/11/16の始値（2158円）です。この始値を前日である2018/11/15の始値へ動かします。
'''


# 終値を1日分移動させる
df_shift = Stock_train_df.copy()
df_shift["Open"] = df_shift.Open.shift(-1)
# 改めてデータを確認
print(df_shift.tail())




'''
              High      Low     Open    Close     Volume  Adj Close
Date
2022-04-04  12830.0  12590.0  12780.0  12650.0  3031900.0    12650.0
2022-04-05  12820.0  12595.0  12530.0  12715.0  2796500.0    12715.0
2022-04-06  12540.0  12390.0  12030.0  12435.0  3164400.0    12435.0
2022-04-07  12210.0  11970.0  12240.0  12045.0  4166200.0    12045.0
2022-04-08  12260.0  12005.0      NaN  12195.0  3531100.0    12195.0
ご覧の通り2022/04/08の始値はNaN（非数）になっており、その代わりに2022/04/07の始値へ移動しています。
'''

#最後の行と時間（id, Date）列 は不要ですので訓練データから落とします
# 最後の行を除外
df_shift = df_shift[:-1]
df = df_shift.copy()

# time（時間）を削除
df21 = Stock_train_df.reset_index()
del df21['Date']
#del df['id']
 
# データセットのサイズを確認
df21.info()
'''
class 'pandas.core.frame.DataFrame'>
RangeIndex: 3370 entries, 0 to 3369
Data columns (total 5 columns):
Open      3370 non-null float64
High      3370 non-null float64
Low       3370 non-null float64
Close     3370 non-null float64
Volume    3370 non-null float64
dtypes: float64(5)
memory usage: 131.7 KB

もともと3371行のデータでしたが、最後尾を削除したので3370行6列のデータとなります。

次にテストデータと訓練データへ分割しましょう。
今回は訓練データを全体の8割、テストデータを残り2割としています。
'''

# データセットの行数と列数を格納
n = df.shape[0] # row
p = df.shape[1] # col
 
# 訓練データとテストデータへ切り分け
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end 
test_end = n
data_train = df.iloc[np.arange(train_start, train_end), :]
data_test = df.iloc[np.arange(test_start, test_end), :]

#テトレーニングデータの先頭を表示
print(data_train.head())
'''
 	Open 	High 	Low 	Close 	Volume
0 	832.140119 	811.030008 	705.479452 	809.920119 	26225022.24
1 	777.700008 	906.580453 	753.260230 	759.919564 	36745274.16
2 	749.919452 	817.700452 	715.479563 	757.699786 	15178217.67
3 	714.369674 	764.370230 	717.710451 	725.479674 	5326732.62
4 	794.359453 	793.249564 	714.369674 	785.480341 	9981998.10
'''
#次に「正規化」を行います。(株価データの正規化について)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train_norm = scaler.transform(data_train)
data_test_norm = scaler.transform(data_test)


#最後に正規化したデータを特徴量（ｘ）とターゲット（ｙ）へ切り分ける
# 特徴量とターゲットへ切り分け
X_train = data_train_norm[:, 1:]
y_train = data_train_norm[:, 0]
X_test = data_test_norm[:, 1:]
y_test = data_test_norm[:, 0]


'''学習と予測'''
#アルゴリズムの選択。
lr = LinearRegression(normalize=True)  # アルゴリズムの選択。


'''
1.X_train: 訓練データ
2.X_test: テストデータ
3.Y_train: 訓練データの正解ラベル
4.Y_test: テストデータの正解ラベル
'''

'''学習'''
'''教師データの数値の配列 (train_X) と結果の配列 (train_y) を学習させる'''
lr.fit(X_train, y_train)


'''予測'''
'''テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくる'''
test_y = lr.predict(X_test)  # テストデータの数値の配列 (test_X)


#訓練データ, テストデータに対する予測
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

#訓練データ
print('accuracy:', accuracy_score(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1 score:', f1_score(y_true=y_train, y_pred=y_pred_train))
print('confusion matrix = \n', confusion_matrix(
    y_true=y_train, y_pred=y_pred_train))




