#昨日の株価から今日の株価予測
#2019-05-08 18:55:45
#テーマ：ブログ
#今回の予測は特徴量として昨日の高値、安値、終値、取引量のデータを使って、明日の始値を予測します。
#ニューラルネットワークの構築に必要なライブラリを読み込みます。

# 株価を扱うためのライブラリのインポート
import sqlite3
import pandas as pd
import numpy as np

# 機械学習のためのライブラリのインポート
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from datetime import datetime
#pandasのdatareaderをwebとしてインポートします。
import pandas_datareader as web


#データを読み込み
#今回は DeNA(2432) を使いました
# データベース接続とカーソル生成
connection = sqlite3.connect("prices.sqlite")
df = pd.read_sql("SELECT * FROM t{}".format(2432), connection)#, index_col='Date', parse_dates=True)
print(df)

'''
         id        Date         Open         High          Low        Close         Volume 
0      8083  2005/02/17   757.699786   811.030008   705.479452   809.920119    26225022.24  
1      8084  2005/02/18   832.140119   906.580453   753.260230   759.919564    36745274.16  
...     ...         ...          ...          ...          ...          ...            ...  
3368  13101  2018/11/14  2123.000000  2159.000000  2117.000000  2143.000000     2209600.00 
3369  13102  2018/11/15  2136.000000  2171.000000  2125.000000  2164.000000     1066200.00   
3370  13103  2018/11/16  2158.000000  2191.000000  2149.000000  2152.000000     1323600.00    

[3371 rows x 7 columns]
'''



today = datetime.today().strftime("%Y-%m-%d")

#次に、2014年から今日までの6年間のデータを取得しましょう。期日を決めて行きます。
start_date="2020-01-01"

#終了日はプログラムの実行日にしたいので、日時と文字列を相互に変換するメソッドstrftime関数を使います。様々なフォーマットの日付や時間を操作することが可能です。
end_date = datetime.today().strftime("%Y-%m-%d")



# csv ファイルからの時系列データ読み込み
filename = 'data/add_column.csv' # 日経平均株価データ
df = pd.read_csv(filename, index_col=0, parse_dates=True)

# ディープラーニングで株価予測
# モデルは 10 日分の平均株価を入力として、1 日後の平均株価を予測することとします。
# ですので、取得したデータを読み込んで日付順にソートした後、終値だけを取り出します。
#df = web.DataReader("TYOYY",data_source="yahoo",start=start_date,end=end_date)
#df = df.sort_index()
print(df)

'''
         id        Date         Open         High          Low        Close         Volume 
0      8083  2005/02/17   757.699786   811.030008   705.479452   809.920119    26225022.24  
1      8084  2005/02/18   832.140119   906.580453   753.260230   759.919564    36745274.16  
...     ...         ...          ...          ...          ...          ...            ...  
3368  13101  2018/11/14  2123.000000  2159.000000  2117.000000  2143.000000     2209600.00 
3369  13102  2018/11/15  2136.000000  2171.000000  2125.000000  2164.000000     1066200.00   
3370  13103  2018/11/16  2158.000000  2191.000000  2149.000000  2152.000000     1323600.00    

[3371 rows x 7 columns]
'''
#データの前処理と正規化
#ニューラルネットワークへ学習させるためにデータの前処理を行いましょう。

#まずは、データの始値を1日ずらします。
#今回の予測は特徴量として昨日の高値、安値、終値、取引量のデータを使って、明日の始値を予測します。
#つまり、始値を一日ずらすことで、前日の特徴量に対して当日の始値が教師データとして出来上がるわけです。説明よりも実際にやってみると解りやすいです。

#一番最後は2018/11/16の始値（2158円）です。この始値を前日である2018/11/15の始値へ動かします。


# 終値を1日分移動させる
df_shift = df.copy()
df_shift["Open"] = df_shift.Open.shift(-1)
# 改めてデータを確認
df_shift.tail()

'''
 	id 	Date 	Open 	High 	Low 	Close 	Volume
3366 	13099 	2018/11/12 	2093.0 	2158.0 	2104.0 	2143.0 	1543100.0
3367 	13100 	2018/11/13 	2123.0 	2152.0 	2072.0 	2140.0 	2218000.0
3368 	13101 	2018/11/14 	2136.0 	2159.0 	2117.0 	2143.0 	2209600.0
3369 	13102 	2018/11/15 	2158.0 	2171.0 	2125.0 	2164.0 	1066200.0
3370 	13103 	2018/11/16 	NaN 	2191.0 	2149.0 	2152.0 	1323600.0
'''

#ご覧の通り2018/11/16の始値はNaN（非数）になっており、その代わりに2018/11/15の始値へ移動しています。
#最後の行と時間（id, Date）列 は不要ですので訓練データから落とします
# 最後の行を除外
df_shift = df_shift[:-1]
df = df_shift.copy()

# time（時間）を削除
#del df['Date']
#del df['id']
 
# データセットのサイズを確認
df.info()

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
'''

#次にテストデータと訓練データへ分割しましょう。
#今回は訓練データを全体の8割、テストデータを残り2割としています。

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
data_train.head()

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

#TensorFlowでニューラルネットワーク構築

# 訓練データの特徴量の数を取得
n_stocks = X_train.shape[1]
 
# ニューロンの数を設定
n_neurons_1 = 256
n_neurons_2 = 128
 
# セッションの開始
net = tf.InteractiveSession()
 
# プレースホルダーの作成
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])
 
# 初期化
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()
#今回のニューラルネットワークのモデルでは、ニューロンの数は256、128と指定しています。

#次に隠れ層の重み（Weights）を書きましょう。
# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
#2層の非常にシンプルなニューラルネットワークです。上で指定したニューロンを各層で処理をします。

#次は出力（ Output）の設定です。
# 出力の重み
W_out = tf.Variable(weight_initializer([n_neurons_2, 1]))
bias_out = tf.Variable(bias_initializer([1]))
#続いて隠れ層の構造を指定します。
#一層目と二層目の活性化関数として「ReLU」が使っています。

# 隠れ層の設定（ReLU＝活性化関数）
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
 
# 出力層の設定
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))

#次にコスト関数、最適化関数を指定しましょう。
#コスト関数とは、簡単に言えば予測した値と実際の値の誤差を計算するものです。
#ニューラルネットワークでは、この誤差を最小にするような処理が内部で行われます。

# コスト関数
mse = tf.reduce_mean(tf.squared_difference(out, Y))
 
# 最適化関数
opt = tf.train.AdamOptimizer().minimize(mse)
 
# 初期化
net.run(tf.global_variables_initializer())

#では、いよいよ訓練データをニューラルネットワークに学習させてみましょう。
#epochsというのは、ニューラルネットワークの反復計算回数です。

# ニューラルネットワークの設定
batch_size = 128
mse_train = []
mse_test = []

 
# 訓練開始(500回の反復処理)
epochs = 500
for e in range(epochs):
    net.run(opt, feed_dict={X: X_train, Y: y_train})

#テストデータを使って訓練したモデルから予測を出力
#モデルは訓練データのみ「学習」していますので、テストデータは全く新しい見たことの無いデータな訳です。

# 復習ですが、テストデータには「始値」が含まれていません。
# 各日にちの「高値」「安値」「終値」「取引量」のみです。
# それらを使って、翌日（次の日）の始値を予測するのが今回の目的です。

# テストデータで予測
pred_test = net.run(out, feed_dict={X: X_test})
test_train = net.run(out, feed_dict={X: X_train}) 

# 予測データの最初の2つを表示
pred_test[0][0:2]
#array([-0.45388696, -0.44489673], dtype=float32)

#scaler.inverse_transform()を使って元の値へ復元してあげましょう。
# 予測値をテストデータに戻そう（値も正規化からインバース）
pred_test = np.concatenate((pred_test.T, X_test), axis=1)
pred_inv = scaler.inverse_transform(pred_test)

test_train = np.concatenate((test_train.T, X_test), axis=1)
test_inv = scaler.inverse_transform(test_train)


#予測が出ました
#では、確認してみましょう。特徴量があっているかの確認を含めて、まずは元データの最後尾のデータを出してあげます。

# 元データの最後尾
print(df_shift.tail(1))
print()
'''
 	id 	Date 	Open 	High 	Low 	Close 	Volume
3369 	13102 	2018/11/15 	2158.0 	2171.0 	2125.0 	2164.0 	1066200.0
ご覧の通り11月15日のDeNAのデータです。
'''

#ただ一つ注意しなくてはいけないのは、始値（Open）は、前処理で1日ずらしたので、これは16日の始値です。

#では、今回訓練したニューラルネットワークは一体16日の終値をいくらで予測したのか確認して見ましょう
#また正規化を戻しましたが、ちゃんと戻っているのか確認のためdata_testとtest_invも表示させます。
# テストデータの最後のデータ（正規化前）
print('data_test',data_test.tail())
print()
'''
	Open 	High 	Low 	Close 	Volume
3365 	2104.0 	2244.0 	2123.0 	2150.0 	4479000.0
3366 	2093.0 	2158.0 	2104.0 	2143.0 	1543100.0
3367 	2123.0 	2152.0 	2072.0 	2140.0 	2218000.0
3368 	2136.0 	2159.0 	2117.0 	2143.0 	2209600.0
3369 	2158.0 	2171.0 	2125.0 	2164.0 	1066200.0
'''
# テストデータの最後のデータ（正規化を戻した後）
print('test_inv', test_inv[-5:-1])
print()
'''
[[2.1040e+03 2.2440e+03 2.1230e+03 2.1500e+03 4.4790e+06]
 [2.0930e+03 2.1580e+03 2.1040e+03 2.1430e+03 1.5431e+06]
 [2.1230e+03 2.1520e+03 2.0720e+03 2.1400e+03 2.2180e+06]
 [2.1360e+03 2.1590e+03 2.1170e+03 2.1430e+03 2.2096e+06]]
'''
# モデルが予測したデータ
print('pred_inv', pred_inv[-5:-1])
print()

'''
[[2.15364204e+03 2.24400000e+03 2.12300000e+03 2.15000000e+03
  4.47900000e+06]
 [2.15388537e+03 2.15800000e+03 2.10400000e+03 2.14300000e+03
  1.54310000e+06]
 [2.15137685e+03 2.15200000e+03 2.07200000e+03 2.14000000e+03
  2.21800000e+06]
 [2.15405782e+03 2.15900000e+03 2.11700000e+03 2.14300000e+03
  2.20960000e+06]]
'''
#実際の15日の始値は2136円でしたが、14日の特徴量から予測した結果「2154円」という予想結果です
#今回はシンプルなニューラルネットワークを使って予測しました。
#予測したレートと実際のレートを簡単にチャートに落としてみましょう。 
# 予測と実際のテストの終値のチャートをプロット
import matplotlib.pyplot as plt
import mpl_finance as mpf
import matplotlib.dates as mdates
plt.ion()
fig = plt.figure(figsize=(18, 9))
ax1 = fig.add_subplot(111)
line1, = ax1.plot(test_inv[:,0])
line2, = ax1.plot(pred_inv[:,0])
plt.show()


# 非常に大きな値幅で予測は外れていましたが、このようにプロットしてみると、意外とそれなりにトレンドを読んで予測しているのがわかります。
# グラフだけでは予想の評価として曖昧なので、MAE（平均絶対誤差）という指標を算出してあげましょう。
# MAEとは、予測した値と実際の値の誤差を指標化したものです。
# つまり値が小さければ誤差が少ない、値が大きければ誤差が大きいを意味します。
# MAEの計算
mae_test = mean_absolute_error(test_inv, pred_inv)
print(mae_test)
'''
4.3656036205285265
MAEは約4.3656でした。

モデルを改善するたびに、このようにMAEを算出することで、改善されたのか？悪くなったのか？を相対的に評価することができます。
'''