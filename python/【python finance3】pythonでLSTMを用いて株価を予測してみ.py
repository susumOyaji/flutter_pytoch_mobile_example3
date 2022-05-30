'''
【python finance3】pythonでLSTMを用いて株価を予測してみよう！ https://drwatson2020.com/stock3/
python finance3
今回はLSTMを用いたAiで実際に株価の予測を行っていきます。具体的にはグーグル社の株価予測と実際の結果との確認まで行っていきます。
今回もpythonを使ってanacondaのspyderで解説致します。
まず、LSTMとは簡単に言いますと、時系列データを扱う目的のために開発されたディープラーニングの一種です。株価や為替の価格を予測するにはとても便利な機能だと思ってください。実際にプログラムを確認してみましょう。
'''


#計算に使用する数学をインポートしていきます。
import math

#pandasのdatareaderをwebとしてインポートします。
import pandas_datareader as web

#行列の計算などに便利なnumpyをnpとしてインポートします。
import numpy as np

#データ集取や加工に便利なpandasをpdとしてインポートします。
import pandas as pd
from sklearn import datasets


'''次にここが重要です。正規化をするためのSklearnからMinMaxScalerをインポートします。※正規化は後で説明致します。'''
from sklearn.preprocessing import MinMaxScaler

#層を積み上げるためkerasからSequentialのインポートします。
from keras.models import Sequential

#LSTMをインポートします。
from keras.layers import Dense,LSTM

#最後にグラフ化させるのにMatplotlib.pyplotをpltでインポートします。
#import Matplotlib.pyplot as plt
from matplotlib import pyplot as plt


'''株価を取得していきます。2012年の1月1日から2020年の6月までの株式情報を取得して行きましょう。'''
google_stock = web.DataReader("GOOG",data_source="yahoo",start="2012-01-01",end="2020-06-01")

#まずは取得できているか確認してみましょう。
print(google_stock)


#使ってない標準ライブラリーをコメントアウトして実行してみます。
#このようにgoogleの株が確認できます。2日あいているのは、株式取引が閉鎖されていた日です。次に、google_stockのデータの型を確認しましょう。

print(google_stock.shape)
#これで実行しますと。

#確認の結果、2128行と6列のデータであることが確認できます。次にデータを視覚化させましょう。図のサイズをきます。

plt.figure(figsize=(18,8))
#図のタイトルを決めます。

plt.title("google stock History")
#終値をプロットさせます。

plt.plot(google_stock["Close"])
#X軸のラベルを書きます

plt.xlabel("google_date" ,fontsize=18)
#Y軸のラベルを書きます

plt.ylabel("Close Price ($)",fontsize=18)
#グラフを表示します

plt.show()
#これで実行し確認してみますと。

#2013年に株を保有していた人は2020年には４倍にまで上がっていることが確認できますね。次にデータを加工するため新しい変数にデータを入れ込みます。

google_data = google_stock.filter(["Close"])
#google_dataの各要素をgoogle_close変数に入れ込みます。

google_close = google_data.values

'''#ここからがすごく重要です。
予測したデータと実際の結果がどれだけ似ているのか検証するためにも機会学習をさせるためデータを分ける必要があります。
つまり、実際にデータの80%を使用して残りの20%のデータを予測させ、実際のデータと比較します。
ですので訓練用データとテスト用データに分けましょう。
訓練用データは、先ほどのデータに0.8かけてtrainig_data変数を用意します。
'''
trainig_data = math.ceil(len(google_close)*0.8)

#trainig_data変数の型を確認しましょう。
print(trainig_data)

#これで実行しますと。
#このように2003×0.8の1603が取れます。次にデータの正規化を行っていきます。正規化とは簡単に言いますと、今のデータをデータ分析しやすい形にするという意味です。イメージとしては、テストの成績をイメージしてください。科目によって難しさは異なりますよね、例えば平均点が40点の中でとる60点と平均点が90点のテストの中の60点では相対的な出来が異なります。要は正規化とは統一するというイメージです。データを一定のルールに基づいて変形し利用しやすくしたものです。今回はMinMaxScalerメソッドを使用します。これは簡単に言いますと計算する前に全体のデータを0〜1の間にそのまま抑えましょうという意味です。本来であればデータとして2や-5などのデータが存在すると思いますが、これをそのまま縮小させ、0から1の間の割合に入れ込むという意味です。まず、正規化変数スケイラーを作成しそれに入れ込みます。
scaler=MinMaxScaler(feature_range=(0,1))

#データセットを保持するスケールデータと呼ばれる変数を作成します。ここでデータを0と1の間で変換しましょう。
scaled_data = scaler.fit_transform(google_close)

#実際にscaled_dataの中身を確認してみましょう。
print(scaled_data)
#これを実行しますと
#このように正規化されたデータが確認できます。


#次に学習のためのデータを作成します。まず初めに行うことは正規化した学習のためのデータをセットすることです。train_dataという変数を作成し、設置します。
train_data = scaled_data[0:trainig_data , : ]

#学習用と予測結果の変数を用意します。
x_train = []
y_train = []

#データを詳細に分けます。
for i in range (60, len(train_data)):
        x_train.append(train_data[ i - 60:i , 0 ]) 
        y_train.append(train_data[ i ,0])

#x_trainとy_trainをそれぞれに計算しやすいよう改めて代入しましょう。
x_train,y_train = np.array(x_train),np.array(y_train)

#ここでx_trainの型を確認してみましょう。
print(x_train.shape)

#このように2次元の(1543,60)行列になっていることが確認できますね。LSTMでの予測は3次元での予測となりますので、3次元配列に戻してあげましょう。ここで計算しやすいように2次元配列を3次元配列に変更します。reshapeを使用して行列改めて作成します。
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#これを差し込みましてもう一度実行してみます。このように
#(1543,60,1)が出力されました。次に、機会学習を行っていきます。
model = Sequential()
model.add(LSTM(50,return_sequences = True , input_shape= (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#モデルをコンパイルして行きます。
model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

#そして80％のデータを機会学習させます。
model.fit(x_train,y_train,batch_size=1,epochs=1)

#次に使用していなかった1543〜2003までのスケーリングされた値を含む新しい配列を作成しましょう。手順はほとんど同じです。
test_data = scaled_data[trainig_data - 60 : ,  : ]
x_test = [ ]
y_test = dataset[trainig_data: , :  ]

for i in range (60, len(test_data)):
        x_test.append(test_data[ i - 60:i , 0 ])

#Yは実際の結果ですのでｙを設定する必要はありません。そして前回と同じように
x_test= np.array(x_test)

#このままだと2次元のデータとなっています。LSTMでの予測は3次元での予測となりますので、３次元配列に戻してあげましょう。
x_test = np.reshape(x_test,(x_test[0],x_test[1],1))

#次にモデルの予測結果を取得します。
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#次に二乗平均平方根誤差も確認しえみます。（略してMSE）
chek_mse = np.sqrt(np.mean(predictions - y_test)**2)
print(chek_mse)

#データをプロットしていきます。
train = data[ : trainig_data_len]
valid = data[trainig_data_len : ]
valid["predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price USD ($)",fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close","Predictions"]])
plt.legend(["Train","Val","Predictions"],loc="lower right")
plt.show()
#これを実行してみますと

#このように予測値と実際の結果が出力されたことが確認できます。青色が機械学習を行った実際の過去のデータです。黄色が機械学習によって予測されたデータです。赤色がその時の実際の株価データです。はい、今回はグーグル社の過去の株価データからLSTMを用いて株価分析まで行ってみました。初心者の方には凄く難しい内容になったと反省しております。また再度わかりやすい講座にリメイクさせていただきます。次回は金融工学のブラックショールズ（ブラウン幾何運動）による株価予測と今回のLSTMを用いたニュラールネットワークの株価予測との違いについての解説を行っています。また、補足説明として、上記を理解した上でのテクニカル指標を用いたトレード方法をご紹介しています。クオンツトレーダー向けの講座となりますので、よりいっそう難しくなりますが、株価や為替トレード以外にも応用できますので、是非、気になる方はご覧ください。以上で、python finance3を終了致します。ご清聴ありがとうございました。(๑╹ω╹๑ )