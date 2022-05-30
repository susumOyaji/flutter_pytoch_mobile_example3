import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM, Dropout, Flatten
#from keras.models import Model




 

'''

ディープラーニングを使って株価予想してみた！
まず、株価分析の手法は、以下の２つに大別できます。

【ファンダメンタル分析】
ファンダメンタル分析は、企業の業績や、市場の方向性を分析して、株価の予想を行う手法です。

【テクニカル分析】
テクニカル分析は、過去の株価を用いて、株価の予想を行う手法です。

今回使うディープラーニングは、説明変数として過去の株価を用いるので、「テクニカル分析」を行うことになります。
ですが、過去の様々な論文で、テクニカル分析の効果がないことが実証済みです。
ですので、過度な期待はせず温かい目でご覧ください。
'''
'''
data = pd.read_csv(r'./data/stocks_price_data/dow_df.csv',
               	encoding='shift_jis',
              	index_col='Date',
               	parse_dates=True,
               	dtype='float64').dropna(axis = 1).dropna()
data.plot()
print(data.head())
'''

X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'
code_dow = '^DJI'
code_nikkei = '^N225'



import datetime
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2021, 1, 1)#教師データ(今までのデータ)
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
data = pdr.get_data_yahoo(f'{code}.T', start_train, end_train) # 教師データのcsvファイルを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)# 試験データのcsvファイルを読み込む。

Dow_df = pdr.get_data_yahoo(code_dow, start_train, end_train)# 試験データのcsvファイルを読み込む。
Nikkei_df = pdr.get_data_yahoo(code_nikkei, start_train, end_train)# 試験データのcsvファイルを読み込む。

	
print(data.head())





'''
引数
各層の一番最初の引数には出力次元を入力し、それ以降は各層固有の引数が続きます。
（様々な引数が用意されているので、是非 Keras公式ドキュメントなどで調べてみてください）

input_shape引数
最初の層には、input_shape引数に入力次元のタプルを指定してください。

activation引数
Dense層には、activation引数がありますが、一般的に中間層ならば ’relu’ を指定すればよいでしょう。

こうすることで、非線形変換が可能になります。
ただし、今回は回帰問題なので、最後の Dense層の activation は、 ’linear’ と指定しておきましょう。

Dropout層
model_1、model_2 共に、途中に Dropout層を挟んでいます。

Dropout層は過学習を抑制する効果があり、ディープラーニングモデルを作成する際には入れておくと良いと思います。

compileメソッド
最後に、compileメソッドを用いる際に、問題に対応した誤差関数（loss）を指定するように注意しましょう。今回で言えば回帰問題を解くので loss=’mae’ を指定しました。
'''

model_1 = Sequential()
model_1.add(Dense(5, activation='relu', input_shape=(20,)))
model_1.add(Dropout(0.5))
model_1.add(Dense(1, activation='linear'))
model_1.summary()
model_1.compile(optimizer='adam',
           	loss='mse',
           	metrics=['mae'])




model_2 = Sequential()
model_2.add(LSTM(10,
            	dropout=0.2,
            	recurrent_dropout=0.2,
            	input_shape=(20,1)))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(1, activation='linear'))
model_2.summary()
model_2.compile(optimizer='adam',
           	loss='mse',
           	metrics=['mae'])






'''
入力用、出力用テンソルの作成
ここで、入力用、出力用のテンソルを作成します。

作成の際には、「入力次元」「出力次元」が、modelと整合的になるように注意してください。
今回は、model_1 に合わせて、出力次元を (sample,timesteps) とします。

入力用、出力用テンソルを出力する関数
pd.dataframe を入力し、入力用、出力用テンソルを出力する関数は以下のようになります。
'''

def getInputLabel(data, period=20):
	period = period
	input_tensor = []
	label_tensor = []
	for i in range(0, len(data) - period, 1):
		input_tensor.append(data.values[i:i + period,0])
		label_tensor.append(data.values[i + period,0])
	input_tensor = np.array(input_tensor)
	label_tensor = np.array(label_tensor)
	return input_tensor, label_tensor





'''
ここで、Topixデータを当てはめてしまいたいところですが、Topix の値は、先ほどプロットしたように変動幅が大きいため、標準化するよう心がけましょう。
（そのままの値を用いると大きな値の影響度が大きくなってしまいます）
Topixデータを標準化
よって、以下のように Topixデータを標準化してから当てはめていきます。
'''
tmp = data - data.mean()
tmp = tmp/data.std()
input_tensor, label_tensor = getInputLabel(data = tmp)


tmp = data - data.mean()
tmp = tmp/data.std()
input_tensor, label_tensor = getInputLabel(data = tmp)






'''
トレーニングデータとテストデータに分割
さて、機械学習の分野では、「トレーニングデータでパラメータの学習」を行い、「テストデータを用いて そのパラメータの評価」を行います。

幸いにも、sklearn.model_selection に、そのような関数が用意されているので使わせて頂きましょう。
'''
X_train, X_test, y_train, y_test = train_test_split(input_tensor, label_tensor, test_size=0.2,random_state=100, shuffle = False)






'''
ディープラーニング実践
それでは、まずは下準備していきます。

モデルの過学習を防ぐために、以下のようにコールバックを設定しましょう。

earlystopping = EarlyStopping(monitor='loss', patience=5)
1
earlystopping = EarlyStopping(monitor='loss', patience=5)
EarlyStopping の引数を簡単に説明すると

monitor： どの値で過学習を判断するのか
patience： どのくらのエポック数改善しなければ学習を終了するか
株価は、1、2エポック学習が進まなくても、その後急に改善することがあるので、 patience=5 に設定しました。
'''
earlystopping = EarlyStopping(monitor='loss', patience=5)





'''
モデルのパラメータを学習させてみる
では、いよいよモデルのパラメータを学習させましょう！

とは言っても難しい作業はなく、以下のように model の fitメソッドを呼び出せばいいだけです。

ただし、model_2 を学習させる際は、次元を一つ増やすために、入力用テンソルを X_train[:,:,np.newaxis]  としていることに注意してください。
'''
model_1.fit(X_train, y_train, batch_size=10, epochs=50, callbacks=[earlystopping])

model_2.fit(X_train[:,:,np.newaxis], y_train, batch_size=10, epochs=50, callbacks=[earlystopping])

'''
学習終了
EarlyStopping が作動し
model_1は17エポック,model_2は14エポック
で学習が終了しました。

今回は「バッチ数10」「エポック数50」で行いました。

ですが、適当なので、自分で行う際は変えていただいて構いません！

学習の過程を見てみると、どちらのモデルも「loss」が減っていて学習が上手く行われていることが分かります。
'''





'''
結果
テストデータで結果を図示してみましょう。
上手く予想できているのでしょうか…
'''
predicted = model_1.predict(X_test)
result = pd.DataFrame(predicted)
result.columns = ['predict(Yosoku)']
result['actual(jiisai)'] = y_test
result.plot()
plt.show()


predicted = model_2.predict(X_test[:,:,np.newaxis])
result = pd.DataFrame(predicted)
result.columns = ['predict']
result['actual'] = y_test
result.plot()
plt.show()

'''
見た感じ、model_1の方が予測の精度がよさそうです。
LSTM を使っている model_2 はどうしたのでしょう…？
株価との相性が悪いのでしょうか？
'''




'''
価格の上げ下げの正解率で評価してみる
別の観点から評価してみましょう。

株価を予想するモチベーションは、なんと言っても「お金を稼ぐ」こと。

価格の上げ下げの正解率は、どのくらいか調べてみましょう。

正解率を調べる関数を、以下のように定義しました。
'''

def updown(actual, predict):
	act = np.sign(np.diff(actual))
	pre = np.sign(np.diff(predict[:,0]))
	tmp =  act*pre>0
	return np.sum(tmp)/len(tmp)


print('model_1:',updown(y_test, model_1.predict(X_test)))
print('model_2:',updown(y_test, model_2.predict(X_test[:,:,np.newaxis])))	


'''
正解率を調べた結果は
では、「model_1」「model_2」のどちらが稼ぐことのできるモデルか調べてみます。
なんと、これでも model_1 に敗北してしまいました！
しかし、正解率が「0.5」を超えないようでは、実際に使用することは出来ません…。
'''






'''
改善案その２
稼げるモデルを目指すため、目的関数を変更してみます！

先ほどは、株価を予想する（つまり回帰）モデルを構築しました。

「改善案その２」では、株価を予想するのではなく、価格の上げ下げを予想するモデルに変更してみます。

データは、Topix のみのものを使用します。

モデルの構築
では、モデルを構築します。
'''
model_2 = Sequential()
model_2.add(LSTM(10,
            	dropout=0.2,
            	recurrent_dropout=0.2,
            	input_shape=(20,1)))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(1, activation='sigmoid'))
model_2.summary()
model_2.compile(optimizer='adam',
           	loss='binary_crossentropy',
           	metrics=['acc'])



'''
モデルの変更点は、最後の Dense層の activetion を ’sigmoid’ にしたこと。
そして、compileメソッドの引数loss を、 ’binaryf_crossentroppy’ としたことです。
このようにすることで、2値分類（価格が上がるか下がるか）を実行できるモデルが構築できます。
'''

'''
出力ラベルの変更
モデルに合わせて、出力ラベルを変更することも忘れずに行いましょう。

出力ラベルを、次の日Topixの値が上がったら「１」、次の日Topixの値が下がったら「０」とする関数は、以下のようになります。
'''

def getInputLabel(data, period=20):
	period = period
	input_tensor = []
	label_tensor = []
	for i in range(0, len(data) - period, 1):
		input_tensor.append(data.values[i:i + period, 0])
		label_tensor.append(np.diff(data.values[:,0])[i + period -1])
	input_tensor = np.array(input_tensor)
	label_tensor = np.sign(np.array(label_tensor))
	label_tensor[label_tensor<0] = 0
	return input_tensor, label_tensor


'''
モデルを学習
先ほどと同様の手順で、モデルを学習していきます。

今回は2値分類なので、結果は図示できません。

テストデータの正解率
代わりに、テストデータの正解率を見てみましょう。
'''
model_2.evaluate(X_test[:,:,np.newaxis], y_test)
print('model_2:',updown(y_test, model_2.predict(X_test[:,:,np.newaxis])))	
'''
evaluate メソッドは、「与えられた入力データ」と「その正解データ」に対してモデルを適用し、(loss, metrics) のタプルを返す関数です。

metrics=’acc’ と指定しているため、metrics の値は正解率となります。

ですので、このモデルは 0.56 の正解率を達成することが出来たことになります！

やはり「目的に合ったモデルを構築する」ということが何よりも大切だということが分かりますね。
'''

'''
さいごに
ディープラーニングを用いて株価予想をしてみました。
「絶対当たる夢のようなモデル」はやはり一筋縄ではいかないようです。

ですが今回の検証で、最終的には56％の正解率にたどり着いたことを考えると、「8割方当たるモデル」というのは工夫次第で実現可能かもしれませんね…！

もっとも、そのようなモデルが発見され広まれば、全員がそのモデルを使用するので、当たらないモデルになってしまいますね（笑）
'''