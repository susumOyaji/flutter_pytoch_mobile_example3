import numpy
#import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation
from keras.layers import Dense, LSTM, Dropout, Flatten


 
IN_NODES = 10			# length of input data
HIDDEN_NODES = 20		# number of hidden nodes
OUT_NODES = 1			# length of output data
MODELFILE = 'model_rnn_nikkei.hdf5'
TESTDATA = 'nikkei_yosoku_data.csv'


 


X_train = []  # 教師データ
y_train = []  # 上げ下げの結果の配列
y_test = []
code = '6758'
code_dow = '^DJI'
code_nikkei = '^N225'

import datetime
from dateutil.relativedelta import relativedelta
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2021, 1, 1)#教師データ(今までのデータ)
end_train= datetime.date.today() + relativedelta(days=-1)#昨日分(today-1日)まで取得できる（当日分は変動しているため）


start_test = datetime.date.today() + relativedelta(days= -10)#試験データ
end_test = datetime.date.today()#昨日分(today-1日)まで取得できる（当日分は変動しているため）


'''学習データの前処理'''
'''使うデータを読み込む。'''
from pandas_datareader import data as pdr
data = pdr.get_data_yahoo(f'{code}.T', start_train, end_train) # 教師データを読み込む。
Stock_test_df = pdr.get_data_yahoo(f'{code}.T', start_test, end_test)# 試験データを読み込む。



Dow_df = pdr.get_data_yahoo(code_dow, start_train, end_train)# 試験データのcsvファイルを読み込む。
Nikkei_df = pdr.get_data_yahoo(code_nikkei, start_train, end_train)# 試験データのcsvファイルを読み込む。

# data load
#data = None
#data = pandas.read_csv(TESTDATA)
##data = pd.read_csv(r'./data/stocks_price_data/nikkei_yosoku_data.csv')
#data.columns = ['date','openprice','highprice','lowprice','closeprice','Adj Close' ]
#data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
 


# hyojunka & reduce data
#StandardScalerはデータセットの標準化機能を提供してくれています。
#標準化を行うことによって、特徴量の比率を揃えることが出来ます。
#例えば偏差値を例にすると、100点満点のテストと50点満点のテストがあったとして
#点数の比率、単位が違う場合でも標準化を利用することでそれらの影響を受けずに点数を評価できます。
scaler = StandardScaler()
# need values
datatmp = Stock_test_df['Close'].values.reshape(-1, 1)
Stock_test_df['Close'] = scaler.fit_transform(datatmp)
data = Stock_test_df.sort_values(by='Date')
data = data.reset_index(drop=True)
data = data.loc[:, ['Close']]
 
feature_test = data['Close'].values.reshape(-1,1)        # need values


#print(feature_test)


 
# build RNN model
# RNNモデルの構造を定義しています。
# これに従ってデータが学習されます。
model = Sequential()
model.add(InputLayer(batch_input_shape=(None, IN_NODES, OUT_NODES)))
model.add(SimpleRNN(units=HIDDEN_NODES, return_sequences=False))
model.add(Dense(OUT_NODES))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")


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
model_2.add(LSTM(10,dropout=0.2,recurrent_dropout=0.2,input_shape=(20,1)))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(1, activation='sigmoid'))
model_2.summary()
model_2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
'''



# load model
model.load_weights(f'./data/stocks_price_data/{MODELFILE}')

print('Train loaded.')  
 
# predict
#入力サンプルに対する予測値の出力を生成します。
predicted = model.predict(feature_test)                       # 1 output
predicted = scaler.inverse_transform(predicted)               # inverse hyoujunka result
feature_test = scaler.inverse_transform(feature_test)         # inverse hyoujunka indata

data = Stock_test_df.reset_index(drop=False)
data = data['Date']
datalast = data[(data.size)-1]
#print(now.strftime("%Y/%m/%d %H:%M:%S %B, %a")) # 2021/01/04 16:01:02 January, Mon
datalast = datalast.strftime("%Y-%m-%d") # 2021/01/04

 
print(f'Last day {datalast} =  ' + str(feature_test[-1,-1]))
print('Next day = ' + str(predicted[-1,-1]))