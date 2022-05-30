import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
import seaborn as sns

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import keras as keras

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

fonts = fm.findSystemFonts()
sns.set(font=['IPAPGothic'])

def stock_calc(stock_code,rows=0):
    data = quandl.get(stock_code,rows=rows)
    #株価を入れる場合はquandlのauthtokenが必要
    #data = quandl.get(stock_code,rows=rows,authtoken='need authtoken!!')
    #カラム変換
    columns = data.columns
    trans_columns =pd.Series(
        ['open','high','low','close','open','high','low','close'],
        index=['Open Price', 'High Price', 'Low Price', 'Close Price','Open', 'High', 'Low', 'Close']
    )
    arr = []
    for i in range(len(columns)):
        if columns[i] in trans_columns.index:
            arr.append(trans_columns[columns[i]])
        else:
            data = data.drop(columns[i], axis=1)
    
    data.columns=arr
    return data


def plot_grap(data):
    fig1=plt.figure(figsize=(50,10))
    ax1=fig1.add_subplot(111)
    plt.title("終値と移動平均")
    ax1.plot(data['close'],label="close")
    ax1.plot(data['ma5'],label="5")
    ax1.plot(data['ma10'],label="10")
    ax1.plot(data['ma20'],label="20")
    ax1.plot(data['ma25'],label="25")
    plt.legend()
    plt.tight_layout()
    
    return

def grap(data):
    fig1=plt.figure(figsize=(50,20))
    ax1=fig1.add_subplot(211)
    plt.title("ボリンジャーバンドとシグナル")
    ax1.plot(data['close'],color='b')
    ax1.plot(data['ma20'],color='orange')
    ax1.plot(data['upper1'],color='red')
    ax1.plot(data['lower1'],color='red')
    ax1.plot(data['upper2'],color='red')
    ax1.plot(data['lower2'],color='red')
    ax1.plot(data['upper3'],color='red')
    ax1.plot(data['lower3'],color='red')

    plt.legend()
    plt.tight_layout()
    return

def calc(data):
    open=data['open'].copy()
    close=data['close'].copy()
    high = data['high'].copy()
    low = data['low'].copy()

    #移動平均の計算
    ma25 = close.rolling(window=25).mean()
    ma20 = close.rolling(window=20).mean()
    ma10 = close.rolling(window=10).mean()
    ma5 = close.rolling(window=5).mean()

    #BB
    window_num=20 #移動平均の日数
    ma = close.rolling(window=window_num).mean()
    rstd = close.rolling(window=window_num).std()
    upper1 = ma + rstd
    lower1 = ma - rstd 
    upper2 = ma + rstd * 2
    lower2 = ma - rstd * 2
    upper3 = ma + rstd * 3
    lower3 = ma - rstd * 3

    #MACD
    ShotEMA_per = 12 #短期EMAの期間
    LongEMA_per = 26 #長期EMAの期間
    SignalSMA_per = 9 #SMAを取る期間
    MACD = close.ewm(span=LongEMA_per).mean() - close.ewm(span=ShotEMA_per).mean()
    MACDSignal = MACD.rolling(SignalSMA_per).mean()

    #Momentum
    mom_period = 10
    shift = close.shift(mom_period)
    mon = close/shift*100


    #RSI
    #「Relative Strength Index」の略であり、日本語で「相対力指数」と呼びます。
    # 一般的に相場の過熱感を分析するインジケーターであり、買われすぎや売られすぎを判断するオシレーター系に分類されます。
    RSI_period = 14
    diff = close.diff(1)
    positive = diff.clip(lower=0).ewm(alpha=1/RSI_period).mean()
    negative = diff.clip(upper=0).ewm(alpha=1/RSI_period).mean()
    RSI = 100-100/(1-positive/negative)

    #HLband(Close)
    band_period = 20 #期間
    Hline = close.rolling(band_period).max()
    Lline = close.rolling(band_period).min()

    #HLband(High/Low)
    HHline = high.rolling(band_period).max()
    LLline = low.rolling(band_period).min()

    #Stochastics
    Kperiod = 14 #%K期間
    Dperiod = 3  #%D期間
    Slowing = 3  #平滑化期間
    Hline = high.rolling(Kperiod).max()
    Lline = low.rolling(Kperiod).min()
    sumlow = (close-Lline).rolling(Slowing).sum()
    sumhigh = (Hline-Lline).rolling(Slowing).sum()
    Stoch = sumlow/sumhigh*100
    StochSignal = Stoch.rolling(Dperiod).mean()

    #未来の10日以内の最大最小値
    window=10
    close10_max = close.copy()
    close10_min = close.copy()
    close10_max.column=['close10_max']
    close10_max.column=['close10_min']
    for i in range(len(close)):
        close10_max[i]=close[i:i+window].max()
        close10_min[i]=close[i:i+window].min()

    #未来の5日以内の最大最小値
    window=5
    close5_max = close.copy()
    close5_min = close.copy()
    close5_max.column=['close5_max']
    close5_max.column=['close5_min']
    for i in range(len(close)):
        close5_max[i]=close[i:i+window].max()
        close5_min[i]=close[i:i+window].min()

        
    #未来の5日以内の1%以上上昇or下落
    window=5
    close5_updown = close.copy()
    close5_updown.column=['close5_updown']
    for i in range(len(close)):
        close5_updown[i]  =  0
        if (close[i:i+window].max().item() / close[i:i+1].item()) > 1.01 and (close[i:i+window].min().item() / close[i:i+1].item()) < 0.99 : #乱高下
            close5_updown[i]  =  1
        if (close[i:i+window].max().item() / close[i:i+1].item()) > 1.01 : #上昇
            close5_updown[i]  =  2
        elif (close[i:i+window].min().item() / close[i:i+1].item()) < 0.99 : #下落
            close5_updown[i]  =  3


    #標準偏差を正規化
    normal_rstd = rstd / rstd.max()
        
    #std変化率
    rstd_sift_num=1
    rstd_mon = rstd / rstd.shift(rstd_sift_num)
    #変化率の前日差
    rstd_mon_diff = rstd_mon / rstd_mon.shift(rstd_sift_num)
    #変化率差を標準化して標準偏差値をもとめ、σ３を超えるところでシグナルを出す
    normal_rstd_mon_diff = rstd_mon_diff / rstd_mon_diff.max()
    rstd_mon_diff_sigma=(normal_rstd_mon_diff.mean()+normal_rstd_mon_diff.std()*3)
    normal_rstd_mon_diff_sig = normal_rstd_mon_diff > rstd_mon_diff_sigma 
    
    #前日との差がプラスの場合＝バンドが広がっている
    rstd_avg_diff = 1
    rstd_Xday_avg = rstd.rolling(window=rstd_avg_diff).mean()
    rstd_diff = (rstd_Xday_avg - rstd_Xday_avg.shift(rstd_sift_num))>0
    #両方満たすものがシグナル
    normal_rstd_mon_diff_plus_dif_sig = normal_rstd_mon_diff_sig & rstd_diff
    
    #長期間窪んでいる個所を見つける
    #-σより小さければ対象とする
    mean_rstd_sig3 = rstd.rolling(window=10).mean() - rstd.std()
    rstd_mainas_sig3 = rstd < mean_rstd_sig3 
    
    tmp = pd.concat([data, ma5,ma10,ma20,ma25,rstd,upper1,lower1,upper2,lower2,upper3,lower3,
                     mon,MACD,MACDSignal,RSI,Hline, Lline,HHline, LLline,Stoch,StochSignal,
                     rstd_mon,rstd_mon_diff,normal_rstd_mon_diff_sig,normal_rstd_mon_diff_plus_dif_sig,normal_rstd,rstd_mainas_sig3,
                     close5_max,close5_min,close10_max,close10_min,close5_updown
                    ],
                    axis=1)
    tmp.columns = [
        'open','high', 'low','close','ma5','ma10','ma20','ma25',
        'rstd','upper1','lower1','upper2','lower2','upper3','lower3',
        'mon','MACD','MACDSignal','RSI',
        'Hline', 'Lline','HHline', 'LLline','Stoch','StochSignal',
        'rstd_mon','rstd_mon_diff','normal_rstd_mon_diff_sig','normal_rstd_mon_diff_plus_dif_sig','normal_rstd','rstd_mainas_sig3',
        'close5_max','close5_min','close10_max','close10_min','close5_updown'
    ]
    
    return tmp




def build_multilayer_perceptron(input_dim,output_class):
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dense(output_class))
    model.add(Activation('softmax'))
    return model

def makedata(data):
	data = calc(src_data)
	data=data.dropna()
	data.reset_index( drop = True )


	SRC_X = data[['close','ma5','ma10','ma20','ma25','rstd','upper1','lower1','upper2','lower2','upper3','lower3','mon','MACD','MACDSignal','RSI','Hline', 'Lline','HHline', 'LLline','Stoch','StochSignal']]
	SRC_Y = data[['close5_updown']]
	SRC_X2 = data[['close']]
	X = SRC_X.values
	Y = SRC_Y.values
	#各次元が属性ごとに標準化したデータなるようにする
	col = X.shape[1]
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	for i in range(col):
		X[:,i] = (X[:,i] - mu[i]) / sigma[i]
	Y = np_utils.to_categorical(Y)
	return X,Y

def fit(model,X,Y):
	# モデル訓練
	#fpath = './logs/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
	fpath = './logs/weights.hdf5'
	cp_cb = keras.callbacks.ModelCheckpoint(filepath = fpath , monitor='acc',verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
	tb_cb=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
	model.fit(X, Y, epochs=5, batch_size=1000, verbose=1, callbacks=[tb_cb,cp_cb])

	loss, accuracy = model.evaluate(X, Y, verbose=0)
	print("Accuracy = {:.2f}".format(accuracy))

	#print(model.predict_classes(X[-5:]))
	#print(Y[-5:])
	#print(model.evaluate(X[-5:],Y[-5:]))
	#print(model.predict_proba(X[-5:-4]))
	return model

def evaluate(model,X,Y):
	return model.evaluate(X, Y, verbose=0)





'''
main()
'''
#src_data = stock_calc('NIKKEI/INDEX',rows=1000)
src_data = stock_calc('WIKI/KO',rows=1000)
print(src_data)
#plot_grap(calc(src_data))
#grap(calc(src_data))

X,Y = makedata(src_data)


# モデル構築
model = build_multilayer_perceptron(X.shape[1],Y.shape[1])
#学習済みモデルをロードする場合
'''
model = keras.models.load_model('nk225_model.h5')
'''
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model=fit(model,X[:-100],Y[:-100])
model.save('nk225_model.h5')

loss_and_metrics = model.evaluate(X[-100:], Y[-100:])
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
print("過去５日間の予想値",model.predict(X[-5:]))





import tensorflow as tf
# load model
#model = tf.keras.models.load_model('./models/model.hdf5')
model = tf.keras.models.load_model('nk225_model.h5')
# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save to *.tflite
open("assets/nk225_model.tflite", "wb").write(tflite_model)
with open('assets/labels.txt', mode='wt', encoding='utf-8') as fp:
    fp.write('Up\n')
    fp.write('Down\n')



#日経225銘柄全体で学習させる場合(要authtoken）
'''
nk225_code =['9984','9983','9766','9735','9681','9613',
'9602','9532','9531','9503','9502',
'9501','9437','9433','9432','9412','9301','9202','9107','9104','9101',
'9064','9062','9022','9021','9020','9009','9008','9007','9005','9001',
'8830','8804','8802','8801','8795','8766','8750','8729','8725','8630',
'8628','8604','8601','8411','8355','8354','8331','8316','8309','8308',
'8306','8304','8303','8267','8253','8252','8233','8058','8053','8035',
'8031','8028','8015','8002','8001','7951','7912','7911','7762','7752',
'7751','7735','7733','7731','7272','7270','7269','7267','7261','7211',
'7205','7203','7202','7201','7186','7013','7012','7011','7004','7003',
'6988','6976','6971','6954','6952','6902','6857','6841','6773','6770',
'6762','6758','6752','6703','6702','6701','6674','6508','6506','6504',
'6503','6502','6501','6479','6473','6472','6471','6367','6366','6361',
'6326','6305','6302','6301','6113','6103','5901','5803','5802','5801',
'5715','5714','5713','5711','5707','5706','5703','5631','5541','5413',
'5411','5406','5401','5333','5332','5301','5233','5232','5214','5202',
'5201','5108','5101','5020','5002','4911','4902','4901','4755','4704',
'4689','4578','4568','4543','4523','4519','4507','4506','4503','4502',
'4452','4324','4272','4208','4188','4183','4151','4063','4061','4043',
'4042','4021','4005','4004','3865','3863','3861','3436','3407','3405',
'3402','3401','3382','3289','3105','3103','3101','3099','3086','2914',
'2871','2802','2801','2768','2531','2503','2502','2501','2432','2282',
'2269','2002','1963','1928','1925','1812','1808','1803','1802','1801',
'1721','1605','1333','1332']

for i in nk225_code:
    code = 'TSE/'+i
    src_data = stock_calc(code,rows=3000)
    tmpX,tmpY = makedata(src_data)
    X = np.vstack((X,tmpX))
    Y = np.vstack((Y,tmpY))
    #loss,accuracy = evaluate(model,X,Y)
    #print(i)
    #print(model.predict_classes(X[-1:]))
    #print(model.predict_proba(X[-1:]))
    #print("Accuracy = {:.2f}".format(accuracy))
    #model=fit(model,X,Y)
    #model.save('nk225_model.h5')
    print(X.shape[0])
# モデル構築
model = build_multilayer_perceptron(X.shape[1],Y.shape[1])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model=fit(model,X,Y)
loss_and_metrics = model.evaluate(X, Y)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
model.save('nk225_model.h5')

'''
