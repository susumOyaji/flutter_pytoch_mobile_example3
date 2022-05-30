import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
import seaborn as sns
sns.set()

#株価データのインポート
data = pd.read_csv('data/stocks_price_data/nikkei-225-index-historical-chart-data.csv', header=8)

#前半の部分を削除
data = data.query('index >9000')
data = data.drop(['date'],axis =1)

#データを確認
data.plot()

# ５０日分のデータを１塊とした窓を作る
def _load_data(data, n_prev = 50):  
   
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 50):  
    
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)

#株価の平均値で割ることで正規化を実施
df = data / data.mean()
length_of_sequences = 50
(X_train, y_train), (X_test, y_test) = train_test_split(df, test_size = 0.1, n_prev = length_of_sequences)

#確認
print("X_train = ",X_train.shape)
print("y_train = ",y_train.shape)
print("X_test  = ",X_test.shape)
print("y_test  = ",y_test.shape)




length_of_sequences = X_train.shape[1]
hidden_neurons = 128
in_out_neurons = 1

#LSTMモデル作成
model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
model.add(Dropout(0.25))#念のためDropout
model.add(Dense(1))  
model.add(Activation("linear")) 

model.compile(loss="mean_squared_error", optimizer="adam")
history = model.fit(X_train, y_train, batch_size=128, epochs=60, validation_split=0.2)


# 予測値算出
predicted = model.predict(X_test) 
# 実績と比較
dataf =  pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["Stock price"] = y_test
dataf.plot(figsize=(15, 5))
# グラフを表示する
plt.show()
