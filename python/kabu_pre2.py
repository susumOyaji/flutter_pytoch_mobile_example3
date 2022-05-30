#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 上場企業４２３社の２０年分を学習
# 終値前日比1.03%以上であるかを２クラス分類
# データ比１１：１を１：１に調整（減らした）
# StandardScaler accuracy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import *
import seaborn as sns
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
import mglearn
import random

from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.feature_selection import SelectPercentile
import os
import glob

import datetime
import pandas_datareader.data as web
import locale

# 実行上問題ない注意は非表示にする
warnings.filterwarnings('ignore') 




#df_data_j = pd.read_excel("./data/data_j.xls")

start = datetime.date(2021,1,1)
end = datetime.date.today()
code = '6758.T'#SONY

#for index, row in df_data_j.iterrows():
try:
    #df = web.DataReader(f'{code}.T', 'yahoo', start, end)
    #CSVファイル読み込み
    #URL = "http://www.data.jma.go.jp/obd/stats/data/mdrr/pre_rct/alltable/pre1h00_rct.csv"
    #df = pd.read_csv(URL,encoding="SHIFT-JIS")
    #df = pd.get_data_yahoo("SPY", start, end)
    df = pd.read_csv(f'https://stooq.com/q/d/l/?s={code}.jp&d1={start}&d2={end}&i=d',index_col=0)
except:
    print("error")
print("web.DataReader",df)
df.to_csv("./data/stocks_price_data/stock_price_data_{}.csv".format(code))
print("df.to_csv",df)


# In[2]:


# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
#files = glob.glob("data/stocks_price_data/stock_price_data_{}.csv".format(code))


# In[3]:


# 説明変数となる行列X, 被説明変数となるy2を作成
base = 100
day_ago = 3
num_sihyou = 8
reset =True
X =[]


#まず、span01,02,03という3つの変数を準備します。
#これに、5と25と50を代入します。
#これは過去5日分の移動平均と、25日の移動平均、50日の移動平均を算出するために使います。
#pythonで移動平均を算出するには、pandasのrolling()メソッドを使います。
#データフレームに単純移動平均の略称sma01というカラムを追加しましょう。
#データフレームに追加したいカラム名を記載します。

#そして、データフレームから1列取り出したpriceを記述します。
#ドットを書いてrollingと記述します。丸括弧の中には移動平均の日数を記述します。
#windowという引数に日数を記述します。
#先ほど、span01に5という変数を代入したのでこれを使いましょう。
#そして、最後の平均という意味のmeanを書いて丸括弧です。
#ちなみに、ここにmaxを書くと、過去5日間の最大値を取得できます。
#またminと書くと過去5日間の最小値を取得できます。
#これと同様にsma02というカラムを作って、過去25日の移動平均を作り、sma03というカラムを作って過去50日間の移動平均を作ります。
#実行してみます。
#さて、表示させてみます。


#5日平均
span01=5
#25日平均
span02=25
#75日平均
span03=75

encoding = locale.getpreferredencoding()
 
df = pd.read_csv("data/stocks_price_data/stock_price_data_{}.csv".format(code), header = 0, encoding = 'cp932')
price = df['Adj Close']
#単純移動平均の略称sma01
df['sma01'] = price.rolling(window=span01).mean()
df['sma02'] = price.rolling(window=span02).mean()
df['sma03'] = price.rolling(window=span03).mean()
df = df.fillna(0)#（NaN）をゼロで置換


df.to_csv('data/stocks_price_data/add_column.csv', index = False, columns=['Date','Open', 'High','Low','Close','sma01','sma02','sma03','Volume','Adj Close'],encoding=encoding)


# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("data/stocks_price_data/add_column.csv")
# すべてのCSVファイルから得微量作成
for file in files:
    temp = pd.read_csv(file, header=0, encoding='cp932')
    print(temp)
    #temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp = temp[['Date','Open', 'High','Low','Close','sma01','sma02','sma03','Volume']]
    temp= temp.iloc[::-1]#上下反対に
    temp2 = np.array(temp)
    
    # 前日比を出すためにbase日後からのデータを取得
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp3), 0] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4]
    temp3[0:len(temp3), 1] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 2] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 3] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float)
    
    # tempX : 現在の企業のデータ
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    
    # 日にちごとに横向きに（day_ago）分並べる
    # sckit-learnは過去の情報を学習できないので、複数日（day_ago）分を特微量に加える必要がある
    # 注：tempX[0:day_ago]分は欠如データが生まれる
    for s in range(0, num_sihyou): 
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
             
    # Xに追加
    # X : すべての企業のデータ
    # tempX[0:day_ago]分は削除
    if reset:
        X = tempX[day_ago:]
        reset = False
    else:
        X = np.concatenate((X, tempX[day_ago:]), axis=0)

# 何日後を値段の差を予測するのか
pre_day = 1
# y : pre_day後の終値/当日終値
y = np.zeros(len(X))

y[0:len(y)-pre_day] = X[pre_day:len(X),0]

up_rate =1.03

# データを一旦分別
X_0 = X[y<=up_rate]
X_1 = X[y>up_rate]
y_0 = y[y<=up_rate]
y_1 = y[y>up_rate]

# X_0をX_1とほぼ同じ数にする
X_drop, X_t, y_drop, y_t = train_test_split(X_0, y_0, test_size=0.09, random_state=0)

# 分別したデータの結合
X_ = np.concatenate((X_1, X_t), axis=0)
y_ = np.concatenate((y_1, y_t))


# In[4]:

# 確認
# 何もしないときの比率
print("X.shape: ", X.shape)

print("yの割合")
# yc：翌日の終値/当日の終値がup_rateより上か
yc = np.zeros(len(y))
for i in range(0, len(yc)):
    if y[i] <= up_rate:
        yc[i] = 0
    else:
        yc[i] = 1

pd_yc = pd.DataFrame(yc)
print(pd_yc[0].value_counts())

'''出力結果
X.shape:  (1734326, 24)
yの割合
0.0    1594644
1.0     139682
Name: 0, dtype: int64
'''


# In[5]:


# 確認
# X_0をX_1の数を調整後の比率
print("X_.shape: ", X_.shape)

print("y_の割合")
# yc：翌日の終値/当日の終値がup_rateより上か
yc_ = np.zeros(len(y_))
for i in range(0, len(yc_)):
    if y_[i] <= up_rate:
        yc_[i] = 0
    else:
        yc_[i] = 1

pd_yc_ = pd.DataFrame(yc_)
print(pd_yc_[0].value_counts())

'''出力結果
X_.shape:  (283200, 24)
y_の割合
0.0    143518
1.0    139682
Name: 0, dtype: int64
'''


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)

# y_train_,y_test2：翌日の終値/当日の終値がup_rateより上か
y_train2 = np.zeros(len(y_train))
for i in range(0, len(y_train2)):
    if y_train[i] <= up_rate:
        y_train2[i] = 0
    else:
        y_train2[i] = 1
        
y_test2 = np.zeros(len(y_test))
for i in range(0, len(y_test2)):
    if y_test[i] <= up_rate:
        y_test2[i] = 0
    else:
        y_test2[i] = 1


# In[7]:


print("start time: ", datetime.datetime.now().strftime("%H:%M:%S"))

X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(max_iter=200000, alpha=0.001))])
param_grid = {'classifier__hidden_layer_sizes': [(10,), (100,), (500,)]}

grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, cv=2 ,return_train_score=False, scoring="accuracy")
grid.fit(X_train, y_train)

print(grid.cv_results_['mean_test_score'])
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("over time: ", datetime.now().strftime("%H:%M:%S"))

conf = confusion_matrix(y_test, grid.predict(X_test))
print(conf)

'''出力結果
start time:  00:01:59
[0.6651177  0.66957156 0.6526177 ]
Best parameters:  {'classifier__hidden_layer_sizes': (100,)}
grid best score,  0.6695715630885123
Test set score: 0.67
over time:  00:21:33
[[24043 11924]
 [11108 23725]]
'''


# In[ ]:


# 一目均衡表,転換線,基準線,先行スパン1,先行スパン2,25日ボリンジャーバンド, 曜日 追加
# 説明変数となる行列X, 被説明変数となるy2を作成
base = 100 
day_ago = 3
num_sihyou = 16
reset =True
# すべてのCSVファイルから得微量作成
for file in files:
    temp = pd.read_csv(file, header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対に
    temp2 = np.array(temp)
    
    # 前日比を出すためにbase日後からのデータを取得
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp3), 0] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4]
    temp3[0:len(temp3), 1] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 2] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 3] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float)
    
     # 一目均衡表を追加します (9,26, 52) 
    para1 =9
    para2 = 26
    para3 = 52
    temp2_2 = np.c_[temp2, np.zeros((len(temp2), 3))]
    p1 = 9
    p2 = 10
    p3 =11
    
    # 転換線 = （過去(para1)日間の高値 + 安値） ÷ 2
    for i in range(para1, len(temp2)):
        tmp_high =temp2[i-para1+1:i+1,2].astype(np.float)
        tmp_low =temp2[i-para1+1:i+1,3].astype(np.float)
        temp2_2[i, p1] = (np.max(tmp_high) + np.min(tmp_low)) / 2 /temp2[i, 4]
        
    temp3[0:len(temp3), 8] = temp2_2[base:len(temp2), p1]

    # 基準線 = （過去(para2)日間の高値 + 安値） ÷ 2
    for i in range(para2, len(temp2)):
        tmp_high =temp2[i-para2+1:i+1,2].astype(np.float)
        tmp_low =temp2[i-para2+1:i+1,3].astype(np.float)
        temp2_2[i, p2] = (np.max(tmp_high) + np.min(tmp_low)) / 2 /temp2[i, 4]
    temp3[0:len(temp3), 9] = temp2_2[base:len(temp2), p2]
        

    # 先行スパン1 = ｛ （転換値+基準値） ÷ 2 ｝を(para2)日先にずらしたもの
    temp3[0:len(temp3), 10] = (temp2_2[base-para2:len(temp2)-para2, p1] + temp2_2[base-para2:len(temp2)-para2, p2]) /2 /temp2[base:len(temp2), 4]

    # 先行スパン2 = ｛ （過去(para3)日間の高値+安値） ÷ 2 ｝を(para2)日先にずらしたもの
    for i in range(para3, len(temp2)):
        tmp_high =temp2[i-para3+1:i+1,2].astype(np.float)
        tmp_low =temp2[i-para3+1:i+1,3].astype(np.float)
        temp2_2[i, p3] = (np.max(tmp_high) + np.min(tmp_low)) / 2 /temp2[i, 4]
    temp3[0:len(temp3), 11] = temp2_2[base-para2:len(temp2)-para2, p3]

    # 25日ボリンジャーバンド（±1, 2シグマ）を追加します
    parab = 25
    for i in range(base, len(temp2)):
        tmp25 = temp2[i-parab+1:i+1,4].astype(np.float)
        temp3[i-base,12] = np.mean(tmp25) + 1.0* np.std(tmp25) 
        temp3[i-base,13] = np.mean(tmp25) - 1.0* np.std(tmp25) 
        temp3[i-base,14] = np.mean(tmp25) + 2.0* np.std(tmp25) 
        temp3[i-base,15] = np.mean(tmp25) - 2.0* np.std(tmp25)
        
    # tempX : 現在の企業のデータ
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    
    # 日にちごとに横向きに（day_ago）分並べる
    # sckit-learnは過去の情報を学習できないので、複数日（day_ago）分を特微量に加える必要がある
    # 注：tempX[0:day_ago]分は欠如データが生まれる
    for s in range(0, num_sihyou): 
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
             
    # 曜日情報の追加
    ddata = pd.to_datetime(temp['日付'], format='%Y%m%d')
    daydata = ddata[base:len(temp2)].dt.dayofweek
    daydata_dummies = pd.get_dummies(daydata, columns=['Yobi'])
    daydata2 = np.array(daydata_dummies)
    tempX = np.concatenate((tempX, daydata2), axis=1)
    
    # Xに追加
    # X : すべての企業のデータ
    # tempX[0:day_ago]分は削除
    if reset:
        X = tempX[day_ago:]
        reset = False
    else:
        X = np.concatenate((X, tempX[day_ago:]), axis=0)

# 何日後を値段の差を予測するのか
pre_day = 1
# y : pre_day後の終値/当日終値
y = np.zeros(len(X))
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]

up_rate =1.03
# y2：翌日の終値/当日の終値がup_rateより上か
y2 = np.zeros(len(y))
for i in range(0, len(y2)):
    if y[i] <= up_rate:
        y2[i] = 0
    else:
        y2[i] = 1

# データを一旦分別
X_0 = X[y2==0]
X_1 = X[y2==1]
y_0 = y2[y2==0]
y_1 = y2[y2==1]

# X_0をX_1とほぼ同じ数にする
X_dummy, X_t, y_dummy, y_t = train_test_split(X_0, y_0, test_size=0.09, random_state=0)

# 分別したデータの結合
X_ = np.concatenate((X_1, X_t), axis=0)
y_ = np.concatenate((y_1, y_t))


# In[ ]:


print("start time: ", datetime.now().strftime("%H:%M:%S"))

X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(max_iter=200000, alpha=0.001))])
param_grid = {'classifier__hidden_layer_sizes': [(10,), (100,), (500,)]}

grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, cv=2 ,return_train_score=False, scoring="accuracy")
grid.fit(X_train, y_train)

print(grid.cv_results_['mean_test_score'])
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("over time: ", datetime.now().strftime("%H:%M:%S"))

conf = confusion_matrix(y_test, grid.predict(X_test))
print(conf)
