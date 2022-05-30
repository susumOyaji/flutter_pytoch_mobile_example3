#!/usr/bin/env python
# coding: utf-8

# In[7]:


# 上場企業４２３社の２０年分を学習
# 終値前日比1.03%以上であるかを２クラス分類
# データ比１１：１を１：１に調整（減らした）
# StandardScaler accuracy
import numpy as np
import pandas as pd
from sklearn import *
import seaborn as sns
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
import mglearn
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import datetime
import pandas_datareader.data as web
#from datetime import datetime
import os
import glob
import locale
import csv
from csv import reader

# 実行上問題ない注意は非表示にする
warnings.filterwarnings('ignore') 


start = datetime.date(2020, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY


import datetime
#2021年から今日までの1年間のデータを取得しましょう。期日を決めて行きます。
start_train = datetime.date(2017, 1, 1)#教師データ(今までのデータ)
#end_train = datetime.date(2021,12,31)
end_train= datetime.date.today()# + relativedelta(days=-1)#昨日分(today-1日)まで取得できる（当日分は変動しているため）

from pandas_datareader import data as pdr

#try:
'''使うデータを読み込む。'''
#closed = pdr.get_data_yahoo(f'{code}.T', start, end)["Close"]  # 株価データの取得
df = pdr.get_data_yahoo(f'{code}.T', start_train, end_train)#["Adj Close"]  # 教師データのcsvファイルを読み込む。
#for index, row in df_data_j.iterrows():
print(df)
#except:
#    print("error")
#print("web.DataReader", df)
#保存
df.to_csv("./data/stocks_price_data/stock_price_data_{}.csv".format(code))


# In[8]:


# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
#files = glob.glob("data/kabu1/*.csv")


# In[9]:


# 説明変数となる行列X, 被説明変数となるy2を作成
base = 100 
day_ago = 3
num_sihyou = 8
reset =True
X = []


span01 =5 #5日間移動平均
span02 = 25 #25日平均
span03 = 75 #75日平均

#encoding = locale.getpreferredencoding()

#df = pd.read_csv("data/stocks_price_data/stock_price_data_{}.csv".format(code), header=0, encoding='cp932')
adj_price = df['Adj Close']
#単純移動平均の略称sma
df['sma01'] = adj_price.rolling(window=span01).mean()
df['sma02'] = adj_price.rolling(window=span02).mean()
df['sma03'] = adj_price.rolling(window=span03).mean()
df = df.fillna(0)#（NaN）をゼロで置換

df=df.reset_index()


#temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close', 'sma01', 'sma02', 'sma03', 'Volume', 'Adj Close']]




df.to_csv('data/stocks_price_data/add_column{}.csv'.format(code),index=False)

# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("data/stocks_price_data/add_column{}.csv".format(code))
print(df)






# すべてのCSVファイルから得微量作成
for file in files:
    # csv をﾘｽﾄに読み込む
    temp = pd.read_csv(file, header=0, encoding='cp932')
    #temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp = temp[['Date', 'Open', 'High', 'Low','Close', 'sma01', 'sma02', 'sma03', 'Volume']]
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
X = X[:-pre_day]
y = y[:-pre_day]

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



#In[9]

pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(max_iter=200000, alpha=0.001))])

param_grid = {'classifier__hidden_layer_sizes': [(10,), (100,), (500,)]}

grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, cv=2 ,return_train_score=False, scoring="accuracy")
grid.fit(X_train, y_train)


print(grid.cv_results_['mean_test_score'])
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
conf = confusion_matrix(y_test, grid.predict(X_test))
print(conf)

'''出力結果
[0.6632097  0.67099965]
Best parameters:  {'classifier__hidden_layer_sizes': (100,)}
grid best score,  0.6709996467679266
Test set score: 0.67
over time:  00:17:02
[[23934 11997]
 [11216 23628]]
'''





# In[10]:

'''隠れ層、alphaを調整'''
param_grid = {'classifier__alpha':[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 ], 
              'classifier__hidden_layer_sizes': [(60, ),(70, ), (80, ),(90, ), (100, ), (110, ), (120, ), (130, ), (140, ), (150, ), (160, ), (170, )]}


print(grid.cv_results_['mean_test_score'])
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test2)))

#ヒートマップで確認
xa = 'classifier__hidden_layer_sizes'
xx = param_grid[xa]
ya = 'classifier__alpha'
yy = param_grid[ya]
plt.figure(figsize=(5,8))
scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(yy), -1)
mglearn.tools.heatmap(scores, xlabel=xa, xticklabels=xx, 
                      ylabel=ya, yticklabels=yy, cmap="viridis")
'''
start time:  09:04:52
[0.66983634 0.66970917 0.66917697 0.6675521  0.66820676 0.6684187
 0.66648299 0.66292712 0.66562581 0.66385965 0.66300247 0.66399623
 0.66965736 0.66888496 0.66758978 0.66698222 0.66795243 0.66711409
 0.66445308 0.66298363 0.66240433 0.66574826 0.66748617 0.66630402
 0.67171553 0.67057577 0.66944543 0.6672601  0.66794772 0.66719416
 0.66722242 0.66848463 0.6693842  0.66848463 0.66216884 0.66649241
 0.6700624  0.66927587 0.66985988 0.67064641 0.6652349  0.67026021
 0.6675992  0.67008595 0.66846109 0.66872483 0.66701519 0.66769339
 0.66932768 0.67108913 0.67236077 0.67157424 0.67078771 0.67164488
 0.67100436 0.67317556 0.67175321 0.67185211 0.67232309 0.67072648
 0.67026492 0.67025079 0.67186153 0.67162604 0.67227128 0.67201696
 0.67221947 0.6719416  0.67177205 0.67346756 0.67292123 0.67198399]
Best parameters:  {'classifier__alpha': 0.1, 'classifier__hidden_layer_sizes': (150,)}
grid best score,  0.6734675615212528
Test set score: 0.67
[[24254 11677]
 [11667 23177]]
Test set precision score(再現率): 0.66
over time:  11:48:24
'''








#In[20]
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


# In[11]:


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







# In[14]:

#シュミレーションを行い、可視化を行う。
# シュミレーション（株価終値前日比＋３％が約２日に１回起こるので正確な結果ではない）
# 予測結果の合計を計算（空売り無し）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# tray_day日間
try_day = 10#50
y_pred = grid.predict(X_test)
for j in range(0, 5):
    a=random.randrange(len(y_test2)-try_day)
    X_test_try = X_test[a:a+try_day]
    y_test2_try = y_test2[a:a+try_day]
    y_test_try = y_test[a:a+try_day]
    y_pred_try = y_pred[a:a+try_day]
    
    c_ = 0
    win_c = 0
    money = 10000

    # 予測結果の総和グラフを描く
    total_return = np.zeros(len(y_test2_try))
    for i in range(0, try_day): 
        if y_pred_try[i] == 1:
            money = money*y_test_try[i]
            c_ +=1
            if y_test_try[i] >= 1:
                win_c +=1
            
        total_return[i] = money
    
    # 混同行列で確認
    conf = confusion_matrix(y_test2_try, grid.predict(X_test_try))
    
    # 上昇予測回数が０のときは、勝率、再現率を９９９にする
    if c_==0:
        win_score=999
    else:
        win_score = win_c / c_
        
    if conf[0,1]==0 and conf[1,1]==0:
        pre_score=999
    else:
        pre_score = conf[1,1]/(conf[0,1]+conf[1,1])

    print("投資結果：10000 円 → %1.3lf" %money, "円", "(買い回数：%1.3lf" %c_, "勝ち：%1.3lf" %win_c, "勝率：%1.3lf" %win_score, "「３％上昇」再現率：%1.3lf" %pre_score, "精度：%1.3lf" %grid.score(X_test_try, y_test2_try), ")") 

plt.figure(figsize=(15, 2))
plt.plot(total_return)

'''出力結果
投資結果：10000 円 → 14808.749 円 (買い回数：17.000 勝ち：12.000 勝率：0.706 「３％上昇」再現率：0.529 精度：0.540 )
投資結果：10000 円 → 31046.279 円 (買い回数：19.000 勝ち：17.000 勝率：0.895 「３％上昇」再現率：0.737 精度：0.580 )
投資結果：10000 円 → 30692.962 円 (買い回数：20.000 勝ち：17.000 勝率：0.850 「３％上昇」再現率：0.800 精度：0.780 )
投資結果：10000 円 → 18082.973 円 (買い回数：15.000 勝ち：14.000 勝率：0.933 「３％上昇」再現率：0.733 精度：0.600 )
投資結果：10000 円 → 15254.049 円 (買い回数：16.000 勝ち：14.000 勝率：0.875 「３％上昇」再現率：0.688 精度：0.580 )
'''


# In[17]:


# 学習データに無い企業でシュミレーション
start = datetime.date(2021,1,1)
end = datetime.date.today()
code = '6976'#TAIYO
Xt = []

#for index, row in df_data_j.iterrows():
try:
    df = web.DataReader(f'{code}.T', 'yahoo', start, end)
except:
    print("error")

#保存
df.to_csv("./data/stocks_price_data/stock_price_data_{}.csv".format(code))

# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("data/stocks_price_data/stock_price_data_{}.csv".format(code))

#5日平均
span01=5
#25日平均
span02=25
#75日平均
span03=75

encoding = locale.getpreferredencoding()
 
df = pd.read_csv("data/stocks_price_data/stock_price_data_{}.csv".format(code), header = 0, encoding = 'cp932')
price=df['Adj Close']
#単純移動平均の略称sma01
df['sma01'] = price.rolling(window=span01).mean()
df['sma02'] = price.rolling(window=span02).mean()
df['sma03'] = price.rolling(window=span03).mean()
df = df.fillna(0)#（NaN）をゼロで置換
df.to_csv('data/stocks_price_data/add_column{}.csv'.format(code), index = False, columns=['Date','Open', 'High','Low','Close','sma01','sma02','sma03','Volume','Adj Close'],encoding=encoding)






# In[2]:


# data/kabu1フォルダ内にあるcsvファイルの一覧を取得
filestest = glob.glob("data/stocks_price_data/add_column{}.csv".format(code))
#filestest = glob.glob("data/kabu2/*.csv") ##モデル作成時とは別のファイル

base = 100
day_ago = 3
num_sihyou = 8
reset =True
for file in filestest:
    temp = pd.read_csv(file, header=0, encoding='cp932')
    #temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp = temp[['Date','Open', 'High','Low','Close','sma01','sma02','sma03','Volume']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp3), 0] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4]
    temp3[0:len(temp3), 1] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 2] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 3] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4]
    temp3[0:len(temp3), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base:len(temp2), 4].astype(np.float)
    temp3[0:len(temp3), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float)
    
        
# 説明変数となる行列Xtを作成
tempX = np.zeros((len(temp3), day_ago*num_sihyou))


# 日にちごとに横向きに（day_ago）分並べる
# sckit-learnは過去の情報を学習できないので、複数日（day_ago）分を特微量に加える必要がある
# 注：tempX[0:day_ago]分は欠如データが生まれる
for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
    for i in range(0, day_ago):
        tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]



# Xに追加
# X : すべての企業のデータ
# tempX[0:day_ago]分は削除

if reset:
    Xt = tempX[day_ago:]
    reset = False
else:
    Xt= np.concatenate((Xt, tempX[day_ago:]), axis=0)
    
# 被説明変数となる Y = pre_day後の終値/当日終値 を作成
yt = np.zeros(len(Xt))
# 何日後を値段の差を予測するのか
pre_day = 1
yt[0:len(yt)-pre_day] = Xt[pre_day:len(Xt),0]
Xt = Xt[:-pre_day]
yt = yt[:-pre_day]
print(Xt.shape)

up_rate =1.03
yt2 = np.zeros(len(yt))
for i in range(0, len(yt2)):
    if yt[i] <= up_rate:
        yt2[i] = 0
    else:
        yt2[i] = 1

'''出力結果
(4188, 24)
'''


# In[18]:


# 予測結果の合計を計算（空売り無し。買いだけ）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# ランダムに日付を選び、tray_day日間運用
try_day = 100
print("全体精度: {:.2f}".format(grid.score(Xt, yt2)))
yt_pred = grid.predict(Xt)
for j in range(0, 5):
    a=random.randrange(len(yt2)-try_day)
    Xt_try = Xt[a:a+try_day]
    yt2_try = yt2[a:a+try_day]
    yt_try = yt[a:a+try_day]
    yt_pred_try = yt_pred[a:a+try_day]
    
    c_ = 0
    win_c = 0
    money = 10000

    # 予測結果の総和グラフを描くーーーーーーーーー
    total_return = np.zeros(len(yt_try))
    for i in range(0, try_day):
        if yt_pred_try[i] == 1:
            money = money*yt_try[i]
            c_ +=1    
            if yt_try[i] >= 1:
                win_c +=1
                   
            
        total_return[i] = money

    # 混同行列で確認
    conf = confusion_matrix(yt2_try, grid.predict(Xt_try))
    # 上昇予測回数が０のときは、勝率、再現率を９９９にする
    if c_==0:
        win_score=999
    else:
        win_score = win_c / c_
        
    if conf[0,1]==0 and conf[1,1]==0:
        pre_score=999
    else:
        pre_score = conf[1,1]/(conf[0,1]+conf[1,1])
    print("投資結果：10000 円 → %1.3lf" %money, "円", "(買い回数：%1.3lf" %c_, "勝ち：%1.3lf" %win_c, "勝率：%1.3lf" %win_score, "「３％上昇」再現率：%1.3lf" %pre_score, "精度：%1.3lf" %grid.score(Xt_try, yt2_try), ")") 

# 最後のシュミレーションだけ可視化
print(conf)
plt.figure(figsize=(15, 2))
plt.plot(total_return)

'''出力結果
全体精度: 0.61
投資結果：10000 円 → 9488.770 円 (買い回数：14.000 勝ち：6.000 勝率：0.429 「３％上昇」再現率：0.143 精度：0.830 )
投資結果：10000 円 → 10305.978 円 (買い回数：22.000 勝ち：11.000 勝率：0.500 「３％上昇」再現率：0.091 精度：0.770 )
投資結果：10000 円 → 12049.243 円 (買い回数：43.000 勝ち：26.000 勝率：0.605 「３％上昇」再現率：0.186 精度：0.580 )
投資結果：10000 円 → 11230.923 円 (買い回数：23.000 勝ち：14.000 勝率：0.609 「３％上昇」再現率：0.217 精度：0.780 )
投資結果：10000 円 → 10230.065 円 (買い回数：3.000 勝ち：2.000 勝率：0.667 「３％上昇」再現率：0.000 精度：0.940 )
[[94  3]
 [ 3  0]]
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



