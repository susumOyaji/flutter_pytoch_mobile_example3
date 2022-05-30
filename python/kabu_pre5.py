import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
#matplotlib inline



dflist = glob.glob("data/stocks_price_data/add_column.csv")
data = pd.DataFrame()
for i in dflist:
    name = pd.read_csv(i,header=0, nrows=1, encoding='cp932')# 日本語を表記する
    stock_name = name.columns[0]
    df = pd.read_csv(i,header=0, encoding='cp932')
    #print(df)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    data = data.append(df)
    #日付、終値を取ります
    date = data.iloc[:,0]
    price = data.iloc[:,4]
data.head(15)
data.info()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,6))
plt.plot(date,price)
plt.title(stock_name,fontsize=20)
plt.xlabel('年月',fontsize=18)
plt.ylabel('株価',fontsize=18)