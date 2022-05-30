import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#scikit-learnより癌のデータを抽出する
data = load_breast_cancer()
df_X = pd.DataFrame(data=data.data,columns=data.feature_names)
df_y = pd.DataFrame(data=data.target,columns=['cancer'])
df = pd.concat([df_X,df_y],axis=1)

#いくつかの変数で変数同士の関係性を見てみる。
sns.pairplot(df.iloc[:,25:], hue='cancer')
plt.show()




#説明変数Xと予測したい変数Yを準備する（yは0,1のカテゴリ変数に変換）
X = df.drop('cancer',axis=1)
y = df.loc[:,'cancer']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.5, random_state = 0) 

#説明変数は標準化しておく(あとで回帰係数を比較するため)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

#モデルの構築と学習
lr = LogisticRegression() 
lr.fit(X_train, y_train)

#訓練データ,テストデータに対する予測
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
#最初の１０サンプルだけ表示してみる
print(y_pred_train[:10])
print(y_pred_test[:10])
"""output
[1 0 1 0 1 1 0 1 0 0]
[0 1 1 1 1 1 1 1 1 1]
"""


#訓練データ
print('accuracy：', accuracy_score(y_true=y_train, y_pred=y_pred_train))
print('precision：', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall：', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1 score：', f1_score(y_true=y_train, y_pred=y_pred_train))
print('confusion matrix = \n', confusion_matrix(y_true=y_train, y_pred=y_pred_train))
"""output
accuracy =  0.9894366197183099
precision =  0.9829545454545454
recall =  1.0
f1 score =  0.9914040114613181
confusion matrix = 
 [[108   3]
 [  0 173]]
"""

#テストデータ
print('accuracy：', accuracy_score(y_true=y_test, y_pred=y_pred_test))
print('precision：', precision_score(y_true=y_test, y_pred=y_pred_test))
print('recall：', recall_score(y_true=y_test, y_pred=y_pred_test))
print('f1 score：', f1_score(y_true=y_test, y_pred=y_pred_test))
print('confusion matrix = \n', confusion_matrix(y_true=y_test, y_pred=y_pred_test))
"""output
accuracy =  0.9754385964912281
precision =  0.9783783783783784
recall =  0.9836956521739131
f1 score =  0.981029810298103
confusion matrix = 
 [[ 97   4]
 [  3 181]]
"""

#ROC曲線の描画、AUCの計算（ROC曲線の下側の面積）の計算
y_score = lr.predict_proba(X_test)[:, 1] # 検証データがクラス1に属する確率
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], linestyle='--', label='random')
plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label='ideal')
plt.legend()
plt.title('ROC curve of test sample',fontsize=15)
plt.xlabel('false positive rate',fontsize=15)
plt.ylabel('true positive rate',fontsize=15)
plt.show()



#切片（intercept）の表示
print("intercept（切片）：", round(lr.intercept_[0],3))
"""output
intercept（切片）： -0.220
"""

#回帰係数を格納したpandasDataFrameの表示
df_coef =  pd.DataFrame({'coefficient':lr.coef_.flatten()}, index=X.columns)
df_coef['coef_abs'] = abs(df_coef['coefficient'])
df_coef.sort_values(by='coef_abs', ascending=True,inplace=True)
df_coef = df_coef.iloc[-10:,:]

#グラフの作成
x_pos = np.arange(len(df_coef))

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.barh(x_pos, df_coef['coefficient'], color='b')
ax1.set_title('coefficient of variables',fontsize=18)
ax1.set_yticks(x_pos)
ax1.set_yticks(np.arange(-1,len(df_coef.index))+0.5, minor=True)
ax1.set_yticklabels(df_coef.index, fontsize=14)
ax1.set_xticks(np.arange(-10,11,2)/10)
ax1.set_xticklabels(np.arange(-10,11,2)/10,fontsize=12)
ax1.grid(which='minor',axis='y',color='black',linestyle='-', linewidth=1)
ax1.grid(which='major',axis='x',linestyle='--', linewidth=1)
plt.show()