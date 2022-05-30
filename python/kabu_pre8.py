from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import random

if __name__ == '__main__':

    # データセットを読み込む
    iris = load_iris()
    x = iris.data
    y = iris.target

    # 読み込んだデータセットをシャッフルする
    p = list(zip(x, y))
    random.shuffle(p)
    x, y = zip(*p)

    # 学習データの件数を指定する
    train_size = 100
    test_size = len(x) - train_size

    # データセットを学習データとテストデータに分割する
    train_x = x[:train_size]
    train_y = y[:train_size]
    test_x = x[train_size:]
    test_y = y[train_size:]

    # 決定木の学習を行う
    tree = DecisionTreeClassifier(criterion='gini', max_depth=None)
    tree.fit(train_x, train_y)

    # 学習させたモデルを使ってテストデータに対する予測を出力する
    count = 0
    pred = tree.predict(test_x)
    for i in range(test_size):
        print('[{0}] correct:{1}, predict:{2}'.format(i, test_y[i], pred[i]))
        if pred[i] == test_y[i]:
            count += 1

    # 予測結果から正答率を算出する
    score = float(count) / test_size
    print('{0} / {1} = {2}'.format(count, test_size, score))
'''
このプログラムを実行すると以下の出力結果が得られます。

[0] correct:0, predict:0
[1] correct:0, predict:0
[2] correct:2, predict:2
...省略...
[47] correct:0, predict:0
[48] correct:1, predict:1
[49] correct:0, predict:0
50 / 50 = 1.0
テスト用データ50件に対する予測は正答率が1.0となりました。非常に高い分類精度が得られたことがわかります。

'''