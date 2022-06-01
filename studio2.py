import torch



'''MyRNNクラスを使っての学習と評価'''
#次に、今定義したMyRNNクラスとNetクラスを使って、学習と評価を実行してみましょう。
# 学習するコードは関数として、この後にも使うことにします。
EPOCHS = 100
num_batch = 25
input_size = 1 #一度に1個のデータを入力していくので、input_sizeは1です。
hidden_size = 32 #（このサイズに合わせて、この後の隠れ状態の初期化を行います）。

def train(epocs, net, X_train, y_train, criterion, optimizer):
    losses = []

    for epoch in range(EPOCHS):
        print('epoch:', epoch)
        optimizer.zero_grad()
        hidden = torch.zeros(num_batch, hidden_size)
        output, hidden = net(X_train, hidden)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss.item() / len(X_train):.6f}')
        losses.append(loss.item() / len(X_train))

    return output, losses

#学習を行うコード