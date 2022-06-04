'''2. LSTMを組む'''
#2-1. ネットワークをつくる
#LSTMもRNNとほぼ全く同じように組むことができます:

import torch
import torch.nn as nn

# 入力2次元, 隠れ状態3次元, 1層のLSTM
dim_i, dim_h, num_l = 2, 3, 1
model = nn.LSTM(input_size=dim_i, hidden_size=dim_h, num_layers=num_l)
#双方向LSTM, 多層LSTMもRNNのときと同様です:

# 入力5次元, 隠れ状態8次元, 1層の双方向LSTM
model2 = nn.LSTM(input_size=5, hidden_size=8, num_layers=1, bidirectional=True)

# 入力3次元, 隠れ状態6次元, 2層のLSTM
model3 = nn.LSTM(3, 6, 2)


'''2-2. LSTMコンストラクタの引数'''
#torch.nn.LSTMのコンストラクタに入れることのできる引数は以下のとおりです。
#RNNのコンストラクタとほぼ変わりありません。
#RNNとの違いは活性化関数を指定する項目がない点くらいでしょう。

#model = torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

    #input_size: int -> 入力ベクトルの次元数
    #idden_size: int -> 隠れ状態の次元数
    #*num_layers: int -> LSTMの層数。多層にしたいときは2以上に
    #*bias: bool -> バイアスを使うかどうか
    #*batch_first: bool
    #*dropout: float -> 途中の隠れ状態にDropoutを適用する確率
    #*bidirectional: bool -> 双方向LSTMにするかどうか

'''2-3. LSTMのパラメーターの確認'''
#LSTMのパラメーターもまた，ネットワークを組んだ時点で適当な値で初期化されています。
#現在のパラメーターの様子をみてみましょう:

for param_name, param in model.named_parameters():
    print(param_name, param)
#次のような出力が得られるはずです:
'''
weight_ih_l0 Parameter containing:
tensor([[ 0.2848, -0.1366],
        [-0.5757,  0.2086],
        [ 0.4995,  0.4271],
        [-0.1745, -0.3294],
        [ 0.4708, -0.0210],
        [ 0.4829, -0.4076],
        [ 0.4412,  0.3948],
        [ 0.4969, -0.0128],
        [ 0.4600,  0.4799],
        [ 0.3268,  0.2755],
        [ 0.2120,  0.0517],
        [ 0.1208, -0.1436]], requires_grad=True)
weight_hh_l0 Parameter containing:
tensor([[-0.0824,  0.3834, -0.0103],
        [ 0.5396,  0.3769,  0.1899],
        [-0.4365, -0.5241, -0.2395],
        [ 0.4210, -0.5123,  0.1195],
        [-0.3324,  0.2434,  0.3067],
        [-0.2196,  0.3060, -0.3943],
        [ 0.1774, -0.2787,  0.0273],
        [-0.2064, -0.4244, -0.0538],
        [ 0.1785,  0.0495,  0.4612],
        [ 0.1111,  0.4128,  0.5325],
        [ 0.0116, -0.2142,  0.3397],
        [ 0.2183, -0.2899,  0.1467]], requires_grad=True)
bias_ih_l0 Parameter containing:
tensor([ 0.2030, -0.3873,  0.5769, -0.3200,  0.0116, -0.0453, -0.5763, -0.0194,
        -0.1736, -0.0692,  0.2100, -0.0362], requires_grad=True)
bias_hh_l0 Parameter containing:
tensor([ 0.1686, -0.3883, -0.3789, -0.3639,  0.1766,  0.0311, -0.4657,  0.3933,
        -0.0357,  0.2844,  0.3898,  0.3525], requires_grad=True)
これらの出力は次のようなことを意味します。
LSTMの隠れ層で行われているのは次のような演算でした:

it=σ((Wiixt+bii)+(Whiht−1+bhi))ft=σ((Wifxt+bif)+(Whfht−1+bhf))gt=tanh((Wigxt+big)+(Whght−1+bhg))ct=ft⋅ct−1+it⋅gtot=σ((Wioxt+bio)+(Whoht−1+bho))ht=ot⋅tanh(ct)
it=σ((Wiixt+bii)+(Whiht−1+bhi))ft=σ((Wifxt+bif)+(Whfht−1+bhf))gt=tanh((Wigxt+big)+(Whght−1+bhg))ct=ft⋅ct−1+it⋅gtot=σ((Wioxt+bio)+(Whoht−1+bho))ht=ot⋅tanh(ct)
LSTMのパラメーターとして扱われるのはWiiWii,WifWif,WigWig,WioWio,WhiWhi,WhfWhf,WhgWhg,WhoWho,biibii,bifbif,bigbig,biobio,bhibhi,bhfbhf,bhgbhg,bhobhoの16の行列もしくはベクトルです。

上の出力例では，
weight_ih_l0 ParameterにWiiWii,WifWif,WigWig,WioWioがまとめて格納されて出力され，
weight_hh_l0 ParameterにWhiWhi,WhfWhf,WhgWhg,WhoWhoがまとめて格納されて出力され，
bias_ih_l0 Parameterにbiibii,bifbif,bigbig,biobioがまとめて格納されて出力され，
bias_hh_l0 Parameterにbhibhi,bhfbhf,bhgbhg,bhobhoがまとめて格納されて出力されています。
'''

'''2-4. LSTMに推論させる'''
#このLSTMに何かを入力して，何らかの出力を得てみましょう。
#もちろんこのLSTMは初期化された状態のままであり，一切の学習を行なっていないため，でたらめな値を吐き出します。

n_sample = 2
seq_length = 5

# 入力
X = torch.randn(seq_length, n_sample, dim_i)
# 初期の隠れ状態
h0 = torch.randn(num_l, n_sample, dim_h)
# 初期のメモリセル
c0 = torch.randn(num_l, n_sample, dim_h)

# Y:出力, hn:最終の隠れ状態, cn:最終のメモリセル
# (h0,c0)を省略するとh0,c0には零ベクトルが代入される
Y, (hn,cn) = model(X, (h0,c0))
print(Y, hn, cn, sep='\n')

'''
次のような出力が得られるはずです:

tensor([[[-0.0608,  0.0157, -0.3091],
         [-0.1908,  0.1270, -0.0131]],

        [[-0.0604,  0.1197, -0.2682],
         [-0.1019,  0.1923, -0.1177]],

        [[-0.0411, -0.0321, -0.2204],
         [-0.1566,  0.3992,  0.1179]],

        [[-0.0693,  0.0297, -0.1263],
         [-0.0999,  0.4723,  0.2208]],

        [[-0.0499,  0.2873,  0.0223],
         [-0.1095,  0.2102,  0.2421]]], grad_fn=<StackBackward>)
tensor([[[-0.0499,  0.2873,  0.0223],
         [-0.1095,  0.2102,  0.2421]]], grad_fn=<StackBackward>)
tensor([[[-0.0972,  0.4610,  0.0448],
         [-0.2396,  0.3879,  0.4673]]], grad_fn=<StackBackward>)
'''