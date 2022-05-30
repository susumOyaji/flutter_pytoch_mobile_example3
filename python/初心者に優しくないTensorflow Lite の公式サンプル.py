'''初心者に優しくないTensorflow Lite の公式サンプル'''
#Python

#TensorFlow

#TensorflowLite
概要
とあるプロジェクトで、TensorFlow Lite の利用検討をすることになりました。
が、そもそもTensorflowにあまり詳しくなく、公式サンプルでも結構詰まってしまいました。
備忘録として、Tensorflow Liteの公式サンプルの簡易な解説と実際のモデルでの検証結果を書いておきます。

環境
PC: Linux Mint 19 Tara
Python: 2.7.15
tensorflow: 1.10.0
About Tensorflow Lite
組み込み・モバイル向けのTensorflowのモデルの軽量版。
Tensorflowで作成した学習モデルをConvertして使用できる。そのため学習モデルの再利用が可能。
TFLiteを利用するためには、ざっくり下記の２ステップが必要になる。
Step １．既存のTFmodelをTFlite用にConvert
　　　　　今回は、Keras modelなのでConvertも簡単。
　　　　　もしckpt形式で保存している場合は、freeze_graph.pyでモデルの変換が必要。freeze_graph.pyに関しては、ここの記事が詳しい
Step ２．TFLiteモデルをInterpretして推論実行

詳しくは、公式を参照。

Tensorflow Convert
公式のサンプルコード通りで実行できる。
コンバート用のAPIが用意されているので、従来のTFモデル保存時点でConvertしてあげればいい。

convert.py
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)




//Tensorflow Interpreter
//公式のサンプルコードに沿って、tfliteモデルの推論実行を解説する。

'''interpret.py'''
import numpy as np
import tensorflow as tf

# TFLiteモデルの読み込み
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
# メモリ確保。これはモデル読み込み直後に必須
interpreter.allocate_tensors()

# 学習モデルの入力層・出力層のプロパティをGet.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_details/output_detailsは、辞書型のリスト構造で、入力層・出力層のプロパティを示している。
この構成の解説が見当たらないけど、皆どうやって理解してるんだろ…

Input_detailsの実例.data
[{
'index': 7, ※ここに入力したいテンソルデータのポインタをセットする
'shape': array([ 1, 50, 50,  3], dtype=int32),  ※入力するテンソルデータの構成。このサンプルだと、50 X 50のRGB画像(3次元)が入力
'quantization': (0.0, 0L), ※量子化パラメータ。これはよくわかってない
'name': 'conv2d_1_input', ※入力層の名前
'dtype': <type 'numpy.float32'> ※入力データのtype
}]
※この実例は公式のサンプルコードのものではない。
output_deailsも構成は変わらない。ただ、index部のみ使い方が変わる。
Input_detailsのindexには入力データをセットするが、output_detailsのindexには推論実行時に出力データがセットされる。
つまり、推論実行後の出力はoutput_detailsのindexから取得すればいい。

ここまで読み込んだモデルの入力層・出力層のプロパティをゲットできたので、続いて入力層へのデータセットと実行、及び結果取得を行う。

interpret_続き.py
# 入力層のテンソルデータ構成の取得
input_shape = input_details[0]['shape']
# テンソルデータ構成から、ランダムな ndArrayを作成
# np.arrayのcall時に、input_detailsのdtypeと整合性が取れるように型をセットしないと、set_tensor時にエラーが発生
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# indexにテンソルデータのポインタをセット
interpreter.set_tensor(input_details[0]['index'], input_data)

# 推論実行
interpreter.invoke()
# 推論結果は、output_detailsのindexに保存されている
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
パフォーマンス計測
せっかくTensorflow Liteのモデル作ったので、ラズパイ(raspberry pi zero w)上で動かしました。

対象モデル
CNN
Keras model
50x50のRGB画像を学習させて作成したモデル

モデルサイズ
> ls
-rw-rw-r-- 1 yohac yohac  31M 11月 13 17:52 keras.h5 　★通常のモデル
-rw-rw-r-- 1 yohac yohac  16M 11月 14 20:41 keras.tflite ★変換後のモデル
モデルサイズ自体は約半減。ここのサイズの圧縮率は、モデル構成次第。今回はたまたま半減できた。

処理速度
モデルのロード・推論実行を含めた処理時間は下記になった。

実行時間
TFLite Model	30sec
TF Model	15 sec
処理速度は、モデルのロード/推論実行のそれぞれが約半分で済んだ。
ここはモデルサイズに依存していると考えられるので、妥当な結果だと思う。

ラズパイ自体ハード性能が限られるので、このモデル変換は有用かも。

精度
そもそものモデルの精度が芳しくないので、ここは割愛。
精度がガタ落ちさえしてなければ、TFLiteは実用的かも。

所感とか
Tensorflow自体かなり情報量が少ないが、実際使ってみるとその有用性はわかる。
機械学習なので精度検証もこれからしないとなー。

余談
RNNを使ったモデルもConvertしようとしたが、エラー発生。
GithubのIssueを見る限り、サポートチケットがOpenなままなので、まだ非サポなのかも
https://github.com/tensorflow/tensorflow/issues/15845

参考
https://blogs.yahoo.co.jp/verification_engineer/71450155.html


