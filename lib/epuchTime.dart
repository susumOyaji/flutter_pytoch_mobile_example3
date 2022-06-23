import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';

import "package:intl/intl.dart";
import 'package:intl/date_symbol_data_local.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:csv/csv.dart';
import 'package:file_picker/file_picker.dart';
//void main() {
//  runApp(const MyAppTime());
//}

class MyAppTime extends StatelessWidget {
  const MyAppTime({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;
  DateTime start = DateTime(2022, 1, 1);
  DateTime end = DateTime.now();
  String predicted_start = '';
  String predicted_end = '';
  String code = '6758';

  String _Time_to_unixcode(DateTime ref) {
    //unixtime
    //int unixtime = DateTime.now().toUtc().millisecondsSinceEpoch;
    int unixtime = ref.toUtc().millisecondsSinceEpoch;
    unixtime = unixtime ~/ 1000;

    debugPrint("unixtime=$unixtime"); // unixtime=1557085819211

    DateTime dd = DateTime.fromMillisecondsSinceEpoch(unixtime);
    debugPrint("date=$dd"); // date=2019-05-06 04:56:30.237
    debugPrint(
        "date:${dd.toString().substring(0, 10)}"); // date=2019-05-06 04:56

    String unixcode = unixtime.toString();
    return unixcode;
  }

  Future getData(String code) async {
    predicted_start = _Time_to_unixcode(start);
    predicted_end = _Time_to_unixcode(end);

    //final response = await http.get(Uri.parse(
    //  'https://query1.finance.yahoo.com/v7/finance/download/6758.T?period1=1609460285&period2=1640996285&interval=1d&events=history&includeAdjustedClose=true')); //^DJI

    final response = await http.get(Uri.parse(
        'https://query1.finance.yahoo.com/v7/finance/download/${code}.T?period1=${predicted_start}&period2=${predicted_end}&interval=1d&events=history&includeAdjustedClose=true')); //^DJI

    String data = response.body;
    debugPrint(data);

    List<List<dynamic>> rowsAsListOfValues =
        const CsvToListConverter().convert(data);
    debugPrint(rowsAsListOfValues.toString());

    String csv = const ListToCsvConverter().convert(rowsAsListOfValues);

    //final result = await http.get(url);
    //String response_j;
    //setState(() {
    //  Map<String, dynamic> jsonData = json.decode(response
    //      .body); //['Date','Open','High,'Low','Close','Adj Close',Volume'];
    //});
  }

/*
''学習データ(train_data)を正規化'''
#特徴量を時系列にグラフ表示に示した通り、割とデータには開きがあります。なので、最小値と最大値を決めた正規化を学習データに行います。

# データセットの正規化を行う。最小値0と最大値1の範囲で行う。
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train_data)


'''Tensor型に変換'''
#PyTorchを使うのでTensor型にします。
train_data_normalized = torch.FloatTensor(train_data_normalized)


'''予測のためのデータ準備'''
#学習によってパラメータが決定したので、このモデルを使って予測してみます。
# そのために学習データとテストデータに分割するにて分割したtest_dataをデータを正規化で作成したscalerを用いて変換します。
# train_dataの正規化で取得した統計情報をtest_data正規化時にも使って変換します。(scaler.fit_transform)
#また、pred_daysは予測する日数です。本記事では、30日分のtest_dataとこれから予測する30日分を比較してみます。
#さらに、test_inputsというリストを用意します。
# test_inputsには学習データを正規化で正規化した学習データ(train_data_normalized)の直近seq_length(=30)日分を入れます。
# これは予測時に使います。

# 予測する日数
pred_days = 30

# テストデータの正規化を行う。最小値0と最大値1の範囲で行ってTensor型に変換する。
test_data_normalized = scaler.fit_transform(test_data)
test_data_normalized = torch.FloatTensor(test_data_normalized)

# 予測するためのデータの最初のseq_length分はtrain_dataを使う
test_inputs = train_data_normalized[-seq_length:].tolist()



*/

  void _incrementCounter() {
    setState(() {
      _counter++;

      getData('6976');
    });
  }

  @override
  void initState() {
    super.initState();

    getData(code);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headline4,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
