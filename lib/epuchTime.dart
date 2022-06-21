import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';

import "package:intl/intl.dart";
import 'package:intl/date_symbol_data_local.dart';
import 'package:http/http.dart' as http;

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

  String _Time_to_unixcode(ref) {
    //unixtime
    //int unixtime = DateTime.now().toUtc().millisecondsSinceEpoch;
    int unixtime = ref.toUtc().millisecondsSinceEpoch;
    unixtime = unixtime ~/ 1000;

    print("unixtime=$unixtime"); // unixtime=1557085819211

    DateTime dd = DateTime.fromMillisecondsSinceEpoch(unixtime);
    //print("date=$dd"); // date=2019-05-06 04:56:30.237
    print("date:${dd.toString().substring(0, 10)}"); // date=2019-05-06 04:56

    String code = unixtime.toString();
    return code;
  }

  Future getData(String code) async {
    //final response = await http.get(Uri.parse(
    //  'https://query1.finance.yahoo.com/v7/finance/download/6758.T?period1=1609460285&period2=1640996285&interval=1d&events=history&includeAdjustedClose=true')); //^DJI

    final response = await http.get(Uri.parse(
        'https://query1.finance.yahoo.com/v7/finance/download/${code}.T?period1=${predicted_start}&period2=${predicted_end}&interval=1d&events=history&includeAdjustedClose=true')); //^DJI

    String data = response.body;
    debugPrint(data);

    //final result = await http.get(url);
    //setState(() {
    //  response = json.decode(result.body)['dataset_data']['data'];
    //});
  }

  void _incrementCounter() {
    setState(() {
      _counter++;
      predicted_start = _Time_to_unixcode(start);
      predicted_end = _Time_to_unixcode(end);
      getData('6976');
    });
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
