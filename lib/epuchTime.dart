import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';

import "package:intl/intl.dart";
import 'package:intl/date_symbol_data_local.dart';

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

  Map map = {
   "sample_date": 2020-11-08 08:33:27 Etc/GMT,
   "sample_date_ms": 1604824407000,
   "sample_date_microseconds": 1604824407000000,
   "firebase_timestamp": 2020/10/11 5:22:03 PM UTC+9
  };

  int millisecondsSinceEpoch = 1640901600000;


  void time() {
    final sample =
        DateTime.fromMicrosecondsSinceEpoch(1640901600000000, isUtc: true);

    final mini =
        DateTime.fromMillisecondsSinceEpoch(1640901600000, isUtc: true);

    //MillisecondsからDateTimeへの変換
    DateTime.fromMillisecondsSinceEpoch(map['sample_date_ms']);

    //MicrosecondsからDateTimeへの変換
    DateTime.fromMicrosecondsSinceEpoch(map['sample_date_microseconds']);

    //Firebase TimestampからDateTimeへの変換
    map['firebase_timestamp'].toDate();

    external DateTime.fromMillisecondsSinceEpoch(millisecondsSinceEpoch,isUtc: false);

    //unixtime
int unixtime = DateTime.now().toUtc().millisecondsSinceEpoch;
print("unixtime=$unixtime"); // unixtime=1557085819211
DateTime dd = new DateTime.fromMillisecondsSinceEpoch(unixtime);
print("date=$dd"); // date=2019-05-06 04:56:30.237
print("date:${dd.toString().substring(0,16)}"); // date=2019-05-06 04:56


    print(sample); // 2021-12-31 19:30:00.000Z
    print(mini);
  }




  void _incrementCounter() {
    setState(() {
      _counter++;
      time();
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
