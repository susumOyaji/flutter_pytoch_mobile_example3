//flutter_pytorch_mobile

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:http/http.dart' as http;
import "package:intl/intl.dart";
import 'package:intl/date_symbol_data_local.dart';
import 'epuchTime.dart';

void main() => runApp(const MyAppTime());

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  _MyAppState createState() => _MyAppState();
}

//unixtimeから見やすいフォーマットに変換する
String convertTime(int t) {
  //int型からDateTime型に変換
  DateTime date = DateTime.fromMillisecondsSinceEpoch(t);
  String time = DateFormat("yyyy年MM月dd日").format(date).toString();
  print(time);
  return time; //例 2020年08月14日
}

epochTime() {
  DateTime now = DateTime.now();
  var epochTime = (now.millisecondsSinceEpoch) * 1000;
  print(epochTime);
  return epochTime;
}

void myconvertTime(int t) {
  var dateUtc = DateTime.fromMillisecondsSinceEpoch(t, isUtc: true);
  var dateInMyTimezone = dateUtc.add(Duration(hours: 9));
  var secondsOfDay = dateInMyTimezone.hour * 3600 +
      dateInMyTimezone.minute * 60 +
      dateInMyTimezone.second;
  print(dateInMyTimezone);
}

Future getData() async {
  //final parameters = {
  //  'api_key': 'xxxxxxxxxx',
  //  'start_date': '2021-01-01',
  //  'end_date': '2021-03-31',
  //};
  //final url = Uri.https('www.quandl.com', '/api/v3/datasets/CHRIS/CME_NK2/data.json', parameters);
  //final url = Uri.https('https://finance.yahoo.com/quote/6758.T/history?period1=1639785600&period2=1655510400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true', parameters);
  //final url = Uri.https('https://query1.finance.yahoo.com/v7/finance/download/6758.T?period1=1623974400&period2=1655510400&interval=1d&events=history&includeAdjustedClose=true')
  //https://query1.finance.yahoo.com/v7/finance/download/6758.T?period1=1623974400&period2=1655510400&interval=1d&events=history&includeAdjustedClose=true

  final response = await http.get(Uri.parse(
      'https://query1.finance.yahoo.com/v7/finance/download/6758.T?period1=1609460285&period2=1640996285&interval=1d&events=history&includeAdjustedClose=true')); //^DJI

  String data = response.body;
  print(data);

  //final result = await http.get(url);
  //setState(() {
  //  response = json.decode(result.body)['dataset_data']['data'];
  //});
}

class _MyAppState extends State<MyApp> {
  Model? _imageModel, _customModel;

  String? _imagePrediction;
  List? _prediction;
  File? _image;
  ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    loadModel().then((value) {
      setState(() {});
    });
    getData();
  }

  @override
  void dispose() {
    //dis function disposes and clears our memory
    super.dispose();
    //PyTorchMobile.close();
  }

  //load your model
  Future loadModel() async {
    //String pathImageModel = "assets/models/resnet.pt";
    String pathCustomModel = "assets/models/custom_model.pt";
    try {
      //_imageModel = await PyTorchMobile.loadModel(pathImageModel);
      _customModel = await PyTorchMobile.loadModel(pathCustomModel);
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  //run an image model
  Future runImageModel() async {
    //pick a random image
    final PickedFile? image = await _picker.getImage(
        source: (Platform.isIOS ? ImageSource.gallery : ImageSource.camera),
        maxHeight: 224,
        maxWidth: 224);
    //get prediction
    //labels are 1000 random english words for show purposes
    _imagePrediction = await _imageModel!.getImagePrediction(
        File(image!.path), 224, 224, "assets/labels/labels.csv");

    setState(() {
      _image = File(image.path);
    });
  }

  //run a custom model with number inputs
  //カスタム予測を取得する
  Future runCustomModel() async {
    _prediction = await _customModel!
        .getPrediction([1, 2, 3, 4], [1, 2, 2], DType.float32);

    setState(() {});
    //epochTime();
    //myconvertTime(1609460285);
    //convertTime(1609460285);
  }

  /*
  ///predicts abstract number input
  Future<List?> getPrediction(
      List<double> input, List<int> shape, DType dtype) async {
    final List? prediction = await _channel.invokeListMethod('predict', {
      "index": _index,
      "data": input,
      "shape": shape,
      "dtype": dtype.toString().split(".").last
    });
    return prediction;
  }
  */

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Pytorch Mobile Example'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            TextButton(
              onPressed: runCustomModel,
              style: TextButton.styleFrom(
                backgroundColor: Colors.blue,
              ),
              child: Text(
                "Run custom model",
                style: TextStyle(
                  color: Colors.white,
                ),
              ),
            ),
            Center(
              child: Visibility(
                visible: _prediction != null,
                child: Text(_prediction != null ? "${_prediction![0]}" : ""),
              ),
            )
          ],
        ),
      ),
    );
  }
}
