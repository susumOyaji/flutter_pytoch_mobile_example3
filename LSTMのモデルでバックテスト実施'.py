'''LSTMのモデルでバックテスト実施'''
#LSTMのモデルを使ってバックテストを実施してみます。
#準備として必要なパッケージのインポート、csvデータの読み込み等を行います。
#pip install backtesting
#from backtesting import Backtest
#from backtesting import Strategy
#from backtesting.lib import crossover

from backtesting import Backtest, Strategy 
from backtesting.lib import crossover
from backtesting.test import SMA
df = pd.read_csv('USD_JPY_201601-201908_M10.csv', index_col='Datetime')
df.index = pd.to_datetime(df.index)


'''続いて、バックテストで利用するためのLSTMのStrategyクラスを定義します。'''
#1. initではLSTMの学習済みモデルを読み込んでおきます。
#2. PyTorchのLSTMに投入するためにデータを整えます。
#3. LSTMのPyTorchのモデルで予測します。
#4. 予測結果が1であれば買い、0であれば売りの指示を出します。今回はst(stop loss）、tp（take profit）も指定してみました。

class myLSTMStrategy(Strategy):
    def init(self):
        # 1. LSTMの学習済みモデルの読み込み
        self.model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
        # load model
        self.model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))

    def next(self): 
        # 過去500ステップ分のデータが貯まるまではスキップ
        # 1日に1回のみ取引するため、hour & minuteが0の時のみ処理するようにする。
        if len(self.data) >= moving_average_num + time_steps and len(self.data) % future_num == 0:
            # 2. 推測用データの用意
            x_array = self.prepare_data()
            x_tensor = torch.tensor(x_array, dtype=torch.float, device=device)
            # 3. 予測の実行
            with torch.no_grad():
                y_pred = self.predict(x_tensor.view(1, time_steps, feature_num))

            # 4. 予測が買い(1)であればbuy()、それ以外はsell()
            if y_pred == 1:
                self.buy(sl=self.data.Close[-1]*0.99, 
                         tp=self.data.Close[-1]*1.01)
            else:
                self.sell(sl=self.data.Close[-1]*1.01, 
                         tp=self.data.Close[-1]*0.99)

    def prepare_data(self):
        # いったんPandasのデータフレームに変換
        tmp_df = pd.concat([
                    self.data.Volume.to_series(), 
                    self.data.Open.to_series(), 
                    self.data.High.to_series(), 
                    self.data.Low.to_series(), 
                    self.data.Close.to_series(), 
                    ], axis=1)

        # 500足の移動平均に対する割合とする。
        cols = tmp_df.columns
        for col in cols:
            tmp_df['Roll_' + col] = tmp_df[col].rolling(window=moving_average_num, min_periods=moving_average_num).mean()
            tmp_df[col] = tmp_df[col] / tmp_df['Roll_' + col] - 1

        #最後のtime_steps分のみの値を返す
        return tmp_df.tail(time_steps)[cols].values

    def predict(self, x_array):
        y_score = self.model(x_array) 
        return np.round(y_score.view(-1).to('cpu').numpy())[0]



'''バックテストを実行します。'''

bt = Backtest(df[100000:], myLSTMStrategy, cash=100000, commission=.00004)
bt.run()

'''
実行結果はどうだったでしょうか。

Start                     2018-09-06 13:10:00
End                       2019-08-01 03:50:00
Duration                    328 days 14:40:00
Exposure [%]                          97.6289
Equity Final [$]                       103012
Equity Peak [$]                        104203
Return [%]                            3.01237
Buy & Hold Return [%]                 1.76702
Max. Drawdown [%]                    -3.79017
Avg. Drawdown [%]                   -0.179441
Max. Drawdown Duration       77 days 13:50:00
Avg. Drawdown Duration        1 days 13:34:00
# Trades                                  228
Win Rate [%]                          50.8772
Best Trade [%]                       0.993839
Worst Trade [%]                      -1.00702
Avg. Trade [%]                        0.01088
Max. Trade Duration           3 days 01:00:00
Avg. Trade Duration           1 days 09:47:00
Expectancy [%]                        0.25287
SQN                                    0.4725
Sharpe Ratio                        0.0330574
Sortino Ratio                       0.0501238
Calmar Ratio                       0.00287057
_strategy                      myLSTMStrategy
dtype: object
228回の取引でReturnが3.01%、10万円が10万3,012円となっていました。
'''