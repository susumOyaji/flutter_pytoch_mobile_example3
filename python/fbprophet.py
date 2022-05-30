# データフレームを扱うライブラリを読み込む
import pandas as pd

# fbprophetのライブラリを読み込む
from fbprophet import Prophet

# R2を計算するためのライブラリを読み込む
from sklearn.metrics import r2_score

# RMSEを計算するためのライブラリを読み込む
from sklearn.metrics import mean_squared_error

# numpyのライブラリを読み込む
import numpy as np

# プログレスバーを表示するためのライブラリを読み込む
from tqdm import tqdm

# グラフ表示関連のライブラリを読み込む
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
from matplotlib.dates import date2num

# 日付を扱うためのライブラリを読み込む
from datetime import datetime

# main
def main():
    # 予測に必要な全データを読み込む
    JPYUSD, DJI, IXIC, N225, shift, MA, df_base, merge_flag = load_data()

    # データの辞書を設定する
    dict_data = {'JPYUSD': JPYUSD,
                 'DJI': DJI,
                 'IXIC': IXIC,
                 'N225': N225,
                 'shift': shift,
                 'MA': MA,
                 'df_base': df_base}

    # 特徴量に応じた評価結果を保存するデータフレームを用意する
    # 自己回帰データを作成するためにshift関数でシフトした値を保存するカラムを追加する
    # r2、rmseの計算結果を保存するカラムを追加する
    df_result = merge_flag.copy()
    df_result['shift'] = 0
    df_result['r2'] = 0
    df_result['rmse'] = 0

    # 株価を予測して、結果を評価する
    for i in tqdm(range(len(merge_flag))):
        # merge_flagから特徴量に追加するデータを選択し、特徴量を作成する
        # 各データのフラグを用意し、追加する場合は1,追加しない場合は0を代入する
        JPYUSD_f = merge_flag.iloc[i, 0]
        DJI_f = merge_flag.iloc[i, 1]
        IXIC_f = merge_flag.iloc[i, 2]
        N225_f = merge_flag.iloc[i, 3]
        shift_f = merge_flag.iloc[i, 4]
        MA_f = merge_flag.iloc[i, 5]

        dict_flag = {'JPYUSD': JPYUSD_f,
                     'DJI': DJI_f,
                     'IXIC': IXIC_f,
                     'N225': N225_f,
                     'shift': shift_f,
                     'MA': MA_f}

        # r2とrmseの計算結果を保存するデータフレームを作成する
        df_result_score = pd.DataFrame(columns=['shift', 'r2', 'rmse'])

        # 特徴量を作成する
        df_feature_base = make_feature(dict_data, dict_flag)

        # シフトする値を設定する
        # 2020/11/11～2020/12/31までを予測するので、shift_startは36以上を設定する
        shift_start = 36
        shift_end = 100

        # シフトを＋1づつ行う
        for s in range(shift_start, shift_end+1):
            print('merge_flag:', i)
            print('shift:', s)
            # 特徴量の説明変数をシフトする
            df_feature = shift_data(df_feature_base, dict_data['df_base'], s)

            # 訓練データと検証データを作成する
            x_train, x_test, dates_test = make_trainData_testData(df_feature, s)

            # モデルを定義して学習する
            model = make_model(x_train, dict_flag)

            # 株価を予測する
            forecast = model_predict(df_feature, model)

            # r2,rmseの計算を行う
            r2, rmse, ypred, ytest = score(forecast, x_test, s)

            # R2とRMSEの計算結果を保存する
            df_result_score = df_result_score.append({'shift': s, 'r2': r2, 'rmse': rmse}, ignore_index=True)

        # r2の計算結果が最も高かったshift,rmseの計算結果を保存する
        r2_max = df_result_score['r2'].idxmax()
        df_result.iloc[i, -3] = df_result_score.iloc[r2_max, 0]
        df_result.iloc[i, -2] = df_result_score.iloc[r2_max, 1]
        df_result.iloc[i, -1] = df_result_score.iloc[r2_max, 2]

    # 結果をCSVに保存する
    df_result.to_csv(f'./result_logistic/shift_{shift_start}_{shift_end}_result.csv')

    # グラフを描画する
    # draw_graph(model, forecast, dates_test, r2, rmse, ypred, ytest)


# データを読み込む関数
def load_data():
    # 予測したい株価(toyota7203)を読み込む
    df_load = pd.read_csv('./stock_info/7203.csv', index_col=0, parse_dates=True)

    # 読み込んだ株価を日付で昇順に変換する
    df_load = df_load[::-1]

    # 予測対象の日付と終値を保存するdf_baseを新規に作成する
    df_base = pd.DataFrame()

    # df_baseにdf_loadから日付と終値を追加する
    df_base['Date'] = df_load.index.values
    df_base['Close'] = df_load['Close'].values

    # 円ドル為替の時系列データを読み込む
    JPYUSD = pd.read_csv('./stock_info/JPY_USD.csv', index_col=0, parse_dates=True)

    # ダウ平均(DJI)の時系列データを読み込む
    DJI = pd.read_csv('./stock_info/DJI.csv', index_col=0, parse_dates=True)

    # NASDAQ(IXIC)の時系列データを読み込む
    IXIC = pd.read_csv('./stock_info/IXIC.csv', index_col=0, parse_dates=True)

    # 日経平均(N225)の時系列データを読み込む
    N225 = pd.read_csv('./stock_info/N225.csv', index_col=0, parse_dates=True)

    # 自己回帰データ用のデータフレームを用意する
    shift = pd.DataFrame(index=df_load.index, columns=[])
    shift['Close'] = df_load.loc[:, 'Close']

    # 終値の移動平均を追加したデータフレームを用意する
    # 7回の取引の移動平均を算出する
    MA = pd.DataFrame(index=df_load.index, columns=[])
    MA['Close'] = df_load.loc[:, 'Close']
    window = 7
    min_periods = 1
    MA = MA.rolling(window=window, min_periods=min_periods).mean()

    # 特徴量に含めるかを判定するフラグを読み込む
    merge_flag = pd.read_csv('./stock_info/merge_flag.csv')

    return JPYUSD, DJI, IXIC, N225, shift, MA, df_base, merge_flag


# 特徴量を作成する関数
def make_feature(dict_data,  dict_flag):
    # 特徴量を作成する
    # dict_dataのdf_base、予測したい株価を読み込む
    df_make_feature = dict_data['df_base'].copy()

    # フラグに応じて、JPYUSD,DJI,IXIC,N225,Close_shift,Close_MAを説明変数として特徴量に追加する
    for i in range(len(dict_flag)):
        if list(dict_flag.values())[i] == 1:
            df_make_feature = pd.merge(df_make_feature,
                                       list(dict_data.values())[i]['Close'],
                                       how='left',
                                       on='Date',
                                       suffixes=['', '_'+list(dict_flag.keys())[i]])

    # 欠損しているセルを後ろの値で補完する
    df_make_feature = df_make_feature.fillna(method='bfill')

    return df_make_feature


# 説明変数をシフトして特徴量に追加する関数
def shift_data(df_feature, df_base, shift):
    # 説明変数だけを抽出する
    df_tmp1 = df_feature.drop(['Date', 'Close'], axis=1)

    # 説明変数をシフトする
    df_shift = df_tmp1.shift(shift)

    # シフトした説明変数をdf_baseに結合する
    df_feature = pd.concat([df_base, df_shift], axis=1)

    # シフトしたので、開始日を変更する
    start_day = df_feature.iloc[shift, 0]

    # 開始日用のindexを作成する
    start_index = df_feature['Date'] >= start_day

    # 開始日を移動したデータを作成する
    df_feature = df_feature[start_index]

    return df_feature


# 訓練データと検証データを作成する関数
def make_trainData_testData(df_feature, shift):
    # fbprophetで扱えるカラム名に変更する
    df_feature = df_feature.rename(columns={'Date': 'ds', 'Close': 'y'})

    # 訓練データと検証データを分割する
    # 訓練データ：2018-02-27 ～ 2020-11-09
    # 検証データ：2020-11-10 ～ 2020-12-30
    # 分割日split_dayを設定する
    split_day = pd.to_datetime('2020-11-10')
    end_day = pd.to_datetime('2020-12-30')

    # 分割日を起点にtrueとfalseを割り当てる
    # train_indexは691行目までTrueで、692～727行目までFalseとなり、
    # test_indexは691行目までFalseで、692～727行目までTrueとなる
    train_index = df_feature['ds'] < split_day test_index = df_feature['ds'] >= split_day

    # 訓練データと検証データを作成する
    x_train = df_feature[train_index]
    x_test = df_feature[test_index]

    # 予測データと検証データをグラフ表示する際に使用する日付を設定する
    # 設定する日付は2020-11-10 ～ 2020-12-30となる
    dates_test = df_feature['ds'][test_index]

    return x_train, x_test, dates_test


# モデルを定義して学習する関数
def make_model(x_train, dict_flag):
    # モデルを定義する
    model = Prophet(growth='logistic',
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative')

    # フラグに応じて、説明変数をモデルに組み込む
    for i in range(len(dict_flag)):
        if list(dict_flag.values())[i] == 1:
            model.add_regressor('Close_'+list(dict_flag.keys())[i])

    # cap(上限値)、floor(下限値)を設定する
    x_train['cap'] = 8500
    x_train['floor'] = 0

    # モデルを学習する
    model.fit(x_train)

    return model


# 予測する関数
def model_predict(df_feature, model):
    # 予測用データを作成する
    df_predicct = df_feature.drop('Close', axis=1)
    df_predicct = df_predicct.rename(columns={'Date': 'ds'})

    # cap(上限値)、floor(下限値)を設定する
    df_predicct['cap'] = 8500
    df_predicct['floor'] = 0

    # 予測する
    forecast = model.predict(df_predicct)

    return forecast


# r2,rmseを計算する関数
def score(forecast, x_test, shift):
    # forecastから予測部分yhatのみ抽出する
    # 2020-11-10から2020-12-30の期間で株取引が行われた日数は36日となる
    ypred = forecast[-36:][['yhat']].values

    # forecastから予測部分yhatのみ抽出する
    # 2020-11-10から2020-12-30の期間で株取引が行われた日数は36日となる
    ypred = forecast[-36:][['yhat']].values

    # ytest:予測期間中の正解データを抽出する
    ytest = x_test['y'].values

    # R2値,RMSEを計算する
    r2 = r2_score(ytest, ypred)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))

    return r2, rmse, ypred, ytest


# グラフを描画する関数
def draw_graph(model, forecast, dates_test, r2, rmse, ypred, ytest):
    # 要素ごとのグラフを描画する
    # トレンド、週周期、年周期を描画する
    fig = model.plot_components(forecast)

    # 訓練データ・検証データ全体のグラフ化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 予測結果のグラフ表示(prophetの関数)
    model.plot(forecast, ax=ax)

    # タイトル設定など
    ax.set_title('終値の予測')
    ax.set_xlabel('日付')
    ax.set_ylabel('終値')

    # 時系列グラフを描画する
    fig, ax = plt.subplots(figsize=(8, 4))

    # グラフを描画する
    ax.plot(dates_test, ytest, label='正解データ', c='k')
    ax.plot(dates_test, ypred, label='予測結果', c='r')

    # 日付目盛間隔を表示する
    # 木曜日ごとに日付を表示する
    weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
    ax.xaxis.set_major_locator(weeks)

    # 日付表記を90度回転する
    ax.tick_params(axis='x', rotation=90)

    # 方眼表示、凡例、タイトルを設定する
    ax.grid()
    ax.legend()
    ax.set_title('終値の予測')

    # x座標：2020年11月19日、y座標：7800にscore_r2とscore_rmseを表示する
    xdata = date2num(datetime(2020, 11, 19))
    ax.text(xdata, 7800, f'socre_R2:{r2:.4f}\nscore_RMSE:{rmse:.4f}', size=15)
    # ax.text(xdata, 6600, f'socre_RMSE:{score_rmse:.4f}', size=15)

    # 画面出力
    plt.show()

if __name__ == "__main__":
    main()