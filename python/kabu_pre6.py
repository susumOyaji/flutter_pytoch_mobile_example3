# %% [markdown]
# # 指定した銘柄から、似た値動きをする銘柄を抽出します。

# %%
# 実行パスをPythonとJupyterでどちらで動いても統一できるようにしておく
import os
current = os.path.dirname(os.path.abspath(__file__))
os.chdir(current)

# %%
# カレントパスも実行パスに通しておく
import sys
from pathlib import Path
import os
parent = Path(current).resolve().parent
sys.path.append(str(parent))

# %%
# 独自モジュールなど
#import python.config_global as config
#import python.search_stock as search_stock

# %%
import time
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import pearsonr
from sklearn import cluster, preprocessing, mixture
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
def get_data(code):
  '''指定した銘柄のデーターをリターンインデックス込みで取得する'''
  data = search_stock.get_stock_data(code)
  if data is None:
    return None
  data = search_stock.get_ret_index(data)
  return data

# %%
def create_stock_table(stock_list, ret_index_table):
  '''渡ってきたデーターをDaraframeに変換する'''
  df = pd.DataFrame.from_dict(stock_list)
  # リターンインデックスのないデーターは除外する
  return df[df['code'].isin(ret_index_table.keys())]

# %%
def create_ret_index_table(stock_list, length):
  '''
  リターンインデックステーブルを作成する
  '''
  ret_index_table = {}
  for data in stock_list:
    code = data['code']
    new_data = get_data(code)
    if new_data is None:
      continue
    if len(new_data['ret_index']) != length:
      # 指定銘柄とデーターの長さが違う場合除外する
      continue
    ret_index_table = new_data['ret_index']
  return ret_index_table

# %%
def set_distance(target_data, stock_table, ret_index_table):
  '''対象とリターンインデックスがどれだけ離れているか計算する'''
  target = target_data['ret_index']
  stock_table['r'] = None
  stock_table['p'] = None
  stock_table['b'] = False
  for code, ret_index in ret_index_table.items():
    # r 相関係数は 1 に近いほど強い相関がある
    # p 有意確率 P 値は 0 に近いほどデータが偶然にそうなった可能性が低い
    r, p = pearsonr(target, ret_index)
    b = float(p) < float(0.05)
    data = stock_table[stock_table['code'] == code]
    stock_table.loc[stock_table['code'] == code, 'r'] = r
    stock_table.loc[stock_table['code'] == code, 'p'] = p
    stock_table.loc[stock_table['code'] == code, 'b'] = b
  # 有意確率の高いものだけに絞り込む
  return stock_table[stock_table['b'] == True]

# %%
def setup_ret_indexes_list(target_code, target_list, stock_table, ret_index_table):
  '''機械学習用にリターンインデックスのリストを作成します'''
  # 最初は必ず検索対象のリターンインデックス
  ret = [target_list]
  codes = [target_code]
  for code in stock_table['code']:
    r = ret_index_table
    ret.append(r)
    codes.append(code)
  return ret, codes

# %%
def check_vbgm(ret_indexes):
  '''機械学習でクラスタリングした結果を返す'''
  # データーをノーマライズさせる
  sc=preprocessing.StandardScaler()
  sc.fit(ret_indexes)
  ret_indexes_normarize = sc.transform(ret_indexes)
  n_components = min(10, len(ret_indexes))
  # VBGMMクラスタリング
  vbgm = mixture.BayesianGaussianMixture(
    n_components=n_components, max_iter=100,
    verbose=0,
  )
  vbgm=vbgm.fit(ret_indexes_normarize)
  labels=vbgm.predict(ret_indexes_normarize)
  return labels

# %%
def stock_group(labels, codes):
  '''銘柄コードを機械学習されたグループごとにわけます'''
  # 空のグループリストを作成
  groups = []
  for i in range(max(labels) + 1):
    groups.append([])
  # ラベルをインデックスとして、それぞれ銘柄コードをグループ化
  for label, code in zip(labels, codes):
    groups[label].append(code)
  return groups

# %%
def create_json_ret_index_table(target_code, target_data, ret_index_table):
  '''リターンインデックステーブルをJSON用に変換する'''
  # 最初に対象銘柄のデーターを挿入する
  ret = {
    target_code: target_data['ret_index'].values.tolist()
  }
  for code in ret_index_table.keys():
    ret = ret_index_table.values.tolist()
  return ret

# %%
def create_json_stock_table(stock_table):
  cols = stock_table.columns.values.tolist()
  t = []
  for row in stock_table.itertuples(name=None):
    t.append(row[1:])
  return {'keys': cols, 'data': t}

# %%
def start(target_code, stock_list_):
  '''API用にデーター生成する'''
  # ターゲットデーター作成
  target_data = get_data(target_code)
  target_length = len(target_data['ret_index'])
  y = target_data['date']
  # 各種リスト作成
  ret_index_table = create_ret_index_table(stock_list_, target_length)
  stock_table = create_stock_table(stock_list_, ret_index_table)
  # ピアソンの積率相関係数を作成
  stock_table = set_distance(target_data, stock_table, ret_index_table)
  # 相関係数の高い順にソート
  stock_table = stock_table.sort_values('r', ascending=False)
  # 機械学習用にリターンインデックスのリストを作る。どのリターンインデックスがどの銘柄コードか検索できるように、同じ並びで銘柄コードのリストも作成
  ret_list, code_list = setup_ret_indexes_list(target_code, target_data['ret_index'], stock_table, ret_index_table)
  # 機械学習でクラス分け
  labels = check_vbgm(ret_list)
  # ターゲット銘柄は必ず先頭なので、先頭のグループIDを取得
  target_group_id = labels[0]
  # グループごとに銘柄コードのリストを作成
  grouped = stock_group(labels, code_list)
  # JSONにできる形式に変換
  # リターンインデックステーブル
  j_ret_index_table = create_json_ret_index_table(target_code, target_data, ret_index_table)
  # グラフ用日付リスト
  j_y_label = y.values.tolist()
  j_target_data = {'code': target_code, 'name': search_stock.get_name(target_code)}
  j_stock_table = create_json_stock_table(stock_table)
  return {
    'grouped': grouped,
    'target_group_id': int(target_group_id),
    'y_label': j_y_label,
    'target_data': j_target_data,
    'stock_table': j_stock_table,
    'ret_index_table': j_ret_index_table,
  }
  
# %%
if __name__ == '__main__':
  time_start = time.time()
  low = 100000
  high = 200000
  target_code = '7974'
  stock_list = search_stock.get_stock_list(low, high, target_code)
  # ここまでがWEBから渡ってくる値
  g = start(target_code, stock_list)
  print(g)
  process_time = time.time() - time_start
  print(process_time)