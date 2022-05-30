%matplotlib inline
 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
 
# use seaborn plotting defaults
import seaborn as sns; sns.set()
次に、2クラスにクラス分けされたトレーニングデータセットを用意します。

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
random_state=3, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='brg');