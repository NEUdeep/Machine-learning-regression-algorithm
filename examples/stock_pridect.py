import numpy as np
from sklearn.svm import SVR
import matplotlib.pylab as plt
import pandas as pd


import time
path = '/Users/kanghaidong/Desktop/haidong/github-reper/Machine-learning-regression-algorithm/data/stock_pinganyinhang.csv'
df=pd.read_csv(path,encoding='GBK')
y = df[['closePrice']].values
df = df.dropna(axis=0,how='all')
df = df.set_index('tradeDate').copy()
dates = df.index
X = np.mat(range(1,len(df.values)+1)).T
X =  np.array(X).reshape(-1,1)
print(X,X.size,y.size)

# regression

svr_rbf = SVR(kernel = 'rbf',gamma = 0.1)
y_ybf = svr_rbf.fit(X,y).predict(X)
print(y_ybf)
_,ax = plt.subplots(figsize=[14,7])
datas  = df.index
ax.scatter(X,y,c='r',label='data')
ax.plot(X,y_ybf,c='g',label='rbf_model')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR')
plt.legend()
# plt.savefig('result_rbf_model')
plt.show()

