import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def _darw(y,pridict,origin_label,pridict_label,colour,xlabel,ylabel,title):
    '''
    :param y:
    :param pridict:
    :param origin_label:
    :param pridict_label:
    :param colour: 'k','g','r','b'
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    '''
    x = [k for k in range(len(y))]
    x = np.sort(x,axis=0)
    # print(len(y),x)
    plt.scatter(x,y,c='k',label=origin_label)
    plt.plot(pridict,c=colour,label=pridict_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # plt.savefig('result')
    '''
    注意，当调用ga_svr的时候，上面一行不要打开，当前代码逻辑是每次ga搜索都会保存图片。
    '''
    plt.show()


def _read_csv_(path):
    '''
    :param path:
    :return:
    '''

    # 加载数据集
    # boston_data = load_boston()
    # print(boston_data)
    df = pd.read_csv(path, encoding='GBK')
    y = df[['closePrice']].values
    df = df.dropna(axis=0, how='all')
    df = df.set_index('tradeDate').copy()
    X = np.mat(range(1, len(df.values) + 1)).T
    X = np.array(X).reshape(-1, 1)
    print(X.size, y.size)

    # 拆分数据集
    # x = boston_data.data
    # y = boston_data.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # 预处理
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    y_train = StandardScaler().fit_transform(y_train).ravel()
    y_test = StandardScaler().fit_transform(y_test).ravel()
    return x_train, x_test, y_train, y_test


def _read_csv_huoyun(path):
    df = pd.read_csv(path, encoding='GBK')
    print(df.head)
    '''use dropna(axis=0,how='all')'''
    df = df.dropna(axis=0, how='all')

    x = df[['GDP', '工业总产值', '铁路运输长度', '复线比例', '公路运输长度', '等级公路比重', '铁路货运数量', '民用载货车辆']]
    y = df[['货运量', '铁路货运量', '公路货运量']]
    return x,y