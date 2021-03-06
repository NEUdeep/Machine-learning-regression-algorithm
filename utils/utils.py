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
    print(len(y),x)
    # x = [2010,2011,2012,2013,2014,2015,2016,2017,2018]

    # y = list(y)
    # y = np.sum([y,x],axis=0)
    plt.scatter(x,y,c='k',label=origin_label)
    plt.plot(pridict,c=colour,label=pridict_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('result')
    '''
    注意，当调用ga_svr的时候，上面一行不要打开，当前代码逻辑是每次ga搜索都会保存图片。
    '''
    plt.show()


# def _read_csv_(path):
#     '''
#     :param path:
#     :return:
#     '''
#
#     # 加载数据集
#     # boston_data = load_boston()
#     # print(boston_data)
#     df = pd.read_csv(path, encoding='GBK')
#     y = df[['closePrice']].values
#     df = df.dropna(axis=0, how='all')
#     df = df.set_index('tradeDate').copy()
#     X = np.mat(range(1, len(df.values) + 1)).T
#     X = np.array(X).reshape(-1, 1)
#     print(X.size, y.size)
#
#     # 拆分数据集
#     # x = boston_data.data
#     # y = boston_data.target
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#     # 预处理
#     y_train = np.array(y_train).reshape(-1, 1)
#     y_test = np.array(y_test).reshape(-1, 1)
#     x_train = StandardScaler().fit_transform(x_train)
#     x_test = StandardScaler().fit_transform(x_test)
#     y_train = StandardScaler().fit_transform(y_train).ravel()
#     y_test = StandardScaler().fit_transform(y_test).ravel()
#     return x_train, x_test, y_train, y_test


def _read_csv_(path):
    '''
    :param path:
    :return:
    '''

    # 加载数据集
    # boston_data = load_boston()
    # print(boston_data)
    df = pd.read_csv(path, encoding='utf8')
    # y = df[['单位GDP通航飞行时间']].values
    y = df[['y3']].values
    df = df.dropna(axis=0, how='all')
    #df = df.set_index('居民消费水平').copy()
    #X = np.mat(range(1, len(df.values) + 1)).T
    # X = df[['社会消费品零售总额','第一产业增加值','第三产业增加值','民航全行业利润总额','通航企业数量','通用航空器数量','通航产业政策绩效']]
    X = df[['x1','x2','x3','x4','x5','x6','x7']]
    print(X)
    X = X.values
    # X = np.array(X).reshape(-1, 1)
    print(X.size, y.size)

    # 拆分数据集
    # x = boston_data.data
    # y = boston_data.target
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # x_train, x_test, y_train, y_test = X,X,y,y
    x_train, x_test, y_train, y_test = X[:-7], X[9:], y[:-7], y[9:]
    # 预处理
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    # x_train = StandardScaler().fit_transform(x_train) # 先拟合、再标准化
    scaler1 = StandardScaler().fit(x_train)
    x_train = scaler1.transform(x_train)
    # x_train = StandardScaler().fit(x_train) # 拟合
    # x_test = StandardScaler().fit_transform(x_test)

    scaler2 = StandardScaler().fit(x_test)
    x_test = scaler2.transform(x_test)
    # x_test = StandardScaler().transform(x_test) # 只标准化、不拟合
    # y_train = StandardScaler().fit_transform(y_train).ravel()
    scaler3 = StandardScaler().fit(y_train)
    y_train = scaler3.transform(y_train)
    # y_test = StandardScaler().fit_transform(y_test).ravel()
    return x_train, x_test, y_train, y_test, scaler3


def _read_csv_huoyun(path):
    df = pd.read_csv(path, encoding='utf-8') #GBK
    print(df.head)
    '''use dropna(axis=0,how='all')'''
    df = df.dropna(axis=0, how='all')

    # x = df[['GDP', '工业总产值', '铁路运输长度', '复线比例', '公路运输长度', '等级公路比重', '铁路货运数量', '民用载货车辆']]
    # y = df[['货运量', '铁路货运量', '公路货运量']]
    x = df[['x1','x2','x3','x4','x5','x6','x7']]
    y = df[['y1','y2','y3']]
    x,y = x[:-7],y[:-7]
    return x,y
