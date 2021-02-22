from utils.utils import _read_csv_,_darw
from sklearn import svm


def _MSE(p, label):
    '''
    :param p:
    :param label:
    :return:
    '''
    MSEERROR = []
    # absError = []
    for i in range(len(p)):
        loss = p[i] - label[i]
        MSEERROR.append(loss * loss)  # 预测值与真实值之差的平方

    print("Square Error: ", MSEERROR)
    print("MSE = ", sum(MSEERROR) / len(MSEERROR))  # 均方误差MSE
    return sum(MSEERROR) / len(MSEERROR)


def _SVR(v, x, b):
    '''
    :param v:
    :param x:
    :param b:
    :return:
    '''
    path = '/Users/kanghaidong/Desktop/market_data/pinganyinhang.csv'
    x_train, x_test, y_train, y_test=_read_csv_(path)

    c = x[0]
    e = x[1]
    g = x[2]
    _SVR = svm.SVR(C=c, epsilon=e, gamma=g, kernel='rbf')
    _SVR.fit(x_train, y_train)
    y_pred = _SVR.predict(x_test) # np.array
    # print("y_pred is", y_pred , "y_true is" , y_test)
    # print(len(y_pred),len(y_test))
    # 返回svm的mse作为适应度值
    # y_test = y_test.values # Pandas中把dataframe和np.array的相互转换
    score = _SVR.score(x_test, y_test)
    print(f"score{score}")
    _darw(y_test, y_pred, 'y_test', 'pridict', 'r', 'days', 'close_price', 'stock_price')

    return _MSE(y_pred, y_test),y_pred, y_test