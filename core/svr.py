from utils.utils import _read_csv_,_darw
from sklearn import svm
import numpy as np


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
    path = '/Users/kanghaidong/Desktop/haidong/github-reper/Machine-learning-regression-algorithm/data/hangkongshuju.csv'
    x_train, x_test, y_train, y_test, scaler =_read_csv_(path)

    c = x[0]
    e = x[1]
    g = x[2]
    _SVR = svm.SVR(C=c, epsilon=e, gamma=g, kernel='rbf')
    _SVR.fit(x_train, y_train)
    y_pred = _SVR.predict(x_train) # np.array
    pridict = _SVR.predict(x_test)
    score = _SVR.score(x_train, y_train)

    # 为了反标转化，需要将数据扩展为2维numpy
    # numpy.array(a).reshape(len(a),1)  # reshape(列的长度，行的长度)

    y_pred = np.array(y_pred).reshape(len(y_pred), 1)
    pridict = np.array(pridict).reshape(len(pridict), 1)

    y_train = scaler.inverse_transform(y_train)
    y_pred = scaler.inverse_transform(y_pred)
    pridict = scaler.inverse_transform(pridict)

    # 打印未来预测值
    print(f'predict futrue: {pridict}')

    # print(f'real values is: {y_test[-1]}')
    # print("y_pred is", y_pred , "y_true is" , y_test)
    # print(len(y_pred),len(y_test))
    # 返回svm的mse作为适应度值
    # y_test = y_test.values # Pandas中把dataframe和np.array的相互转换
    print(f"score{score}")
    _darw(y_train, y_pred, 'y_test', 'pridict', 'r', 'years', 'Industrial aviation unit GDP flight hours', 'Industrial aviation unit GDP flight hours')

    return _MSE(y_pred, y_train),y_pred, y_train