from sklearn.svm import SVR
from utils.utils import _read_csv_,_darw
from core.gs_alg import _GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score


def _gs_svr(path):
    '''
    :param path:
    :return:
    '''
    x_train, x_test, y_train, y_test, scalar = _read_csv_(path)

    # 设置超参数
    C = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 5, 10]
    kernel = 'rbf'
    gamma = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]
    epsilon = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    # 参数字典
    params_dict = {
        'C': C,
        'gamma': gamma,
        'epsilon': epsilon
    }
    svr = SVR()
    gs = _GridSearchCV(params_dict,svr)


    # 调用模型
    gs.fit(x_train, y_train)

    # 输出参数信息
    print("最佳度量值:", gs.best_score_)
    print("最佳参数:", gs.best_params_)
    print("最佳模型:", gs.best_estimator_)

    # 用最佳参数生成模型
    svr = SVR(C=gs.best_params_['C'], kernel=kernel, gamma=gs.best_params_['gamma'],
              epsilon=gs.best_params_['epsilon'])

    # 获取在训练集的模型
    svr.fit(x_train, y_train)
    # 预测结果
    svr_predict = svr.predict(x_train)

    pridict = svr.predict(x_test)

    # 打印未来预测值
    print(f'predict future: {pridict}')

    #predict
    print(f'predict exit values is: {svr_predict[-6:-1]}')
    print(f'real values is: {y_test[-6:-1]}')
    score = svr.score(x_train, y_train)
    # 模型评测
    mae = mean_absolute_error(y_train, svr_predict)
    mse = mean_squared_error(y_train, svr_predict)
    evs = explained_variance_score(y_train, svr_predict)
    r2 = r2_score(y_train, svr_predict)

    # draw
    _darw(y_train, svr_predict, 'y_test', 'pridict', 'r', 'days', 'close_price', 'stock_price')
    return score,mae,mse,evs,r2