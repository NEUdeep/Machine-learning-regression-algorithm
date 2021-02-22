from core._gs_svr import _gs_svr


if __name__ == "__main__":
    '''
    程序主入口
    path说明：
    要仔细看你的csv的内容。本例子展示的是股票价格走势预测，自变量是时间，因变量是交易价格。
    假设你要预测3个因变量，和多个相同的自变量的关系，请一个一个因变量带入预测。
    如有疑问,如具体如何替换因变量，请call khd.
    '''
    path = '/Users/kanghaidong/Desktop/haidong/github-reper/Machine-learning-regression-algorithm/data/stock_pinganyinhang.csv'
    score,mae,mse,evs,r2 = _gs_svr(path)
    print("MAE：", mae)
    print("MSE：", mse)
    print("EVS：", evs)
    print("R2：", r2)
    print(f"score{score}")
    print(f'done')