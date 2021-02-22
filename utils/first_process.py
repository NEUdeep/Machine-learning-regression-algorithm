from sklearn.preprocessing import  MinMaxScaler
from utils.utils import _read_csv_,_read_csv_huoyun


def _first_process(path):
    x_train, y_train = _read_csv_huoyun(path)
    # 对数据进行最大最小值归一化
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    x = x_scaler.fit_transform(x_train)
    y = y_scaler.fit_transform(y_train)

    # 对样本进行转置，矩阵运算
    sample_in = x.T
    sample_out = y.T

    return sample_in,sample_out,x_scaler,y_scaler,x,y