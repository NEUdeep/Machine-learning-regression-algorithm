from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from utils.utils import _read_csv_,_darw


#加载数据集
# boston=load_boston()
# print(boston.feature_names)
# x=boston.data
# y=boston.target
path = '/Users/kanghaidong/Desktop/haidong/github-reper/Machine-learning-regression-algorithm/data/hangkongshuju.csv'
df=pd.read_csv(path,encoding='utf-8') # GBK
print(df.head)
'''use dropna(axis=0,how='all')'''
df = df.dropna(axis=0,how='all')

# x=df[['GDP','工业总产值','铁路运输长度','复线比例','公路运输长度','等级公路比重','铁路货运数量','民用载货车辆']]
# y=df[['货运量']]
x = df[['x1','x2','x3','x4','x5','x6','x7']]
y = df[['y1']]
# 拆分数据集
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7,random_state=10)
x_train,x_test,y_train,y_test = x[:-7],x[9:],y[:-7],y[9:]
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

#创建svR实例
svr=SVR(C=1, kernel='rbf', epsilon=0.2)
svr=svr.fit(x_train,y_train)
#预测
svr_predict=svr.predict(x_train)
score = svr.score(x_train,y_train)
pridict = svr.predict(x_test)

# 为了反标转化，需要将数据扩展为2维numpy
# numpy.array(a).reshape(len(a),1)  # reshape(列的长度，行的长度)

svr_predict = np.array(svr_predict).reshape(len(svr_predict),1)
pridict = np.array(pridict).reshape(len(pridict),1)

# 使用sklearn进行数据的归一化和归一化还原
# # invers_tainsform()
# pridict = StandardScaler().fit(pridict.reshape(-1, 1)).inverse_transform(pridict.reshape(-1, 1))
# y_train = StandardScaler().fit(y_train.reshape(-1, 1)).inverse_transform(y_train.reshape(-1, 1))
# svr_predict = StandardScaler().fit(svr_predict.reshape(-1, 1)).inverse_transform(svr_predict.reshape(-1, 1))
y_train = scaler3.inverse_transform(y_train)
svr_predict = scaler3.inverse_transform(svr_predict)
pridict = scaler3.inverse_transform(pridict)

#打印未来预测值
print(f'predict futrue: {pridict}')

#评价结果
mae = mean_absolute_error(y_train, svr_predict)
mse = mean_squared_error(y_train, svr_predict)
evs = explained_variance_score(y_train, svr_predict)
r2 = r2_score(y_train, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)
print("score",score)

# 绘图
_darw(y_train,svr_predict,'y_test','pridict','r','years','Flight hours per unit GDP','Flight hours per unit GDP')



