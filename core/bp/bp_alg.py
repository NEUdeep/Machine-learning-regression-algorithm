import numpy as np
from utils.utils import _read_csv_,_read_csv_huoyun
from utils.first_process import _first_process
import  matplotlib.pyplot as  plt



def _knn_bp(path,save):
    # BP神经网络网络参数
    sample_in, sample_out,x_scaler,y_scaler,x,y =_first_process(path)
    max_epochs = 60000  # 循环迭代次数
    learn_rate = 0.00012  # 学习率
    mse_final = 6.5e-4  # 设置一个均方误差的阈值，小于它则停止迭代
    sample_number = x.shape[0]  # 样本数
    input_number = x.shape[1]  # 输入特征数
    output_number = y.shape[1]  # 输出目标个数
    hidden_units = 8  # 隐含层神经元个数
    print(sample_number, input_number, output_number)

    # 定义激活函数Sigmod
    # import math
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_delta(z):  # 偏导数
        return 1 / ((1 + np.exp(-z)) ** 2) * np.exp(-z)

    print(sigmoid(0), sigmoid_delta(0))

    # 一层隐含层
    # W1矩阵:M行N列，M等于该层神经元个数，N等于输入特征个数
    W1 = 0.5 * np.random.rand(hidden_units, input_number) - 0.1
    b1 = 0.5 * np.random.rand(hidden_units, 1) - 0.1

    W2 = 0.5 * np.random.rand(output_number, hidden_units) - 0.1
    b2 = 0.5 * np.random.rand(output_number, 1) - 0.1

    mse_history = []  # 空列表，存储迭代的误差
    # 不设置激活函数
    for i in range(max_epochs):
        # FP
        hidden_out = sigmoid(np.dot(W1, sample_in) + b1)  # np.dot矩矩阵相乘,hidden_out1结果为8行20列
        network_out = np.dot(W2, hidden_out) + b2  # np.dot矩阵相乘,W2是2行8列，则output结果是2行20列
        # 误差
        err = sample_out - network_out
        mse_err = np.average(np.square(err))  # 均方误差
        mse_history.append(mse_err)
        if mse_err < mse_final:
            break
        # BP
        # 误差向量
        delta2 = -err  # 最后一层的误差
        delta1 = np.dot(W2.transpose(), delta2) * sigmoid_delta(
            hidden_out)  # 前一层的误差向量,这一层对hidden_out用了sigmoid激活函数,要对hidden_out求偏导数；注意最后一步是两个矩阵的点乘，是两个完全相同维度矩阵
        # 梯度：损失函数的偏导数
        delta_W2 = np.dot(delta2, hidden_out.transpose())
        delta_W1 = np.dot(delta1, sample_in.transpose())
        delta_b2 = np.dot(delta2, np.ones((sample_number, 1)))
        delta_b1 = np.dot(delta1, np.ones((sample_number, 1)))
        W2 -= learn_rate * delta_W2
        b2 -= learn_rate * delta_b2
        W1 -= learn_rate * delta_W1
        b1 -= learn_rate * delta_b1

    # 损失值画图
    print(mse_history)
    loss = np.log10(mse_history)
    min_mse = min(loss)
    plt.plot(loss, label='loss')
    plt.plot([0, len(loss)], [min_mse, min_mse], label='min_mse')
    plt.xlabel('iteration')
    plt.ylabel('MSE loss')
    plt.title('Log10 MSE History', fontdict={'fontsize': 18, 'color': 'red'})
    plt.legend()
    plt.savefig(save+'Log10 MSE History')
    plt.show()

    # 模型预测输出和实际输出对比图
    hidden_out = sigmoid(np.dot(W1, sample_in) + b1)
    network_out = np.dot(W2, hidden_out) + b2

    # 反转获取实际值：
    network_out = y_scaler.inverse_transform(network_out.T)
    sample_out = y_scaler.inverse_transform(y)
    print(network_out)
    print(sample_out)
    return network_out,sample_out