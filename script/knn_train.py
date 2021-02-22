import  matplotlib.pyplot as  plt
from core.bp.bp_alg import _knn_bp


if __name__ == "__main__":
    path = '//data/train.csv'
    save = '/Users/kanghaidong/Desktop/Machine learning regression algorithm/result/'
    network_out,sample_out = _knn_bp(path,save)


    #解决图片中文无法显示
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(8, 6))
    plt.plot(network_out[:,0],label='pred')
    plt.plot(sample_out[:,0],'r.',label='actual')
    plt.title('huoyunliang')
    plt.legend()
    plt.savefig(save + 'huoyunliang')
    plt.show()



    plt.figure(figsize=(8, 6))
    plt.plot(network_out[:,1],label='pred')
    plt.plot(sample_out[:,1],'r.',label='actual')
    plt.title('tieluhuoyunliang')
    plt.legend()
    plt.savefig(save+'tieluhuoyunliang.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(network_out[:,2],label='pred')
    plt.plot(sample_out[:,2],'r.',label='actual')
    plt.title('gongluhuoyunliang')
    plt.legend()
    plt.savefig(save+'gongluhuoyunliang')
    plt.show()