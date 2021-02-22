import numpy as np
from core.ga_alg import GeneticAlgorithm


if __name__ == "__main__":
    '''
    程序函数入口
    '''
    bound = np.tile([[0.0000001], [1]], 3)
    ga = GeneticAlgorithm(60, 3, bound, 100, [0.9, 0.1, 0.5])
    ga.solve()