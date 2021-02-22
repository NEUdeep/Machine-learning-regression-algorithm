from core.svr import _SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GAIndividual:
    '''
    individual of genetic algorithm
    创建pop中的单个个体
    '''
    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness,y_pred, y_test = _SVR(self.vardim, self.chrom, self.bound)
        print(self.fitness)