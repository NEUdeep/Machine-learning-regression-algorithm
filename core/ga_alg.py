import numpy as np
import random
import copy
from core.ga_profile import GAIndividual



class GeneticAlgorithm:
    '''
    The class for genetic algorithm
    '''
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting
               of crossover rate, mutation rate, alpha
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        self.t = 0 # 迭代次数
        self.initialize() # 初始化种群
        self.evaluate() # 计算适应度
        best = np.max(self.fitness) #选出适应度最大的个体
        bestIndex = np.argmax(self.fitness) # 最大适应度的索引
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness) # 平均适应度
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print(
            "Generation %d: optimal function value is: %f; average function value is %f"
            % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation() # 选择
            self.crossoverOperation() # 交叉
            self.mutationOperation()  # 变异
            self.evaluate()           # 重新计算新种群适应度
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            # 种群中表现最好的个体的适应度变化
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            # 种群平均适应度变化
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print(
                "Generation %d: optimal function value is: %f; average function value is %f"
                % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:",self.best.chrom)

    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))

        # 适应度的累进占比
        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

        # 随机选出新种群的索引
        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        # 选出两个个体进行交换
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            # params[0]=>crossover_rate;
            # params[1]=>mutation_rate;
            # params[2]=>alpha
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    # chrom[j1]=chrom[j1]*α+(1-α)*chrom[j2]
                    newpop[i].chrom[j] = newpop[i].chrom[j] * self.params[2] +\
                        (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    # chrom[j2]=chrom[j2]*α+(1-α)*chrom[j1]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    # chrom=chrom-(chrom-lowerlimit)*(1-rand^(1-t/N))
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] - \
                        (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * \
                            (1 - random.random()**(1 - self.t / self.MAXGEN))
                else:
                    # chrom=chrom+(uperlimit-chrom)*(1-rand^(1-t/N))
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] + \
                        (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * \
                            (1 - random.random()**(1 - self.t / self.MAXGEN))
        self.population = newpop