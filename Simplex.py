import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import operator
from vis import visualize


class Simplex:

    def __init__(self, n: int, m: int, e: int, func: any) -> None:
        self.history = []
        self.n = n
        self.m = m
        self.x = np.array([1., 2.])#np.ones(self.n)
        self.e = e
        self.func = func


    def sigma1(self) -> float:
        return (((self.n + 1) ** (1/2) - 1) / (self.n * 2 ** (1/2))) * self.m
    

    def sigma2(self) -> float:
        return (((self.n + 1) ** (1/2) + self.n - 1) / (self.n * 2 ** (1/2))) * self.m
        

    def calculate_other_points(self):
        result = deepcopy(self.x)
        for i in range(self.n):
            point = np.zeros(self.n)
            for j, c in enumerate(self.x):
                if i == j:
                    point[j] = c + self.sigma1()
                else:
                    point[j] = c + self.sigma2()
            result = np.vstack((result, point))
        return result
    

    def max_k(self, X):
        k, f_max = max(enumerate(self.func(X.T)), key=operator.itemgetter(1))
        return k, f_max
    

    def min_k(self, X):
        k, f_min = min(enumerate(self.func(X.T)), key=operator.itemgetter(1))
        return k, f_min


    def exclude_k(self, X, k):
        return np.concatenate((X[:k], X[k+1:]))


    def grv_center(self, X, k):
        new_X = self.exclude_k(X, k)
        return (new_X.sum(axis=0)) / self.n


    def reflect(self, X, k, Xc):
        return 2 * Xc - X[k]
    

    def reduction(self, X):
        k, _ = self.min_k(X)
        for i in range(len(X)):
            if i != k:
                X[i] = X[k] + 0.5 * (X[i] - X[k])
        return X
    

    def create_df(self, X):
        columns = list(map(lambda x: str(x), range(X.shape[1])))
        columns.append('f(x)')
        return pd.DataFrame(
            columns=columns,
            index=list(range(X.shape[0])),
            
            data=np.c_[X, self.func(X.T)]
        )
    

    def fit(self, iterations=50, verbose=True):
        X = self.calculate_other_points()
        print(X)
        for i in range(iterations):
            print(X)
            K, f_max = self.max_k(X)
            print(K, f_max)
            Xc = self.grv_center(X, K)
            print('grav = ', Xc)
            Xv = self.reflect(X, K, Xc)
            print('reflect = ', Xv)
            new_f = self.func(Xv)
            print('new_f = ', new_f)
            if new_f < f_max:
                X[K] = Xv
                f_max = new_f
            else:
                X = self.reduction(X)

            Xc = np.sum(X, axis=0) / (self.n + 1)
            fXc = self.func(Xc)
            K_min, f_min = self.min_k(X)
            self.history.append((*X[K_min], f_min))
            if verbose:
                print(f'\nITERATION {i + 1}')
            print(self.create_df(X))
            
            if all(round(np.abs(i - fXc), 2) < self.e for i in self.func(X)):
                return self.history

        return self.history


obj = Simplex(2, 0.5, 0.1, lambda x: 10 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2 + 10 * x[1])
history = obj.fit(100)
print(history[-1])
visualize(history)

