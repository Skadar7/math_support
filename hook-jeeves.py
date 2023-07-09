import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from vis import visualize


class Hook_Jeeves:
    def __init__(self, n, d, m, e, h, func):
        self.n = n
        self.d = d
        self.m = m
        self.h = np.array(h)
        self.e = e
        self.func = func
        self.X = np.array([1., 2.])#np.ones(self.n)
        self.history = []
        #self.h = np.random.uniform(0.1, 0.9, self.n)
    
    def check_new_point(self, h):
        res, tmp = deepcopy(self.X), deepcopy(self.X)
        
        for i in range(self.n):
            tmp[i] = tmp[i] - h[i]
            if self.func(tmp) < self.func(self.X):
                res = tmp
            else:
                tmp[i] = tmp[i] + 2 * h[i]
                if self.func(tmp) < self.func(self.X):
                    res = tmp
        print(res)
        return res
    
    def fit(self, iter):
        self.history = [(*self.X, self.func(self.X))]
        for i in range(iter):
            new_X = self.check_new_point(self.h)
            if all(self.X == new_X):
                self.h /= self.d
                new_X = self.check_new_point(self.h)
                self.history.append((*new_X, self.func(new_X)))
            else:
                Xp = new_X + self.m * (new_X - self.X)
                self.X = Xp if self.func(Xp) < self.func(self.X) else new_X
                self.history.append((*self.X, self.func(self.X)))
            if all(self.h <= self.e):
                return self.history
        return self.history
        

    
obj = Hook_Jeeves(2, 2, 2, 0.1, [0.3, 0.3], lambda x: 10 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2 + 10 * x[1])
history = obj.fit(100)
visualize(history)
