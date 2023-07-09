import numpy as np
import pandas as pd
import numdifftools as nd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from vis import visualize

class GradientDescent:

    def __init__(self, n, learning_rate=0.01, num_iterations=100, alpha=0.01):
        self.n = n
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.history = []

    
    def fit(self, func):
        self.X = np.array([1, 2])
        delta = lambda x: np.sqrt(np.sum(i ** 2 for i in x))
        history = [(*self.X, func(self.X))]
        for i in range(self.num_iterations):
            print('ITERATIONS {}'.format(i))
            result_func = func(self.X)
            grad = nd.Gradient(func)(self.X)
            print('X = ', self.X)
            print('f(X) = ', result_func)
            print('grad = ', grad)
            if delta(grad) > self.alpha:
                self.X = self.X - self.learning_rate * grad
                new_f = func(self.X)
                print('new_X = ', self.X)
                print('new_f(X) = ', new_f)
                print('delta_grad = ', delta(grad))
                if new_f < result_func:
                    result_func = new_f
                else:
                    self.learning_rate /= 2
            else:
                return self.history
            history.append((*self.X, func(self.X)))
        return history
            

obj = GradientDescent(n=2)
history = obj.fit(lambda x: 10 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2 + 10 * x[1])
visualize(history)
