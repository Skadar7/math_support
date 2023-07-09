import vis
import numdifftools as nd
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian


class SteepestGradientDescend:
    def __init__(self):
        self.n = None
        self.__norm2 = lambda x: np.sqrt(np.sum([i ** 2 for i in x]))

    def get_start_point_(self):
        return np.array([1., 2.])#np.ones(self.n)

    def fit(self, func, n, e, iterations=100, verbose=True):
        k = 0
        self.n = n
        X_k = self.get_start_point_()
        history = [(*X_k, func(X_k))]
        for k in range(iterations):
            f_x = func(X_k)
            grad = nd.Gradient(func)(X_k)
            if self.__norm2(grad) <= e:
                return history
            hessian_k = jacobian(egrad(func))(X_k)
            h_k = (grad @ grad.T) / ((hessian_k @ grad) @ grad)

            if verbose:
                print(f'ITERATION {k + 1}\nx1 = {X_k[0]} \t\t x2 = {X_k[1]} \t y = {func(X_k)}')
                print('grad = ', grad)
                print('h_k = ', h_k)

            X_k = X_k - h_k * grad
            print('X_k = ', X_k)
            print(self.__norm2(grad))
            history.append((*X_k, func(X_k)))
        return history


if __name__ == '__main__':
    func = lambda x: 10 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2 + 10 * x[1]
    sgd = SteepestGradientDescend()
    history = sgd.fit(
        func,
        n=2,
        e=0.01,
    )
    vis.visualize(history)