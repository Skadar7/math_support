import vis
import numdifftools as nd
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian


class Newton:
    def __init__(self):
        self.n = None
        self.__norm2 = lambda x: np.sqrt(np.sum([i ** 2 for i in x]))

    def get_start_point_(self):
        return np.array([1., 2.])#np.ones(self.n)

    def __is_positive(self, X):
        if X.shape[0] != X.shape[1]:
            return False

        for i in range(1, X.shape[0] + 1):
            minor = X[:i, :i]
            determinant = np.linalg.det(minor)
            if determinant <= 0:
                return False

        return True

    def fit(self, func, n, e, iterations=100, verbose=True):
        k = 0
        self.n = n
        X_k = self.get_start_point_()
        history = [(*X_k, func(X_k))]
        for k in range(iterations):
            f_x = func(X_k)
            grad = nd.Gradient(func)(X_k)
            if self.__norm2(grad) < e:
                return history
            hessian_k = np.array([[20, 3], [3, 2]])#jacobian(egrad(func))(X_k)
            p_k = - np.linalg.inv(hessian_k) @ grad
            if verbose:
                print(f'ITERATION {k + 1}\nx1 = {X_k[0]} \t\t x2 = {X_k[1]} \t y = {func(X_k)}')
                print('hessian_k = ', hessian_k)
                print('p_k = ',p_k)
            if self.__is_positive(hessian_k):
                X_k = X_k + p_k
                print('new_K = ', X_k)
            else:
                h_k = (grad @ p_k) / ((hessian_k @ p_k) @ grad)
                X_k = X_k - h_k * grad
                print('new_K = ', X_k)
            print('new_f(X) = ', func(X_k))
            history.append((*X_k, func(X_k)))
        return history


if __name__ == '__main__':
    func = lambda x: 10 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2 + 10 * x[1]
    newton = Newton()
    history = newton.fit(
        func,
        n=2,
        e=0.01,
    )
    vis.visualize(history)