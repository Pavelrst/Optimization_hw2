from scipy.optimize import rosen
import numpy as np
from numdiff_utils import numdiff
from scipy.io import loadmat

class Rosenbrock():
    def __init__(self, N):
        '''
        :param N: length of input vector
        '''
        self.N = N
        self.name = 'Rosenbrock(' + str(N) + ') function'

    def name(self):
        return self.name

    def starting_point(self):
        return np.array([0]*self.N)

    def optimal(self):
        '''
        Optimal value of Rosenbrock function at [1,1,1,1....1,1]
        :param N: num of variables in Rosenbrock function
        :return: global minimum position.
        '''
        return 0

    def val(self, x, test=False):
        '''
        This method returns Rosenbroc_function value
        '''
        assert len(x) == self.N
        val = 0
        for idx in range(len(x)-1):
            val += (1-x[idx])**2 + 100*(x[idx+1]-(x[idx])**2)**2
        if test:
            if val != rosen(x):
                np.testing.assert_almost_equal(val, rosen(x))
        return val

    def grad(self, x, test=False):
        '''
        This method returns rosengrad function gradient
        '''
        g = np.zeros_like(x)
        # [-2(1-x_1 )+200(x_2-x_1^2 )(-2x_1 )]
        g[0] = -2*(1-x[0])+200*(x[1]-x[0]**2)*(-2*x[0])
        for i in range(1, len(x)-1):
            g[i] = (200*(x[i]-x[i-1]**2))+(-2*(1-x[i])+200*(x[i+1]-x[i]**2)*(-2*x[i]))
            # same: g[i] = 202*x[i]-200*x[i-1]**2-2-400*x[i]*x[i+1]+400*x[i]**3
        # [200(x_N-x_(N-1)^2)]
        g[len(x)-1] = 200*(x[len(x)-1]-x[len(x)-2]**2)

        if test:
            numeric_par = {'epsilon': 2 * pow(10, -16),
                           'f_par': {"dummy": -666},
                           'gradient': self.grad}
            num_grad = numdiff(self.val, x, numeric_par)
            np.testing.assert_almost_equal(g, num_grad, 0)
        return g

    def hess(self, x, test=False):
        assert len(x) == self.N
        h = np.zeros((self.N, self.N))

        # Fill main diagonal
        for i in range(self.N):
            if i == self.N - 1:
                h[i, i] = 200
            elif i == 0:
                h[i, i] = 2 + 200 * (-2 * x[i + 1] + 6 * x[i] ** 2)
            else:
                h[i, i] = 202 + 200 * (-2 * x[i + 1] + 6 * x[i] ** 2)

        # Fill upper diagonal
        for i in range(self.N-1):
            row = i
            col = i+1
            h[row, col] = 200*(-2*x[i])

        # Fill lower diagonal:
        for i in range(self.N-1):
            row = i+1
            col = i
            h[row, col] = 200*(-2*x[i])

        return h


class Quadratic:
    def __init__(self, htype='well'):
        self.name = 'Quadratic ' + htype + ' conditioned function'
        self.mat_file = loadmat("h.mat")
        if htype == 'well':
            self.H = self.mat_file['H_well']
        elif htype == 'ill':
            '''
            A matrix is ill-conditioned if the condition number is too large
            (and singular if it is infinite)
            condition number is the ratio C of the largest to smallest singular value
            in the singular value decomposition of a matrix.
            '''
            self.H = self.mat_file['H_ill']
        else:
            print("define htype 'well' or 'ill'!")
        self.H_T = np.transpose(self.H)

    def name(self):
        return self.name

    def starting_point(self):
        return self.mat_file['x0']

    def optimal(self):
        '''
        Optimal value of f
        '''
        return 0

    def val(self, x):
        assert len(x) == len(self.mat_file['x0'])
        val = np.matmul(np.matmul(np.transpose(x), self.H), x)
        return 0.5 * val[0, 0]

    def grad(self, x, test=False):
        assert len(x) == len(self.mat_file['x0'])
        g = np.matmul(self.H+self.H_T, x)
        if test:
            numeric_par = {'epsilon': 2 * pow(10, -16),
                           'f_par': {"dummy": -666},
                           'gradient': self.grad}
            num_grad = numdiff(self.val, x, numeric_par)
            np.testing.assert_almost_equal(g, num_grad, 0)
        return 0.5 * g

    def hess(self, x='dummy'):
        return 0.5 * (self.H+self.H_T)



