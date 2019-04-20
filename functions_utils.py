from scipy.optimize import rosen
import numpy as np
from numdiff_utils import numdiff

def rosen_val(x):
    '''
    This function implements Rosenbroc_function
    :param x:
    :return:
    '''
    val = 0
    for idx in range(len(x)-1):
        val += (1-x[idx])**2 + 100*(x[idx+1]-(x[idx])**2)**2

    assert val == rosen(x)
    return val


def rosen_grad(x):
    g = np.zeros_like(x)
    # [-2(1-x_1 )+200(x_2-x_1^2 )(-2x_1 )]
    g[0] = -2*(1-x[0])+200*(x[1]-x[0]**2)*(-2*x[0])
    for i in range(1, len(x)-1):
        g[i] = (200*(x[i]-x[i-1]**2))+(-2*(1-x[i])+200*(x[i+1]-x[i]**2)*(-2*x[i]))
        # same: g[i] = 202*x[i]-200*x[i-1]**2-2-400*x[i]*x[i+1]+400*x[i]**3
    # [200(x_N-x_(N-1)^2)]
    g[len(x)-1] = 200*(x[len(x)-1]-x[len(x)-2]**2)

    # Testing:
    # numeric_par = {'epsilon': 2 * pow(10, -16),
    #                'f_par': {"dummy": -666},
    #                'gradient': rosen_grad}
    # num_grad = numdiff(rosen_val, x, numeric_par)
    # np.testing.assert_almost_equal(g, num_grad, 0)

    return g


