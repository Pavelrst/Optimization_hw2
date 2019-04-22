import numpy as np
import matplotlib.pyplot as plt
from functions_utils import *

from functions_utils import quad_val_well
from functions_utils import quad_grad_well
from functions_utils import quad_val_ill
from functions_utils import quad_grad_ill
from numdiff_utils import numdiff


def gradient_descent(x, func_val, func_grad, step_size, acc=0.00001, max_steps=100000):

    f_list = []
    for step in range(max_steps):
        f_list.append(func_val(x))
        x = x - step_size * func_grad(x)

    plot_convergence(f_list, rosen_optim_pos(len(x)))


def plot_convergence(f_list, val_optimal):
    '''
    plots the convergence rate
    :param f_list: list of values of f during gradient descent algo
    :param val_optimal: the global minimum value of the function
    '''
    converg_list = []
    iterations_list = []
    for idx, val in enumerate(f_list):
        converg_list.append(val - val_optimal)
        iterations_list.append(idx)

    plt.plot(iterations_list, converg_list)
    plt.ylabel('f(x)-f*')
    plt.xlabel('iterations')
    plt.yscale('log')
    plt.show()



def main():
    x = np.array([0,0,0,0,0,0,0,0,0,0])
    gradient_descent(x, rosen_val, rosen_grad, 0.0001)
    # quad_val_well(x)
    # quad_grad_well(x)
    # quad_val_ill(x)
    # quad_grad_ill(x)

if __name__ == "__main__":
    main()

