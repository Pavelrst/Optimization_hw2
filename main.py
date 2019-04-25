import numpy as np
import matplotlib.pyplot as plt
from functions_utils import *

from functions_utils import quad_val_well
from functions_utils import quad_grad_well
from functions_utils import quad_val_ill
from functions_utils import quad_grad_ill
from numdiff_utils import numdiff
from armijo_utils import calc_step_size

def gradient_descent(x, func_val, func_grad, step_size, acc=0.00001, max_steps=1000, graphic=True):
    '''
    :param x: starting point of algo
    :param func_val: pointer to function we wish to optimize
    :param func_grad: pointer to grad of function we wish to optimize
    :param step_size: ste size of gradient descent
    :param acc: accuracy threshold for early stopping
    :param max_steps: max iterations of gradient descent
    :return: position of optimal point
    '''
    f_list = []
    steps_list = []

    step_size = calc_step_size(x, func_val, func_grad, graphic=False)

    for step in range(max_steps):
        f_list.append(func_val(x))
        x = x - step_size * func_grad(x)
        step_size = calc_step_size(x, func_val, func_grad, graphic=False)
        steps_list.append(step_size)

        print("next step size = ", step_size, "  current point =", x)

    if graphic:
        plot_convergence(f_list, rosen_optimal(len(x)))
        plot_stepsizes(steps_list)

    return x


def plot_stepsizes(steps_list):
    '''
    plots the behavior of step sizes.
    :param f_list: list of values of f during gradient descent algo
    :param val_optimal: the global minimum value of the function
    '''
    iterations_list = range(len(steps_list))

    a, = plt.plot(iterations_list, steps_list, label='step size')
    plt.legend(handles=[a])
    plt.ylabel('step size')
    plt.xlabel('iterations')
    plt.show()


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
    #x = np.array([1.9, 2, 1.7, 2, 2.6, 1.9, 2, 2.2, 2, 1])
    gradient_descent(x, rosen_val, rosen_grad, 0.0001)
    # quad_val_well(x)
    # quad_grad_well(x)
    # quad_val_ill(x)
    # quad_grad_ill(x)

if __name__ == "__main__":
    main()

