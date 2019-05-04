import numpy as np
import matplotlib.pyplot as plt
from functions_utils import *

from functions_utils import Rosenbrock
from functions_utils import Quadratic
from numdiff_utils import numdiff
from Optimizer_utils import Gradient_descent
from armijo_utils import Armijo_method

def main():
    opt_grad = Gradient_descent(method_type='steepest_descent', max_steps=100000, verbose=False)
    opt_newton = Gradient_descent(method_type='newton', max_steps=100000, verbose=False)

    f = Rosenbrock(10)
    opt_grad.optimize(f, f.starting_point())
    iterations_list1, converg_list1 = opt_grad.get_convergence(f.optimal())
    opt_newton.optimize(f, f.starting_point())
    iterations_list2, converg_list2 = opt_newton.get_convergence(f.optimal())

    plt.plot(iterations_list1, converg_list1, label='Steepest descent')
    plt.plot(iterations_list2, converg_list2, label='Newton method')
    plt.legend()
    plt.title('Rosenbrock(10)')
    plt.ylabel('f(x)-f* / log')
    plt.xlabel('iterations')
    plt.yscale('log')
    plt.show()

    f = Quadratic('well')
    opt_grad.optimize(f, f.starting_point())
    iterations_list1, converg_list1 = opt_grad.get_convergence(f.optimal())
    opt_newton.optimize(f, f.starting_point())
    iterations_list2, converg_list2 = opt_newton.get_convergence(f.optimal())

    plt.plot(iterations_list1, converg_list1, label='Steepest descent')
    plt.plot(iterations_list2, converg_list2, label='Newton method')
    plt.legend()
    plt.title('Well conditioned Quadratic')
    plt.ylabel('f(x)-f* / log')
    plt.xlabel('iterations')
    plt.yscale('log')
    plt.show()

    f = Quadratic('ill')
    opt_grad.optimize(f, f.starting_point())
    iterations_list1, converg_list1 = opt_grad.get_convergence(f.optimal())
    opt_newton.optimize(f, f.starting_point())
    iterations_list2, converg_list2 = opt_newton.get_convergence(f.optimal())

    plt.plot(iterations_list1, converg_list1, label='Steepest descent')
    plt.plot(iterations_list2, converg_list2, label='Newton method')
    plt.legend()
    plt.title('Ill conditioned Quadratic')
    plt.ylabel('f(x)-f* / log')
    plt.xlabel('iterations')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()

