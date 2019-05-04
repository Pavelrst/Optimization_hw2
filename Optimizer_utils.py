import numpy as np
import matplotlib.pyplot as plt
from functions_utils import *
from armijo_utils import Armijo_method
from newton_utils import NewtonMethod


class Gradient_descent():
    def __init__(self, method_type='steepest_descent', threshold=0.00001, step_size_estimator=Armijo_method(),
                 max_steps=100000, verbose=True):
        self.threshold = threshold
        self.max_steps = max_steps
        self.verbose = verbose
        self.f_val_list = []
        self.step_sizes_list = []
        self.step_size_estimator = step_size_estimator
        self.method_type = method_type

    def optimize(self, func, start_point):
        self.f_val_list = []
        self.step_sizes_list = []

        step_size = self.step_size_estimator.calc_step_size(start_point, func)
        x = start_point

        for step in range(self.max_steps):
            prev_x = x
            if self.method_type == 'steepest_descent':
                step_size, x = self.optimizer_step(x, func, step_size)
            elif self.method_type == 'newton':
                step_size, x = self.optimizer_step_newton(x, func, step_size)
            else:
                print("Direction method not selected")
                break
            if self.verbose:
                print("f(x)=", func.val(x), "next step size= ~", np.round(step_size, 5), " current point= ~", np.round(x, 5))
            else:
                print("step:",step)

            if np.linalg.norm(func.grad(x)) < self.threshold:
                print("Optimizer reached accuracy threshold after", step, "iterations!")
                break

    def optimizer_step(self, x, func, step_size):
        self.f_val_list.append(func.val(x))
        x = x - step_size * func.grad(x)
        next_step_size = self.step_size_estimator.calc_step_size(x, func)
        self.step_sizes_list.append(next_step_size)
        return next_step_size, x

    def optimizer_step_newton(self, x, func, step_size):
        newton = NewtonMethod()
        self.f_val_list.append(func.val(x))
        d = newton.direction(x, func)
        x = x + step_size * d
        next_step_size = self.step_size_estimator.calc_step_size(x, func)
        self.step_sizes_list.append(next_step_size)
        return next_step_size, x


    def plot_stepsizes(self):
        iterations_list = range(len(self.step_sizes_list))

        a, = plt.plot(iterations_list, self.step_sizes_list, label='step size')
        plt.legend(handles=[a])
        plt.ylabel('step size')
        plt.xlabel('iterations')
        plt.show()

    def get_convergence(self, val_optimal):
        '''
        gets converg rates list
        :param f_list: list of values of f during gradient descent algo
        :param val_optimal: the global minimum value of the function
        '''
        converg_list = []
        iterations_list = []
        for idx, val in enumerate(self.f_val_list):
            converg_list.append(val - val_optimal)
            iterations_list.append(idx)

        return iterations_list, converg_list

    def plot_convergence(self, val_optimal):
        '''
        plots the convergence rate
        :param f_list: list of values of f during gradient descent algo
        :param val_optimal: the global minimum value of the function
        '''
        converg_list = []
        iterations_list = []
        for idx, val in enumerate(self.f_val_list):
            converg_list.append(val - val_optimal)
            iterations_list.append(idx)

        plt.plot(iterations_list, converg_list)
        plt.ylabel('f(x)-f* / log')
        plt.xlabel('iterations')
        plt.yscale('log')
        plt.show()

