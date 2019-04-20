import numpy as np
import matplotlib.pyplot as plt
from functions_utils import rosen_val
from functions_utils import rosen_grad
from numdiff_utils import numdiff


def gradient_descent(x, func_val, func_grad, step_size, acc=0.00001, max_steps=100000):

    for step in range(max_steps):
        x = x - step_size * func_grad(x)
        print(x)



def main():
    x = np.array([0, 0, 0])
    gradient_descent(x, rosen_val, rosen_grad, 0.0001)


if __name__ == "__main__":
    main()

