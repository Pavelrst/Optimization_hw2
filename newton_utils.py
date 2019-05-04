import numpy as np
import matplotlib.pyplot as plt
from mcholmz import modifiedChol


class NewtonMethod:
    @staticmethod
    def direction(self, x, func):
        L, d, e = modifiedChol(func.hess(x))
        D = np.diag(d.flatten())
        L_T = np.transpose(L)

        # backward substitution
        y = np.linalg.solve(L, func.grad(x))

        # diagonal system
        z = np.linalg.solve(D, y)

        # backward substitution
        d = np.linalg.solve(L_T, z)
        return d

    @staticmethod
    def direction_cheat(self, x, func):
        # We are not using this function as we don't want to inverse Hessian.
        H_inv = np.linalg.inv(func.hess(x))
        return np.matmul(H_inv, func.grad(x))

