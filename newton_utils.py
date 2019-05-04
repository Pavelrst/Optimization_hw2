import numpy as np
import matplotlib.pyplot as plt
from mcholmz import modifiedChol


class NewtonMethod:
    def direction(self, x, func):
        L, d, e = modifiedChol(func.hess(x))
        D = np.diag(d.flatten())
        L_T = np.transpose(L)

        # backward substitution
        y = np.linalg.solve(L, -func.grad(x))

        # daigonal system
        z = np.linalg.solve(D, y)

        # backward substitution
        d = np.linalg.solve(L_T, z)
        return d

    def direction_cheat(self, x, func):
        H_inv = np.linalg.inv(func.hess(x))
        return np.matmul(-H_inv, func.grad(x))

