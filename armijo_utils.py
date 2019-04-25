import numpy as np


def calc_step_size(x, func_val, func_grad, alpha=1, sigma=0.25, beta=0.5):
    '''
    Denotre Phi(alpha) as function of step size: Phi(alpha)=f(x+alpha*d)-f(x)
    while d is direction. Deriviate Phi by alpha and we get:
    Phi'(alpha) =  f'(x+alpha*d)*d so,
    Phi'(0) =  f'(x)*d i.e. directional derivative, in our case - gradient
    Denote: c = Phi'(0) = f_grad(x), so
    alpha*c is a tangent line to Phi(alpha) at alpha=0.
    Now lets make a new line: sigma*alpha*c
    Finally let's state to Armijo condition:
        f(x+alpha*d)-f(x) <= sigma*alpha*c
    Also we assume search direction d is gradient direction.
    :param x: the point from which we want make a step
    :param alpha: Initial step size
    :param sigma: Factor for a slope of tangent
    :param beta: Step size decrement factor
    :return: step size which is fulfilling Armijo condition.
    '''

    # f(x+alpha*d)-f(x)
    left_armijo = func_val(x + alpha*func_grad(x)) - func_val(x)

    # sigma*alpha*c
    c = np.matmul(np.ones(len(x)), func_grad(x))
    right_armijo = sigma*alpha*c

    # f(x+alpha*d)-f(x) <= sigma*alpha*c
    while left_armijo > right_armijo:
        print("left_armijo > right_armijo:", left_armijo, " > ", right_armijo)
        # Decrement alpha by factor beta
        alpha = alpha * beta
        right_armijo = sigma * alpha * c
        left_armijo = func_val(x + alpha * func_grad(x)) - func_val(x)
