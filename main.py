from functions_utils import Rosenbrock
from functions_utils import Quadratic
from Optimizer_utils import Gradient_descent


def main():
    opt_grad = Gradient_descent(method_type='steepest_descent', max_steps=100000, verbose=False)
    opt_newton = Gradient_descent(method_type='newton_method', max_steps=100000, verbose=True)

    f = Rosenbrock(10)
    opt_grad.optimize(f, f.starting_point())
    opt_grad.plot_convergence(f.optimal(), f.name)
    opt_newton.optimize(f, f.starting_point())
    opt_newton.plot_convergence(f.optimal(), f_name=f.name, marker=(13, 1.708))

    f = Quadratic('well')
    opt_grad.optimize(f, f.starting_point())
    opt_grad.plot_convergence(f.optimal(), f.name)
    opt_newton.optimize(f, f.starting_point())
    opt_newton.plot_convergence(f.optimal(), f.name)

    f = Quadratic('ill')
    opt_grad.optimize(f, f.starting_point())
    opt_grad.plot_convergence(f.optimal(), f.name)
    opt_newton.optimize(f, f.starting_point())
    opt_newton.plot_convergence(f.optimal(), f.name)


if __name__ == "__main__":
    main()

