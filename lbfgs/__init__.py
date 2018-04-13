"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.
"""

from ._lowlevel import LBFGS, LBFGSError


def fmin_lbfgs(f, x0, progress=None, args=(), orthantwise_c=0,
               orthantwise_start=0, orthantwise_end=-1, m=6,
               epsilon=1e-5, past=0, delta=0., max_iterations=0,
               line_search="default", max_linesearch=20, min_step=1e-20,
               max_step=1e+20, ftol=1e-4, wolfe=0.9, gtol=0.9, xtol=1e-30):
    """Minimize a function using LBFGS or OWL-QN

     Parameters
    ----------
    f : callable(x, g, *args)
        Computes function to minimize and its gradient.
        Called with the current position x (a numpy.ndarray), a gradient
        vector g (a numpy.ndarray) to be filled in and *args.
        Must return the value at x and set the gradient vector g.
    x0 : array-like
        Initial values. A copy of this array is made prior to optimization.

    progress : callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args),
               optional
        If not None, called at each iteration after the call to f with the
        current values of x, g and f(x), the L2 norms of x and g, the line
        search step, the iteration number, the number of evaluations at
        this iteration and args (see below).
        If the return value from this callable is not 0 and not None,
        optimization is stopped and LBFGSError is raised.

    args : sequence
        Arbitrary list of arguments, passed on to f and progress as *args.

    orthantwise_c: float, optional (default=0)
        Coefficient for the L1 norm of variables.
        This parameter should be set to zero for standard minimization
        problems. Setting this parameter to a positive value activates
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
        minimizes the objective function F(x) combined with the L1 norm |x|
        of the variables, {F(x) + C |x|}. This parameter is the coefficient
        for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
        zero, the library modifies function and gradient evaluations from
        a client program suitably; a client program thus have only to return
        the function value F(x) and gradients G(x) as usual. The default value
        is zero.

    orthantwise_start: int, optional (default=0)
        Start index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter b (0 <= b < N)
        specifies the index number from which the library computes the
        L1 norm of the variables x,
            |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
        In other words, variables x_1, ..., x_{b-1} are not used for
        computing the L1 norm. Setting b (0 < b < N), one can protect
        variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
        regression) from being regularized. The default value is zero.

    orthantwise_end: int, optional (default=-1)
        End index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter e (0 < e <= N)
        specifies the index number at which the library stops computing the
        L1 norm of the variables x,

    m: int, optional, default=6
        The number of corrections to approximate the inverse hessian matrix.
        The L-BFGS routine stores the computation results of previous `m`
        iterations to approximate the inverse hessian matrix of the current
        iteration. This parameter controls the size of the limited memories
        (corrections). The default value is 6. Values less than 3 are
        not recommended. Large values will result in excessive computing time.

    epsilon: float, optional (default=1e-5)
        Epsilon for convergence test.
        This parameter determines the accuracy with which the solution is to
        be found. A minimization terminates when
            ||g|| < \ref epsilon * max(1, ||x||),
        where ||.|| denotes the Euclidean (L2) norm. The default value is
        1e-5.

    past: int, optional (default=0)
        Distance for delta-based convergence test.
        This parameter determines the distance, in iterations, to compute
        the rate of decrease of the objective function. If the value of this
        parameter is zero, the library does not perform the delta-based
        convergence test. The default value is 0.

    delta: float, optional (default=0.)
        Delta for convergence test.
        This parameter determines the minimum rate of decrease of the
        objective function. The library stops iterations when the
        following condition is met:
            (f' - f) / f < delta,
        where f' is the objective value of `past` iterations ago, and f is
        the objective value of the current iteration.
        The default value is 0.

    max_iterations: int, optional (default=0)
        The maximum number of iterations. Setting this parameter to zero continues an
        optimization process until a convergence or error. The default value
        is 0.

    line_search: str, optional (default="default")
        The line search algorithm.
        This parameter specifies a line search algorithm to be used by the
        L-BFGS routine. Possible values are:

        - 'default': same as 'morethuente'
        - 'morethuente': Method proposed by More and Thuente
        - 'armijo': backtracking with Armijo's conditions
        - 'wolfe': backtracking with Wolfe's conditions
        - 'strongwolfe': backtracking with strong Wolfe's conditions

    max_linesearch: int, optional (default=20)
        The maximum number of trials for the line search.
        This parameter controls the number of function and gradients evaluations
        per iteration for the line search routine. The default value is 20.

    min_step: float, optional (default=1e-20)
        The minimum step of the line search routine.
        The default value is 1e-20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    max_step: float, optional (default=1e20)
        The maximum step of the line search.
        The default value is 1e+20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    ftol: float, optional (default=1e-4)
        A parameter to control the accuracy of the line search routine.
        The default value is 1e-4. This parameter should be greater
        than zero and smaller than 0.5.

    wolfe: float, optional (default=0.9)
        A coefficient for the Wolfe condition. This parameter is valid only
        when the backtracking line-search algorithm is used with the Wolfe
        condition (`line_search='wolfe'` or `line_search='strongwolfe'`),
        The default value is 0.9. This parameter should be greater
        the `ftol` parameter and smaller than 1.0.

    gtol: float, optional (default=0.9)
        A parameter to control the accuracy of the line search routine.
        The default value is 0.9. If the function and gradient
        evaluations are inexpensive with respect to the cost of the
        iteration (which is sometimes the case when solving very large
        problems) it may be advantageous to set this parameter to a small
        value. A typical small value is 0.1. This parameter should be
        greater than the ftol parameter (1e-4) and smaller than
        1.0.


    xtol: float, optional (default=1e-30)
        The machine precision for floating-point values.
        This parameter must be a positive value set by a client program to
        estimate the machine precision. The line search routine will terminate
        with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
        of the interval of uncertainty is less than this parameter.


    """
    opt = LBFGS()
    opt.orthantwise_c = orthantwise_c
    opt.orthantwise_start = orthantwise_start
    opt.orthantwise_end = orthantwise_end
    opt.m = m
    opt.epsilon = epsilon
    opt.past = past
    opt.delta = delta
    opt.max_iterations = max_iterations
    opt.linesearch = line_search
    opt.max_linesearch = max_linesearch
    opt.min_step = min_step
    opt.max_step = max_step
    opt.ftol = ftol
    opt.wolfe = wolfe
    opt.gtol = gtol
    opt.xtol = xtol

    return opt.minimize(f, x0, progress=progress, args=args)
