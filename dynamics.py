import torch
import warnings

# import logging
# logger = logging.getLogger('logger')
# logger.warning('prova')

__all__ = ['dynamics']


def dynamics(W, X, tol=1e-6, max_iter=5, mode='replicator', **kwargs):
    """
    Selector for dynamics
    Input:
    W:  the pairwise nxn similarity matrix (with zero diagonal)
    X:  an (n,m)-array whose rows reside in the n-dimensional simplex. If
        an (n,)-array is provided, it must reside in the n-dimensional simplex
    tol:  error tolerance
    max_iter:  maximum number of iterations
    mode: 'replicator' to run classical replicator dynamics,
          'exponential' to run exponential replicator dynamics.
          'inf_imm' to run infection-immunization dynamics (not implemented).
    """

    if mode == 'replicator':
        X = _replicator(W, X, tol, max_iter)
    elif mode == 'exponential':
        X = _exponential(W, X, tol, max_iter, kwargs.get('k', 1.))
    elif mode == 'inf_imm':
        if X.dim == 2:
            raise ValueError('Currently, only one-dimensional vectors are '
                             'accepted with \'inf_imm\' mode')
        X = _inf_imm(W, X, tol, max_iter)
    else:
        raise ValueError('mode \'' + mode + '\' is not defined.')

    return X


def _replicator(W, X, tol, max_iter):
    """
    Replicator Dynamics
    Output:
    X:  the population(s) at convergence
    i:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """

    i = 0
    while i < max_iter:
        X = X * torch.matmul(W, X)
        # z = X.register_hook(lambda g: print(g))
        # print(z)
        X /= X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1)
        # z = X.register_hook(lambda g: print(g))
        # print(z)

        i += 1

    return X


def _exponential(W, X, tol, max_iter, k):
    """
    Exponential Replicator Dynamics
    Input:
    k: the "acceleration" parameter of the dynamical system
    Output:
    x:  the population(s) at convergence
    iter:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """

    err = 2. * tol
    i = 0
    while err > tol and i < max_iter:
        X_old = X
        X = X * torch.exp(k * (torch.matmul(W, X) - 1.))  # softmax trick
        X /= X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1)

        err = torch.norm(X - X_old)
        i += 1

    if i == max_iter:
        warnings.warn("Maximum number of iterations reached.")

    return X


def _inf_imm(W, X, tol, max_iter):
    """
    Infection Immunization Dynamics
    Output:
    x:  the population(s) at convergence
    iter:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """
    dtype = X.dtype  # casting dtype for ByteTensors

    WX = torch.matmul(W, X)
    XWX = torch.matmul(WX, X)
    r = WX - XWX

    # TODO: check Nash error
    err = (torch.max(X, r) ** 2.).sum()
    i = 0

    while err > tol and i < max_iter:
        max_, imax = torch.max(r, dim=0)
        min_, imin = torch.min(r * (X > 0.).to(dtype), dim=0)  # TODO: check
        infective = imax if max_ > -min_ else imin
        den = W[infective, infective] - WX[infective] - r[infective]

        do_remove = False
        if r[infective] >= 0.:
            mu = 1.
            if den < 0:
                opt_delta = -r[infective] / den
                if opt_delta < mu:
                    mu = opt_delta
                # if mu < 0.: mu = 0.
        else:
            do_remove = True
            mu = X[infective] / (X[infective] - 1.)
            if den < 0.:
                opt_delta = -r[infective] / den
                if opt_delta >= mu:
                    mu = opt_delta
                    do_remove = False

        n = X.shape[0]
        X = mu * ((torch.arange(n, device=X.device) == infective).to(
            dtype) - X) + X

        if do_remove:
            X[infective] = 0.

        WX = mu * (W[infective, :] - WX) + WX

        XWX = torch.matmul(X, WX)
        r = WX - XWX

        err = (torch.max(X, r) ** 2.).sum()
        i += 1

    if i == max_iter:
        warnings.warn("Maximum number of iterations reached.")

    return X