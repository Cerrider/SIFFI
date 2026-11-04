#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Antti Kukkurainen
# Finnish Meteorological Institute
# Correspondence: antti.kukkurainen@fmi.fi

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lapack
from scipy.optimize import Bounds, OptimizeResult, minimize


ModelFunction = Callable[[np.ndarray, np.ndarray, dict], np.ndarray]

# Default solver is 'GN'. If you need bounded solution, use 'GN-B'.
SOLVERS = ('GN', 'GN-B', 'BFGS', 'L-BFGS-B')
BOUNDED_SOLVERS = ('GN-B', 'L-BFGS-B')


class BoundWarning(UserWarning):
    """The estimate is at a box constraint for some wavelengths."""


#%% Settings

@dataclass
class RetrievalConfig:
    """
    Configuration options for SIFFI retrieval.

    Parameters
    ----------
    wavelength : ndarray, shape (n,)
        Wavelength values.
    radiance : ndarray, shape (n,)
        Observed radiance values.
    error_mean : ndarray, shape (n,) or float, optional
        Mean of the instrument specific error model. Default is 0.
    error_cov : ndarray, shape (n, n)
        Covariance of the instrument specific error model.
        With standard deviation ``sigma`` of the noise, use::

            numpy.diag(sigma**2)

    surf_prior_mean : ndarray, shape (n,)
        Mean of the surface reflectance prior model.
    surf_prior_cov : ndarray, shape (n, n)
        Covariance of the surface reflectance prior model.
    sif_prior_mean : ndarray, shape (n,)
        Mean of the SIF prior model.
    sif_prior_cov : ndarray, shape (n, n)
        Covariance of the SIF prior model.
    forward_model : callable
        Forward model returning the modelled radiance, shape (n,)::

            forward_model(surface_reflectance, sif, fmodel_kwargs)

        The model must be wavelength-wise, i.e. the radiance at wavelength
        ``i`` depends only on ``surface_reflectance[i]`` and ``sif[i]``.
    jacobian : callable
        Derivatives of the forward model, shape (2, n)::

            jacobian(surface_reflectance, sif, fmodel_kwargs)

        Row 0 is dL/d(surface reflectance) and row 1 is dL/d(SIF).
    fmodel_kwargs : dict, optional
        Additional parameters passed to ``forward_model`` and ``jacobian``.
    use_approximation_error_model : bool, optional
        Whether to use the approximation error method. Default is False.
    ae_mean : ndarray, shape (n,) or float, optional
        Mean of the approximation error model.
    ae_cov : ndarray, shape (n, n), optional
        Covariance of the approximation error model.
    surf_initial_guess : ndarray, shape (n,) or float, optional
        Initial guess for surface reflectance. Default is 0.
    sif_initial_guess : ndarray, shape (n,) or float, optional
        Initial guess for SIF. Default is 0.
    solver : {'GN', 'GN-B', 'BFGS', 'L-BFGS-B'}, optional
        Optimization method. Default is 'GN', Gauss-Newton with
        Levenberg-Marquardt damping. 'GN-B' is Gauss-Newton with the box
        constraints below (projected Newton). 'BFGS' and 'L-BFGS-B' use
        ``scipy.optimize.minimize``. Only 'GN-B' and 'L-BFGS-B' apply the
        box constraints.
    surf_bounds : (lower, upper), optional
        Box constraints for surface reflectance, used with 'GN-B' and
        'L-BFGS-B'. Each limit is a float or an array of shape (n,).
        Default is (0, 1).
    sif_bounds : (lower, upper), optional
        Box constraints for SIF, used with 'GN-B' and 'L-BFGS-B'.
        Each limit is a float or an array of shape (n,). Default is (0, 6).
    gn_tol : float, optional
        Convergence tolerance for 'GN' and 'GN-B'. Iteration stops when the
        Gauss-Newton step, measured in units of posterior standard deviation
        (Mahalanobis norm), is below ``gn_tol``. Default is 1e-5.
    gtol : float, optional
        Gradient norm tolerance for 'BFGS' and 'L-BFGS-B'. Default is 1e-10.
    ftol : float, optional
        Relative cost function tolerance, used only with 'L-BFGS-B'.
        Default is 2.22e-14.
    maxiter : int, optional
        Maximum number of solver iterations. Default is 1000000.
    verbose : bool, optional
        Print progress and a summary of the optimization. Default is False.
    """

    # Data
    wavelength: np.ndarray | None = None
    radiance: np.ndarray | None = None

    # Instrument specific error model
    error_mean: np.ndarray | float = 0
    error_cov: np.ndarray | None = None

    # Prior model
    surf_prior_mean: np.ndarray | None = None
    surf_prior_cov: np.ndarray | None = None
    sif_prior_mean: np.ndarray | None = None
    sif_prior_cov: np.ndarray | None = None

    # Forward model
    forward_model: ModelFunction | None = None
    jacobian: ModelFunction | None = None
    fmodel_kwargs: dict[str, Any] = field(default_factory=dict)

    # Approximation error model
    use_approximation_error_model: bool = False
    ae_mean: np.ndarray | float = 0
    ae_cov: np.ndarray | float = 0

    # Initial guess
    surf_initial_guess: np.ndarray | float = 0
    sif_initial_guess: np.ndarray | float = 0

    # Solver
    solver: str = 'GN'
    surf_bounds: tuple = (0, 1)
    sif_bounds: tuple = (0, 6)
    gn_tol: float = 1e-5
    gtol: float = 1e-10
    ftol: float = 2.22e-14
    maxiter: int = 1000000
    verbose: bool = False


#%% Retrieval code

def retrieval(configuration: RetrievalConfig) -> dict:
    """
    SIFFI retrieval main routine.

    Finds the maximum a posteriori estimate of surface reflectance and SIF
    and their posterior standard deviations (Laplace approximation).

    Parameters
    ----------
    configuration : siffi.RetrievalConfig
        Contains inputs and options for the retrieval.

    Returns
    -------
    dict
        Retrieval output with keys ``wavelength``, ``surf_estimate``,
        ``sif_estimate``, ``surf_std``, ``sif_std``, ``radiance_estimate``,
        ``opt_output`` (scipy ``OptimizeResult``), ``cost_history`` (cost
        function value at every evaluation), ``optimization_time`` [s],
        ``surf_at_bound`` and ``sif_at_bound``.

        The ``*_at_bound`` arrays flag wavelengths where a bounded solver
        stopped at a box constraint: -1 at the lower bound, +1 at the upper
        bound, 0 otherwise (always 0 for unbounded solvers). The posterior
        standard deviation is unreliable at flagged wavelengths, because the
        Gaussian (Laplace) approximation assumes an interior minimum.
    """
    
    cfg = configuration
    _validate(cfg)

    if cfg.verbose:
        print('Running retrieval...')

    problem = _Problem(cfg)
    n = problem.n
    x0 = np.concatenate((np.ones(n) * cfg.surf_initial_guess,
                         np.ones(n) * cfg.sif_initial_guess))

    t_start = time.perf_counter()
    if cfg.solver == 'GN':
        opt = _gauss_newton(problem, x0, cfg.gn_tol, cfg.maxiter)
    elif cfg.solver == 'GN-B':
        opt = _gauss_newton_bounded(problem, x0, _bounds(cfg, n), cfg.gn_tol, cfg.maxiter)
    elif cfg.solver == 'BFGS':
        opt = minimize(problem.cost_and_gradient, x0, jac=True, method='BFGS',
                       options={'maxiter': cfg.maxiter, 'gtol': cfg.gtol})
    else:
        opt = minimize(problem.cost_and_gradient, x0, jac=True, method='L-BFGS-B',
                       bounds=_bounds(cfg, n),
                       options={'maxiter': cfg.maxiter, 'maxfun': 10 * cfg.maxiter,
                                'ftol': cfg.ftol, 'gtol': cfg.gtol,
                                'maxcor': 140})
    optimization_time = time.perf_counter() - t_start

    if cfg.verbose:
        print(opt.message)
        print(f'Total number of iterations: {opt.nit}')
        print(f'Cost function value: {opt.fun}')
        print(f'Runtime of optimization was {optimization_time:.3f} s')

    surf_est, sif_est = opt.x[:n], opt.x[n:]
    radiance_est = cfg.forward_model(surf_est, sif_est, cfg.fmodel_kwargs)
    if 'hess_factor' in opt:
        factor = opt.pop('hess_factor')  # GN: Hessian already factored at opt.x
    else:
        d_surf, d_sif = cfg.jacobian(surf_est, sif_est, cfg.fmodel_kwargs)
        factor = _cho_factor(problem.hessian(d_surf, d_sif), 'posterior precision')
    post_std = np.sqrt(_inv_diag(factor))

    at_bound = np.zeros(2 * n, dtype=np.int8)
    if cfg.solver in BOUNDED_SOLVERS:
        at_bound = _at_bound(opt.x, _bounds(cfg, n), post_std)
        if np.any(at_bound):
            warnings.warn('Estimate is at a box constraint for some wavelengths; '
                          'posterior std is unreliable there '
                          '(see surf_at_bound and sif_at_bound).',
                          BoundWarning, stacklevel=2)

    return {'wavelength': cfg.wavelength,
            'surf_estimate': surf_est,
            'sif_estimate': sif_est,
            'surf_std': post_std[:n],
            'sif_std': post_std[n:],
            'radiance_estimate': radiance_est,
            'opt_output': opt,
            'cost_history': problem.cost_history,
            'optimization_time': optimization_time,
            'surf_at_bound': at_bound[:n],
            'sif_at_bound': at_bound[n:],
            }


class _Problem:
    """
    Cost function of the retrieval: two times the negative log-posterior

        C(x) = |y - F(x)|^2_W + |x_surf - mu_surf|^2_P_surf + |x_sif - mu_sif|^2_P_sif

    where x = (surface reflectance, SIF) and W, P_* are precision matrices.
    """

    def __init__(self, cfg: RetrievalConfig):
        self.n = len(cfg.radiance)
        self.forward_model = cfg.forward_model
        self.jacobian = cfg.jacobian
        self.fmodel_kwargs = cfg.fmodel_kwargs

        # Total noise model: instrument error (+ approximation error)
        noise_mean = cfg.error_mean
        noise_cov = cfg.error_cov
        if cfg.use_approximation_error_model:
            if cfg.verbose:
                print('Using approximation error method!')
            noise_mean = noise_mean + cfg.ae_mean
            noise_cov = noise_cov + cfg.ae_cov

        self.y = cfg.radiance - noise_mean
        self.noise_prec = _inv_spd(noise_cov, 'noise covariance')
        self.surf_prec = _inv_spd(cfg.surf_prior_cov, 'surface reflectance prior covariance')
        self.sif_prec = _inv_spd(cfg.sif_prior_cov, 'SIF prior covariance')
        self.surf_mean = cfg.surf_prior_mean
        self.sif_mean = cfg.sif_prior_mean
        self.cost_history = []

    def evaluate(self, x):
        """Cost, gradient and forward model derivatives (d_surf, d_sif) at x."""
        surf, sif = x[:self.n], x[self.n:]
        residual = self.y - self.forward_model(surf, sif, self.fmodel_kwargs)
        w_residual = self.noise_prec @ residual
        surf_dev = self.surf_prec @ (surf - self.surf_mean)
        sif_dev = self.sif_prec @ (sif - self.sif_mean)

        cost = (residual @ w_residual
                + (surf - self.surf_mean) @ surf_dev
                + (sif - self.sif_mean) @ sif_dev)

        # The Jacobian is diagonal in wavelength, so K^T W r is elementwise
        d_surf, d_sif = self.jacobian(surf, sif, self.fmodel_kwargs)
        gradient = 2 * np.concatenate((surf_dev - d_surf * w_residual,
                                       sif_dev - d_sif * w_residual))

        self.cost_history.append(cost)
        return cost, gradient, (d_surf, d_sif)

    def cost_and_gradient(self, x):
        cost, gradient, _ = self.evaluate(x)
        return cost, gradient

    def hessian(self, d_surf, d_sif):
        """Gauss-Newton Hessian K^T W K + prior precision (half of the cost Hessian)."""
        
        W = self.noise_prec
        return np.block([
            [d_surf[:, None] * W * d_surf + self.surf_prec, d_surf[:, None] * W * d_sif],
            [d_sif[:, None] * W * d_surf, d_sif[:, None] * W * d_sif + self.sif_prec],
        ])


def _gauss_newton(problem: _Problem, x0: np.ndarray, tol: float,
                  maxiter: int) -> OptimizeResult:
    """
    Minimize the cost with Gauss-Newton iterations and Levenberg-Marquardt
    damping (Nielsen's update rule). Plain Gauss-Newton steps are used until
    a step fails to decrease the cost; only then is damping switched on.
    
    Reference: H. B. Nielsen, Damping parameter in Marquardt’s method,
    Tech. Rep. IMM-REP-1999-05, Technical University of Denmark, (1999).
    
    """
    
    x = x0.astype(float)
    cost, gradient, jac = problem.evaluate(x)
    if not (np.isfinite(cost) and np.all(np.isfinite(jac))):
        raise ValueError('Cost function or Jacobian is not finite at the initial guess')
    nfev = 1
    damping, growth = 0.0, 2.0
    status, message = 1, 'Maximum number of iterations reached.'

    for nit in range(1, maxiter + 1):
        hess = problem.hessian(*jac)
        factor = cho_factor(hess)
        step = cho_solve(factor, -gradient / 2)

        # Step length in posterior standard deviations: sqrt(step^T H step)
        if np.sqrt(-gradient @ step / 2) < tol:
            status, message = 0, 'Optimization terminated successfully.'
            nit -= 1  # converged before taking a step on this round
            break

        while True:
            if damping > 0:
                damped = hess + damping * np.diag(np.diag(hess))
                step = cho_solve(cho_factor(damped), -gradient / 2)
            trial_cost, trial_gradient, trial_jac = problem.evaluate(x + step)
            nfev += 1

            # Gain ratio: actual vs. predicted decrease of the quadratic model
            predicted = -(gradient @ step + step @ hess @ step)
            gain = (cost - trial_cost) / predicted
            if gain > 0:  # also False when trial_cost is nan
                break
            damping = 1e-3 if damping == 0 else growth * damping
            growth *= 2
            if damping > 1e10:
                return OptimizeResult(x=x, fun=cost, jac=gradient, nit=nit, nfev=nfev,
                                      status=2, success=False, hess_factor=factor,
                                      message='Damping too large: cost cannot be decreased.')

        x = x + step
        cost, gradient, jac = trial_cost, trial_gradient, trial_jac
        damping *= max(1 / 3, 1 - (2 * gain - 1)**3)
        growth = 2.0
        if damping < 1e-7:
            damping = 0.0

    if status != 0:  # the last accepted step moved x, refactor there
        factor = cho_factor(problem.hessian(*jac))
    return OptimizeResult(x=x, fun=cost, jac=gradient, nit=nit, nfev=nfev,
                          status=status, success=status == 0, message=message,
                          hess_factor=factor)


def _gauss_newton_bounded(problem: _Problem, x0: np.ndarray, bounds: Bounds,
                          tol: float, maxiter: int) -> OptimizeResult:
    """
    Minimize the cost subject to box constraints with Bertsekas' projected
    Newton method, using the Gauss-Newton Hessian.

    Each iteration holds the variables that are at (or within a shrinking
    band of) a bound with the gradient pushing outward, takes a Newton step
    in the remaining free variables, and shortens the step along the path
    projected onto the box until the cost decreases enough (Armijo rule).
    Without active bounds this is plain Gauss-Newton with a line search.

    Reference: D. P. Bertsekas, Projected Newton methods for optimization
    problems with simple constraints, SIAM J. Control Optim. 20 (1982).
    """
    
    lower, upper = bounds.lb, bounds.ub
    x = np.clip(x0.astype(float), lower, upper)
    cost, gradient, jac = problem.evaluate(x)
    if not (np.isfinite(cost) and np.all(np.isfinite(jac))):
        raise ValueError('Cost function or Jacobian is not finite at the initial guess')
    nfev = 1
    band_max = 1.0        # largest band width, in posterior standard deviations
    armijo = 1e-4         # required fraction of the predicted decrease
    status, message = 1, 'Maximum number of iterations reached.'

    for nit in range(1, maxiter + 1):
        hess = problem.hessian(*jac)
        h = np.diag(hess)
        scale = np.sqrt(h)  # converts distances to (approximate) posterior std units

        # Band width: length of a diagonally scaled projected gradient step;
        # it goes to zero at the solution
        band = min(band_max, np.linalg.norm((x - np.clip(x - gradient / (2 * h), lower, upper)) * scale))
        held = ((((x - lower) * scale <= band) & (gradient > 0))
                | (((upper - x) * scale <= band) & (gradient < 0)))
        free = ~held

        direction = np.empty_like(x)
        direction[held] = -gradient[held] / (2 * h[held])
        if free.any():
            direction[free] = cho_solve(cho_factor(hess[np.ix_(free, free)]), -gradient[free] / 2)

        # Step length in posterior standard deviations, as in _gauss_newton
        step = np.clip(x + direction, lower, upper) - x
        if np.sqrt(step @ hess @ step) < tol:
            status, message = 0, 'Optimization terminated successfully.'
            nit -= 1  # converged before taking a step on this round
            break

        alpha = 1.0
        while True:
            x_trial = np.clip(x + alpha * direction, lower, upper)
            trial_cost, trial_gradient, trial_jac = problem.evaluate(x_trial)
            nfev += 1
            predicted = (-alpha * gradient[free] @ direction[free]
                         + gradient[held] @ (x[held] - x_trial[held]))
            if cost - trial_cost >= armijo * predicted:  # also False when trial_cost is nan
                break
            alpha /= 2
            if alpha < 1e-12:
                return OptimizeResult(x=x, fun=cost, jac=gradient, nit=nit, nfev=nfev,
                                      status=2, success=False,
                                      message='Line search failed: cost cannot be decreased.')

        x = x_trial
        cost, gradient, jac = trial_cost, trial_gradient, trial_jac

    return OptimizeResult(x=x, fun=cost, jac=gradient, nit=nit, nfev=nfev,
                          status=status, success=status == 0, message=message)


#%% Helpers

def _validate(cfg: RetrievalConfig) -> None:
    """Check that the required settings are provided and consistent."""
    required = [('wavelength', 'wavelength'),
                ('radiance', 'radiance'),
                ('error_mean', 'error mean'),
                ('error_cov', 'error covariance'),
                ('surf_prior_mean', 'surface reflectance prior mean'),
                ('surf_prior_cov', 'surface reflectance prior covariance'),
                ('sif_prior_mean', 'SIF prior mean'),
                ('sif_prior_cov', 'SIF prior covariance'),
                ('forward_model', 'forward model'),
                ('jacobian', 'jacobian')]
    
    if cfg.use_approximation_error_model:
        required += [('ae_mean', 'approximation error mean'),
                     ('ae_cov', 'approximation error covariance')]
    
    for attr, name in required:
        if getattr(cfg, attr) is None:
            raise ValueError(f'{name} must be provided in the config!')

    if cfg.solver not in SOLVERS:
        raise ValueError(f'solver must be one of {SOLVERS}, got {cfg.solver!r}')

    n = len(cfg.radiance)
    if len(cfg.wavelength) != n:
        raise ValueError('wavelength and radiance must have the same length')
    
    for attr in ('error_cov', 'surf_prior_cov', 'sif_prior_cov'):
        if np.shape(getattr(cfg, attr)) != (n, n):
            raise ValueError(f'{attr} must have shape ({n}, {n})')


def _cho_factor(matrix: np.ndarray, name: str) -> tuple:
    """Cholesky factor of a symmetric positive definite matrix."""
    try:
        return cho_factor(matrix)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(f'{name} is not positive definite') from None


def _inv_spd(matrix: np.ndarray, name: str) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix."""
    diagonal = np.diag(matrix)
    if np.count_nonzero(matrix) == np.count_nonzero(diagonal):
        if np.any(diagonal <= 0):
            raise np.linalg.LinAlgError(f'{name} is not positive definite')
        return np.diag(1 / diagonal)
    c, lower = _cho_factor(matrix, name)
    inverse, _ = lapack.dpotri(c, lower=lower)
    # dpotri fills only one triangle
    return np.triu(inverse) + np.triu(inverse, 1).T if not lower else np.tril(inverse) + np.tril(inverse, -1).T


def _inv_diag(factor: tuple) -> np.ndarray:
    """Diagonal of the inverse of a matrix from its Cholesky factor."""
    c, lower = factor
    inverse, _ = lapack.dpotri(c, lower=lower)
    return np.diag(inverse).copy()


def _bounds(cfg: RetrievalConfig, n: int) -> Bounds:
    """Box constraints for the stacked (surface reflectance, SIF) vector."""
    lower = np.concatenate((np.broadcast_to(cfg.surf_bounds[0], n),
                            np.broadcast_to(cfg.sif_bounds[0], n)))
    upper = np.concatenate((np.broadcast_to(cfg.surf_bounds[1], n),
                            np.broadcast_to(cfg.sif_bounds[1], n)))
    return Bounds(lower, upper)


def _at_bound(x: np.ndarray, bounds: Bounds, std: np.ndarray) -> np.ndarray:
    """
    Flag the elements of x that are at a bound: -1 lower, +1 upper, 0 free.
    An element is at a bound when it is within 0.001 posterior standard
    deviations of it.
    """
    
    tol = 1e-3 * std
    flags = np.zeros(len(x), dtype=np.int8)
    flags[x - bounds.lb <= tol] = -1
    flags[bounds.ub - x <= tol] = 1
    return flags
