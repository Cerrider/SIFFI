# SIFFI
SIFFI retrieval algorithm for retrieving solar-induced fluorescence and surface reflectance from observed radiance data.

The retrieval is Bayesian: given Gaussian prior models for reflectance and SIF
and a Gaussian noise model, SIFFI finds the maximum a posteriori (MAP) estimate
and its posterior standard deviation (Laplace approximation). The forward model
is supplied by the user, so SIFFI can be paired with any radiative transfer
model or instrument. The method is described in
[Kukkurainen et al. (2025)](https://doi.org/10.1016/j.rse.2024.114558).

## Requirements

- Python ≥ 3.10
- NumPy
- SciPy

## Installation

The code can be imported without installing by adding the directory to the `PYTHONPATH` but it's recommended to install it to your (virtual) environment.

Install directly from GitHub:

```bash
pip install git+https://github.com/Cerrider/SIFFI.git
```
Or clone the repository:

```bash
git clone https://github.com/Cerrider/SIFFI.git
cd SIFFI
pip install .
```

Check the installation:

```bash
python3 -c "import siffi; print(siffi.__version__)"
```

## Usage

A retrieval has two steps:

1. Initialize retrieval input using:

    `retrieval_input = siffi.RetrievalConfig()`

The input configuration behaves like a dictionary.
All initialized `None` values need to be replaced.

2. Run the retrieval by passing the configuration:

    `retrieval_results = siffi.retrieval(retrieval_input)`
    
The results are returned as a dictionary.

### Configuration

Required inputs (all arrays have length `n`, covariances are `n × n`):

| Option | Description |
|---|---|
| `wavelength` | Wavelengths |
| `radiance` | Observed radiance |
| `error_cov` | Noise covariance, e.g. `numpy.diag(sigma**2)` |
| `surf_prior_mean`, `surf_prior_cov` | Prior model for surface reflectance |
| `sif_prior_mean`, `sif_prior_cov` | Prior model for SIF |
| `forward_model`, `jacobian` | Forward model and its derivatives, see [Forward model](#Forward model)) |

Optional settings:

| Option | Default | Description |
|---|---|---|
| `fmodel_kwargs` | `{}` | Extra parameters passed to the forward model and Jacobian |
| `error_mean` | `0` | Mean of the noise model |
| `use_approximation_error_model` | `False` | Account for forward model errors with the approximation error method ([Kukkurainen et al., 2025](https://doi.org/10.1109/TGRS.2025.3590496)) |
| `ae_mean`, `ae_cov` | `0` | Mean and covariance of the approximation error |
| `surf_initial_guess`, `sif_initial_guess` | `0` | Starting point of the optimization |
| `solver` | `'GN'` | Optimization method, see [Solvers](#solvers) |
| `surf_bounds` | `(0, 1)` | Lower and upper bounds for reflectance (bounded solvers only) |
| `sif_bounds` | `(0, 6)` | Lower and upper bounds for SIF (bounded solvers only) |
| `gn_tol` | `1e-5` | Convergence tolerance for `'GN'` and `'GN-B'` |
| `gtol`, `ftol` | `1e-10`, `2.2e-14` | Tolerances for the SciPy solvers |
| `maxiter` | `1000000` | Maximum number of iterations |
| `verbose` | `False` | Print progress and a summary |

Full descriptions: `help(siffi.RetrievalConfig)`.

### Forward model

The forward model and its Jacobian are called as

```python
forward_model(surface_reflectance, sif, fmodel_kwargs)  # -> shape (n,)
jacobian(surface_reflectance, sif, fmodel_kwargs)       # -> shape (2, n)
```

where `n` is the number of wavelengths. Row 0 of the Jacobian is
dL/d(surface reflectance) and row 1 is dL/d(SIF). Any extra inputs (irradiance,
transmittances, geometry, ...) go in the `fmodel_kwargs` dictionary.

The model must be **wavelength-wise**: the radiance at wavelength `i` depends
only on `surface_reflectance[i]` and `sif[i]`.

### Solvers

| Solver | Description |
|---|---|
| `'GN'` | Gauss–Newton with Levenberg–Marquardt damping. Fast; recommended default. |
| `'GN-B'` | Gauss–Newton with box constraints (projected Newton). |
| `'BFGS'` | `scipy.optimize.minimize` with BFGS. |
| `'L-BFGS-B'` | `scipy.optimize.minimize` with L-BFGS-B, box constrained. |

With the bounded solvers, a `siffi.BoundWarning` is issued if the estimate
stops at a bound. The posterior standard deviation is unreliable at those
wavelengths, because the Laplace approximation assumes the minimum is inside
the bounds.

### Output

`siffi.retrieval` returns a dictionary:

| Key | Description |
|---|---|
| `wavelength` | Wavelengths |
| `surf_estimate`, `sif_estimate` | MAP estimates |
| `surf_std`, `sif_std` | Posterior standard deviations |
| `radiance_estimate` | Modelled radiance at the estimate |
| `surf_at_bound`, `sif_at_bound` | −1 / +1 where the estimate is at the lower / upper bound, 0 elsewhere |
| `opt_output` | Solver output (`scipy.optimize.OptimizeResult`) |
| `cost_history` | Cost function value at every evaluation |
| `optimization_time` | Run time of the optimization [s] |

## Citation

If you use SIFFI in your work, please cite:

> Kukkurainen, A., Lipponen, A., Kolehmainen, V., Arola, A., Cogliati, S.,
> & Sabater, N. (2025). SIFFI: Bayesian solar-induced fluorescence retrieval
> algorithm for remote sensing of vegetation. *Remote Sensing of Environment*,
> 318, 114558. https://doi.org/10.1016/j.rse.2024.114558

If you use the approximation error model, please also cite:

> Kukkurainen, A., Lipponen, A., Sabater, N., Arola, A., & Kolehmainen, V.
> (2025). Simulation study on improving spectrally resolved solar-induced
> fluorescence satellite retrievals under uncertain atmospheric states.
> *IEEE Transactions on Geoscience and Remote Sensing*, 63, 1–9.
> https://doi.org/10.1109/TGRS.2025.3590496

## License

MIT License, see [LICENSE](LICENSE).


## Contact

Antti Kukkurainen\
Finnish Meteorological Institute\
(antti.kukkurainen@fmi.fi)

