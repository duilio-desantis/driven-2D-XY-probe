# Driven two-dimensional XY model

Overdamped Model-A Langevin dynamics on a square lattice with periodic boundaries, sinusoidal coupling modulation, and a monochromatic electromagnetic probe. Conductivity is evaluated at one probe frequency per run.

This is a reference for the code used in the paper [https://doi.org/10.48550/arXiv.2511.12287], accepted in Physical Review Letters [https://doi.org/10.1103/p9t6-8nzn]

## Installation and usage

Python 3.10 or newer and NumPy are required.

```bash
python -m pip install -r requirements.txt
python xy_model.py --output results
```

The defaults provide a demonstration run. An ensemble can be computed with `--N 10`. Use `--A 0` for constant coupling and `--probe_dir y` for a probe along y. All parameters are available through `python xy_model.py --help`. Output directories must be empty.

## Model and response

Units are chosen so that the equilibrium coupling, kinetic coefficient, and Boltzmann constant are one. Frequencies are angular frequencies. The Hamiltonian is

$$H(t)=-J(t)\sum_{\mathbf r,\mu=x,y}\cos[\theta_{\mathbf r}-\theta_{\mathbf r+\hat\mu}-A_\mu(t)],$$

with $J(t)=1+A f(t)\sin(\omega t+\delta)$ and $f(t)=1-\exp[-(t/\tau_s)^2]$, where $\tau_s=2\pi n_{\mathrm{switch}}/\omega$. Setting `--n_switch 0` gives an immediate turn-on. `--A` is the coupling-modulation amplitude, distinct from the probe vector potential.

Euler–Maruyama updates integrate $d\theta_{\mathbf r}=-\partial_{\theta_{\mathbf r}}H\,dt+\sqrt{2T}\,dW_{\mathbf r}$. Each realization starts from independent uniformly distributed phases and evolves for `burn` steps with $J=1$ and no probe. The clock then starts at zero, and drive and probe are applied for `steps` updates. The first `discard` production steps are excluded from measurement; every `meas` updates thereafter supplies a sample.

For the selected probe direction,

$$A_p(t)=\frac{E_*}{\omega_p}\cos(\omega_p t+\phi_p),\qquad E_p(t)=-\dot A_p(t)=E_*\sin(\omega_p t+\phi_p),$$

where $E_*=E_0\sqrt{1+(\omega_p/\omega_c)^2}$. The parameters are `probe_E0`, `probe_omega`, `probe_phase`, and `probe_omega_c`; the default cutoff is 100. For $\omega_p\ll\omega_c$, $E_*\simeq E_0$.

The spatially averaged current is $j_p(t)=J(t)\langle\sin(\theta_{\mathbf r}-\theta_{\mathbf r+\hat p}-A_p)\rangle_{\mathbf r}$. Using the positive-exponent Fourier convention,

$$\sigma(\omega_p)=\frac{\sum_k j_p(t_k)e^{i\omega_p t_k}}{\sum_k E_p(t_k)e^{i\omega_p t_k}}.$$

The uniform sampling interval cancels in this ratio. Measurements use the updated phase field and the coupling and vector potential at that same time. Ensemble results average the complex conductivity across seeds `seed` through `seed + N - 1`.

## Output

- `conductivity.json`: parameters, mean real and imaginary conductivity, their standard errors, and measurement duration.
- `response.npz`: `time`, `electric_field`, `current_mean`, `current_sem`, and complex `sigma_samples`.

Standard errors are estimated across independent realizations, not time samples. They are unavailable for a single realization (`null` in JSON and `NaN` in `current_sem`). Load arrays with `numpy.load("results/response.npz")`.

## Numerical convergence

The demonstration parameters do not establish convergence. Check equilibration time, transient exclusion, lattice size, timestep, sampling interval, ensemble size, and measurement duration. Use many probe and drive cycles, preferably integer numbers of both, to reduce finite-window leakage and drive-sideband contamination. Verify linear response by varying the probe amplitude.
