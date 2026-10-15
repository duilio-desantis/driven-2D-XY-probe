from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parameters:
    L: int = 32
    T: float = 1.2
    dt: float = 0.05
    A: float = 0.3
    omega: float = 0.1
    delta: float = 0.0
    n_switch: float = 1.0
    burn: int = 2000
    steps: int = 20000
    discard: int = 4000
    meas: int = 5
    probe_E0: float = 0.002
    probe_omega: float = 1.0
    probe_phase: float = 0.0
    probe_dir: str = "x"
    probe_omega_c: float = 100.0
    N: int = 1
    seed: int = 0

    def validate(self):
        for name in ("L", "burn", "steps", "discard", "meas", "N", "seed"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.L < 2 or self.N < 1 or self.meas < 1 or self.seed < 0:
            raise ValueError("Require L >= 2, N >= 1, meas >= 1, seed >= 0")
        if self.burn < 0 or not 0 <= self.discard < self.steps:
            raise ValueError("Require burn >= 0 and 0 <= discard < steps")
        if self.T < 0 or self.dt <= 0 or self.n_switch < 0:
            raise ValueError("Require T >= 0, dt > 0 and n_switch >= 0")
        if not 0 <= self.A < 1 or self.omega < 0:
            raise ValueError("Require 0 <= A < 1 and omega >= 0")
        if self.A > 0 and self.omega == 0:
            raise ValueError("A nonzero drive requires omega > 0")
        if min(self.probe_E0, self.probe_omega, self.probe_omega_c) <= 0:
            raise ValueError("Probe amplitude, frequency and cutoff must be positive")
        if self.probe_dir not in ("x", "y"):
            raise ValueError("probe_dir must be x or y")
        if max(self.omega, self.probe_omega)*self.dt*self.meas >= np.pi:
            raise ValueError("Sampling must resolve both drive and probe frequencies")
        if (self.steps-self.discard-1)//self.meas+1 < 4:
            raise ValueError("At least four measurement samples are required")


class ModelAXY2D:
    def __init__(self, parameters: Parameters, seed: int):
        self.p = parameters
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(-np.pi, np.pi, (parameters.L, parameters.L))
        self.t = 0.0
        self.noise = np.sqrt(2*parameters.T*parameters.dt)

    def coupling(self, t):
        p = self.p
        if p.A == 0:
            return 1.0
        envelope = 1.0 if p.n_switch == 0 else -np.expm1(
            -(t/(p.n_switch*2*np.pi/p.omega))**2
        )
        return 1+p.A*envelope*np.sin(p.omega*t+p.delta)

    def probe(self, t):
        p = self.p
        amplitude = p.probe_E0*np.sqrt(1+(p.probe_omega/p.probe_omega_c)**2)
        phase = p.probe_omega*t+p.probe_phase
        potential = amplitude/p.probe_omega*np.cos(phase)
        electric = amplitude*np.sin(phase)
        return potential, electric

    def force(self, Ax=0.0, Ay=0.0):
        theta = self.theta
        return (
            np.sin(theta-np.roll(theta, 1, 0)+Ay)
            + np.sin(theta-np.roll(theta, -1, 0)-Ay)
            + np.sin(theta-np.roll(theta, 1, 1)+Ax)
            + np.sin(theta-np.roll(theta, -1, 1)-Ax)
        )

    def step(self, production=True):
        coupling, Ax, Ay = 1.0, 0.0, 0.0
        if production:
            coupling = self.coupling(self.t)
            potential, _ = self.probe(self.t)
            Ax, Ay = (potential, 0.0) if self.p.probe_dir == "x" else (0.0, potential)
        self.theta += -self.p.dt*coupling*self.force(Ax, Ay)
        self.theta += self.noise*self.rng.standard_normal(self.theta.shape)
        self.theta = (self.theta+np.pi) % (2*np.pi)-np.pi
        if production:
            self.t += self.p.dt

    def current(self):
        potential, _ = self.probe(self.t)
        axis = 1 if self.p.probe_dir == "x" else 0
        return self.coupling(self.t)*np.sin(
            self.theta-np.roll(self.theta, -1, axis)-potential
        ).mean()


def conductivity(time, current, electric_field, omega):
    phase = np.exp(1j*omega*time)
    denominator = np.sum(electric_field*phase)
    if abs(denominator) <= np.finfo(float).eps*np.sum(np.abs(electric_field)):
        raise ValueError("The probe Fourier component is zero or numerically unresolved")
    return np.sum(current*phase)/denominator


def run_realization(p: Parameters, seed: int):
    model = ModelAXY2D(p, seed)
    for _ in range(p.burn):
        model.step(production=False)
    samples = (p.steps-p.discard-1)//p.meas+1
    time, current, electric = np.empty((3, samples))
    k = 0
    for step in range(p.steps):
        model.step()
        if step >= p.discard and (step-p.discard) % p.meas == 0:
            time[k] = model.t
            current[k] = model.current()
            electric[k] = model.probe(model.t)[1]
            k += 1
    sigma = conductivity(time, current, electric, p.probe_omega)
    return time, current, electric, sigma


def simulate(p: Parameters):
    p.validate()
    sigma = np.empty(p.N, dtype=complex)
    mean = second_moment = None
    for i in range(p.N):
        time, current, electric, sigma[i] = run_realization(p, p.seed+i)
        if mean is None:
            mean = np.zeros_like(current)
            second_moment = np.zeros_like(current)
        difference = current-mean
        mean += difference/(i+1)
        second_moment += difference*(current-mean)
    sem = np.sqrt(np.maximum(second_moment, 0)/(p.N*(p.N-1))) if p.N > 1 else np.full_like(mean, np.nan)
    return dict(time=time, current_mean=mean, current_sem=sem,
                electric_field=electric, sigma_samples=sigma)


def main():
    parser = argparse.ArgumentParser(description="Driven two-dimensional XY model conductivity")
    defaults = Parameters()
    for name, value in asdict(defaults).items():
        options = {"type": type(value), "default": value}
        if name == "probe_dir":
            options["choices"] = ("x", "y")
        parser.add_argument("--"+name, **options)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = vars(parser.parse_args())
    output = args.pop("output")
    p = Parameters(**args)
    try:
        p.validate()
    except ValueError as error:
        parser.error(str(error))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be empty; choose another --output")
    result = simulate(p)
    sigma = result["sigma_samples"]
    summary = dict(
        parameters=asdict(p),
        sigma_real=float(sigma.real.mean()),
        sigma_imag=float(sigma.imag.mean()),
        sigma_sem_real=float(sigma.real.std(ddof=1)/np.sqrt(p.N)) if p.N > 1 else None,
        sigma_sem_imag=float(sigma.imag.std(ddof=1)/np.sqrt(p.N)) if p.N > 1 else None,
        measurement_samples=len(result["time"]),
        measurement_duration=len(result["time"])*p.meas*p.dt,
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"response.npz", **result)
    (output/"conductivity.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
