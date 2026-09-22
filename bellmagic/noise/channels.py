"""Amplitude damping (T1) and coherent (systematic unitary) single-qubit
error channels, applied per qubit via the density-matrix pipeline in
core.density. Unlike the global depolarizing channel in depolarizing.py,
neither of these has a closed form on the Bell outcome distribution P, so
they go through an exact density-matrix simulation instead of an analytic
rescaling."""
import numpy as np
from ..core.density import density_from_state, apply_kraus_1q


def amplitude_damping_kraus(gamma):
    """T1-style decay towards |0> with probability gamma per qubit."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]


def coherent_rotation_error(epsilon):
    """A systematic (non-random) over-rotation about X by angle epsilon --
    a single unitary, so this channel is non-dissipative: it never reduces
    purity on its own, unlike amplitude damping or depolarizing noise."""
    c, s = np.cos(epsilon / 2), -1j * np.sin(epsilon / 2)
    return np.array([[c, s], [s, c]], dtype=complex)


def noisy_density(psi, kind, strength):
    """Apply `kind` ("amplitude_damping" or "coherent") at the given
    strength to every qubit of the pure state psi, independently."""
    N = psi.ndim
    rho = density_from_state(psi)
    for n in range(N):
        if kind == "amplitude_damping":
            rho = apply_kraus_1q(rho, amplitude_damping_kraus(strength), n, N)
        elif kind == "coherent":
            rho = apply_kraus_1q(rho, [coherent_rotation_error(strength)], n, N)
        else:
            raise ValueError(kind)
    return rho
