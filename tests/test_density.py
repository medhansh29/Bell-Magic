import numpy as np
import pytest

from bellmagic import bell_probs
from bellmagic.core import (
    A_phi, product_state, apply_circuit, doped_clifford_gates,
    density_from_state, bell_probs_density,
)
from bellmagic.noise import purity_from_probs, amplitude_damping_kraus, coherent_rotation_error, noisy_density


def _random_instance(N, rng, n_t=2, depth=3):
    gates = doped_clifford_gates(N, n_t, rng, depth)
    return apply_circuit(product_state([[1, 0]] * N), gates)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_noiseless_density_matches_pure_state_pipeline(N):
    rng = np.random.default_rng(0)
    for _ in range(3):
        psi = _random_instance(N, rng)
        rho = density_from_state(psi)
        assert bell_probs_density(rho) == pytest.approx(bell_probs(psi), abs=1e-10)


def test_amplitude_damping_kraus_is_trace_preserving():
    for gamma in (0.0, 0.2, 0.7, 1.0):
        E0, E1 = amplitude_damping_kraus(gamma)
        total = E0.conj().T @ E0 + E1.conj().T @ E1
        assert total == pytest.approx(np.eye(2), abs=1e-12)


def test_amplitude_damping_gamma_zero_is_identity():
    rng = np.random.default_rng(1)
    psi = _random_instance(3, rng)
    rho = noisy_density(psi, "amplitude_damping", 0.0)
    assert bell_probs_density(rho) == pytest.approx(bell_probs(psi), abs=1e-10)


def test_amplitude_damping_gamma_one_collapses_to_pure_zero_state():
    rng = np.random.default_rng(1)
    psi = _random_instance(3, rng)
    rho = noisy_density(psi, "amplitude_damping", 1.0)
    P = bell_probs_density(rho)
    # every qubit decays to |0>: this is deterministic, so the state stays
    # pure (purity 1) even though it's a different state than psi.
    assert purity_from_probs(P) == pytest.approx(1.0, abs=1e-10)
    zero_probs = bell_probs(product_state([[1, 0]] * 3))
    assert P == pytest.approx(zero_probs, abs=1e-10)


def test_amplitude_damping_reduces_purity_for_generic_gamma():
    rng = np.random.default_rng(2)
    psi = _random_instance(3, rng)
    P = bell_probs_density(noisy_density(psi, "amplitude_damping", 0.3))
    assert purity_from_probs(P) < 1.0 - 1e-6


def test_coherent_rotation_is_unitary():
    for eps in (0.0, 0.3, 1.0, 2.5):
        U = coherent_rotation_error(eps)
        assert U.conj().T @ U == pytest.approx(np.eye(2), abs=1e-12)


def test_coherent_error_preserves_purity():
    # a purely unitary (coherent) error never mixes the state.
    rng = np.random.default_rng(3)
    psi = _random_instance(3, rng)
    for eps in (0.1, 0.5, 1.5):
        P = bell_probs_density(noisy_density(psi, "coherent", eps))
        assert purity_from_probs(P) == pytest.approx(1.0, abs=1e-9)


def test_coherent_epsilon_zero_is_identity():
    rng = np.random.default_rng(4)
    psi = _random_instance(3, rng)
    rho = noisy_density(psi, "coherent", 0.0)
    assert bell_probs_density(rho) == pytest.approx(bell_probs(psi), abs=1e-10)
