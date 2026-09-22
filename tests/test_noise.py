import numpy as np
import pytest

from bellmagic import bell_probs, exact_B
from bellmagic.core import A_phi, product_state
from bellmagic.core.sampling import sample_bitstrings
from bellmagic.noise import (
    depolarized_probs, purity_from_probs, purity_from_samples, dp_purity,
    p_from_purity, mitigate_B, mitigate_B_samples, mitigate_from_moments,
)
from conftest import pauli_from_index, haar_state


def density_bell_probs(psi, p):
    """Independent: Bell projections on rho_dp (x) rho_dp with explicit matrices."""
    N = psi.ndim
    d = 2 ** N
    v = psi.reshape(-1)
    rho = (1 - p) * np.outer(v, v.conj()) + p * np.eye(d) / d
    rr = np.kron(rho, rho)
    phi_plus = np.eye(d, dtype=complex).reshape(-1) / np.sqrt(d)
    out = np.zeros(d * d)
    for r in range(d * d):
        bell = np.kron(pauli_from_index(r, N), np.eye(d)) @ phi_plus
        out[r] = np.real(bell.conj() @ rr @ bell)
    return out


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("p", [0.0, 0.15, 0.6])
def test_depolarized_probs_match_density_matrix(N, p, rng):
    psi = haar_state(N, rng)
    assert np.allclose(depolarized_probs(bell_probs(psi), p),
                       density_bell_probs(psi, p), atol=1e-12)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_purity_from_probs(N, rng):
    psi = haar_state(N, rng)
    P = bell_probs(psi)
    assert purity_from_probs(P) == pytest.approx(1.0)
    for p in (0.05, 0.3):
        assert purity_from_probs(depolarized_probs(P, p)) == pytest.approx(dp_purity(p, N))


def test_purity_matches_density_matrix(rng):
    N, p = 2, 0.2
    psi = haar_state(N, rng).reshape(-1)
    rho = (1 - p) * np.outer(psi, psi.conj()) + p * np.eye(4) / 4
    assert dp_purity(p, N) == pytest.approx(np.real(np.trace(rho @ rho)))


def test_p_roundtrip():
    for N in (1, 3, 5):
        for p in (0.0, 0.1, 0.5, 0.9):
            assert p_from_purity(dp_purity(p, N), N) == pytest.approx(p)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("p", [0.05, 0.15, 0.4])
def test_exact_mitigation_recovers_noise_free_B(N, p, rng):
    for psi in (haar_state(N, rng), product_state([A_phi(np.pi / 4)] * N)):
        P = bell_probs(psi)
        B0 = exact_B(P)
        P_dp = depolarized_probs(P, p)
        assert mitigate_B(P_dp) == pytest.approx(B0, abs=1e-9)          # p from purity
        assert mitigate_B(P_dp, p=p) == pytest.approx(B0, abs=1e-9)     # p known


def test_plan_identity_1_minus_B():
    """1-B_dp = a^8(1-B) + 2a^4(1-a^4) Q0(0) + (1-a^4)^2/4^N."""
    N, p = 2, 0.15
    a = 1 - p
    P = bell_probs(product_state([A_phi(np.pi / 4)] * 2))
    Q00 = float(np.dot(P, P))
    lhs = 1 - exact_B(depolarized_probs(P, p))
    rhs = a ** 8 * (1 - exact_B(P)) + 2 * a ** 4 * (1 - a ** 4) * Q00 + (1 - a ** 4) ** 2 / 4 ** N
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_naive_rescaling_fails():
    """Plan warning: B_dp / a^8 is not the mitigated value (N=2, p=0.15)."""
    N, p = 2, 0.15
    P = bell_probs(product_state([A_phi(np.pi / 4)] * 2))
    B0 = exact_B(P)
    naive = exact_B(depolarized_probs(P, p)) / (1 - p) ** 8
    assert abs(naive - B0) > 0.05
    assert mitigate_B(depolarized_probs(P, p), p=p) == pytest.approx(B0, abs=1e-9)


def test_sampled_mitigation_converges(rng):
    N, p = 3, 0.1
    P = bell_probs(product_state([A_phi(np.pi / 4)] * 3))
    B0 = exact_B(P)
    P_dp = depolarized_probs(P, p)
    est = [mitigate_B_samples(sample_bitstrings(P_dp, 4000, rng), N, 40000, rng)[0]
           for _ in range(20)]
    sem = np.std(est, ddof=1) / np.sqrt(len(est))
    assert abs(np.mean(est) - B0) < 5 * sem + 5e-3


def test_sampled_purity_estimates_p(rng):
    N, p = 3, 0.15
    P_dp = depolarized_probs(bell_probs(haar_state(N, rng)), p)
    s = sample_bitstrings(P_dp, 200000, rng)
    assert p_from_purity(purity_from_samples(s, N), N) == pytest.approx(p, abs=0.01)
