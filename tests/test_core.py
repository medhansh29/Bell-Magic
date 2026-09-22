import numpy as np
import pytest

from bellmagic import bell_probs, exact_B, sample_B, additive, check_commute
from bellmagic.core import (
    A_phi, product_state, apply_1q, apply_circuit, random_clifford_circuit, T_GATE,
)
from bellmagic.core.states import H
from bellmagic.core.sampling import sample_bitstrings
from bellmagic.core.exact import walsh_hadamard
from conftest import pauli_from_index, haar_state


def zero(N):
    return product_state([[1, 0]] * N)


# ---------------- Phase 1: Bell distribution ----------------

def test_probs_normalised(rng):
    for N in (1, 2, 3):
        assert bell_probs(haar_state(N, rng)).sum() == pytest.approx(1.0)


def test_zero_state_outcomes():
    p = bell_probs(zero(1))
    # index x*2^N+z: "00" -> 0, "10" -> 2
    assert p[0] == pytest.approx(0.5) and p[2] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0) and p[3] == pytest.approx(0)


def test_A_phi_matches_paper_p21():
    phi = 0.7
    p = bell_probs(A_phi(phi))
    assert p[0] == pytest.approx(0.5)                       # p00
    assert p[2] == pytest.approx(0.5 * np.cos(phi) ** 2)    # p10
    assert p[1] == pytest.approx(0.5 * np.sin(phi) ** 2)    # p01 (paper's 2nd "p10")
    assert p[3] == pytest.approx(0)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_probs_match_bell_basis_projection(N, rng):
    """Circuit result equals |<Phi_r|psi(x)psi>|^2 built from explicit Pauli matrices."""
    psi = haar_state(N, rng)
    d = 2 ** N
    phi_plus = np.eye(d, dtype=complex).reshape(-1) / np.sqrt(d)   # sum_i |i>|i>
    two = np.kron(psi.reshape(-1), psi.reshape(-1))
    expected = np.zeros(d * d)
    for r in range(d * d):
        bell = np.kron(pauli_from_index(r, N), np.eye(d)) @ phi_plus
        expected[r] = abs(np.vdot(bell, two)) ** 2
    assert np.allclose(bell_probs(psi), expected, atol=1e-12)


# ---------------- Phase 2: exact oracle ----------------

def test_wht_involution(rng):
    a = rng.random(16)
    assert np.allclose(walsh_hadamard(walsh_hadamard(a)) / 16, a)


def brute_force_B(P, N):
    """B straight from the definition with real commutator operator norms."""
    d2 = 4 ** N
    Q = np.zeros(d2)
    for n in range(d2):
        for r in range(d2):
            Q[n] += P[r] * P[r ^ n]
    paulis = [pauli_from_index(r, N) for r in range(d2)]
    B = 0.0
    for r in range(d2):
        for q in range(d2):
            comm = paulis[r] @ paulis[q] - paulis[q] @ paulis[r]
            B += Q[r] * Q[q] * np.linalg.norm(comm, 2)
    return B


@pytest.mark.parametrize("N", [1, 2])
def test_exact_B_matches_definition(N, rng):
    for _ in range(3):
        P = bell_probs(haar_state(N, rng))
        assert exact_B(P) == pytest.approx(brute_force_B(P, N), abs=1e-10)


def test_exact_B_on_arbitrary_distribution(rng):
    P = rng.random(16); P /= P.sum()
    assert exact_B(P) == pytest.approx(brute_force_B(P, 2), abs=1e-10)


def test_maximally_mixed_value():
    for N in (1, 2, 3):
        U = np.full(4 ** N, 4.0 ** -N)
        assert exact_B(U) == pytest.approx(1 - 4.0 ** -N)


def test_stabilizer_states_zero(rng):
    for N in (1, 2, 3, 4):
        assert exact_B(bell_probs(zero(N))) == pytest.approx(0, abs=1e-12)
        plus = product_state([[1, 1]] * N) / 2 ** (N / 2)
        assert exact_B(bell_probs(plus)) == pytest.approx(0, abs=1e-12)
        psi = apply_circuit(zero(N), random_clifford_circuit(N, 6, rng))
        assert exact_B(bell_probs(psi)) == pytest.approx(0, abs=1e-12)


def test_T_state_paper_values():
    B = exact_B(bell_probs(A_phi(np.pi / 4)))
    assert B == pytest.approx(0.5) and additive(B) == pytest.approx(1.0)


def test_small_angle_scaling():
    """Paper p.7: B ~ 2 phi^2 for |phi| << 1."""
    phi = 0.02
    assert exact_B(bell_probs(A_phi(phi))) == pytest.approx(2 * phi ** 2, rel=1e-2)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_additivity_in_T_states(k):
    singles = [A_phi(np.pi / 4)] * k + [[1, 0]] * (3 - k)
    Ba = additive(exact_B(bell_probs(product_state(singles))))
    assert Ba == pytest.approx(k, abs=1e-9)


def test_clifford_invariance(rng):
    N = 3
    psi = haar_state(N, rng)
    B0 = exact_B(bell_probs(psi))
    for _ in range(5):
        phi = apply_circuit(psi, random_clifford_circuit(N, 5, rng))
        assert exact_B(bell_probs(phi)) == pytest.approx(B0, abs=1e-10)


def test_T_gate_on_plus_is_magic():
    psi = apply_1q(product_state([[1, 1]]) / np.sqrt(2), T_GATE, 0)
    assert exact_B(bell_probs(psi)) > 0.1


# ---------------- Phase 3: sampling estimator ----------------

def test_check_commute_matches_matrices(rng):
    N = 2
    d2 = 4 ** N
    for r in range(d2):
        for q in range(d2):
            A, Bm = pauli_from_index(r, N), pauli_from_index(q, N)
            commute = np.allclose(A @ Bm, Bm @ A)
            assert bool(check_commute(r, q, N)) == commute


def test_sampler_vs_exact_oracle(rng):
    """Sampler is checked against the exact oracle, never against itself."""
    N = 2
    psi = product_state([A_phi(np.pi / 4), A_phi(np.pi / 3)])
    P = bell_probs(psi)
    B = exact_B(P)
    est = [
        sample_B(sample_bitstrings(P, 4000, rng), N, 40000, rng) for _ in range(20)
    ]
    sem = np.std(est, ddof=1) / np.sqrt(len(est))
    assert abs(np.mean(est) - B) < 5 * sem + 5e-3


def test_sampler_zero_for_stabilizer(rng):
    P = bell_probs(apply_circuit(zero(3), random_clifford_circuit(3, 5, rng)))
    assert sample_B(sample_bitstrings(P, 500, rng), 3, 5000, rng) == 0.0


def test_error_scales_as_inverse_sqrt_NQ(rng):
    """Paper Fig. 2: Delta B ~ N_Q^{-1/2}. Fit slope over N_Q=100..6400."""
    N = 2
    P = bell_probs(product_state([A_phi(np.pi / 4)] * 2))
    B = exact_B(P)
    nqs = [100, 400, 1600, 6400]
    errs = []
    for nq in nqs:
        e = [abs(sample_B(sample_bitstrings(P, nq, rng), N, 10 * nq, rng) - B)
             for _ in range(60)]
        errs.append(np.mean(e))
    slope = np.polyfit(np.log(nqs), np.log(errs), 1)[0]
    assert -0.65 < slope < -0.35


# ---------------- Phase 5a helpers: paper states & circuit builder ----------------

def test_paper_R_state_value():
    th = np.arccos(1 / np.sqrt(3))
    R = np.array([np.cos(th / 2), np.exp(-1j * np.pi / 4) * np.sin(th / 2)])
    assert additive(exact_B(bell_probs(R))) == pytest.approx(np.log2(27 / 11), abs=1e-12)


def test_paper_T_state_value():
    T = np.array([1, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)
    assert additive(exact_B(bell_probs(T))) == pytest.approx(1.0, abs=1e-12)


def test_maximal_magic_matches_paper_appendix_G():
    """Paper App. G: B_a^max = 1.29545588, 2.67807, 4.651794 for N = 1, 2, 3."""
    from bellmagic.experiments.max_magic import maximise
    for N, target in ((1, 1.29545588), (2, 2.67807), (3, 4.651794)):
        B, _ = maximise(N, restarts=40 if N == 3 else 10)
        assert additive(B) == pytest.approx(target, abs=2e-5)


def test_doped_circuit_magic_bounded_and_zero_at_nt0(rng):
    from bellmagic.core import doped_clifford_state
    N = 3
    assert exact_B(bell_probs(doped_clifford_state(N, 0, rng))) == pytest.approx(0, abs=1e-12)
    for nt in (1, 2, 3):
        for _ in range(4):
            psi = doped_clifford_state(N, nt, rng)
            assert np.linalg.norm(psi) == pytest.approx(1.0)
            assert additive(exact_B(bell_probs(psi))) <= nt + 1e-9   # B_a <= N_T


def test_single_T_gate_gives_one_unit_of_magic(rng):
    from bellmagic.core import doped_clifford_state
    for _ in range(5):
        psi = doped_clifford_state(3, 1, rng)
        assert additive(exact_B(bell_probs(psi))) == pytest.approx(1.0, abs=1e-9)
