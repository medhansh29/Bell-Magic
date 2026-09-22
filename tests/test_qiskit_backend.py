import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from bellmagic import bell_probs, exact_B
from bellmagic.core import doped_clifford_gates
from bellmagic.backends import bell_circuit, exact_probs, counts_to_probs, run_bell_circuit, prep_depolarizing_noise


def test_zero_state_convention_n1():
    # |0>: CNOT(A->B) leaves |00>, H(A) gives (|00>+|10>)/sqrt2 -> outcomes
    # (x=0,z=0) and (x=1,z=0), each with probability 1/2, index x*2+z.
    probs = exact_probs(1, [])
    assert probs == pytest.approx([0.5, 0.0, 0.5, 0.0], abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_zero_state_matches_numpy_oracle(n):
    from bellmagic.core import product_state
    oracle = bell_probs(product_state([[1, 0]] * n))
    assert exact_probs(n, []) == pytest.approx(oracle, abs=1e-12)


def test_doped_clifford_matches_numpy_oracle():
    from bellmagic.core import product_state, apply_circuit
    rng = np.random.default_rng(0)
    n = 3
    for _ in range(5):
        gates = doped_clifford_gates(n, n_t=2, rng=rng, depth=3)
        psi = apply_circuit(product_state([[1, 0]] * n), gates)
        oracle = bell_probs(psi)
        assert exact_probs(n, gates) == pytest.approx(oracle, abs=1e-9)


def test_sampled_counts_converge_to_exact():
    rng = np.random.default_rng(1)
    n = 2
    gates = doped_clifford_gates(n, n_t=2, rng=rng, depth=3)
    exact = exact_probs(n, gates)
    qc = bell_circuit(n, gates)
    counts = run_bell_circuit(qc, shots=40000, seed=7)
    empirical = counts_to_probs(counts, n)
    assert np.abs(empirical - exact).max() < 0.02


def test_prep_noise_pushes_raw_B_toward_maximally_mixed_value():
    # Raw (unmitigated) B is NOT a monotone-decreasing "magic" readout under
    # noise: the maximally mixed state has B = 1 - 4^-n (near-maximal, per
    # the Phase-3 physics checks in test_core.py), so per-qubit depolarizing
    # prep noise should push the measured B *up*, toward that value, not
    # down toward 0. This is exactly why the paper needs mitigation
    # (bellmagic.noise.mitigation) rather than reading raw B off noisy data.
    rng = np.random.default_rng(2)
    n = 2
    gates = doped_clifford_gates(n, n_t=2, rng=rng, depth=3)

    from bellmagic.core import product_state, apply_circuit
    psi = apply_circuit(product_state([[1, 0]] * n), gates)
    B_exact = exact_B(bell_probs(psi))

    counts_clean = run_bell_circuit(bell_circuit(n, gates), shots=20000, seed=3)
    B_clean = exact_B(counts_to_probs(counts_clean, n))

    noisy_qc = bell_circuit(n, gates, prep_noise=prep_depolarizing_noise(0.2))
    counts_noisy = run_bell_circuit(noisy_qc, shots=20000, seed=3)
    B_noisy = exact_B(counts_to_probs(counts_noisy, n))

    assert B_clean == pytest.approx(B_exact, abs=0.02)
    assert B_clean < B_noisy < 1 - 4 ** -n + 1e-9
