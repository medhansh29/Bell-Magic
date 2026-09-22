"""Phase 6: check the numpy Bell-magic pipeline against a real Qiskit/Aer
circuit, then look at what gate-level (hardware-shaped) noise does to B,
as opposed to the paper's state-depolarizing channel already covered in
fig2.py.

Qiskit's bitstrings/statevector indices are little-endian (qubit 0 least
significant); bellmagic.backends.qiskit_backend converts to the bellmagic
convention (qubit 0 most significant) before anything is compared.
"""
import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.core import doped_clifford_gates, product_state, apply_circuit
from bellmagic.backends import bell_circuit, exact_probs, counts_to_probs, run_bell_circuit, prep_depolarizing_noise


def check_zero_state():
    probs = exact_probs(1, [])
    print("|0> convention check, n=1: nonzero outcomes",
          {i: round(p, 4) for i, p in enumerate(probs) if p > 1e-9},
          "(expect index 0 = x0z0 and index 2 = x1z0, each 0.5)")
    expected = [0.5, 0.0, 0.5, 0.0]
    if not np.allclose(probs, expected, atol=1e-12):
        raise AssertionError(f"convention check failed: {probs} != {expected}")


def check_exact_match(n=3, n_t=2, depth=3, trials=5, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        gates = doped_clifford_gates(n, n_t, rng, depth)
        psi = apply_circuit(product_state([[1, 0]] * n), gates)
        oracle = bell_probs(psi)
        qiskit_exact = exact_probs(n, gates)
        worst = max(worst, np.abs(qiskit_exact - oracle).max())
    print(f"n={n}, {trials} random doped-Clifford instances: "
          f"max |qiskit_exact - numpy_oracle| = {worst:.2e}")


def check_sampling(n=3, n_t=2, depth=3, shots=40000, seed=1):
    rng = np.random.default_rng(seed)
    gates = doped_clifford_gates(n, n_t, rng, depth)
    exact = exact_probs(n, gates)
    counts = run_bell_circuit(bell_circuit(n, gates), shots=shots, seed=seed)
    empirical = counts_to_probs(counts, n)
    print(f"n={n}, {shots} shots on AerSimulator: "
          f"max |empirical - exact| = {np.abs(empirical - exact).max():.4f}")


def prep_noise_sweep(n=2, n_t=2, depth=3, shots=20000, seed=2):
    """Sweep a per-qubit depolarizing channel applied after state prep and
    before an otherwise-ideal Bell measurement.

    Two things worth flagging, both found while building this:
    - The noise must be injected as an explicit instruction right after
      prep, not via a NoiseModel keyed on gate names -- the Bell-measurement
      circuit itself uses the same H and CNOT gate names as state prep, so a
      name-matched NoiseModel corrupts the measurement too. That first
      attempt gave a nonsensical result (raw B jumping around with no
      relation to the input noise), because it breaks the algebraic
      identity exact_B relies on -- it stops measuring anything physical.
    - Raw (unmitigated) B is not "how much magic survives the noise": the
      maximally mixed state has B = 1 - 4^-n, near-maximal (Phase 3's
      physics checks, test_core.py), so B should rise toward that value as
      noise increases, not fall toward 0. That's exactly reproduced below,
      now from a real Qiskit circuit rather than only the closed-form
      formula in bellmagic.noise -- and it's the reason the paper needs
      mitigation (bellmagic.noise.mitigation) instead of reading B raw."""
    rng = np.random.default_rng(seed)
    gates = doped_clifford_gates(n, n_t, rng, depth)
    psi = apply_circuit(product_state([[1, 0]] * n), gates)
    B_exact = exact_B(bell_probs(psi))
    ceiling = 1 - 4 ** -n
    print(f"n={n}, exact B={B_exact:.4f}, maximally-mixed ceiling 1-4^-n={ceiling:.4f}. "
          f"Per-qubit prep-noise sweep, {shots} shots:")
    for p in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7):
        qc = bell_circuit(n, gates, prep_noise=prep_depolarizing_noise(p) if p else None)
        counts = run_bell_circuit(qc, shots=shots, seed=seed)
        B = exact_B(counts_to_probs(counts, n))
        print(f"  p={p:<5} B={B:.4f}  (ceiling - B = {ceiling - B:.4f})")


def run():
    check_zero_state()
    check_exact_match()
    check_sampling()
    prep_noise_sweep()
    print("\nA real IBM Quantum hardware run needs account credentials that "
          "aren't configured here, so it's left as a follow-up rather than attempted.")


if __name__ == "__main__":
    run()
