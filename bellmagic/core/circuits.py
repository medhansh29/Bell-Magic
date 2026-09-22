import numpy as np
from .states import apply_circuit, random_clifford_circuit, product_state


def doped_clifford_gates(N, n_t, rng, depth=6):
    """U_C prod_{k=1}^{n_t} (T_{q_k} U_C^k) acting on |0>^N (paper Sec. VI, Fig. 3b):
    n_t T gates, each on a random qubit, separated by random Clifford circuits."""
    gates = []
    for _ in range(n_t):
        gates += random_clifford_circuit(N, depth, rng)
        gates.append(("t", int(rng.integers(N))))
    gates += random_clifford_circuit(N, depth, rng)
    return gates


def doped_clifford_state(N, n_t, rng, depth=6):
    zero = product_state([[1, 0]] * N)
    return apply_circuit(zero, doped_clifford_gates(N, n_t, rng, depth))
