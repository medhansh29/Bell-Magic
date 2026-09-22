import numpy as np
from .states import H, apply_1q, apply_cnot


def bell_probs(psi, phi=None):
    """Outcome distribution of the Bell measurement on |psi> (x) |phi> (two copies
    of |psi> when phi is None; the mixed form is used by the shift-rule gradient).

    Circuit: for each n, CNOT(A_n -> B_n) then H(A_n); measure all 2N qubits.
    Returns a flat array of length 4^N indexed by x*2^N + z, where x is the
    bit string read from the A registers and z from the B registers
    (qubit 0 most significant in each).
    """
    psi = np.asarray(psi, dtype=complex)
    N = psi.ndim
    phi = psi if phi is None else np.asarray(phi, dtype=complex)
    two = np.multiply.outer(psi, phi)  # axes: A_0..A_{N-1}, B_0..B_{N-1}
    for n in range(N):
        two = apply_cnot(two, n, N + n)
    for n in range(N):
        two = apply_1q(two, H, n)
    p = np.abs(two.reshape(-1)) ** 2
    return p / p.sum()
