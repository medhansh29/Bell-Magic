"""Statevector helpers. A state of N qubits is a complex array of shape (2,)*N,
axis n = qubit n (qubit 0 is the most significant bit when flattened)."""
import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
T_GATE = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)


def A_phi(phi):
    """Single-qubit |A_phi> = cos(phi/2)|0> + sin(phi/2)|1> (paper, p. 5)."""
    return np.array([np.cos(phi / 2), np.sin(phi / 2)], dtype=complex)


def product_state(singles):
    """Tensor product of single-qubit vectors -> shape (2,)*N."""
    psi = np.ones((), dtype=complex)
    for v in singles:
        psi = np.multiply.outer(psi, np.asarray(v, dtype=complex))
    return psi


def apply_1q(psi, U, n):
    out = np.tensordot(U, psi, axes=([1], [n]))
    return np.moveaxis(out, 0, n)


def apply_cnot(psi, c, t):
    out = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[c] = 1
    sub = out[tuple(idx)]
    # after fixing axis c, target axis shifts down by one if t > c
    ta = t - 1 if t > c else t
    out[tuple(idx)] = np.flip(sub, axis=ta)
    return out


def random_clifford_circuit(N, depth, rng):
    """Random circuit of H, S, CNOT layers as a list of gate tuples."""
    gates = []
    for _ in range(depth):
        for n in range(N):
            gates.append((rng.choice(["h", "s"]), n))
        if N > 1:
            c, t = rng.choice(N, size=2, replace=False)
            gates.append(("cx", int(c), int(t)))
    return gates


def apply_circuit(psi, gates):
    for g in gates:
        if g[0] == "h":
            psi = apply_1q(psi, H, g[1])
        elif g[0] == "s":
            psi = apply_1q(psi, S, g[1])
        elif g[0] == "t":
            psi = apply_1q(psi, T_GATE, g[1])
        elif g[0] == "cx":
            psi = apply_cnot(psi, g[1], g[2])
        else:
            raise ValueError(g)
    return psi
