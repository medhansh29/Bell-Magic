"""Density-matrix tensor helpers, generalising core.states to mixed states.

A density matrix of N qubits is a complex array of shape (2,)*N + (2,)*N:
axes 0..N-1 are ket indices, axes N..2N-1 the matching bra indices, in the
same qubit-0-most-significant convention as states.py. Flattened to a
2^N x 2^N matrix via `.reshape(2**N, 2**N)`, axis order gives the ordinary
rho[i, j].

`apply_1q`/`apply_cnot` from states.py are generic tensor ops (they act on
whichever axis index they're given, regardless of the tensor's overall
rank), so they're reused directly here for both the ket- and bra-side
action, rather than reimplemented.
"""
import numpy as np
from .states import H, apply_1q, apply_cnot


def density_from_state(psi):
    """Pure-state density matrix |psi><psi|, shape (2,)*N + (2,)*N."""
    psi = np.asarray(psi, dtype=complex)
    return np.multiply.outer(psi, psi.conj())


def apply_kraus_1q(rho, kraus_ops, n, N):
    """rho -> sum_k K_k rho K_k^dagger, each K_k a 2x2 op on qubit n (ket
    axis n, bra axis N+n). Kraus operators need not be unitary; a single
    unitary K applies a coherent (non-dissipative) error."""
    out = None
    for K in kraus_ops:
        term = apply_1q(rho, K, n)
        term = apply_1q(term, K.conj(), N + n)
        out = term if out is None else out + term
    return out


def apply_unitary_1q(rho, U, n, N):
    return apply_kraus_1q(rho, [U], n, N)


def apply_cnot_dm(rho, c, t, N):
    """CNOT is real and an involution (its own adjoint and inverse), so the
    same index-flip implements both the ket- and bra-side conjugation."""
    rho = apply_cnot(rho, c, t)
    rho = apply_cnot(rho, N + c, N + t)
    return rho


def probs_from_density(rho):
    """Diagonal of the (2^N x 2^N) density matrix as an outcome distribution."""
    N = rho.ndim // 2
    d = 2 ** N
    p = np.real(np.diagonal(np.asarray(rho).reshape(d, d)))
    return p / p.sum()


def two_copies_density(rho):
    """rho_A (x) rho_B for rho_A = rho_B = rho, as a single (2N)-qubit
    density-matrix tensor with axes ordered [A_ket, B_ket, A_bra, B_bra]."""
    N = rho.ndim // 2
    combined = np.multiply.outer(rho, rho)  # axes: A_ket, A_bra, B_ket, B_bra
    perm = list(range(N)) + list(range(2 * N, 3 * N)) + list(range(N, 2 * N)) + list(range(3 * N, 4 * N))
    return np.transpose(combined, perm)


def bell_probs_density(rho):
    """Mixed-state analogue of core.bell.bell_probs: outcome distribution of
    the Bell measurement on rho (x) rho (two independent noisy copies).
    Reduces to bell_probs(psi) when rho = density_from_state(psi)."""
    N = rho.ndim // 2
    rho2 = two_copies_density(rho)
    Np = 2 * N
    for n in range(N):
        rho2 = apply_cnot_dm(rho2, n, N + n, Np)
    for n in range(N):
        rho2 = apply_unitary_1q(rho2, H, n, Np)
    return probs_from_density(rho2)
