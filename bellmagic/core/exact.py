import numpy as np


def walsh_hadamard(a):
    """Unnormalised Walsh-Hadamard transform of a length-2^k array."""
    a = np.array(a, dtype=float)
    n = a.size
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.concatenate([a[:, 0:1] + a[:, 1:2], a[:, 0:1] - a[:, 1:2]], axis=1)
        h *= 2
    return a.reshape(-1)


def swap_halves(N):
    """Permutation mapping index r=x*2^N+z to z*2^N+x."""
    d = 1 << N
    return np.arange(d * d).reshape(d, d).T.reshape(-1)


def exact_B(P):
    """Infinite-shot Bell magic from the Bell outcome distribution P (length 4^N).

    Q = P (*) P is the XOR self-convolution, Q^ = WHT(P)^2 (convolution theorem).
    Two Paulis r, q anticommute iff omega(r,q)=1 with omega the symplectic form,
    and sum_q Q(q)(-1)^omega(r,q) = Q^(swap(r)). With B = 2*Pr[anticommute]:
        B = 1 - sum_r Q(r) Q^(swap(r)).
    """
    P = np.asarray(P, dtype=float)
    N = int(round(np.log2(P.size) / 2))
    if 4 ** N != P.size:
        raise ValueError("P must have length 4^N")
    Phat = walsh_hadamard(P)
    Qhat = Phat ** 2
    Q = walsh_hadamard(Qhat) / P.size
    return float(1.0 - np.dot(Q, Qhat[swap_halves(N)]))


def additive(B):
    """B_a = -log2(1 - B)."""
    return float(-np.log2(1.0 - B)) if B < 1.0 else float("inf")
