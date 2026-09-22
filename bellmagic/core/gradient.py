"""Shift-rule gradient of Bell magic (paper Sec. VIII, Algorithm 3, p. 8).

For each parameter k the extra Bell distributions are those of
|psi(theta +- pi/2 e_k)> (x) |psi(theta)>. With Q = P*P and R_pm = P*P_pm (XOR
convolutions), P_anti(A, B) = (1 - sum_a A(a) B^(swap a)) / 2 is the probability
that a draw from A anticommutes with a draw from B, and
    dB/dtheta_k = GRAD_SCALE * (P_anti(Q, R_+) - P_anti(Q, R_-)).
GRAD_SCALE is pinned against finite differences in tests/test_gradient.py."""
import numpy as np
from .exact import walsh_hadamard, swap_halves
from .sampling import check_commute

GRAD_SCALE = 8.0


def anti_prob_exact(P_a, P_b, P_c, P_d):
    """Pr[(r1^r2) anticommutes with (r3^r4)], r1,r2~P_a,P_b and r3,r4~P_c,P_d."""
    N = int(round(np.log2(P_a.size) / 2))
    Q = walsh_hadamard(walsh_hadamard(P_a) * walsh_hadamard(P_b)) / P_a.size
    Rhat = walsh_hadamard(P_c) * walsh_hadamard(P_d)
    return float(0.5 * (1.0 - np.dot(Q, Rhat[swap_halves(N)])))


def grad_component_exact(P, P_plus, P_minus):
    """Infinite-shot dB/dtheta_k from Bell distributions of psi(x)psi and the two
    shifted pairs; the shifted state takes the place of one of the four copies."""
    return GRAD_SCALE * (anti_prob_exact(P, P, P, P_plus) - anti_prob_exact(P, P, P, P_minus))


def grad_component_sampled(r, q_plus, q_minus, N, n_r, rng):
    """Algorithm 3: r holds 3*N_Q samples of psi(x)psi, q_plus/q_minus N_Q each."""
    r = np.asarray(r, dtype=np.uint64)
    n3 = r.size
    out = []
    for q in (q_plus, q_minus):
        q = np.asarray(q, dtype=np.uint64)
        idx = np.empty((0, 3), dtype=np.int64)
        while idx.shape[0] < n_r:
            d = rng.integers(0, n3, size=(2 * (n_r - idx.shape[0]) + 16, 3))
            s = np.sort(d, axis=1)
            idx = np.concatenate([idx, d[(np.diff(s, axis=1) != 0).all(axis=1)]])
        idx = idx[:n_r]
        m = rng.integers(0, q.size, size=n_r)
        out.append(np.mean(~check_commute(r[idx[:, 0]] ^ r[idx[:, 1]], r[idx[:, 2]] ^ q[m], N)))
    return GRAD_SCALE * (out[0] - out[1])
