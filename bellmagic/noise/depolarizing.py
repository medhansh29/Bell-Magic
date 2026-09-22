"""Global depolarizing model rho_dp = (1-p)|psi><psi| + p I/2^N with a noiseless
Bell measurement (paper Sec. V, App. I)."""
import numpy as np
from ..core.sampling import _parity


def depolarized_probs(P, p):
    """Bell distribution of rho_dp (x) rho_dp: (1-p)^2 P + (1-(1-p)^2) / 4^N."""
    P = np.asarray(P, dtype=float)
    a2 = (1.0 - p) ** 2
    return a2 * P + (1.0 - a2) / P.size


def _swap_sign(N):
    """(-1)^{popcount(a & b)} over outcome index r = a*2^N + b: the eigenvalue of
    SWAP on the Bell outcome (each singlet (1,1) contributes -1)."""
    r = np.arange(4 ** N, dtype=np.uint64)
    a, b = r >> np.uint64(N), r & np.uint64((1 << N) - 1)
    return 1.0 - 2.0 * _parity(a & b).astype(float)


def purity_from_probs(P):
    """tr(rho^2) = <SWAP> on rho (x) rho = sum_r P(r) (-1)^{popcount(a&b)}."""
    P = np.asarray(P, dtype=float)
    N = int(round(np.log2(P.size) / 2))
    return float(np.dot(P, _swap_sign(N)))


def purity_from_samples(samples, N):
    s = np.asarray(samples, dtype=np.uint64)
    a, b = s >> np.uint64(N), s & np.uint64((1 << N) - 1)
    return float(np.mean(1.0 - 2.0 * _parity(a & b).astype(float)))


def dp_purity(p, N):
    """Analytic purity of rho_dp: (1-p)^2 + p(2-p)/2^N."""
    return (1 - p) ** 2 + p * (2 - p) / 2 ** N


def p_from_purity(purity, N):
    """Invert dp_purity: p = 1 - sqrt((purity - 1/d)/(1 - 1/d)), clipped to [0,1]."""
    d = 2.0 ** N
    x = (purity - 1.0 / d) / (1.0 - 1.0 / d)
    return float(1.0 - np.sqrt(np.clip(x, 0.0, 1.0)))
