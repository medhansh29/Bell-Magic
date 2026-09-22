"""Bell-magic error mitigation for global depolarizing noise.

Derivation (matches paper App. I; the paper's Sec. V p_c = 1-(1-p)^4 is the
error probability, App. I uses (1-p)^4 for the same symbol):
    Q_dp = a^4 Q_0 + (1-a^4) U,   a = 1-p,   U = 4^-N uniform,
    sum_q U(q) W(r,q) = [r != 0]  (W = 2*[anticommute]),
so, bilinearly, with pc = 1-a^4 and B_R = 1 - Q_0(0):
    B_dp = a^8 B_0 + 2 a^4 pc B_R + pc^2 (1 - 4^-N)
and Q_0(0) is recovered from the measured collision probability
    sum_r P_dp(r)^2 = Q_dp(0) = a^4 Q_0(0) + pc 4^-N.
"""
import numpy as np
from ..core.exact import exact_B
from ..core.sampling import sample_B
from .depolarizing import purity_from_probs, purity_from_samples, p_from_purity


def mitigate_from_moments(B_dp, collision, p, N):
    """B_0 from noisy B_dp, collision = sum_r P_dp(r)^2, and depolarizing p."""
    a4 = (1.0 - p) ** 4
    pc = 1.0 - a4
    Q0_0 = (collision - pc * 4.0 ** -N) / a4
    B_R = 1.0 - Q0_0
    return (B_dp - 2 * a4 * pc * B_R - pc ** 2 * (1 - 4.0 ** -N)) / a4 ** 2


def mitigate_B(P_dp, p=None):
    """Exact (infinite-shot) mitigation from the noisy Bell distribution.
    If p is None it is estimated from the purity."""
    N = int(round(np.log2(len(P_dp)) / 2))
    if p is None:
        p = p_from_purity(purity_from_probs(P_dp), N)
    return mitigate_from_moments(exact_B(P_dp), float(np.dot(P_dp, P_dp)), p, N)


def collision_from_samples(samples):
    """Unbiased estimate of sum_r P(r)^2 (fraction of equal pairs)."""
    n = len(samples)
    _, c = np.unique(samples, return_counts=True)
    return float(np.sum(c * (c - 1)) / (n * (n - 1)))


def mitigate_B_samples(samples, N, n_r, rng, p=None):
    """Sampled mitigation from N_Q noisy Bell outcomes. Returns (B_mtg, p_used).
    p is estimated from the purity of the same samples when not given."""
    if p is None:
        p = p_from_purity(purity_from_samples(samples, N), N)
    B_dp = sample_B(samples, N, n_r, rng)
    return mitigate_from_moments(B_dp, collision_from_samples(samples), p, N), p
