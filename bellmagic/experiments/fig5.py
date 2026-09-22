"""Simulated reproduction of paper Fig. 5 (p. 8): classify stabilizer vs highly
magical N=3 states from noisy Bell-magic estimates with a learned threshold B*.
Circuit (App. H, Fig. 7): d layers of R_y, R_z on every qubit then a nearest-
neighbour CNOT chain (final rotation layer added by us - the paper's K is not
retrieved). Magical: angles ~ U[0, 2pi). Stabilizer: angles in {n pi/2}.
Noise: global depolarizing p=0.15 (paper's measured average); mitigation uses
that known p. Learner (App. J): B* maximising training accuracy. Error is on a
20% test split averaged over 10 random splits; 20 states per class."""
import argparse
import numpy as np
from bellmagic import bell_probs
from bellmagic.core import product_state, apply_1q, apply_cnot
from bellmagic.core.sampling import sample_bitstrings, sample_B
from bellmagic.noise import depolarized_probs
from bellmagic.noise.mitigation import mitigate_from_moments, collision_from_samples

N, D, P_NOISE = 3, 2, 0.15


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def hea_state(theta):
    """theta: shape (d+1, n, 2) of (y, z) angles; n and d are read from the shape."""
    d, n = theta.shape[0] - 1, theta.shape[1]
    psi = product_state([[1, 0]] * n)
    for layer in range(d + 1):
        for q in range(n):
            psi = apply_1q(apply_1q(psi, ry(theta[layer, q, 0]), q), rz(theta[layer, q, 1]), q)
        if layer < d:
            for q in range(n - 1):
                psi = apply_cnot(psi, q, q + 1)
    return psi


def random_theta(magical, rng, n=N, d=D):
    if magical:
        return rng.uniform(0, 2 * np.pi, size=(d + 1, n, 2))
    return rng.integers(0, 4, size=(d + 1, n, 2)) * (np.pi / 2)


def learn_threshold(b, y):
    """App. J: B* maximising training accuracy (y=+1 magical if b > B*)."""
    order = np.sort(b)
    cands = np.concatenate([[order[0] - 1], (order[:-1] + order[1:]) / 2, [order[-1] + 1]])
    acc = [np.mean(np.where(b > t, 1, -1) == y) for t in cands]
    best = np.flatnonzero(np.isclose(acc, max(acc)))
    return float(cands[best[len(best) // 2]])  # middle of the tied optimal range


def estimate(psi_P, n_q, p, rng):
    s = sample_bitstrings(depolarized_probs(psi_P, p), n_q, rng)
    b = sample_B(s, N, 10 * n_q, rng)
    return mitigate_from_moments(b, collision_from_samples(s), p, N)


def run(n_qs=(4, 10, 30, 100, 300, 1000), n_per=20, splits=10, p=P_NOISE, seed=0):
    rng = np.random.default_rng(seed)
    Ps = [bell_probs(hea_state(random_theta(m, rng))) for m in (False, True) for _ in range(n_per)]
    y = np.array([-1] * n_per + [1] * n_per)
    errs = []
    for n_q in n_qs:
        b = np.array([estimate(P, n_q, p, rng) for P in Ps])
        e = []
        for _ in range(splits):
            test = rng.permutation(len(y))[: len(y) // 5]
            train = np.setdiff1d(np.arange(len(y)), test)
            t = learn_threshold(b[train], y[train])
            e.append(np.mean(np.where(b[test] > t, 1, -1) != y[test]))
        errs.append(float(np.mean(e)))
    return list(n_qs), errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig5.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_qs, errs = run(seed=a.seed)
    for n, e in zip(n_qs, errs):
        print("N_Q", n, "P_E", round(e, 3))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogx(n_qs, errs, "o-", color="#eb6834")
    ax.axhline(0.5, color="#52514e", ls="--", lw=1)
    ax.set_xlabel(r"$N_Q$"), ax.set_ylabel(r"$P_E$"), ax.set_ylim(0, 0.55)
    fig.tight_layout(), fig.savefig(a.out, dpi=160)
