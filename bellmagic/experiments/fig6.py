"""Simulated reproduction of paper Fig. 6 (p. 9): variational maximisation of Bell
magic with Adam and shift-rule gradients from N_Q samples per setting (Alg. 3).
N=4, depth d=6, learning rate 0.1 (paper). Plotted: B_max - <B> where B is the
EXACT Bell magic of the trained state and B_max = 1 - 2^-6.221364 is the paper's
reported near-maximal N=4 value (App. G). Init: Clifford angles + Gaussian noise
(paper only says 'close to a stabilizer state'; sigma is our choice).
Samples per epoch: 3 N_Q for psi(x)psi plus N_Q per shifted setting."""
import argparse
import time
import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.core.gradient import grad_component_exact, grad_component_sampled
from bellmagic.core.sampling import sample_bitstrings
from bellmagic.experiments.fig5 import hea_state, random_theta

N, D, LR, INIT_SIGMA = 4, 6, 0.1, 0.1
B_MAX = 1 - 2.0 ** -6.221364


def gradient(theta, n_q, rng):
    """dB/dtheta; n_q=None gives the exact (infinite-shot) gradient."""
    psi = hea_state(theta)
    P = bell_probs(psi)
    if n_q is not None:
        r = sample_bitstrings(P, 3 * n_q, rng)
    g = np.zeros(theta.size)
    flat = theta.reshape(-1)
    for k in range(flat.size):
        e = np.zeros_like(flat)
        e[k] = np.pi / 2
        Pp = bell_probs(hea_state((flat + e).reshape(theta.shape)), psi)
        Pm = bell_probs(hea_state((flat - e).reshape(theta.shape)), psi)
        if n_q is None:
            g[k] = grad_component_exact(P, Pp, Pm)
        else:
            g[k] = grad_component_sampled(r, sample_bitstrings(Pp, n_q, rng),
                                          sample_bitstrings(Pm, n_q, rng), N, 10 * n_q, rng)
    return g.reshape(theta.shape)


def train(n_q, epochs, rng, lr=LR):
    theta = random_theta(False, rng, N, D) + rng.normal(0, INIT_SIGMA, size=(D + 1, N, 2))
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    hist = []
    for t in range(1, epochs + 1):
        hist.append(exact_B(bell_probs(hea_state(theta))))
        g = gradient(theta, n_q, rng)  # ascent on B
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        theta = theta + lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    hist.append(exact_B(bell_probs(hea_state(theta))))
    return np.array(hist)


def haar_mean_B(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
        out.append(exact_B(bell_probs((v / np.linalg.norm(v)).reshape((2,) * N))))
    return float(np.mean(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--nq", type=int, nargs="*", default=[10, 100, 1000])
    ap.add_argument("--exact-instances", type=int, default=2)
    ap.add_argument("--out", default="fig6.png")
    ap.add_argument("--time-only", action="store_true")
    a = ap.parse_args()
    if a.time_only:
        rng = np.random.default_rng(0)
        t0 = time.time()
        train(100, 2, rng)
        print("s/epoch (N_Q=100):", (time.time() - t0) / 2)
        raise SystemExit
    rng = np.random.default_rng(0)
    curves = {}
    for n_q in [None] + a.nq:
        curves[n_q] = np.mean([train(n_q, a.epochs, rng) for _ in range(a.exact_instances if n_q is None else a.instances)], axis=0)
        print("N_Q", n_q, "final B", round(curves[n_q][-1], 4), "gap", f"{B_MAX - curves[n_q][-1]:.2e}", flush=True)
    haar = haar_mean_B()
    print("Haar mean B", round(haar, 4), "B_max", round(B_MAX, 4))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for c, (k, y) in zip(["#52514e", "#2a78d6", "#eb6834", "#1baf7a"], curves.items()):
        ax.loglog(np.arange(1, len(y) + 1), B_MAX - y, color=c, label="exact gradient" if k is None else f"$N_Q$={k}")
    ax.axhline(B_MAX - haar, color="#8a5cd0", ls="--", label="Haar random")
    ax.set_xlabel("epoch"), ax.set_ylabel(r"$B_{max} - \langle B\rangle$"), ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(a.out, dpi=160)
