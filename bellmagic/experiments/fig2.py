"""Scaled-down reproduction of paper Fig. 2 (p. 6): estimation error of the
mitigated Bell magic, Delta_B = <|B_mtg - B_exact|>, on U_C |A_phi>^{N_A} |0>^{N-N_A}.
Paper uses N=50 (tensor networks); here N=8 so the exact oracle is available.
  (a) Delta_B vs N_Q for several p        (paper slopes ~ -0.5; N_A=3, N_R=10 N_Q)
  (b) Delta_B vs (1-p), fit vs (1-p)^b    (paper b ~ -6..-8;   N_A=1,3,5, N_Q=1e4)
  (c) Delta_B vs N_R for several N_Q      (paper: saturates for N_R >> N_Q; N_A=1)
The Clifford circuit is deeper than the paper's depth-4 hardware circuit so that
the state is actually scrambled with our sparse (one CNOT per layer) generator."""
import argparse
import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.core import A_phi, product_state, apply_circuit, random_clifford_circuit
from bellmagic.core.sampling import sample_bitstrings
from bellmagic.noise import depolarized_probs, mitigate_B_samples

N = 8
PHI = np.pi / 4


def instance(n_a, rng, n=N, depth=16):
    singles = [A_phi(PHI)] * n_a + [[1, 0]] * (n - n_a)
    psi = apply_circuit(product_state(singles), random_clifford_circuit(n, depth, rng))
    P = bell_probs(psi)
    return P, exact_B(P)


def delta_B(n_a, p, n_q, n_r_factor, reps, rng, n_r=None, stat=np.mean):
    """`stat` (mean, as in the paper, or median) of |B_mtg - B_exact| over `reps`
    fresh random circuits. The mean is heavy-tailed at large p and small N_Q: a
    purity-estimated p near 1 makes the (1-p)^-8 factor explode."""
    errs = []
    for _ in range(reps):
        P, B0 = instance(n_a, rng)
        s = sample_bitstrings(depolarized_probs(P, p), n_q, rng)
        b, _ = mitigate_B_samples(s, N, n_r if n_r else int(n_r_factor * n_q), rng)
        errs.append(abs(b - B0))
    return float(stat(errs))


def slope(x, y):
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


def run(reps=100, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    out = {}
    nqs = np.array([40, 100, 250, 600, 1500, 4000])
    ps = [0.0, 0.1, 0.2, 0.3]
    out["a"] = (nqs, ps, {p: np.array([delta_B(3, p, q, 10, reps, rng, stat=stat) for q in nqs]) for p in ps})
    ones = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    out["b"] = (ones, [1, 3, 5], {na: np.array([delta_B(na, 1 - a, 10 ** 4, 10, reps, rng, stat=stat) for a in ones])
                                  for na in (1, 3, 5)})
    nrs = np.array([10, 100, 1000, 10 ** 4, 10 ** 5])
    nqc = [40, 100, 200, 1000]
    out["c"] = (nrs, nqc, {q: np.array([delta_B(1, 0.1, q, 0, reps, rng, n_r=r, stat=stat) for r in nrs]) for q in nqc})
    return out


def plot(res, path):
    import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0"]
    ink, mute = "#0b0b0b", "#52514e"
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axs:
        ax.set_facecolor("#fcfcfb")
        ax.set_xscale("log"), ax.set_yscale("log")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=mute)
        ax.grid(color="#e6e5e0", lw=0.8), ax.set_axisbelow(True)
    x, keys, d = res["a"]
    for c, k in zip(pal, keys):
        b = slope(x, d[k])
        axs[0].plot(x, d[k], "o-", color=c, label=f"p={k} (slope {b:.2f})")
    axs[0].set_xlabel(r"$N_Q$"), axs[0].set_ylabel(r"$\Delta B$"), axs[0].set_title("(a) N_A=3", loc="left")
    x, keys, d = res["b"]
    for c, k in zip(pal, keys):
        b = slope(x, d[k])
        axs[1].plot(x, d[k], "o-", color=c, label=f"N_A={k} (slope {b:.1f})")
    axs[1].set_xlabel("1 - p"), axs[1].set_title(r"(b) $N_Q=10^4$", loc="left")
    x, keys, d = res["c"]
    for c, k in zip(pal, keys):
        axs[2].plot(x, d[k], "o-", color=c, label=f"N_Q={k}")
    axs[2].set_xlabel(r"$N_R$"), axs[2].set_title("(c) N_A=1, p=0.1", loc="left")
    for ax in axs:
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig2.png")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean")
    args = ap.parse_args()
    stat = np.mean if args.stat == "mean" else np.median
    res = run(args.reps, stat=stat)
    for name in "abc":
        x, keys, d = res[name]
        for k in keys:
            print(name, k, "slope", round(slope(x, d[k]), 3), d[k].round(4))
    plot(res, args.out)
