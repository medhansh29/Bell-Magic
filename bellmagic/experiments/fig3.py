"""Reproduce paper Fig. 3 (noisy simulation, not hardware): additive Bell magic
for (a) |+>^3, |T>^3, |R>^3, |psi_max> and (b) B_a vs number of T gates N_T.
N=3, N_Q=1000, 6 instances, global depolarizing p=0.1 (paper's IonQ estimate).
B_hat is clipped to [0, B_cap] with B_cap the maximal pure-state value for N=3."""
import argparse
import numpy as np
from bellmagic import bell_probs, exact_B, additive
from bellmagic.core import doped_clifford_state, product_state
from bellmagic.core.sampling import sample_bitstrings, sample_B
from bellmagic.noise import depolarized_probs, mitigate_B_samples
from bellmagic.experiments.max_magic import maximise

N = 3
B_CAP = 1 - 2.0 ** -4.651794


def paper_states():
    th = np.arccos(1 / np.sqrt(3))
    T = np.array([1, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)
    R = np.array([np.cos(th / 2), np.exp(-1j * np.pi / 4) * np.sin(th / 2)])
    plus = np.array([1, 1]) / np.sqrt(2)
    _, psi_max = maximise(3, restarts=40)
    return {
        r"$|+\rangle^{\otimes 3}$": product_state([plus] * 3),
        r"$|T\rangle^{\otimes 3}$": product_state([T] * 3),
        r"$|R\rangle^{\otimes 3}$": product_state([R] * 3),
        r"$|\psi_{\max}\rangle$": psi_max,
    }


def measure(psi, p, n_q, rng):
    """(exact, unmitigated, mitigated) additive Bell magic from one noisy run."""
    P = bell_probs(psi)
    s = sample_bitstrings(depolarized_probs(P, p), n_q, rng)
    ba = lambda B: additive(float(np.clip(B, 0.0, B_CAP)))
    return (
        additive(exact_B(P)),
        ba(sample_B(s, N, 10 * n_q, rng)),
        ba(mitigate_B_samples(s, N, 10 * n_q, rng)[0]),  # p estimated from purity
    )


def haar_mean_Ba(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        out.append(additive(exact_B(bell_probs((v / np.linalg.norm(v)).reshape((2,) * 3)))))
    return float(np.mean(out))


def run(p=0.1, n_q=1000, instances=6, max_nt=8, seed=0):
    rng = np.random.default_rng(seed)
    a = {k: np.array([measure(psi, p, n_q, rng) for _ in range(instances)])
         for k, psi in paper_states().items()}
    b = {nt: np.array([measure(doped_clifford_state(N, nt, rng), p, n_q, rng)
                       for _ in range(instances)]) for nt in range(max_nt + 1)}
    return a, b, haar_mean_Ba()


def plot(a, b, haar, path):
    import matplotlib.pyplot as plt
    C = {"exact": "#2a78d6", "unmitigated": "#eb6834", "mitigated": "#1baf7a"}
    MK = {"exact": "o", "unmitigated": "s", "mitigated": "^"}
    ink, mute = "#0b0b0b", "#52514e"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c2")
        ax.tick_params(colors=mute)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
    names = list(a)
    x = np.arange(len(names))
    for j, k in enumerate(("exact", "unmitigated", "mitigated")):
        m = [a[n][:, j].mean() for n in names]
        e = [a[n][:, j].std() for n in names]
        ax1.errorbar(x + (j - 1) * 0.16, m, yerr=e, fmt=MK[k], color=C[k], ms=7,
                     mec="#fcfcfb", mew=1.5, capsize=3, lw=1.5, label=k)
    ax1.set_xticks(x, names, color=ink)
    ax1.set_ylabel(r"additive Bell magic $B_a$", color=ink)
    ax1.set_title("(a) states, N = 3", loc="left", color=ink, fontsize=11)
    nts = sorted(b)
    for j, k in enumerate(("exact", "unmitigated", "mitigated")):
        m = np.array([b[n][:, j].mean() for n in nts])
        e = np.array([b[n][:, j].std() for n in nts])
        ax2.plot(nts, m, MK[k] + "-", color=C[k], lw=2, ms=7, mec="#fcfcfb", mew=1.5, label=k)
        ax2.fill_between(nts, m - e, m + e, color=C[k], alpha=0.15, lw=0)
        ax2.annotate(k, (nts[-1], m[-1]), xytext=(6, {"exact": -9, "unmitigated": 0, "mitigated": 9}[k]),
                     textcoords="offset points",
                     color=ink, fontsize=9, va="center")
    ax2.axhline(haar, color=mute, ls="--", lw=1)
    ax2.text(0.05, haar + 0.08, f"Haar average {haar:.2f}", color=mute, fontsize=9)
    ax2.set_xlabel(r"number of T gates $N_T$", color=ink)
    ax2.set_title("(b) doped Clifford circuit, N = 3", loc="left", color=ink, fontsize=11)
    ax2.set_xlim(-0.3, nts[-1] + 1.9)
    ax1.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig3.png")
    args = ap.parse_args()
    a, b, haar = run()
    for k, v in a.items():
        print(k, v.mean(0).round(3), "(exact, unmit, mtg)")
    for nt, v in b.items():
        print("N_T", nt, v.mean(0).round(3))
    print("Haar mean B_a", round(haar, 3))
    plot(a, b, haar, args.out)
