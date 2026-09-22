"""Phase 7 (extension, not from the paper): "break the mitigation."

bellmagic.noise.mitigation was derived for one specific noise model -- a
global depolarizing channel (Sec. V / App. I of the paper). It estimates a
single effective p from the measured purity and inverts a formula that's
only exact for that channel. This experiment checks what mitigate_B does
when it's fed noise that has the *same purity loss* but a different physical
origin: amplitude damping (T1-style decay towards |0>) and a coherent
(systematic, non-random unitary) rotation error, both applied per-qubit via
the exact density-matrix pipeline in core.density -- there's no closed form
for either on the Bell outcome distribution the way there is for global
depolarizing, so there's no shortcut around simulating the density matrix.

Everything here is exact (no sampling shots), so any bias found is the
mitigation formula's own model-mismatch bias, not Monte Carlo noise. For
each noisy instance we also build a *matched* depolarizing control at the
same effective purity, mitigated with the true p -- this arm recovers B0 to
numerical precision by construction (see test_mitigation_bias.py), and is
the sanity check that isolates the real channels' error as a genuine
model-mismatch effect rather than a bug in mitigate_B itself.

What actually shows up, at 30 trials per point:
- Amplitude damping DOES reduce purity, so mitigate_B applies a nonzero
  correction. Averaged over instances the correction is roughly unbiased
  (mean close to 0), but its trial-to-trial spread grows sharply with
  strength (std ~0.004 at strength 0.02, ~0.27 at 0.35): a single
  experiment's "corrected" B can be far off even though the method isn't
  systematically wrong on average.
- Coherent (purely unitary) error NEVER reduces purity -- it's
  non-dissipative by construction -- so mitigate_B always infers p_eff=0
  and applies literally no correction: mitigate_B(P_noisy) equals
  exact_B(P_noisy) bit-for-bit (see
  test_coherent_error_gets_zero_mitigation_correction). The "bias" plotted
  for this channel is just the raw, entirely uncorrected effect of the
  miscalibration -- a clean, one-sided, monotonically growing error, not
  noise. Purity-based mitigation is structurally blind to this failure mode.
"""
import argparse
import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.core import product_state, apply_circuit, doped_clifford_gates
from bellmagic.core.density import bell_probs_density
from bellmagic.noise import mitigate_B, purity_from_probs, p_from_purity, depolarized_probs, noisy_density

CHANNEL_STRENGTHS = {
    "amplitude_damping": [0.0, 0.02, 0.05, 0.1, 0.2, 0.35],
    "coherent": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5],
}


def instance(N, n_t, rng, depth):
    gates = doped_clifford_gates(N, n_t, rng, depth)
    return apply_circuit(product_state([[1, 0]] * N), gates)


def one_point(N, n_t, depth, kind, strength, rng):
    psi = instance(N, n_t, rng, depth)
    P0 = bell_probs(psi)
    B0 = exact_B(P0)

    rho = noisy_density(psi, kind, strength)
    P_noisy = bell_probs_density(rho)
    B_mtg = mitigate_B(P_noisy)  # p estimated from P_noisy's own purity, as a real experiment would

    purity = purity_from_probs(P_noisy)
    p_eff = p_from_purity(purity, N)
    P_dep = depolarized_probs(P0, p_eff)
    B_mtg_dep = mitigate_B(P_dep, p=p_eff)  # matched depolarizing control, true p given

    return B0, purity, B_mtg - B0, B_mtg_dep - B0


def run(N=4, n_t=3, depth=4, trials=8, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for kind, strengths in CHANNEL_STRENGTHS.items():
        rows = []
        for x in strengths:
            pts = np.array([one_point(N, n_t, depth, kind, x, rng) for _ in range(trials)])
            rows.append({
                "strength": x,
                "purity": float(pts[:, 1].mean()),
                "bias_real": float(pts[:, 2].mean()),
                "bias_real_std": float(pts[:, 2].std()),
                "bias_matched_depolarizing": float(pts[:, 3].mean()),
            })
        out[kind] = rows
    return out


def report(out):
    for kind, rows in out.items():
        print(f"\n{kind}:")
        print(f"  {'strength':>9} {'purity':>8} {'bias (real channel)':>22} {'bias (matched dep. control)':>28}")
        for r in rows:
            print(f"  {r['strength']:>9.3f} {r['purity']:>8.4f} "
                  f"{r['bias_real']:>+14.4f} (+/-{r['bias_real_std']:.4f}) "
                  f"{r['bias_matched_depolarizing']:>+20.4f}")


def plot(out, path):
    import matplotlib.pyplot as plt
    pal = {"amplitude_damping": "#eb6834", "coherent": "#2a78d6"}
    ink, mute = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=mute)
    ax.grid(color="#e6e5e0", lw=0.8), ax.set_axisbelow(True)
    ax.axhline(0.0, color=mute, lw=1, ls="--")
    for kind, rows in out.items():
        purity = [r["purity"] for r in rows]
        bias = [r["bias_real"] for r in rows]
        std = [r["bias_real_std"] for r in rows]
        ax.errorbar(purity, bias, yerr=std, fmt="o-", color=pal[kind], ms=6,
                     mec="#fcfcfb", mew=1.2, capsize=3, lw=1.8, label=kind)
    dep_purity = [r["purity"] for r in out["amplitude_damping"]]
    dep_bias = [r["bias_matched_depolarizing"] for r in out["amplitude_damping"]]
    ax.plot(dep_purity, dep_bias, "^--", color="#1baf7a", ms=6, mec="#fcfcfb",
             mew=1.2, lw=1.5, label="matched depolarizing (control)")
    ax.set_xlabel("measured purity of the noisy state", color=ink)
    ax.set_ylabel(r"mitigation bias  $\hat{B}_{\mathrm{mtg}} - B_0$", color=ink)
    ax.set_title("Phase 7: mitigate_B applied outside its depolarizing assumption", loc="left", color=ink, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mitigation_bias.png")
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    res = run(trials=args.trials)
    report(res)
    plot(res, args.out)
