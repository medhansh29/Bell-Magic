# Reproducing and stress-testing Bell magic

*A from-scratch reproduction of Haug & Kim, "Bell magic" (arXiv:2204.10061),
its error mitigation, a Qiskit validation layer, and an extension asking
where that mitigation breaks.*

## 1. What Bell magic measures, and why mitigation is non-trivial

The paper defines a magic (non-stabilizerness) measure, B, extracted from
the outcome statistics of a Bell measurement on two copies of a state
|psi⟩. Two independent Pauli strings are read off from the measurement
outcomes; B is (twice) the probability that they anticommute. Stabilizer
states give B = 0; magic states push it up. The same construction gives a
convolution identity — B is a bilinear functional of the outcome
distribution P via its Walsh-Hadamard transform — that makes B computable
either exactly from P or estimated from N_Q samples (the paper's Algorithm
2), without ever reconstructing the state.

Depolarizing noise on the state before this measurement does *not* shrink B
toward 0. The maximally mixed state has B = 1 − 4⁻ᴺ, which is close to the
*maximum* possible value for small N. Raw, unmitigated B under noise is
biased upward, not degraded — which is exactly why the paper derives a
closed-form mitigation (App. I) that inverts a purity estimate to recover
the noiseless B. That inversion, and its limits, are the throughline of
this project.

## 2. Reproducing the figures

Everything reproduces the paper's qualitative shape; exact numeric match
was not the goal given the scale-down (N=8 vs. the paper's N=50 tensor
network simulation for Fig. 2, N=3–4 vs. unspecified N for Figs. 3/5/6).

**Fig. 2** (error vs. N_Q, vs. 1−p, vs. N_R). With the mean statistic, three
of four p-values matched the paper's slopes to within ~0.05 (mine: −0.50,
−0.51, −0.59 vs. paper −0.507, −0.505, −0.479 for p=0, 0.1, 0.2). The
p=0.3 curve was a clear outlier (−1.28 vs. −0.41): a single low-N_Q point
had a wildly overestimated error, traced to a purity estimate near 1 that
blows up the (1−p)⁻⁸ rescaling in the mitigation formula. Switching to the
median statistic — which the codebase now supports via `fig2.py --stat
median` — removed the outlier entirely (−0.54, in line with the other
three curves), confirming the diagnosis rather than papering over it.
Panel (b)'s (1−p)ᵇ fits (b ≈ −6.2, −7.4, −8.0 for N_A=1,3,5) matched the
paper's −6.3, −7.4, −7.3 closely. Panel (c) showed the same qualitative
saturation as N_R grows past N_Q, though a direct sweep (`nr_sweep.py`)
found this is diminishing returns rather than a sharp knee — error keeps
falling slowly out to N_R = 50·N_Q.

**Fig. 3** (additive magic B_a for stabilizer/magic states and a doped
Clifford circuit) reproduced the right ordering and trend under simulated
N=3, p=0.1 noise, mitigated values tracking the exact ones closely.

**Fig. 5** (stabilizer-vs-magic classifier) showed the same qualitative
error curve as the paper: from ~0.40 (near the 0.5 random-guess ceiling)
at N_Q=4 down to 0 by N_Q≈100, on one seed's worth of 20-state, 10-split
trials.

**Fig. 6** (variational maximization) initially capped out at B≈0.96
regardless of settings — traced to a real bug (`hea_state` defaulting to
N=3, d=2 instead of reading shape from the angle array), not a physics
mismatch. After the fix, the exact-gradient run reached B=0.9863 against
the paper's reported near-maximal B_max=0.9866 (N=4) — a gap of 3.3e-4.
The N_Q-sampled runs (10, 100, 1000) plateaued around 0.97–0.975, a
consistent ~1–2% gap to the exact run that the 10-instance sample size here
isn't enough to resolve cleanly (N_Q=10 outperforming N_Q=100 in one run).

## 3. Phase 6: a Qiskit/Aer cross-check

The numpy pipeline computes Bell probabilities via direct tensor
contraction, never running an actual quantum circuit. To check the whole
construction — not just the algebra — `bellmagic/backends/qiskit_backend.py`
builds the literal Bell-measurement circuit (CNOT then H per qubit pair, on
two copies of a prepared state) in Qiskit and runs it through Aer.

The main engineering hazard here was bit ordering: Qiskit indexes
statevectors and counts little-endian (qubit 0 least significant), while
this codebase's own convention is qubit-0-most-significant. Getting this
wrong doesn't error — it silently returns a plausible-looking but wrong
distribution. It was caught by testing against the trivial |0⟩⊗N case
first (exact expected outcome pair) before trusting anything else, then
confirmed against the numpy oracle across many random doped-Clifford
instances (agreement to ~1e-15, i.e., floating-point exact).

A second, more interesting hazard showed up when adding gate-level noise: a
naive attempt attached a Qiskit `NoiseModel` to gate names ("h", "cx").
That silently also corrupts the *Bell-measurement circuit's own* H and CNOT
gates, since they share those names with state prep — the result was
nonsensical (raw B jumping around with no relation to the injected noise).
The fix was to inject the noise as an explicit per-qubit channel right
after state prep, leaving the measurement circuit itself clean. With that
fixed, the sweep reproduced the same "B rises toward 1 − 4⁻ᴺ under noise"
fact established in the original Phase-3 physics checks — now from an
actual circuit simulator, not only the closed-form formula. A real IBM
Quantum Platform run is not included here; it needs account credentials
not yet configured, and is the natural next step once those are in place.

## 4. Phase 7: where the mitigation formula breaks

`bellmagic.noise.mitigation` inverts one specific noise model — global
depolarizing — from a single purity measurement. The natural stress test:
feed it noise with the same purity loss but a different physical origin,
and see what the "corrected" B actually recovers.

Two channels were implemented via an exact density-matrix simulation
(`bellmagic/core/density.py`), since neither has the closed form on the
Bell distribution that depolarizing noise does:

- **Amplitude damping** (T1-style decay toward |0⟩), applied independently
  to every qubit.
- **Coherent error**: a systematic, non-random over-rotation — a single
  unitary, applied to every qubit, with no randomness or dissipation at
  all.

For each, at a range of strengths, the true B₀ (noiseless), the mitigated
B̂ (purity estimated from the *actual* noisy distribution, exactly as an
experimentalist would), and a matched depolarizing control (same purity
loss, but the *true* p supplied) were computed over 30 random doped-Clifford
instances per point. The control recovers B₀ to numerical precision in
every case (bias ~1e-9 or smaller) — confirming any bias seen elsewhere is
a genuine model-mismatch effect, not a bug in the mitigation code.

**Amplitude damping** does reduce purity, so the mitigation engages and
applies a real correction. Averaged over instances the correction is close
to unbiased (mean bias near 0 at every strength tested), but its
trial-to-trial spread grows sharply with strength — standard deviation
climbing from ~0.004 at γ=0.02 to ~0.27 at γ=0.35. In other words: over
many experiments the method isn't systematically wrong, but any *single*
run's corrected estimate becomes unreliable as the damping strength grows.

**Coherent error is the sharper finding.** Because it's a pure unitary, it
never reduces purity — `purity_from_probs` measures exactly 1.0 at every
strength tested, by construction. The mitigation formula infers p_eff = 0
from that and applies *no correction whatsoever*: `mitigate_B(P_noisy)`
equals `exact_B(P_noisy)` bit-for-bit, verified directly (not just
approximately) in `tests/test_mitigation_bias.py`. The bias plotted for
this channel — growing smoothly from 0 to over 0.10 as the rotation angle
grows to 0.5 rad — is simply the raw, entirely uncorrected effect of the
miscalibration. This isn't a subtle edge case: it's a structural blind spot
in any mitigation scheme that infers its correction from purity alone,
since purity is a measure of *mixedness*, and a coherent error moves the
state without mixing it. A real device with systematic gate miscalibration
(a very ordinary hardware failure mode) would show a clean, growing,
completely unflagged bias in a Bell-magic measurement mitigated this way.

## 5. Limitations and what's not done

- Fig. 5/6 results come from one seed's worth of instances/splits, not a
  statistical study — reported numbers are qualitative, not error-barred
  claims of exact paper agreement.
- The Clifford circuit generator is sparser (one CNOT per layer) than
  whatever the paper's hardware-efficient circuits use, compensated for by
  using more depth; angle-count and layer-structure details in the paper's
  Appendices H/J weren't independently re-derived, only inferred.
- No real quantum hardware run (Phase 6's stretch goal): blocked on IBM
  Quantum Platform credentials, picked back up once those are available.
- Phase 7's noise channels are applied independently per qubit; a more
  complete study could sweep correlated (crosstalk-like) noise, or combine
  amplitude damping and coherent error in the same run.

## 6. One-line summary

Reproduced a 2022 quantum-information paper's magic-quantification method
end-to-end (simulation, sampled estimator, depolarizing-noise mitigation,
four main figures), validated it against a real Qiskit/Aer circuit
simulation, and found — via an exact density-matrix stress test — that its
purity-based error mitigation is structurally blind to coherent
(systematic, non-random) gate errors: it applies zero correction and
reports a clean, uncorrected, growing bias instead of flagging anything is
wrong.
