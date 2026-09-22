# bell-magic

A from-scratch reproduction of ["Quantifying Bell nonlocality with the
Robustness of Magic"](https://arxiv.org/abs/2204.10061)-style Bell magic
(Haug & Kim, arXiv:2204.10061): a numpy simulation of the paper's Bell-basis
magic measure, its sampled estimator, depolarizing-noise mitigation, the
paper's main figures, a Qiskit/Aer validation layer, and an extension
studying what happens when that mitigation is applied outside the noise
model it was derived for.

## Install

Core pipeline (numpy/matplotlib only):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Phase 6 (Qiskit/Aer) needs the `qiskit` extra on top of that:

```bash
pip install -e ".[dev,qiskit]"
```

## Layout

- `bellmagic/core/` — statevectors (`states.py`), the exact Bell-measurement
  circuit and Walsh-Hadamard `exact_B` (`bell.py`, `exact.py`), the sampled
  estimator (`sampling.py`, Algorithm 2), doped-Clifford circuit builders
  (`circuits.py`), the variational gradient (`gradient.py`), and a
  density-matrix tensor backend for noise models that don't have a closed
  form on the Bell distribution (`density.py`).
- `bellmagic/noise/` — the global depolarizing channel and its closed-form
  mitigation (`depolarizing.py`, `mitigation.py`, App. I of the paper), plus
  amplitude damping and coherent-error channels for Phase 7
  (`channels.py`).
- `bellmagic/backends/` — a Qiskit/Aer reproduction of the Bell-measurement
  circuit, used to validate the numpy pipeline against a real quantum
  circuit simulator (Phase 6).
- `bellmagic/experiments/` — one script per figure/study, each with a
  `run()` you can import and a CLI (`python3 -m bellmagic.experiments.X`).
- `tests/` — 80 tests: unitarity/physics sanity checks, oracle comparisons,
  and the Phase 6/7 validations below.

## Reproducing the figures

All scripts default to seed 0. Fig. 3 is scaled to N=3 (paper: hardware, no
N given), Fig. 2 to N=8 (paper: N=50, tensor networks), Fig. 5/6 match the
paper's N=3/4. See each script's module docstring for the exact settings
and every deliberate deviation from the paper.

```bash
python3 -m bellmagic.experiments.fig3         # --out fig3.png
python3 -m bellmagic.experiments.fig2 --reps 200 --stat median   # --out fig2.png
python3 -m bellmagic.experiments.fig5 --seed 0                   # --out fig5.png
python3 -m bellmagic.experiments.fig6 --epochs 100                # --out fig6.png
python3 -m bellmagic.experiments.nr_sweep                         # prints only, no figure
python3 -m bellmagic.experiments.max_magic                        # variational B_max targets
python3 -m bellmagic.experiments.qiskit_check                     # Phase 6, needs the qiskit extra
python3 -m bellmagic.experiments.mitigation_bias --trials 30      # Phase 7, --out mitigation_bias.png
```

`fig2.py --stat median` matters: with the mean, the p=0.3 panel-(a) curve
picks up a large-error outlier at low N_Q from a purity estimate blowing up
the (1-p)^-8 rescaling, giving a slope of -1.28 against the paper's ~-0.5.
The median removes it (-0.54), consistent with the other three curves.

## Where this matches and diverges from the paper

Quantitative slope comparisons (mine vs. the paper's) live in each script's
docstring and were logged during development; the headline picture:

- **Fig. 3-style scaling, Fig. 5 classifier, Fig. 6 variational training**:
  right qualitative shape; Fig. 6's exact-gradient run reaches the paper's
  reported B_max at N=4 to within 3e-4.
- **Fig. 2 slopes**: match the paper to within ~0.05 for three of four
  p-values with the mean statistic; the fourth (p=0.3) needed the median
  fix above. The N_R saturation trend in panel (c) is qualitatively right
  but "near-optimal at 10*N_Q" is a loose description, not a sharp knee
  (`nr_sweep.py`).
- **Known deviations**: our Clifford generator is sparser than the paper's
  circuits, so we use greater depth to reach comparable scrambling; we used
  200 repetitions per Fig. 2 point against the paper's 1000; Fig. 5/6 use a
  single seed's worth of splits/instances rather than a full statistical
  study. None of these are hidden — each script's docstring says so.

## Phase 6: Qiskit/Aer validation

`bellmagic/backends/qiskit_backend.py` builds the identical Bell-measurement
circuit in Qiskit and converts between Qiskit's little-endian bit order and
this codebase's qubit-0-most-significant convention. Validated three ways
(`qiskit_check.py`, `tests/test_qiskit_backend.py`):

1. Exact statevector simulation matches the numpy oracle to ~1e-15 across
   random doped-Clifford instances.
2. Shot-sampled counts converge to that exact distribution.
3. A per-qubit depolarizing channel injected right after state prep (not
   via a name-matched `NoiseModel`, which would also corrupt the
   Bell-measurement circuit's own H/CNOT gates — an early version of this
   check did exactly that and produced a meaningless result) pushes raw,
   unmitigated B *up* toward `1 - 4^-N`, matching the codebase's own
   Phase-3 fact that the maximally mixed state has near-maximal raw B. This
   reproduces that fact from a real circuit simulator, not just the
   closed-form formula.

A real IBM Quantum Platform run was left for a follow-up pending account
credentials.

## Phase 7 (extension): breaking the mitigation

`bellmagic.noise.mitigation` was derived for exactly one noise model — a
global depolarizing channel — and estimates its single parameter p from
measured purity. `bellmagic/experiments/mitigation_bias.py` asks what it
does when the real noise isn't depolarizing, using an exact density-matrix
simulation (`bellmagic/core/density.py`) since neither channel below has a
closed form on the Bell distribution:

- **Amplitude damping** (T1-style decay) does reduce purity, so mitigation
  applies a real, mostly-unbiased-on-average correction — but its
  trial-to-trial spread grows sharply with noise strength (std ~0.004 at
  gamma=0.02 to ~0.27 at gamma=0.35). A single run's corrected estimate can
  be far off even though the method isn't systematically wrong.
- **Coherent (systematic, non-random unitary) error never reduces purity**
  — it's non-dissipative by construction — so mitigation always infers
  p_eff=0 and applies *no correction whatsoever*: `mitigate_B(P_noisy)`
  equals `exact_B(P_noisy)` bit-for-bit. The purity diagnostic this
  mitigation formula relies on is structurally blind to this entire class
  of error. The "bias" plotted for this channel is just the raw,
  completely uncorrected effect of the miscalibration.
- A matched-depolarizing control (same purity loss, true p given)
  recovers B0 to numerical precision in both cases, confirming the bias
  above is a genuine model-mismatch effect and not a bug in `mitigate_B`.

## Testing

```bash
pytest              # 80 tests, ~6s, works without the qiskit extra
                     # (qiskit tests self-skip via pytest.importorskip)
```
