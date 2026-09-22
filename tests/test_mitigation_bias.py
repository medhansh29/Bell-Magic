import numpy as np
import pytest

from bellmagic import exact_B
from bellmagic.experiments.mitigation_bias import instance, one_point


def test_matched_depolarizing_control_recovers_B0():
    # sanity check: when mitigate_B is given the TRUE p for a genuinely
    # depolarized distribution, it should recover B0 to numerical precision
    # -- this isolates the other channels' bias as a real model-mismatch
    # effect, not a bug in mitigate_B itself.
    rng = np.random.default_rng(0)
    for kind in ("amplitude_damping", "coherent"):
        for _ in range(4):
            _, _, _, bias_dep = one_point(4, 3, 4, kind, 0.2, rng)
            assert bias_dep == pytest.approx(0.0, abs=1e-9)


def test_coherent_error_gets_zero_mitigation_correction():
    # a purely unitary (coherent) error never shows up in the purity signal
    # (purity_from_probs stays exactly 1), so mitigate_B infers p_eff=0 and
    # applies NO correction: it just returns exact_B(P_noisy) unchanged.
    from bellmagic.core.density import bell_probs_density
    from bellmagic.noise import mitigate_B, noisy_density

    rng = np.random.default_rng(1)
    psi = instance(4, 3, rng, 4)
    for eps in (0.1, 0.3, 0.5):
        P_noisy = bell_probs_density(noisy_density(psi, "coherent", eps))
        assert mitigate_B(P_noisy) == pytest.approx(exact_B(P_noisy), abs=1e-12)


def test_amplitude_damping_bias_grows_with_strength():
    rng = np.random.default_rng(2)
    trials = 12
    mean_abs_bias = {}
    for strength in (0.05, 0.3):
        biases = [one_point(4, 3, 4, "amplitude_damping", strength, rng)[2] for _ in range(trials)]
        mean_abs_bias[strength] = np.mean(np.abs(biases))
    assert mean_abs_bias[0.3] > mean_abs_bias[0.05]
