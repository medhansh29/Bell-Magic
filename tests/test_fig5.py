import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.experiments.fig5 import hea_state, random_theta, learn_threshold


def test_hea_normalised(rng):
    psi = hea_state(random_theta(True, rng))
    assert np.isclose(np.linalg.norm(psi), 1)


def test_clifford_angles_give_stabilizer(rng):
    for _ in range(10):
        B = exact_B(bell_probs(hea_state(random_theta(False, rng))))
        assert abs(B) < 1e-9


def test_random_angles_are_magical(rng):
    Bs = [exact_B(bell_probs(hea_state(random_theta(True, rng)))) for _ in range(10)]
    assert np.mean(Bs) > 0.3


def test_threshold_separates():
    b = np.array([0.0, 0.1, 0.05, 0.9, 1.0, 0.8])
    y = np.array([-1, -1, -1, 1, 1, 1])
    t = learn_threshold(b, y)
    assert 0.1 < t < 0.8
