import numpy as np
from bellmagic import bell_probs, exact_B
from bellmagic.core.gradient import grad_component_exact, grad_component_sampled
from bellmagic.core.sampling import sample_bitstrings
from bellmagic.experiments.fig5 import hea_state

N, D = 3, 2


def _dists(th, e):
    P = bell_probs(hea_state(th))
    Pp = bell_probs(hea_state(th + np.pi / 2 * e), hea_state(th))
    Pm = bell_probs(hea_state(th - np.pi / 2 * e), hea_state(th))
    return P, Pp, Pm


def test_exact_shift_gradient_matches_finite_difference(rng):
    th = rng.uniform(0, 2 * np.pi, (D + 1, N, 2))
    for idx in [(0, 0, 0), (1, 2, 1), (2, 1, 0)]:
        e = np.zeros_like(th)
        e[idx] = 1
        h = 1e-5
        fd = (exact_B(bell_probs(hea_state(th + h * e))) - exact_B(bell_probs(hea_state(th - h * e)))) / (2 * h)
        assert np.isclose(grad_component_exact(*_dists(th, e)), fd, atol=1e-6)


def test_bell_probs_two_state_reduces_to_one(rng):
    psi = hea_state(rng.uniform(0, 2 * np.pi, (D + 1, N, 2)))
    assert np.allclose(bell_probs(psi), bell_probs(psi, psi))


def test_sampled_gradient_is_unbiased(rng):
    th = rng.uniform(0, 2 * np.pi, (D + 1, N, 2))
    e = np.zeros_like(th)
    e[1, 1, 0] = 1
    P, Pp, Pm = _dists(th, e)
    exact = grad_component_exact(P, Pp, Pm)
    ests = [grad_component_sampled(sample_bitstrings(P, 3000, rng), sample_bitstrings(Pp, 1000, rng),
                                   sample_bitstrings(Pm, 1000, rng), N, 20000, rng) for _ in range(40)]
    assert abs(np.mean(ests) - exact) < 4 * np.std(ests) / np.sqrt(40) + 0.01


def test_hea_state_uses_shape_of_theta(rng):
    for n, d in [(3, 2), (4, 6)]:
        assert hea_state(rng.uniform(0, 2 * np.pi, (d + 1, n, 2))).shape == (2,) * n
