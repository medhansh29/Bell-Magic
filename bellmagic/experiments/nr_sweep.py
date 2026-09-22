"""Sweep resampling steps N_R for fixed N_Q and report mean |B_hat - B_exact|.
Paper (pp. 4-5): error improves for N_R > N_Q/4 and saturates near 10*N_Q."""
import numpy as np
from bellmagic import bell_probs, exact_B, sample_B
from bellmagic.core import A_phi, product_state
from bellmagic.core.sampling import sample_bitstrings


def run(N_Q=200, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    N = 3
    P = bell_probs(product_state([A_phi(np.pi / 4)] * 3))
    B = exact_B(P)
    print(f"N={N}, exact B={B:.4f}, N_Q={N_Q}, trials={trials}")
    for f in (0.25, 0.5, 1, 2, 5, 10, 20, 50):
        n_r = int(f * N_Q)
        err = np.mean([
            abs(sample_B(sample_bitstrings(P, N_Q, rng), N, n_r, rng) - B)
            for _ in range(trials)
        ])
        print(f"N_R = {f:>5} * N_Q  ->  mean error {err:.4f}")


if __name__ == "__main__":
    run()
