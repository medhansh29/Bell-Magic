"""Variationally maximise exact Bell magic over pure N-qubit states (paper App. G)."""
import numpy as np
from scipy.optimize import minimize
from bellmagic import bell_probs, exact_B, additive


def _state(x, N):
    d = 2 ** N
    v = x[:d] + 1j * x[d:]
    return (v / np.linalg.norm(v)).reshape((2,) * N)


def maximise(N, restarts=30, seed=0):
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    for _ in range(restarts):
        x0 = rng.normal(size=2 * 2 ** N)
        res = minimize(lambda x: -exact_B(bell_probs(_state(x, N))), x0, method="BFGS")
        if -res.fun > best[0]:
            best = (-res.fun, _state(res.x, N))
    return best  # (B, state)


if __name__ == "__main__":
    targets = {1: 1.29545588, 2: 2.67807, 3: 4.651794}
    for N in (1, 2, 3):
        B, psi = maximise(N, restarts=40 if N == 3 else 15)
        print(f"N={N}: B_a={additive(B):.6f}  paper={targets[N]}")
