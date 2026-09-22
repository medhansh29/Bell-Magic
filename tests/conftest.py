import numpy as np
import pytest

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def pauli_from_index(r, N):
    """Independent convention check: outcome index r = a*2^N + b (a: A-register
    bits, b: B-register bits) corresponds to the Bell state (sigma_r (x) I)|Phi+>
    with sigma_r = (x)_n X^{b_n} Z^{a_n}. Derived by hand for N=1:
    00->Phi+, 01->Psi+, 10->Phi-, 11->Psi-."""
    a, b = r >> N, r & ((1 << N) - 1)
    op = np.eye(1, dtype=complex)
    for n in range(N):
        bit = N - 1 - n
        m = I2
        if (b >> bit) & 1:
            m = X @ m
        if (a >> bit) & 1:
            m = m @ Z
        op = np.kron(op, m)
    return op


def haar_state(N, rng):
    v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    v /= np.linalg.norm(v)
    return v.reshape((2,) * N)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
