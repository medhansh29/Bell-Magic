import numpy as np


def _parity(v):
    v = v.astype(np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(s)
    return v & np.uint64(1)


def check_commute(r, q, N):
    """True where Pauli strings sigma_r, sigma_q commute (r, q are indices x*2^N+z).

    Symplectic product x_r.z_q + z_r.x_q mod 2.
    """
    r = np.asarray(r, dtype=np.uint64)
    q = np.asarray(q, dtype=np.uint64)
    mask = np.uint64((1 << N) - 1)
    xr, zr = r >> np.uint64(N), r & mask
    xq, zq = q >> np.uint64(N), q & mask
    return (_parity(xr & zq) ^ _parity(zr & xq)) == 0


def sample_bitstrings(P, n_q, rng):
    return rng.choice(len(P), size=n_q, p=P).astype(np.uint64)


def sample_B(samples, N, n_r, rng):
    """Algorithm 2 (paper p. 4). `samples`: N_Q Bell outcomes (indices).

    Each of N_R steps draws four distinct outcomes without replacement and checks
    whether (r1^r2) and (r3^r4) commute; B = 2 * fraction that do not.
    """
    samples = np.asarray(samples, dtype=np.uint64)
    n_q = samples.size
    if n_q < 4:
        raise ValueError("need at least 4 samples")
    idx = np.empty((0, 4), dtype=np.int64)
    while idx.shape[0] < n_r:
        draw = rng.integers(0, n_q, size=(2 * (n_r - idx.shape[0]) + 16, 4))
        s = np.sort(draw, axis=1)
        ok = (np.diff(s, axis=1) != 0).all(axis=1)
        idx = np.concatenate([idx, draw[ok]])
    idx = idx[:n_r]
    r = samples[idx[:, 0]] ^ samples[idx[:, 1]]
    q = samples[idx[:, 2]] ^ samples[idx[:, 3]]
    return float(2.0 * np.mean(~check_commute(r, q, N)))
