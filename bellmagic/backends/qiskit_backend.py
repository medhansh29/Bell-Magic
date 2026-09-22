"""Qiskit/Aer reproduction of the Bell-measurement circuit in
bellmagic.core.bell.bell_probs: CNOT(A_n -> B_n) then H(A_n) for each n,
then measure all 2N qubits, on two identical copies of a prepared state.

Qiskit's classical bitstrings and Statevector.probabilities() are indexed
little-endian (qubit 0 is the least-significant bit), the opposite of
bellmagic.core.states' "qubit 0 is most significant" convention. The
`_qiskit_index_to_bell_index` map below converts between the two so results
can be compared directly against the numpy oracle.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error


def apply_gates(qc, gates, offset=0):
    """Apply the gate tuples used by bellmagic.core.states.apply_circuit
    (plus ("ry", angle, q) for A_phi state prep) to qubits [offset, offset+n)."""
    for g in gates:
        if g[0] == "h":
            qc.h(offset + g[1])
        elif g[0] == "s":
            qc.s(offset + g[1])
        elif g[0] == "t":
            qc.t(offset + g[1])
        elif g[0] == "ry":
            qc.ry(g[1], offset + g[2])
        elif g[0] == "cx":
            qc.cx(offset + g[1], offset + g[2])
        else:
            raise ValueError(g)


def bell_circuit(n, prep_gates, measure=True, prep_noise=None):
    """2N-qubit circuit: `prep_gates` applied identically to registers A
    (qubits 0..n-1) and B (qubits n..2n-1), then the Bell measurement.

    `prep_noise`, if given, is a per-qubit QuantumError instruction (e.g.
    from `prep_depolarizing_noise`) inserted on all 2n qubits right after
    state prep and before the Bell-measurement gates. It must NOT be
    attached via a NoiseModel matched by gate name ("h", "cx", ...): the
    Bell-measurement circuit itself uses those same gate names, and a
    NoiseModel can't tell the two apart, so it would corrupt the measurement
    rather than the state (see qiskit_check.py's note on this)."""
    qc = QuantumCircuit(2 * n, 2 * n if measure else 0)
    apply_gates(qc, prep_gates, offset=0)
    apply_gates(qc, prep_gates, offset=n)
    if prep_noise is not None:
        for q in range(2 * n):
            qc.append(prep_noise, [q])
    for i in range(n):
        qc.cx(i, n + i)
    for i in range(n):
        qc.h(i)
    if measure:
        qc.measure(range(2 * n), range(2 * n))
    return qc


def _qiskit_index_to_bell_index(k, n):
    """k has qubit i at bit i (Qiskit's little-endian convention). Returns
    x*2^n + z with x, z read qubit-0-most-significant within each register,
    matching bellmagic.core.bell.bell_probs."""
    bits = [(k >> i) & 1 for i in range(2 * n)]
    x = 0
    for i in range(n):
        x = (x << 1) | bits[i]
    z = 0
    for i in range(n):
        z = (z << 1) | bits[n + i]
    return x * (2 ** n) + z


def reorder_to_bell(qiskit_probs, n):
    out = np.empty(4 ** n)
    for k in range(4 ** n):
        out[_qiskit_index_to_bell_index(k, n)] = qiskit_probs[k]
    return out


def exact_probs(n, prep_gates):
    """Noiseless Bell outcome distribution via statevector simulation,
    reordered to match bellmagic.core.bell.bell_probs's index convention."""
    qc = bell_circuit(n, prep_gates, measure=False)
    sv = Statevector.from_instruction(qc)
    return reorder_to_bell(sv.probabilities(), n)


def counts_to_probs(counts, n):
    """Qiskit counts dict -> empirical Bell outcome distribution, length 4^n,
    same index convention as bellmagic.core.bell.bell_probs. A count key's
    bits already have qubit 0 as the least-significant character (Qiskit
    convention), so int(bitstring, 2) is directly the Qiskit index k."""
    out = np.zeros(4 ** n)
    total = sum(counts.values())
    for bitstring, c in counts.items():
        out[_qiskit_index_to_bell_index(int(bitstring, 2), n)] = c / total
    return out


def run_bell_circuit(qc, shots=20000, seed=None):
    sim = AerSimulator(seed_simulator=seed)
    return sim.run(qc, shots=shots).result().get_counts()


def prep_depolarizing_noise(p):
    """Local single-qubit depolarizing channel, probability p, meant to be
    passed as `bell_circuit`'s `prep_noise`. Models accumulated gate
    infidelity during state preparation as a per-qubit channel, with an
    otherwise-ideal Bell measurement -- distinct from the paper's global
    N-qubit depolarizing channel already modelled analytically in
    bellmagic.noise.depolarizing_probs."""
    return depolarizing_error(p, 1).to_instruction()
