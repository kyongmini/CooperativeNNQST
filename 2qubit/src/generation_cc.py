### @ generation_cc.py

# It makes simulation data of coincidence count on 2 qubits

import numpy as np
import itertools

def generate_ideal_counts_2qubit(target_psi, shots_per_basis=500):

    if target_psi.shape != (4,):
        raise ValueError("Only 2-qubit (4-dimensional) states are supported.")

    if not np.isclose(np.linalg.norm(target_psi), 1.0):
        target_psi = target_psi / np.linalg.norm(target_psi)

    rho = np.outer(target_psi, np.conj(target_psi))

    # Define polarization states
    H = np.array([1, 0], dtype=complex)
    V = np.array([0, 1], dtype=complex)
    D = (H + V) / np.sqrt(2)
    A = (H - V) / np.sqrt(2)
    R = (H + 1j * V) / np.sqrt(2)
    L = (H - 1j * V) / np.sqrt(2)

    def projector(ket):
        return np.outer(ket, np.conj(ket))

    single_projector = {
        'H': projector(H), 'V': projector(V),
        'D': projector(D), 'A': projector(A),
        'R': projector(R), 'L': projector(L)
    }

    outcome_map = {'Z': ['H', 'V'], 'X': ['D', 'A'], 'Y': ['R', 'L']}

    basis_list = ["ZZ", "ZX", "ZY", "XZ", "XX", "XY", "YZ", "YX", "YY"]
    result = {}

    for basis in basis_list:
        basis_outcomes = [outcome_map[b] for b in basis]
        all_outcomes = ["".join(p) for p in itertools.product(*basis_outcomes)]

        probs = []
        for o in all_outcomes:
            proj = np.kron(single_projector[o[0]], single_projector[o[1]])
            prob = np.real(np.trace(proj @ rho))
            probs.append(max(0.0, prob))
        probs = np.array(probs)
        probs /= probs.sum() if probs.sum() > 1e-9 else 1.0

        float_counts = probs * shots_per_basis
        rounded = np.round(float_counts).astype(int)
        diff = shots_per_basis - rounded.sum()
        residuals = float_counts - rounded
        order = np.argsort(residuals)
        for i in range(abs(diff)):
            if diff > 0:
                rounded[order[-(i + 1)]] += 1
            else:
                rounded[order[i]] -= 1
        rounded = np.maximum(0, rounded)

        result[basis] = {o: int(c) for o, c in zip(all_outcomes, rounded)}

    return result

