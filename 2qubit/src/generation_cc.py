### @ generation_cc.py

# It makes simulation data of coincidence count on 2 qubits

import numpy as np
import itertools

def generation_cc(target_psi, shots_per_basis=5000):
    """
    {
        "ZZ": {"HH": 3708, "HV": 77, "VH": 51, "VV": 3642},
        ...
    }
    """
    if target_psi.shape != (4,):
        raise ValueError("Only 2-qubit (4-dimensional) states are supported.")

    if not np.isclose(np.linalg.norm(target_psi), 1.0):
        target_psi = target_psi / np.linalg.norm(target_psi)

    rho = np.outer(target_psi, np.conj(target_psi))

    H = np.array([1, 0], dtype=complex)
    V = np.array([0, 1], dtype=complex)
    D = (H + V) / np.sqrt(2)
    A = (H - V) / np.sqrt(2)
    R = (H + 1j * V) / np.sqrt(2)
    L = (H - 1j * V) / np.sqrt(2)

    def proj(ket):
        return np.outer(ket, np.conj(ket))

    single_proj = {
        'H': proj(H), 'V': proj(V),
        'D': proj(D), 'A': proj(A),
        'R': proj(R), 'L': proj(L)
    }

    outcome_map = {'Z': ['H', 'V'], 'X': ['D', 'A'], 'Y': ['R', 'L']}
    basis_list = ["ZZ", "ZX", "ZY", "XZ", "XX", "XY", "YZ", "YX", "YY"]

    result = {}
    for basis in basis_list:
        basis_outcomes = [outcome_map[b] for b in basis]
        all_outcomes = ["".join(p) for p in itertools.product(*basis_outcomes)]

        probs = []
        for outcome in all_outcomes:
            P = np.kron(single_proj[outcome[0]], single_proj[outcome[1]])
            prob = np.real(np.trace(P @ rho))
            probs.append(max(0.0, prob))
        probs = np.array(probs)
        total = probs.sum()
        if total > 1e-9:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

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

    lines = ["{"] 
    basis_items = list(result.items())
    for idx, (basis, outcomes) in enumerate(basis_items):
        inner_parts = [f"\"{k}\": {v}" for k, v in outcomes.items()]
        inner_str = ", ".join(inner_parts)
        comma = "," if idx < len(basis_items) - 1 else ""
        lines.append(f'    "{basis}": {{{inner_str}}}{comma}')
    lines.append("}")
    return "\n".join(lines)