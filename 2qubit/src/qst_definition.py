### @ qst_definition.py

import numpy as np
import torch
from dataclasses import dataclass
import random
import os
from mystyle import print_text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

I = torch.tensor([[1,0], [0,1]], dtype = torch.complex64, device=device)
X = torch.tensor([[0,1], [1,0]], dtype = torch.complex64, device=device)
Y = torch.tensor([[0,-1j], [1j,0]], dtype = torch.complex64, device=device)
Z = torch.tensor([[1,0], [0,-1]], dtype = torch.complex64, device=device)

single_pauli = {'I':I, 'X':X, 'Y':Y, 'Z':Z,}
multi_pauli = {}

sqrt2 = torch.sqrt(torch.tensor(2.0, dtype = torch.complex64, device=device))
H = torch.tensor([1,0], dtype = torch.complex64, device=device)
V = torch.tensor([0,1], dtype = torch.complex64, device=device)

single_state = {
    'H' : H, 'V' : V,
    'D' : (H+V)/sqrt2,
    'A' : (H-V)/sqrt2,
    'R' : (H+1j*V)/sqrt2,
    'L' : (H-1j*V)/sqrt2,
}
multi_state = {}

pauli_to_state = {
    'Z' : {1 : 'H', -1 : 'V'},
    'X' : {1 : 'D', -1 : 'A'},
    'Y' : {1 : 'R', -1 : 'L'}
}

# ex) MeasurementRecord(Pauli_basis='XX', polarization='DA')
@dataclass
class MeasurementRecord:
    Pauli_basis: str
    polarization: str

# fidelity for pure state
def fidelity(psi1, psi2):
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = psi2 / np.linalg.norm(psi2)
    return np.abs(np.vdot(psi1, psi2)) ** 2

def set_seed(seed: int = 42):
    random.seed(seed)                           # Python random
    np.random.seed(seed)                        # NumPy
    torch.manual_seed(seed)                     # CPU용 torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)            # GPU 단일 카드
        torch.cuda.manual_seed_all(seed)        # GPU 멀티 카드
    torch.backends.cudnn.deterministic = True   # CuDNN 고정
    torch.backends.cudnn.benchmark = False      # 연산자 선택 비활성화 (속도 ↓)
    os.environ["PYTHONHASHSEED"] = str(seed)    # 해시 기반 연산 고정 (예: set, dict 순서)

    print_text(f"[Seed : {seed}]")