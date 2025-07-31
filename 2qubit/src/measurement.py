### @ measurement.py

from mystyle import print_text
import random
from qst_definition import MeasurementRecord
import torch
from qst_definition import single_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# cc : coincidence counts
# exp_cc : experimental coincidence counts
## ex) {"AD": 290, "DA": 265}
def MeasurementRecord_from_cc(exp_cc):

    """Converting from experimental data(coincidence count) to MeasurementRecord class"""

    print_text("Reading your coincidence count.")
    measurements_list = []
    for Pauli_basis, cc_polarization in exp_cc.items():
        for polarization, cc in cc_polarization.items():
            for _ in range(cc):
                measurements_list.append(MeasurementRecord(Pauli_basis=Pauli_basis, polarization=polarization))
    random.shuffle(measurements_list)
    print_text(f"Total coincidence counts : {len(measurements_list)}")
    return measurements_list


# generation of samples of cooperative learning in phase network
# Not yet calculate the tensor product
# n_visible : n qubit
# pol : polarization state
# vec : vector of polarization state
def cooperative_samples(measurements_list, n_visible):
    all_single_polarization = {}
    for pol, vec in single_state.items():
        all_single_polarization[pol] = vec.clone().to(device)

    vecs = []
    for record in measurements_list:
        qubit_vectors = [all_single_polarization[pol] for pol in record.polarization]
        vecs.append(torch.stack(qubit_vectors))
    return torch.stack(vecs) # ex) H (2 X 2), V (2 X 2)-> H,V (2 X 2 X 2) tensor 
