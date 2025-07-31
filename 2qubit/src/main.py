# main.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mystyle import my_style, print_banner, print_text
from models import WavefunctionRBM
from qst_definition import fidelity, set_seed
from measurement import MeasurementRecord_from_cc
from training_amplitude import train_amplitude_network
from training_phase import train_phase_network
from visualization import plot_with_fill, qdisplay, qanimation

if __name__ == "__main__":
    # Setting Environment

    set_seed(2025) #if you do not want seed, remove this code
    my_style(fontsize=12)
    os.makedirs("figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_banner(f"Using device: {device}")

    # Target state you want
    n_qubits = 2
    target_state = np.array([1, -1, -1j, 1j], dtype=np.complex64) / np.sqrt(4)
    target_rho = np.outer(target_state, np.conjugate(target_state))

    # Experimental Data (Simulation data : generation_cc.py)
    experimental_counts = {
        "ZZ": {"HH": 139, "HV": 122, "VH": 116, "VV": 123},
        "ZX": {"HD": 0, "HA": 254, "VD": 0, "VA": 246},
        "ZY": {"HR": 140, "HL": 134, "VR": 117, "VL": 109},
        "XZ": {"DH": 107, "DV": 121, "AH": 129, "AV": 143},
        "XX": {"DD": 0, "DA": 240, "AD": 0, "AA": 260},
        "XY": {"DR": 125, "DL": 134, "AR": 123, "AL": 118},
        "YZ": {"RH": 0, "RV": 0, "LH": 251, "LV": 249},
        "YX": {"RD": 0, "RA": 0, "LD": 0, "LA": 500},
        "YY": {"RR": 0, "RL": 0, "LR": 253, "LL": 247}
    }

    all_measurements = MeasurementRecord_from_cc(experimental_counts)
    z_measurements = [m for m in all_measurements if m.Pauli_basis == "ZZ"]

    # initialize the model
    model = WavefunctionRBM(n_visible=n_qubits, n_hidden_amp=4, n_hidden_phase=4).to(device)

    # amplitude training
    amp_dm_log, amp_loss_log, amp_fid_log = train_amplitude_network(
        wf_rbm=model,
        z_measurements=z_measurements,
        n_epochs=200,
        lr=0.01,
        target_rho=target_rho,
        k_cd=5,
        batch_size=128
    )

    # phase training
    phase_dm_log, phase_loss_log, phase_fid_log = train_phase_network(
        wf_rbm=model,
        measurements=all_measurements,
        n_epochs=300,
        lr=0.01,
        target_rho=target_rho,
        n_mcmc=1024,
        k_steps=10,
        burn_in=50,
        batch_size=1024
    )

    # loss and fidelity graph
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    total_epochs = np.arange(1, len(amp_fid_log + phase_fid_log) + 1)
    combined_fid = amp_fid_log + phase_fid_log
    combined_loss = [np.nan] * len(amp_loss_log) + phase_loss_log

    plot_with_fill(ax1, total_epochs, combined_fid, label="Fidelity", color="tab:green")
    ax1.set_ylabel("Fidelity", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_ylim(-0.05, 1.05)

    plot_with_fill(ax2, total_epochs, combined_loss, label="Loss", color="tab:blue")
    ax2.set_ylabel("Loss", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_xlabel("Total Epochs")
    ax1.axvline(x=len(amp_loss_log), color='black', linestyle='--')
    ax1.set_title("QRBM Training: Amplitude + Phase")
    ax1.text(len(amp_loss_log) / 2, 1.02, 'Amplitude Training', ha='center', style='italic')
    ax1.text(len(amp_loss_log) + len(phase_loss_log) / 2, 1.02, 'Phase Training', ha='center', style='italic')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig("figures/training_dynamics_combined.png")
    plt.show()

    # 3D draw
    final_rho = phase_dm_log[-1] if phase_dm_log else amp_dm_log[-1]
    final_fid = fidelity(final_rho, target_rho)

    print_text(f"\nFinal Fidelity vs Target: {final_fid:.6f}")
    qdisplay(final_rho, n_qubits=n_qubits, title_prefix=f"Final Reconstructed State (Fid {final_fid:.4f})")
    qdisplay(target_rho, n_qubits=n_qubits, title_prefix="Target State")

    # animation of QST evolution
    qanimation(
        amp_dm_hist=amp_dm_log,
        phase_dm_hist=phase_dm_log,
        ideal_rho=target_rho,
        n_qubits=n_qubits,
        output_filepath="figures/qrbm_training_animation.gif"
    )
