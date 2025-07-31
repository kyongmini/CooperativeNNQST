### @ training_amplitude.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from mystyle import print_banner, print_text
from models import RBM, WavefunctionRBM
from qst_definition import MeasurementRecord, fidelity

# k_cd : # of steps of CD
def train_amplitude_network(wf_rbm, z_measurements, n_epochs, lr, target_rho, k_cd, batch_size):
    print_banner("Amplitude Network")
    z_eigenvalues = [[{"H": 1.0, "V": -1.0}[p] for p in record.polarization] for record in z_measurements]
    device = next(wf_rbm.parameters()).device
    data_tensor = torch.tensor(z_eigenvalues, dtype=wf_rbm.rbm_amp.a.dtype, device=device)
    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(wf_rbm.rbm_amp.parameters(), lr=lr)

    dm_log, loss_log, fid_log = [], [], []

    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0

        for (v_data,) in dataloader:
            # of batch = 0 -> pass

            
            if v_data.shape[0] == 0:
                continue

            v_model = v_data.clone().detach()

            for _ in range(k_cd):
                h_sample = wf_rbm.rbm_amp.h_given_v(v_model)
                v_model = wf_rbm.rbm_amp.v_given_h(h_sample)

            # model data & real(experimental) data of CD
            cd_loss = torch.mean(wf_rbm.rbm_amp.log_p(v_model.detach())) - torch.mean(wf_rbm.rbm_amp.log_p(v_data))
            optimizer.zero_grad()
            if torch.isfinite(cd_loss):
                cd_loss.backward()
                optimizer.step()
                epoch_loss += cd_loss.item()
                n_batches +=1
        # expectation of CD loss for each batches
        loss_log.append(epoch_loss / n_batches)

        # evaluation of pure RBM state
        with torch.no_grad(): 
            configs, psi = wf_rbm.psi_complex_amplitude()
            normalized_psi = psi / (torch.norm(psi) if torch.norm(psi)> 1e-14 else 1)
            rbm_rho = np.outer(normalized_psi.cpu().numpy(), np.conjugate(normalized_psi.cpu().numpy()))
            dm_log.append(rbm_rho.copy())
            fid_log.append(fidelity(rbm_rho, target_rho))

    
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            print_text(f"Amplitude Training : Epoch {epoch + 1}/{n_epochs} | CD loss: {loss_log[-1]:.4f} | Fidelity: {fid_log[-1]:.4f}")

    return dm_log, loss_log, fid_log

