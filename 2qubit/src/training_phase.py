### @ training_phase.py

### @ training_phase.py

import torch
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple
from mystyle import print_banner, print_text
from qst_definition import MeasurementRecord, fidelity
from measurement import cooperative_samples
from models import WavefunctionRBM 


# Help for analytic gradient
def flatten_parameters(parameter_list):
    return torch.cat([p.reshape(-1) for p in parameter_list if p is not None])

def unflatten_parameters(flat_tensor, flat_parameter_list):
    unflattened, current_index = [], 0
    for original_form in flat_parameter_list:
        num_elements = original_form.numel()
        unflattened.append(flat_tensor[current_index : current_index + num_elements].reshape_as(original_form))
        current_index += num_elements
    return unflattened


# Cooperative Loss
def train_phase_network(
    wf_rbm,
    measurements,
    n_epochs,
    lr,
    target_rho=None,
    n_mcmc=256,
    k_steps=10,
    burn_in=50,
    batch_size=1024,
    verbose=True
):

    print_banner("Cooperative Phase Network with analytic gradient")

    device = next(wf_rbm.parameters()).device
    n_vis = wf_rbm.n_visible
    rbm_dtype = wf_rbm.rbm_amp.a.dtype

    # Freeze amplitude parameters (Necessary code!)
    for p in wf_rbm.rbm_amp.parameters():
        p.requires_grad = False

    #If you use autograd,
    phase_parameters = [p for p in wf_rbm.rbm_phase.parameters() if p.requires_grad]
    optimizer = optim.Adam(phase_parameters, lr=lr)

    full_cooperative_batch = cooperative_samples(measurements, n_vis).to(device).to(torch.complex64)
    dm_log, loss_log, fid_log = [], [], []

    for epoch in range(n_epochs):
        # assign the indeces for shuffling and selection mini-batch
        indices = torch.randperm(full_cooperative_batch.shape[0], device=device)[:batch_size] 
        k_cooperative_batch = full_cooperative_batch[indices]
        current_batch_size = k_cooperative_batch.shape[0]
        if current_batch_size <= 1:
            continue

        v_samples = wf_rbm.rbm_amp.gibbs_sampler(
            num_samples=n_mcmc,
            burn_in=burn_in,
            k_steps=k_steps,
            batch_size=n_mcmc
        ).to(device=device, dtype=rbm_dtype)

        log_p_phase = wf_rbm.rbm_phase.log_p(v_samples)
        phase_factors = torch.exp(0.5 * 1j * log_p_phase.to(torch.complex64))
        # +1 or -1
        Z_eigenvalues = ((1.0 - v_samples) / 2.0).long() # convert int

        # overlap_no_phase = overlap on computational basis without phase
        overlap_no_phase  = torch.zeros(current_batch_size, n_mcmc, dtype=torch.complex64, device=device)
        for k in range(current_batch_size):
            phi_k = k_cooperative_batch[k]
            phi_k_expanded = phi_k.expand(n_mcmc, -1, -1)
            v_indices = Z_eigenvalues.unsqueeze(-1)
            gathered = torch.gather(phi_k_expanded, 2, v_indices).squeeze(-1)
            overlap_no_phase[k] = torch.prod(gathered, dim=1)



        C_k = torch.mean(overlap_no_phase.conj() * phase_factors.unsqueeze(0), dim=1).detach()

        overlap = torch.abs(C_k) ** 2 # overlap is role of scores!

        lse = torch.logsumexp(overlap, dim=0)
        cooperative_loss = (lse / current_batch_size) - torch.mean(overlap)
        loss_log.append(cooperative_loss.item())

        optimizer.zero_grad()


        # Calculate Analytic Gradient of Cooperative Loss
        grad_matrix = []
        for m in range(n_mcmc):
            v_sample = v_samples[m:m+1]
            log_p = wf_rbm.rbm_phase.log_p(v_sample)
            grads = torch.autograd.grad(log_p, phase_parameters, retain_graph=False, allow_unused=True)
            # no gradient -> 0
            grads_clean = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(grads, phase_parameters)]
            grad_matrix.append(flatten_parameters(grads_clean))
        grad_matrix = torch.stack(grad_matrix, dim=0)

        # unsqueeze(2) : [batch_size, n_mcmc, 1]
        term = overlap_no_phase.conj().unsqueeze(2) * phase_factors.unsqueeze(0).unsqueeze(2) * grad_matrix.unsqueeze(0) 
        grad_C_k = 0.5*1j * torch.mean(term, dim=1)

        grad_overlap = 2.0 * torch.real(C_k.conj().unsqueeze(1) * grad_C_k)
        p_k = torch.softmax(overlap, dim=0)
        weighted_grad = (1.0 - p_k).unsqueeze(1) * grad_overlap
        # optimizer : gradient descent : the reason why we derive negative sign
        total_grad = - torch.sum(weighted_grad, dim=0) / current_batch_size

        grad_unflat = unflatten_parameters(total_grad, phase_parameters)
        with torch.no_grad():
            for p, g in zip(phase_parameters, grad_unflat):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(g)
        optimizer.step()

        with torch.no_grad():
            _, psi = wf_rbm.psi_complex_amplitude()
            psi_normed = psi / torch.norm(psi)
            rho = np.outer(psi_normed.cpu().numpy(), np.conjugate(psi_normed.cpu().numpy()))
            dm_log.append(rho.copy())
            fid_log.append(fidelity(rho, target_rho))

        if verbose and ((epoch + 1) % 20 == 0 or epoch == n_epochs - 1):
            print_text(f"Phase Training : Epoch {epoch + 1}/{n_epochs} | Loss: {loss_log[-1]:.6f} | Fidelity: {fid_log[-1]:.4f}")

    return dm_log, loss_log, fid_log