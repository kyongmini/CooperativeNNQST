### @ models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Restricted Boltzmann Machines (RBMs) class
# n_visible : n qubits
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.a = nn.Parameter(torch.zeros(n_visible, dtype = torch.float32)) # visible - visible
        self.b = nn.Parameter(torch.zeros(n_hidden, dtype = torch.float32)) # hidden - hidden
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden, dtype = torch.float32) * 0.01)

# log (unnormalized probability)
    def log_p(self, v):
        v = v.to(device = self.a.device, dtype = self.a.dtype)
        if v.dim() ==1:
            v = v.unsqueeze(0) # (1, n_visible) 

        linear_term =  torch.matmul(v, self.a)
        hidden_input = torch.matmul(v, self.W) + self.b # (batch_size, n_hidden)
        # softplus (log(1+exp(~)))
        hidden_contribution = torch.sum(F.softplus(hidden_input), dim=1)
        log_pv = linear_term + hidden_contribution
        return log_pv.squeeze(0) if v.dim() == 1 else log_pv

    def h_given_v(self, v):
        v_proj = torch.matmul(v.to(self.W.dtype), self.W) + self.b
        return torch.bernoulli(torch.sigmoid(v_proj)).to(self.W.dtype)

# eigenvalue of Pauli matrix is +1 or -1
    def v_given_h(self, h):
        h_proj = torch.matmul(h.to(self.W.dtype), self.W.T) + self.a
        return (2 * torch.bernoulli(torch.sigmoid(2 * h_proj)) - 1).to(self.W.dtype)
    
    def gibbs_sampler(self, num_samples, burn_in, k_steps, initial_v=None, batch_size=128):
        device = self.a.device
        dtype = self.a.dtype

        if initial_v is None:
            # +1, -1 initial samples are generated
            v_current = (torch.randint(0, 2, (batch_size, self.n_visible), device=device).float() * 2.0 - 1.0).to(dtype)
        else:
            v_current = initial_v.to(device=device, dtype=dtype).view(1,-1) # view(1,-1) : Reshaping a single sample for batch processing

        # k-th burn in
        for _ in range(k_steps * burn_in):
            v_current = self.v_given_h(self.h_given_v(v_current))

        # Buffer
        samples = torch.empty((num_samples, self.n_visible), device=device, dtype=dtype)
        collected_samples = 0
    
        while collected_samples < num_samples:
            for _ in range(k_steps):
                v_current = self.v_given_h(self.h_given_v(v_current))
            take_n = min(v_current.shape[0], num_samples - collected_samples)
            samples[collected_samples : collected_samples + take_n] = v_current[:take_n]
            collected_samples += take_n
        return samples
    

# Pure state Wavefunction RBMs class
# Two Network (Amplitude Network & Phase Network)
class WavefunctionRBM(nn.Module):
    def __init__(self, n_visible, n_hidden_amp, n_hidden_phase):
        super(WavefunctionRBM, self).__init__()
        self.n_visible = n_visible
        self.rbm_amp = RBM(n_visible, n_hidden_amp)
        self.rbm_phase = RBM(n_visible, n_hidden_phase)

    def psi(self, v_states):
        v = v_states.to(device=self.rbm_amp.a.device, dtype=self.rbm_amp.a.dtype) 
        unnormalized_amp = torch.sqrt(torch.clamp(torch.exp(self.rbm_amp.log_p(v)), min=0.0))
        normalized_amp = unnormalized_amp / torch.sqrt(torch.clamp(torch.sum(unnormalized_amp ** 2), min=1e-18))
        phase = self.rbm_phase.log_p(v)
        wavefunction = unnormalized_amp.to(torch.complex64) * torch.exp(0.5 * 1j * phase.to(torch.complex64))
        return wavefunction

# computational basis is composed with tensor product of Z

    def all_computational_basis(self):
        configs = list(itertools.product([1.0, -1.0], repeat = self.n_visible))
        return torch.tensor(configs, dtype=self.rbm_amp.a.dtype, device=self.rbm_amp.a.device)

    def psi_complex_amplitude(self):
        all_configs = self.all_computational_basis()
        return all_configs, self.psi(all_configs)

