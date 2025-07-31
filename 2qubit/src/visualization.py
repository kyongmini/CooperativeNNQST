### @ visualization.py

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from mystyle import my_style, print_text, print_banner


# EWMA smoothing 
def plot_with_fill(ax, x_data, y_data, label, color, smoothing_span=25):
    """Plot smoothed curve with filled uncertainty band."""
    series = pd.Series(y_data, index=x_data).dropna()


    y_smooth = series.ewm(span=smoothing_span).mean()
    y_std = series.ewm(span=smoothing_span).std()

    ax.plot(y_smooth.index, y_smooth.values, color=color, label=label, linewidth=2.0)
    ax.fill_between(
        y_smooth.index,
        y_smooth - y_std,
        y_smooth + y_std,
        color=color,
        alpha=0.2,
        linewidth=0
    )


def qdraw(
        ax, rho_component, title, dim, n_qubits, comp_type, vmax_abs=None, cmap_pos='Reds', cmap_neg='Blues'):

    """Draws a 3D bar chart of a real or imaginary part of a density matrix."""
    ax.cla()

    x_pos, y_pos = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    x_flat = x_pos.flatten() - 0.5
    y_flat = y_pos.flatten() - 0.5
    z_base = np.zeros_like(x_flat)
    dz = rho_component.flatten()

    finite_mask = np.isfinite(dz)
    x_f = x_flat[finite_mask]
    y_f = y_flat[finite_mask]
    z_f = z_base[finite_mask]
    dz_f = dz[finite_mask]

    norm_val = vmax_abs if vmax_abs and vmax_abs > 1e-9 else 1.0
    cmap_pos_obj = plt.get_cmap(cmap_pos)
    cmap_neg_obj = plt.get_cmap(cmap_neg)

    colors = ['lightgrey'] * len(dz_f)
    for i, val in enumerate(dz_f):
        if val > 1e-9:
            colors[i] = cmap_pos_obj(np.clip(val / norm_val, 0, 1))
        elif val < -1e-9:
            colors[i] = cmap_neg_obj(np.clip(abs(val) / norm_val, 0, 1))

    ax.bar3d(x_f, y_f, z_f, 0.68, 0.68, dz_f,
            color=colors, shade=True, alpha=0.95,
            edgecolor='black', linewidth=0.33)

    ax.set_zlim(-vmax_abs, vmax_abs)
    basis_labels = [''.join(s) for s in itertools.product("HV", repeat=n_qubits)]
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.set_xticklabels(basis_labels, rotation=0)
    ax.set_yticklabels(basis_labels, rotation=0)
    ax.set_zlabel(comp_type, rotation=90, labelpad=1)
    ax.set_title(title, pad=15)
    ax.view_init(elev=30., azim=-125.)
    ax.grid(False)


def qdisplay(final_rho, n_qubits, title_prefix, out_dir="figures"):
    """Displays and saves a 3D plot of a complex density matrix (real + imag)."""
    my_style(fontsize=10)

    dim = final_rho.shape[0]
    rho_re = np.real(final_rho)
    rho_im = np.imag(final_rho)

    vmax_abs = np.max(np.abs(np.concatenate([rho_re.flatten(), rho_im.flatten()])))

    fig = plt.figure(figsize=(12, 5.8))
    ax_re = fig.add_subplot(121, projection='3d')
    ax_im = fig.add_subplot(122, projection='3d')

    qdraw(ax_re, rho_re, f"{title_prefix} - Real", dim, n_qubits, "Re", vmax_abs)
    qdraw(ax_im, rho_im, f"{title_prefix} - Imag", dim, n_qubits, "Im", vmax_abs, cmap_pos='Greens', cmap_neg='Purples')

    plt.tight_layout(pad=2.0)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{title_prefix.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(out_dir, filename))

    plt.show()


def qanimation(amp_dm_hist, phase_dm_hist, ideal_rho, n_qubits, output_filepath, frame_interval_ms=100):
    """Creates and saves an animation showing density matrix evolution."""
    print_banner("Creating Animation")

    all_dms = amp_dm_hist + phase_dm_hist
    my_style(fontsize=10)

    fig = plt.figure(figsize=(12.5, 6.2))
    ax_re = fig.add_subplot(121, projection='3d')
    ax_im = fig.add_subplot(122, projection='3d')
    plt.subplots_adjust(wspace=0.3)

    global_vmax = np.max(np.abs(np.array(all_dms)))
    epoch_text = fig.text(0.5, 0.975, "", ha='center', va='top', fontsize=12, fontweight='bold')
    fid_text = fig.text(0.5, 0.015, "", ha='center', va='bottom', fontsize=10)

    n_amp_frames = len(amp_dm_hist)

    def update(frame):
        from qst_definition import fidelity

        current_rho = all_dms[frame]
        if frame < n_amp_frames:
            stage, epoch = "Amplitude", frame + 1
        else:
            stage, epoch = "Phase", frame - n_amp_frames + 1

        qdraw(ax_re, np.real(current_rho), "Real Part", current_rho.shape[0], n_qubits, "Re", global_vmax)
        qdraw(ax_im, np.imag(current_rho), "Imag Part", current_rho.shape[0], n_qubits, "Im", global_vmax, cmap_pos='Greens', cmap_neg='Purples')

        fid = fidelity(current_rho, ideal_rho)
        epoch_text.set_text(f"{stage} Training - Epoch {epoch}")
        fid_text.set_text(f"Fidelity: {fid:.4f}")
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_dms),
        interval=frame_interval_ms, blit=False, repeat=False
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Tight layout not applied.*")
        anim.save(output_filepath, writer=animation.PillowWriter(fps=(1000 // frame_interval_ms)))

    print_text(f"Animation saved to {output_filepath}")
    plt.close(fig)
