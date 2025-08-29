import nibabel as nib
import numpy as np
dvf_path = "C:\\Users\P096350\Documents\output\\06-18_17-31_bs2_ncc_mi_final.pt\\pt002\\MR1\\motility_dynamics\\warp\\002_to_001_warp.nii.gz"
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#dvf_path = "033_to_032_warp.nii.gz"
dvf = nib.load(dvf_path).get_fdata()

print(f"DVF shape: {dvf.shape}, dtype: {dvf.dtype}")
print(f"DVF min: {dvf.min()}, max: {dvf.max()}")

# Unique values per channel
if dvf.ndim == 4 and dvf.shape[-1] in [2, 3]:
    for c in range(dvf.shape[-1]):
        unique_vals = np.unique(dvf[..., c].round(3))
        print(f"Channel {c} unique values (rounded to 3 decimals): {len(unique_vals)}")
else:
    print("Unexpected DVF shape.")

# Compute the norm
motion_map = np.linalg.norm(dvf, axis=-1)
unique_motion = np.unique(motion_map.round(3))
print(f"Unique motion magnitudes (rounded to 3 decimals): {len(unique_motion)}")
print(f"Motion map min: {motion_map.min()}, max: {motion_map.max()}")

# Plot
import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, dvf.shape[-1]+1, figsize=(15, 4))
# for i in range(dvf.shape[-1]):
#     axes[i].hist(dvf[..., i].flatten(), bins=1000)
#     axes[i].set_title(f"DVF Channel {i}")
# axes[-1].hist(motion_map.flatten(), bins=1000)
# axes[-1].set_title("Motion Map Magnitude")
# plt.tight_layout()
# # plt.show()
# import matplotlib.pyplot as plt

# mid_ax = dvf.shape[2] // 2  # middle slice
# plt.subplot(1, 2, 1)
# plt.imshow(dvf[:, :, mid_ax, 2], cmap='bwr')
# plt.title('DVF Channel 2, Mid Slice')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(motion_map[:, :, mid_ax], cmap='hot')
# plt.title('Motion Map, Mid Slice')
# plt.colorbar()
# plt.show()

# mask: (X, Y, Z) or (X, Y, Z, 1)
# motion_map = np.linalg.norm(dvf, axis=-1)
# # Load your anatomical mask (binary 1=anatomy, 0=background/pad)
# mask = nib.load('C:\\Users\\P096350\\Documents\\segmentation_preprocessed\\pt002\MR1\\18991230_000000WIP3DMOTILITY25mm160dyns201a1002_001_preprocessed.nii.gz').get_fdata().astype(bool)

# # Use only the anatomy voxels:
# mvh_vals = motion_map[mask]
# # Then plot histogram or MVH only for mvh_vals
# plt.hist(mvh_vals.flatten(), bins=100)
# plt.title("Masked Motion Map Histogram")
# plt.show()

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_dvf(path):
    dvf = nib.load(path).get_fdata()
    # Adjust axes if needed: (3, H, W, D) -> (H, W, D, 3)
    if dvf.shape[0] == 3 and dvf.ndim == 4:
        dvf = np.moveaxis(dvf, 0, -1)
    return dvf

def compute_harmonic_energy(dvf):
    """Compute Harmonic Energy per voxel as Frobenius norm of Jacobian."""
    # dvf shape: (H, W, D, 3)
    gradients = [np.gradient(dvf[..., i]) for i in range(3)]  # X, Y, Z component
    # Build Jacobian matrix J (H, W, D, 3, 3)
    J = np.stack([
        [gradients[i][j] for j in range(3)]  # dDVF_i/dx_j
        for i in range(3)
    ], axis=-1)  # shape: (3, 3, H, W, D)
    J = np.moveaxis(J, [0,1], [-2,-1])      # (H, W, D, 3, 3)
    # Frobenius norm per voxel
    HE = np.linalg.norm(J, axis=(-2, -1))
    return HE

def interactive_scroll_heatmap(he, axis=2, vmin=None, vmax=None):
    """Scroll through HE slices interactively along given axis (default: axial)."""
    he = np.swapaxes(he, axis, 2)  # Now [H, W, D], axis=2 is the scrolled one
    n_slices = he.shape[2]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img = ax.imshow(he[..., 0], cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title('Harmonic Energy (slice 0)')

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Slice', 0, n_slices-1, valinit=0, valfmt='%d', valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(he[..., idx])
        ax.set_title(f'Harmonic Energy (slice {idx})')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# === USAGE EXAMPLE ===
he = compute_harmonic_energy(dvf[:, :, 7:57])
interactive_scroll_heatmap(he, axis=2)  # axis=2: axial, 0: sagittal, 1: coronal

