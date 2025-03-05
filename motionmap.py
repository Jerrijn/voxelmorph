import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------------------------------------------
# 1) Load the 3D displacement field and compute the motion map
# -------------------------------------------------------------------------
# Adjust this path to point to your vector field file:
vector_field_path = r"C:\Users\P096350\OneDrive - Amsterdam UMC\Documenten\actualtest2_vector.nii"

# Load the vector field: shape expected (X, Y, Z, 3)
vec_img = nib.load(vector_field_path)
vec_data = vec_img.get_fdata()  # e.g., shape (160, 160, 64, 3)
print("Vector field shape:", vec_data.shape)

# Compute the motion map (the Euclidean norm of the 3D displacement at each voxel).
# Result is a 3D array (X, Y, Z).
motion_map = np.linalg.norm(vec_data, axis=-1)
print("Motion map shape:", motion_map.shape)

# -------------------------------------------------------------------------
# 2) Visualize the motion map slice-by-slice with a slider
# -------------------------------------------------------------------------
def visualize_motion_map(motion_map):
    """
    Displays axial slices of a 3D motion map with an interactive slider.
    motion_map should be shape (X, Y, Z).
    """
    X, Y, Z = motion_map.shape

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    init_slice = 0
    im = ax.imshow(motion_map[:, :, init_slice], cmap='jet', origin='lower')
    ax.set_title(f"Motion Map - Slice {init_slice}")
    plt.colorbar(im, ax=ax, label='Motion magnitude (units)')

    # Slider to move through slices in the Z dimension
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(ax_slider, 'Slice', 0, Z - 1, valinit=init_slice, valstep=1)

    def update(val):
        slice_idx = int(slice_slider.val)
        im.set_data(motion_map[:, :, slice_idx])
        ax.set_title(f"Motion Map - Slice {slice_idx}")
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()

visualize_motion_map(motion_map)

# -------------------------------------------------------------------------
# 3) Generate and display the Motion-Volume Histogram (MVH)
# -------------------------------------------------------------------------
def motion_volume_histogram(motion_map):
    """
    Plots the Motion-Volume Histogram (MVH) of a 3D motion map.
    The MVH shows for each motion threshold M, the fraction of voxels
    that have motion >= M.
    """
    # Flatten the motion map to get all voxel magnitudes in 1D
    motion_vals = motion_map.flatten()

    # Sort in ascending order
    motion_vals_sorted = np.sort(motion_vals)

    # Number of voxels
    N = len(motion_vals_sorted)

    # Fraction of voxels that have motion >= motion_vals_sorted[i]
    # For index i in ascending order, that fraction is (N - i)/N
    fraction = (N - np.arange(N)) / N * 100.0  # convert to percentage

    # Plot the MVH
    plt.figure()
    plt.plot(motion_vals_sorted, fraction, color='green', linewidth=2)
    plt.title('Motion-Volume Histogram (MVH)')
    plt.xlabel('Motion (units)')
    plt.ylabel('Volume (%)')
    plt.ylim([0, 100])
    # x-axis goes from 0 to the maximum motion value
    plt.xlim([0, motion_vals_sorted.max()])
    plt.show()

motion_volume_histogram(motion_map)
