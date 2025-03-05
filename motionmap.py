import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------------------------------------------
# 1) Load the 3D displacement field and remove singleton dimensions
# -------------------------------------------------------------------------
vector_field_path = r"C:\Users\P096350\OneDrive - Amsterdam UMC\Documenten\Data\pt018\MR1\DVF\DVF_20190719_ZKN__P18_MR1_MMT_25_2_S_F1_M2.nii"

# Load data and squeeze out any dims of size 1
vec_img = nib.load(vector_field_path)
vec_data = np.squeeze(vec_img.get_fdata())
print("Vector field shape after squeeze:", vec_data.shape)
# Expected shape now: (X, Y, Z, 3)

# Compute motion map: Euclidean norm over the last dimension
motion_map = np.squeeze(np.linalg.norm(vec_data, axis=-1))
print("Motion map shape after norm & squeeze:", motion_map.shape)
# Expected shape: (X, Y, Z)

# -------------------------------------------------------------------------
# 2) Prepare data for the Motion-Volume Histogram (MVH)
# -------------------------------------------------------------------------
motion_vals = motion_map.flatten()
motion_vals_sorted = np.sort(motion_vals)
N = len(motion_vals_sorted)
fraction = (N - np.arange(N)) / N * 100.0  # Fraction in percent

# -------------------------------------------------------------------------
# 3) Create a figure with two subplots:
#    Left  -> interactive slider for axial slices
#    Right -> Motion-Volume Histogram (x-axis limited to 15)
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 6))

# -- LEFT SUBPLOT (interactive slice viewer) --
ax_left = fig.add_subplot(1, 2, 1)
plt.subplots_adjust(bottom=0.25)

init_slice = 0
img_display = ax_left.imshow(motion_map[:, :, init_slice],
                             cmap='jet', origin='lower')
ax_left.set_title(f"Motion Map - Slice {init_slice}")
cb = plt.colorbar(img_display, ax=ax_left, fraction=0.046, pad=0.04)
cb.set_label('Motion magnitude')

slider_ax = fig.add_axes([0.15, 0.1, 0.3, 0.03], facecolor='lightgoldenrodyellow')
slice_slider = Slider(slider_ax, 'Slice', 0, motion_map.shape[2] - 1,
                      valinit=init_slice, valstep=1)

def update_slice(val):
    slice_idx = int(slice_slider.val)
    img_display.set_data(motion_map[:, :, slice_idx])
    ax_left.set_title(f"Motion Map - Slice {slice_idx}")
    fig.canvas.draw_idle()

slice_slider.on_changed(update_slice)

# -- RIGHT SUBPLOT (MVH) --
ax_right = fig.add_subplot(1, 2, 2)
ax_right.plot(motion_vals_sorted, fraction, color='green', linewidth=2)
ax_right.set_title('Motion-Volume Histogram (MVH)')
ax_right.set_xlabel('Motion (units)')
ax_right.set_ylabel('Volume (%)')
# Set x-axis limit to 15
ax_right.set_xlim([0, 10])
ax_right.set_ylim([0, 100])

plt.show()
