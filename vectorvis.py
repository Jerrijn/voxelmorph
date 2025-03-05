import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Load the vector field ---
vec_img = nib.load("C:\\Users\\P096350\\OneDrive - Amsterdam UMC\\Documenten\\actualtest2_vector.nii")
vec_data = vec_img.get_fdata()  # expected shape is (160, 160, 64, 3)
print("Original vector field shape:", vec_data.shape)
# If needed, convert from channel-first to channel-last (in this case it's already correct)
if vec_data.shape[-1] != 3:
    vector_field = np.moveaxis(vec_data, 0, -1)
else:
    vector_field = vec_data
print("Converted vector field shape:", vector_field.shape)

# --- Load the background (bowels) image ---
bg_img = nib.load("C:\\Users\\P096350\\OneDrive - Amsterdam UMC\\Documenten\\actualtest2.nii")
bg_data = bg_img.get_fdata()  # expected shape: (160, 160, 64)
print("Background image shape:", bg_data.shape)

def visualize_overlay(vector_field, background, step=5):
    """
    Visualize an anatomical background image with overlaid vector field arrows.
    
    Parameters:
        vector_field (np.ndarray): Array of shape (H, W, D, 3) representing the displacement field.
        background (np.ndarray): 3D image array of shape (H, W, D) as the background.
        step (int): Sampling step for quiver arrows to reduce clutter.
    """
    H, W, D, _ = vector_field.shape

    # Create the figure and adjust layout for slider.
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Initial slice index.
    init_slice = 0
    bg_slice = background[:, :, init_slice]
    im = ax.imshow(bg_slice, cmap='gray', origin='lower')
    
    # Prepare grid for quiver plot.
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    vec_slice = vector_field[:, :, init_slice, :]
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              vec_slice[::step, ::step, 0],
              vec_slice[::step, ::step, 1],
              color='r')
    ax.set_title(f"Slice {init_slice}")
    
    # Create a slider to navigate through slices.
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(ax_slice, 'Slice', 0, D - 1, valinit=init_slice, valstep=1)

    def update(val):
        slice_idx = int(slice_slider.val)
        bg_slice = background[:, :, slice_idx]
        im.set_data(bg_slice)
        ax.set_title(f"Slice {slice_idx}")
        
        # Remove previous quiver arrows by iterating over a copy of ax.collections.
        for coll in list(ax.collections):
            coll.remove()
        
        # Get updated vector field slice.
        vec_slice = vector_field[:, :, slice_idx, :]
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  vec_slice[::step, ::step, 0],
                  vec_slice[::step, ::step, 1],
                  color='lime')
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()

# Run the visualization.
visualize_overlay(vector_field, bg_data, step=5)
