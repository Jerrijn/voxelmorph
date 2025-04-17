import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def visualize_overlay(vector_field, background, step=5, output_path="visualization.png"):
    """
    Visualize an anatomical background image with overlaid vector field arrows and save the visualization.
    
    Parameters:
        vector_field (np.ndarray): Array of shape (H, W, D, 3) representing the displacement field.
        background (np.ndarray): 3D image array of shape (H, W, D) as the background.
        step (int): Sampling step for quiver arrows to reduce clutter.
        output_path (str): Path to save the visualization.
    """
    vector_field = np.squeeze(vector_field)
    background = np.squeeze(background)
    
    H, W, D, _ = vector_field.shape

    # Create the figure
    fig, ax = plt.subplots()
    
    # Set slice to 50 (or the closest valid slice if D < 50)
    slice_idx = min(43, D - 1)
    bg_slice = background[:, :, slice_idx]
    im = ax.imshow(bg_slice, cmap='gray', origin='lower')
    
    # Prepare grid for quiver plot.
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    vec_slice = vector_field[:, :, slice_idx, :]
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              vec_slice[::step, ::step, 0],
              vec_slice[::step, ::step, 1],
              color='r')
    ax.set_title(f"Slice {slice_idx}")
    
    # Save the visualization
    fig.savefig(output_path)
    print(f"Visualization saved as {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize vector field overlay on MRI scan.")
    parser.add_argument("vector_field_path", type=str, help="Path to the vector field NIfTI file.")
    parser.add_argument("background_path", type=str, help="Path to the background NIfTI file.")
    args = parser.parse_args()

    # Load the vector field
    vec_img = nib.load(args.vector_field_path)
    vec_data = np.squeeze(vec_img.get_fdata())
    print("Original vector field shape:", vec_data.shape)
    
    # Ensure the correct format
    if vec_data.shape[-1] != 3:
        vector_field = np.moveaxis(vec_data, 0, -1)
    else:
        vector_field = vec_data
    print("Converted vector field shape:", vector_field.shape)
    
    # Load the background image
    bg_img = nib.load(args.background_path)
    bg_data = np.squeeze(bg_img.get_fdata())
    print("Background image shape:", bg_data.shape)

    # Run visualization and save the result
    visualize_overlay(vector_field, bg_data, step=5, output_path="visualization.png")

if __name__ == "__main__":
    main()
