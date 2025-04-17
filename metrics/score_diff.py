import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider

def dice_score(nifti_path1: str, nifti_path2: str) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between two NIfTI images.
    
    Parameters:
        nifti_path1 (str): File path to the first NIfTI image.
        nifti_path2 (str): File path to the second NIfTI image.
    
    Returns:
        float: The Dice score indicating the spatial overlap between the two binary segmentation masks.
    """
    # Load image data using nibabel.
    img1 = nib.load(nifti_path1).get_fdata()
    img2 = nib.load(nifti_path2).get_fdata()
    
    # Generate binary masks (voxels > 0.01 are considered part of the segmented region).
    mask1 = img1 > 0.2
    mask2 = img2 > 0.2

    # Compute the intersection and sum of both masks.
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    # Return 1.0 if both masks are empty (no segmented voxels).
    if sum_masks == 0:
        return 1.0
    
    # Calculate and return the Dice Similarity Coefficient.
    dice = 2.0 * intersection / sum_masks
    return dice

def plot_masks(nifti_path1: str, nifti_path2: str) -> None:
    """
    Plot the binary segmentation masks from two NIfTI files on a representative axial slice.
    A slider is added to scroll through all slices (e.g., 64 layers) interactively.
    
    Parameters:
        nifti_path1 (str): File path to the first NIfTI image.
        nifti_path2 (str): File path to the second NIfTI image.
    """
    # Load image data and generate binary masks.
    img1 = nib.load(nifti_path1).get_fdata()
    img2 = nib.load(nifti_path2).get_fdata()
    mask1 = img1 > 0.2
    mask2 = img2 > 0.2

    # Determine the number of axial slices (assumes third dimension represents the axial axis).
    n_slices = mask1.shape[2]

    # Create a side-by-side plot of the two masks.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Select the initial slice (middle slice).
    initial_slice = n_slices // 2
    im1 = axes[0].imshow(mask1[:, :, initial_slice], cmap='gray')
    axes[0].set_title('Segmentation Mask 1 (Axial Slice)')
    axes[0].axis('off')
    
    im2 = axes[1].imshow(mask2[:, :, initial_slice], cmap='gray')
    axes[1].set_title('Segmentation Mask 2 (Axial Slice)')
    axes[1].axis('off')

    # Adjust the layout to make room for the slider.
    plt.subplots_adjust(bottom=0.15)

    # Create slider axis [left, bottom, width, height] and initialize slider.
    slider_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    slice_slider = Slider(slider_ax, "Slice", 0, n_slices - 1, valinit=initial_slice, valstep=1)

    # Define the update function to refresh the plots on slider movement.
    def update(val):
        slice_index = int(slice_slider.val)
        im1.set_data(mask1[:, :, slice_index])
        im2.set_data(mask2[:, :, slice_index])
        fig.canvas.draw_idle()

    # Register the update function with the slider.
    slice_slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    # Set up command-line arguments to receive file paths.
    parser = argparse.ArgumentParser(
        description="Compute the Dice Similarity Coefficient and plot segmentation masks for two MRI NIfTI files."
    )
    parser.add_argument("nifti_path1", type=str, help="Path to the first NIfTI file.")
    parser.add_argument("nifti_path2", type=str, help="Path to the second NIfTI file.")
    args = parser.parse_args()
    
    # Compute and print the Dice score.
    score = dice_score(args.nifti_path1, args.nifti_path2)
    print(f"Dice Similarity Coefficient: {score:.4f}")
    
    # Plot the binary segmentation masks with interactive scrolling.
    plot_masks(args.nifti_path1, args.nifti_path2)
