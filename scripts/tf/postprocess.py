import os
import numpy as np
import nibabel as nib
import argparse
import traceback

"""
Restore Preprocessed Medical Images

This script restores a preprocessed NIfTI (.nii) medical image back to its original shape using metadata stored in a .npy file.

Usage:
    python restore_image.py --file path/to/image_preprocessed.nii --output_dir path/to/output_folder

The restored image will be saved in the specified output folder.
"""

def crop_to_original_shape(vol, original_shape):
    """Crops the volume back to its original shape."""
    current_shape = vol.shape
    crop_slices = []
    for i in range(3):
        start = (current_shape[i] - original_shape[i]) // 2
        end = start + original_shape[i]
        crop_slices.append(slice(start, end))
    return vol[crop_slices[0], crop_slices[1], crop_slices[2]]

def restore_image(preprocessed_file, output_dir):
    """Restores a preprocessed image to its original shape using metadata and saves it in the output directory."""
    try:
        # Ensure metadata file exists
        metadata_file = preprocessed_file.replace('.nii', '.nii.npy')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load preprocessed image
        img = nib.load(preprocessed_file)
        vol = img.get_fdata()

        # Load metadata
        metadata = np.load(metadata_file, allow_pickle=True).item()
        original_shape = metadata.get("original_shape")
        affine = metadata.get("affine")
        
        if original_shape is None or affine is None:
            raise ValueError("Metadata file is missing required information.")

        # Crop back to original shape
        restored_vol = crop_to_original_shape(vol, original_shape)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save restored image in the output directory
        restored_filename = os.path.basename(preprocessed_file).replace('_preprocessed.nii', '_restored.nii')
        restored_file = os.path.join(output_dir, restored_filename)
        nib.save(nib.Nifti1Image(restored_vol, affine), restored_file)

        print(f"✅ Restored: {preprocessed_file} -> {restored_file}")
    
    except Exception as e:
        print(f"❌ Error restoring {preprocessed_file}: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Restore a preprocessed medical image to its original shape.")
    parser.add_argument('--file', required=True, help='Path to the preprocessed .nii file')
    parser.add_argument('--output_dir', required=True, help='Directory to save the restored image')
    args = parser.parse_args()
    
    restore_image(args.file, args.output_dir)


if __name__ == "__main__":
    main()
