#!/usr/bin/env python

"""
Postprocess script for restoring original image dimensions.
- Unpads spatial dimensions (x, y, z) back to original shape.
- Supports 3D and 4D NIfTI files.
- Saves the final output.

Usage:
    python postprocess.py --input_dir path/to/preprocessed_data --output_dir path/to/final_output
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob


def unpad_spatial_dimensions(vol, original_shape):
    """Restores the first 3 (spatial) dimensions of a volume to their original shape after padding."""
    current_shape = np.array(vol.shape)
    spatial_dims = len(original_shape)

    # If no change needed
    if np.array_equal(current_shape[:spatial_dims], original_shape):
        return vol

    pad_total = current_shape[:spatial_dims] - original_shape
    pad_before = pad_total // 2

    # Build slice for each dimension: unpad first 3, keep rest as is
    slices = [slice(pb, pb + orig) for pb, orig in zip(pad_before, original_shape)]
    slices.extend([slice(None)] * (vol.ndim - spatial_dims))  # keep remaining dims untouched

    return vol[tuple(slices)]


def postprocess_image(file_path, output_dir):
    """Restores original spatial dimensions using metadata and saves the final image."""
    img = nib.load(file_path)
    vol = img.get_fdata()
    affine = img.affine

    metadata_file = file_path.replace('_preprocessed.nii.gz', '_metadata.npy')
    if not os.path.exists(metadata_file):
        print(f"Skipping (no metadata): {file_path}")
        return

    metadata = np.load(metadata_file, allow_pickle=True).item()
    original_shape = metadata.get('original_shape', vol.shape[:3])

    vol = unpad_spatial_dimensions(vol, original_shape)

    # Save final image
    final_file = os.path.join(output_dir, os.path.basename(file_path).replace('_preprocessed', '_final'))
    nib.save(nib.Nifti1Image(vol, affine), final_file)
    print(f"Restored: {file_path} -> {final_file}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess medical images to restore original spatial dimensions.")
    parser.add_argument('--input_dir', required=True, help='Directory containing preprocessed images')
    parser.add_argument('--output_dir', required=True, help='Directory to save final restored images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob(os.path.join(args.input_dir, '*_preprocessed.nii.gz'))
    for file in files:
        postprocess_image(file, args.output_dir)


if __name__ == "__main__":
    main()
