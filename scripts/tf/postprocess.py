#!/usr/bin/env python

"""
Postprocess script for restoring original image dimensions.
- Unpads images back to their original shape.
- Saves the final output.

Usage:
    python postprocess.py --input_dir path/to/preprocessed_data --output_dir path/to/final_output
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob


def unpad_volume(vol, original_shape):
    """Restores a volume to its original shape after padding."""
    current_shape = np.array(vol.shape)
    pad_total = current_shape - original_shape
    pad_before = pad_total // 2

    slices = []
    for pb, orig in zip(pad_before, original_shape):
        slices.append(slice(pb, pb + orig))

    return vol[tuple(slices)]


def postprocess_image(file_path, output_dir):
    """Restores original dimensions using metadata and saves the final image."""
    img = nib.load(file_path)
    vol = img.get_fdata()

    # Load metadata
    metadata_file = file_path.replace('_preprocessed.nii.gz', '_metadata.npy')
    if not os.path.exists(metadata_file):
        print(f"Skipping {file_path} (metadata not found)")
        return

    metadata = np.load(metadata_file, allow_pickle=True).item()
    original_shape = metadata['original_shape']
    affine = metadata['affine']

    # Unpad to restore original dimensions
    vol_restored = unpad_volume(vol, original_shape)

    # Save final image
    final_file = os.path.join(output_dir, os.path.basename(file_path).replace('_preprocessed', '_final'))
    nib.save(nib.Nifti1Image(vol_restored, affine), final_file)

    print(f"Restored: {file_path} -> {final_file}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess medical images to restore original dimensions.")
    parser.add_argument('--input_dir', required=True, help='Directory containing preprocessed images')
    parser.add_argument('--output_dir', required=True, help='Directory to save final restored images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob(os.path.join(args.input_dir, '*_preprocessed.nii.gz'))
    for file in files:
        postprocess_image(file, args.output_dir)


if __name__ == "__main__":
    main()
