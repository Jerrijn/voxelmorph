#!/usr/bin/env python

"""
Preprocess script for medical images.
- Normalizes intensity values.
- Resizes images to a common shape.
- Pads/crops to ensure spatial dimensions are multiples of 16.
- Saves metadata for postprocessing.

Usage:
    python preprocess.py --input_dir path/to/raw_data --output_dir path/to/preprocessed_data
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import nibabel as nib
from glob import glob


def pad_or_crop_volume(vol, factor=16):
    """Pads or crops a 3D volume to ensure dimensions are multiples of `factor`."""
    original_shape = np.array(vol.shape)
    target_shape = np.ceil(original_shape / factor).astype(int) * factor
    pad_crop = target_shape - original_shape
    pad_before = pad_crop // 2
    pad_after = pad_crop - pad_before

    slices = []
    for d in range(3):
        if pad_crop[d] >= 0:
            slices.append((pad_before[d], pad_after[d]))  # Padding
        else:
            slices.append((slice(-pad_before[d], target_shape[d] + pad_after[d]),))  # Cropping

    # Apply padding
    vol_padded = np.pad(vol, slices, mode='constant', constant_values=0)
    return vol_padded, original_shape


def preprocess_image(file_path, output_dir):
    """Loads, normalizes, and preprocesses an image."""
    img = nib.load(file_path)
    vol = img.get_fdata()
    
    # Normalize intensity between 0 and 1
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-5)

    # Pad/crop to multiples of 16
    vol_padded, original_shape = pad_or_crop_volume(vol, factor=16)

    # Save preprocessed image
    preprocessed_file = os.path.join(output_dir, os.path.basename(file_path).replace('.nii.gz', '_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(vol_padded, img.affine), preprocessed_file)

    # Save metadata for undoing preprocessing
    metadata_file = preprocessed_file.replace('.nii.gz', '_metadata.npy')
    np.save(metadata_file, {'original_shape': original_shape, 'affine': img.affine})

    print(f"Processed: {file_path} -> {preprocessed_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess medical images for training.")
    parser.add_argument('--input_dir', required=True, help='Directory containing raw images (.nii)')
    parser.add_argument('--output_dir', required=True, help='Directory to save preprocessed images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob(os.path.join(args.input_dir, '*.nii'))
    print(f"Found {len(files)} files in {args.input_dir}: {files}")  # Add this for debugging

    for file in files:
        preprocess_image(file, args.output_dir)


if __name__ == "__main__":
    main()
