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
import nibabel as nib


def pad_or_crop_volume(vol, factor=16):
    """
    Pads or crops a 3D volume so that each spatial dimension becomes a multiple of `factor`.
    
    Parameters:
        vol (np.ndarray): 3D volume.
        factor (int): The factor to which dimensions must be a multiple.
    
    Returns:
        vol_padded (np.ndarray): The padded/cropped volume.
        original_shape (tuple): The original volume shape.
    """
    original_shape = np.array(vol.shape)
    target_shape = np.ceil(original_shape / factor).astype(int) * factor
    pad_crop = target_shape - original_shape

    # If cropping is needed (i.e., pad_crop[d] is negative), perform cropping first.
    cropped = vol
    for d in range(3):
        if pad_crop[d] < 0:
            start = (-pad_crop[d]) // 2
            end = start + target_shape[d]
            slices = [slice(None)] * 3
            slices[d] = slice(start, end)
            cropped = cropped[tuple(slices)]
    
    new_shape = np.array(cropped.shape)
    pad_needed = target_shape - new_shape
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    pad_width = [(int(pad_before[i]), int(pad_after[i])) for i in range(3)]
    vol_padded = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    
    return vol_padded, original_shape


def preprocess_image(file_path, output_dir):
    """
    Loads, normalizes, and preprocesses an image, then saves the result and metadata.
    Skips non-3D images.
    """
    img = nib.load(file_path)
    vol = img.get_fdata()

    # Skip non-3D volumes
    if vol.ndim != 3:
        print(f"Skipping (non-3D): {file_path} with shape {vol.shape}")
        return

    # Normalize intensity between 0 and 1
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-5)

    # Pad/crop to multiples of 16
    vol_padded, original_shape = pad_or_crop_volume(vol, factor=16)

    # Determine the output file name based on the input file name.
    basename = os.path.basename(file_path)
    if basename.endswith('.nii.gz'):
        out_name = basename.replace('.nii.gz', '_preprocessed.nii.gz')
    elif basename.endswith('.nii'):
        out_name = basename.replace('.nii', '_preprocessed.nii.gz')
    else:
        out_name = basename + '_preprocessed.nii.gz'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_file = os.path.join(output_dir, out_name)
    nib.save(nib.Nifti1Image(vol_padded, img.affine), preprocessed_file)

    # Save metadata for undoing preprocessing
    metadata_file = preprocessed_file.replace('.nii.gz', '_metadata.npy')
    np.save(metadata_file, {'original_shape': original_shape, 'affine': img.affine})

    print(f"Processed: {file_path} -> {preprocessed_file}")



def main():
    parser = argparse.ArgumentParser(description="Preprocess medical images for training.")
    parser.add_argument('--input_dir', required=True, help='Directory containing raw images (.nii or .nii.gz)')
    parser.add_argument('--output_dir', required=True, help='Directory to save preprocessed images, preserving folder structure')
    args = parser.parse_args()

    # Walk through the input directory recursively.
    for root, dirs, files in os.walk(args.input_dir):
        if 'DVF' in root:
            print(f"Skipping folder with DVF: {root}")
            continue
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                # Compute the file's relative directory with respect to the input directory.
                relative_dir = os.path.relpath(root, args.input_dir)
                # Build the corresponding output directory (preserving subfolder structure)
                output_subfolder = os.path.join(args.output_dir, relative_dir)
                preprocess_image(file_path, output_subfolder)


if __name__ == "__main__":
    main()
