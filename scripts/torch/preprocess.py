#!/usr/bin/env python
"""
Preprocess script for medical images.
- Normalizes intensity values (only for image data).
- Resizes images to a common shape.
- Pads/crops to ensure spatial dimensions are multiples of 16.
- Saves metadata for postprocessing (only for image data).
- Automatically detects DVFs based on filename.

Usage:
    python preprocess.py --input_dir path/to/raw_data --output_dir path/to/preprocessed_data
"""

import os
import argparse
import numpy as np
import nibabel as nib


def pad_or_crop_volume(vol, factor=16):
    """
    Pads or crops a 3D or 4D volume (DVF) so that spatial dims (first 3) are multiples of `factor`.

    Returns:
        vol_padded (np.ndarray): The padded/cropped volume.
        original_shape (tuple): The original volume shape.
    """
    is_dvf = (vol.ndim == 4 and vol.shape[-1] == 3)
    spatial_shape = np.array(vol.shape[:3])
    target_shape = np.ceil(spatial_shape / factor).astype(int) * factor
    pad_crop = target_shape - spatial_shape

    # Crop if needed
    cropped = vol
    for d in range(3):
        if pad_crop[d] < 0:
            start = (-pad_crop[d]) // 2
            end = start + target_shape[d]
            slices = [slice(None)] * vol.ndim
            slices[d] = slice(start, end)
            cropped = cropped[tuple(slices)]

    # Pad if needed
    new_shape = np.array(cropped.shape[:3])
    pad_needed = target_shape - new_shape
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    pad_width = [(int(pad_before[i]), int(pad_after[i])) for i in range(3)]

    # If DVF, don't pad vector dim
    if is_dvf:
        pad_width.append((0, 0))  # No padding on last dim

    vol_padded = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    return vol_padded, spatial_shape

def preprocess_image(file_path, output_dir):
    """
    Loads and preprocesses an image (cine-MRI or DVF) and saves the result.
    """
    basename = os.path.basename(file_path)
    is_dvf = 'DVF' in basename.upper()

    img = nib.load(file_path)
    vol = img.get_fdata()
    vol = np.squeeze(vol)  # removes dims like (160,160,50,1,3) → (160,160,50,3)

    if is_dvf:
        if vol.ndim != 4 or vol.shape[-1] != 3:
            print(f"Skipping (unexpected DVF shape): {file_path} with shape {vol.shape}")
            return
        vol_padded, _ = pad_or_crop_volume(vol, factor=16)

    else:
        if vol.ndim != 3:
            print(f"Skipping (non-3D image): {file_path} with shape {vol.shape}")
            return
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-5)
        vol_padded, original_shape = pad_or_crop_volume(vol, factor=16)

    # Build output name
    if basename.endswith('.nii.gz'):
        out_name = basename.replace('.nii.gz', '_preprocessed.nii.gz')
    elif basename.endswith('.nii'):
        out_name = basename.replace('.nii', '_preprocessed.nii.gz')
    else:
        out_name = basename + '_preprocessed.nii.gz'

    os.makedirs(output_dir, exist_ok=True)
    preprocessed_file = os.path.join(output_dir, out_name)
    nib.save(nib.Nifti1Image(vol_padded, img.affine), preprocessed_file)

    # Save metadata only if not DVF
    if not is_dvf:
        metadata_file = preprocessed_file.replace('.nii.gz', '_metadata.npy')
        np.save(metadata_file, {'original_shape': original_shape, 'affine': img.affine})

    print(f"Processed ({'DVF' if is_dvf else 'IMG'}): {file_path} -> {preprocessed_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess medical images for training.")
    parser.add_argument('--input_dir', required=True, help='Directory containing raw images (.nii or .nii.gz)')
    parser.add_argument('--output_dir', required=True, help='Directory to save preprocessed images, preserving folder structure')
    args = parser.parse_args()

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                relative_dir = os.path.relpath(root, args.input_dir)
                output_subfolder = os.path.join(args.output_dir, relative_dir)
                preprocess_image(file_path, output_subfolder)


if __name__ == "__main__":
    main()
