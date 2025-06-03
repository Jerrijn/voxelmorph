#!/usr/bin/env python

import os
import re
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

def normalize_dvf(dvf: np.ndarray) -> np.ndarray:
    """
    Squeeze out any singleton dimensions and ensure shape (X,Y,Z,3).
    Accepts:
      - (1,X,Y,Z,3)
      - (X,Y,Z,3,1)
      - (3,X,Y,Z)
      - (X,Y,Z,3)
    """
    # remove all size-1 dims
    dvf = np.squeeze(dvf)
    # if channel-first (3,X,Y,Z) → channel-last
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)
    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        raise ValueError(f"DVF must be shape (X,Y,Z,3) or (3,X,Y,Z) after squeezing; got {dvf.shape}")
    return dvf

def apply_dvf_to_image(mri_data: np.ndarray, dvf: np.ndarray,
                       order: int = 1, mode: str = 'nearest') -> np.ndarray:
    """
    Warp a 3D MRI by a displacement vector field.
    - mri_data: shape (X, Y, Z)
    - dvf: raw loaded array (will be normalized to (X,Y,Z,3))
    """
    dvf = normalize_dvf(dvf)
    X, Y, Z = mri_data.shape

    # create coordinate grid
    grid = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z),
        indexing='ij'
    )
    # add displacements
    coords = [
        grid[0] + dvf[..., 0],
        grid[1] + dvf[..., 1],
        grid[2] + dvf[..., 2],
    ]
    coords_flat = [c.flatten() for c in coords]

    warped_flat = map_coordinates(
        mri_data, coords_flat,
        order=order, mode=mode
    )
    return warped_flat.reshape(mri_data.shape)


def main():
    parser = argparse.ArgumentParser(
        description="Apply DVFs to the matching ‘F’ MRI in a parallel folder tree"
    )
    parser.add_argument('--dvf_folder',   required=True,
                        help='Root of DVF NIfTIs (.nii/.nii.gz)')
    parser.add_argument('--mri_folder',   required=True,
                        help='Root of original MRI NIfTIs')
    parser.add_argument('--output_folder', required=True,
                        help='Where to save warped MRIs (mirrors structure)')
    parser.add_argument('--order', type=int, default=1,
                        help='Interpolation order (0=nearest, 1=linear, etc.)')
    parser.add_argument('--mode', default='nearest',
                        choices=['reflect','constant','nearest','mirror','wrap'],
                        help='Boundary mode for interpolation')
    args = parser.parse_args()

    # matches '_F###_Mxxx.nii' (### = digits)
    dvf_pattern = re.compile(r'_F(\d+)_M\d+\.nii(?:\.gz)?$', re.IGNORECASE)

    for dvf_root, _, files in os.walk(args.dvf_folder):
        rel_dir = os.path.relpath(dvf_root, args.dvf_folder)
        mri_dir = os.path.join(args.mri_folder, rel_dir)
        out_dir = os.path.join(args.output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for fn in sorted(files):
            if not fn.lower().endswith(('.nii', '.nii.gz')):
                continue

            m = dvf_pattern.search(fn)
            if not m:
                print(f"[WARN] Skipping unrecognized DVF name '{fn}'")
                continue

            fixed_idx = int(m.group(1))
            padded = f"{fixed_idx:03d}"  # zero-pad to three digits

            if not os.path.isdir(mri_dir):
                print(f"[WARN] MRI folder '{mri_dir}' missing; skipping '{fn}'")
                continue

            # look for MRI ending in _<padded>.nii or .nii.gz
            candidates = [
                f for f in os.listdir(mri_dir)
                if f.endswith(f"_{padded}.nii") or f.endswith(f"_{padded}.nii.gz")
            ]
            if not candidates:
                print(f"[WARN] No MRI matching _{padded} in '{mri_dir}'; skipping '{fn}'")
                continue
            if len(candidates) > 1:
                print(f"[WARN] Multiple MRIs end with _{padded} in '{mri_dir}': {candidates}; using first")
            mri_fname = candidates[0]

            dvf_path = os.path.join(dvf_root, fn)
            mri_path = os.path.join(mri_dir, mri_fname)
            print(f"Applying DVF '{fn}' → MRI '{mri_fname}'")

            # load DVF & MRI
            dvf_img  = nib.load(dvf_path)
            dvf_data = dvf_img.get_fdata()
            mri_img  = nib.load(mri_path)
            mri_data = mri_img.get_fdata()

            # warp
            warped = apply_dvf_to_image(
                mri_data, dvf_data,
                order=args.order, mode=args.mode
            )

            base = os.path.splitext(os.path.splitext(mri_fname)[0])[0]
            out_fname = f"{base}_warped.nii.gz"
            out_path  = os.path.join(out_dir, out_fname)
            nib.save(nib.Nifti1Image(warped, mri_img.affine, mri_img.header), out_path)

    print("✅ All DVFs applied and warped MRIs saved.")

if __name__ == '__main__':
    main()
