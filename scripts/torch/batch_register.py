#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import voxelmorph as vxm
from voxelmorph.torch.networks import VxmDense

def process_directory(input_dir, output_dir, model, device, stride, multichannel):
    """
    Register all scans in input_dir (non-recursive) and save results into output_dir.
    Keeps the same directory structure.
    """
    # Ensure output subdirectory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather and sort all .nii / .nii.gz files in this directory
    nii_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    )
    n = len(nii_files)
    if n < stride + 1:
        # Nothing to do if fewer files than needed for a single pair
        return

    for i in range(0, n - stride):
        mov_fname = nii_files[i]
        fix_fname = nii_files[i + stride]
        mov_path = os.path.join(input_dir, mov_fname)
        fix_path = os.path.join(input_dir, fix_fname)

        print(f"Registering: {mov_path} → {fix_path}")

        add_feat_axis = not multichannel

        # Load volumes as numpy arrays [B, X, Y, Z, C]
        moving = vxm.py.utils.load_volfile(
            mov_path, add_batch_axis=True, add_feat_axis=add_feat_axis
        )
        fixed, fixed_affine = vxm.py.utils.load_volfile(
            fix_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True
        )

        # Convert to tensors [B, C, X, Y, Z]
        input_moving = (
            torch.from_numpy(moving)
                 .to(device)
                 .float()
                 .permute(0, 4, 1, 2, 3)
        )
        input_fixed = (
            torch.from_numpy(fixed)
                 .to(device)
                 .float()
                 .permute(0, 4, 1, 2, 3)
        )

        # Run registration: (warped_image, displacement_field)
        moved, warp = model(input_moving, input_fixed, registration=True)

        # Build output paths
        base_m = os.path.splitext(os.path.splitext(mov_fname)[0])[0]
        base_f = os.path.splitext(os.path.splitext(fix_fname)[0])[0]
        moved_out = os.path.join(output_dir, f"{base_m}_to_{base_f}_moved.nii.gz")
        warp_out  = os.path.join(output_dir, f"{base_m}_to_{base_f}_warp.nii.gz")

        # Save warped image
        moved_np = moved.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved_np, moved_out, fixed_affine)

        # Save displacement field (channel-first → last)
        warp_np = warp.detach().cpu().numpy().squeeze()
        warp_np = np.moveaxis(warp_np, 0, -1)
        vxm.py.utils.save_volfile(warp_np, warp_out, fixed_affine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recursively batch-register NIfTI scans preserving directory structure'
    )
    parser.add_argument('--input_folder',  required=True,
                        help='Root folder containing NIfTI images (.nii, .nii.gz)')
    parser.add_argument('--output_folder', required=True,
                        help='Root of output folder (will mirror input structure)')
    parser.add_argument('--model',         required=True,
                        help='Path to trained VoxelMorph model file (.pt)')
    parser.add_argument('--stride',   type=int, default=1,
                        help='Interval between each pair of scans (default: 1)')
    parser.add_argument('--gpu',      default='-1',
                        help='GPU ID(s) to use; set to -1 for CPU')
    parser.add_argument('--multichannel', action='store_true',
                        help='Set this if inputs have multiple channels')
    args = parser.parse_args()

    # Decide device
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    # Load model (torch.load handles persistent IDs correctly)
    model = VxmDense.load(args.model, device=device)
    model.to(device).eval()

    # Walk input_folder recursively
    for root, dirs, files in os.walk(args.input_folder):
        # Compute relative path from the input root
        rel_path = os.path.relpath(root, args.input_folder)
        # Determine corresponding output directory
        out_dir = os.path.join(args.output_folder, rel_path)
        # Process this directory (only its own .nii files)
        process_directory(
            input_dir=root,
            output_dir=out_dir,
            model=model,
            device=device,
            stride=args.stride,
            multichannel=args.multichannel
        )

    print("✅ Recursive batch registration complete.")
