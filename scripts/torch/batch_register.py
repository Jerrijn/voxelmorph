#!/usr/bin/env python

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import voxelmorph as vxm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', required=True, help='folder containing NIfTI images')
parser.add_argument('--output_folder', required=True, help='folder to save warped images and warps')
parser.add_argument('--model', required=True, help='trained VoxelMorph model file')
parser.add_argument('--stride', type=int, default=1, help='interval between each pair of scans')
parser.add_argument('--gpu', help='GPU number(s) to use, default is CPU')
parser.add_argument('--multichannel', action='store_true', help='set if input images have multiple channels')
args = parser.parse_args()

# Set device
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from voxelmorph.torch.networks import VxmDense

# Minimal patch to force CPU loading without changing VoxelMorph's load code
import torch
torch.load = lambda f, *args, **kwargs: torch.serialization._legacy_load(f, map_location='cpu')

# Now this works without error
model = VxmDense.load(args.model, device="cpu")
model.to('cpu').eval()

# Prepare output folder
os.makedirs(args.output_folder, exist_ok=True)

# List and sort files
nii_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
num_files = len(nii_files)

# Loop over pairs with the given stride
for i in range(0, num_files - args.stride):
    moving_file = os.path.join(args.input_folder, nii_files[i])
    fixed_file = os.path.join(args.input_folder, nii_files[i + args.stride])

    print(f"Registering: {nii_files[i]} -> {nii_files[i + args.stride]}")

    add_feat_axis = not args.multichannel

    # Load moving and fixed images
    moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    # Prepare tensors
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

    # Run model
    moved, warp = model(input_moving, input_fixed, registration=True)

    # Prepare output filenames
    basename_moving = os.path.splitext(os.path.splitext(nii_files[i])[0])[0]
    basename_fixed = os.path.splitext(os.path.splitext(nii_files[i + args.stride])[0])[0]

    moved_path = os.path.join(args.output_folder, f"{basename_moving}_to_{basename_fixed}_moved.nii.gz")
    warp_path = os.path.join(args.output_folder, f"{basename_moving}_to_{basename_fixed}_warp.nii.gz")

    # Save results
    moved = moved.detach().cpu().numpy().squeeze()
    warp = warp.detach().cpu().numpy().squeeze()
    warp = np.moveaxis(warp, 0, -1)

    vxm.py.utils.save_volfile(moved, moved_path, fixed_affine)
    vxm.py.utils.save_volfile(warp, warp_path, fixed_affine)
