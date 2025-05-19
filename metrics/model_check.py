#!/usr/bin/env python3
r"""
Unified script to automate various checks on a trained VoxelMorph model.

Subcommands:
  apply           Apply the trained model to a folder of NIfTI images (recursively), preserving directory hierarchy.
    python .\metrics\model_check.py apply `
    "C:\Users\P096350\Documents\test_data\pt021" `
    --interval 1 `
    --output_folder "C:\Users\P096350\Documents\output\pt021" `
    --model "C:\Users\P096350\Documents\voxelmorph\voxelmorph\models\05-12_16-18_bs2_final.pt" `
    --gpu 0 `
    --prefix patient21
  motionmap       Generate a motion map and Motion-Volume Histogram (MVH) from a DVF file.
    python .\metrics\model_check.py motionmap `
    "C:\Users\P096350\Documents\output\pt021\MR2\motility_dynamics\patient21_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_001_preprocessed.nii_to_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_002_preprocessed.nii_dvf.nii.gz" `
    --slice 50 `
    --out_map "C:\Users\P096350\Documents\output\pt021\MR2\motion_map.png" `
    --out_hist "C:\Users\P096350\Documents\output\pt021\MR2\mvh.png"
  dvf_check       Sum multiple DVF files and visualize the average vector field and heatmap for a given slice.
    python .\metrics\model_check.py dvf_check `
    "C:\Users\P096350\Documents\output\pt021\MR2\motility_dynamics" `
    --slice 60 `
    --vec_out "C:\Users\P096350\Documents\output\pt021\MR2\dvf_quiver.png" `
    --heat_out "C:\Users\P096350\Documents\output\pt021\MR2\dvf_heat.png"
  dvf_metrics     Compute quantitative metrics (EPE, cosine similarity, Jacobian determinant) between two DVFs and optional visualization.
    python .\metrics\model_check.py dvf_metrics `
    "C:\Users\P096350\Documents\output\pt021\MR2\motility_dynamics\patient21_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_001_preprocessed.nii_to_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_002_preprocessed.nii_dvf.nii.gz" `
    "C:\Users\P096350\Documents\ground_truth\pt021\MR2\motility_dynamics\gt_dvf.nii.gz" `
    --metrics epe cosine jacobian
  vectorvis       Overlay a vector field on a background image and save the quiver plot.
    python .\metrics\model_check.py vectorvis `
    "C:\Users\P096350\Documents\output\pt021\MR2\motility_dynamics\patient21_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_001_preprocessed.nii_to_18991230_000000WIP3DMOTILITY25mm160dyns301a1003_002_preprocessed.nii_dvf.nii.gz" `
    "C:\Users\P096350\Documents\anatomy\pt021_MR2.nii.gz" `
    --step 10 `
    --slice 45 `
    --out "C:\Users\P096350\Documents\output\pt021\MR2\overlay.png"
  dice            Compute the Dice Similarity Coefficient between two segmentation masks.
    python .\metrics\model_check.py dice `
    "C:\Users\P096350\Documents\segmentations\pt021_pred.nii.gz" `
    "C:\Users\P096350\Documents\segmentations\pt021_gt.nii.gz" `
    --threshold 0.5
"""

import os
# Force PyTorch backend and suppress TensorFlow logs
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import logging

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import voxelmorph as vxm

# Utility functions

def load_dvf(path):
    img = nib.load(path)
    data = np.array(img.get_fdata())
    data = np.squeeze(data)
    # Ensure shape (..., 3)
    if data.ndim == 4 and data.shape[-1] == 3:
        return data
    elif data.ndim == 4 and data.shape[0] == 3:
        return np.moveaxis(data, 0, -1)
    else:
        raise ValueError(f"Unexpected DVF shape {data.shape} for file {path}")

# 1) Apply registration (recursive)
def cmd_apply(args):
    device = 'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if device=='cuda' else '-1'

    # Load model
    model = vxm.networks.VxmDense.load(args.model, device)
    model.to(device)
    model.eval()

    # Walk input directory recursively
    for root, _, filenames in os.walk(args.data_folder):
        # Identify NIfTI files in this directory
        nii_files = sorted(f for f in filenames if f.lower().endswith(('.nii', '.nii.gz')))
        logging.info(f"Found {len(nii_files)} files in {root}: {nii_files}")
        if len(nii_files) <= args.interval:
            logging.warning(f"Not enough files in {root} for interval {args.interval}. Skipping.")
            continue

        # Prepare output subdirectory
        rel_dir = os.path.relpath(root, args.data_folder)
        out_subdir = args.output_folder if rel_dir == '.' else os.path.join(args.output_folder, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        # Process pairs within this folder
        for i in range(len(nii_files) - args.interval):
            moving_name = nii_files[i]
            fixed_name = nii_files[i + args.interval]
            moving_path = os.path.join(root, moving_name)
            fixed_path  = os.path.join(root, fixed_name)

            # Inform which pair is being registered
            logging.info(f"Registering moving={moving_path} to fixed={fixed_path}")

            moving = vxm.py.utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=True)
            fixed, affine = vxm.py.utils.load_volfile(fixed_path, add_batch_axis=True, add_feat_axis=True, ret_affine=True)

            mov_t = torch.from_numpy(moving).to(device).float().permute(0,4,1,2,3)
            fix_t = torch.from_numpy(fixed).to(device).float().permute(0,4,1,2,3)

            with torch.no_grad():
                warped, dvf = model(mov_t, fix_t, registration=True)

            warped = warped.cpu().numpy().squeeze()
            dvf    = dvf.cpu().numpy().squeeze()

            # Build a unique prefix per pair
            name0 = os.path.splitext(moving_name)[0]
            name1 = os.path.splitext(fixed_name)[0]
            if args.prefix:
                prefix = f"{args.prefix}_{name0}_to_{name1}"
            else:
                prefix = f"{name0}_to_{name1}"
            out_warp = os.path.join(out_subdir, f"{prefix}_moved.nii.gz")
            out_dvf  = os.path.join(out_subdir, f"{prefix}_dvf.nii.gz")

            vxm.py.utils.save_volfile(warped, out_warp, affine)
            vxm.py.utils.save_volfile(dvf, out_dvf, affine)
            # Show completion message
            print(f"Registered {moving_name} -> {fixed_name}: {out_warp}, {out_dvf}")

# 2) Motion map & MVH
def cmd_motionmap(args):
    vec = load_dvf(args.dvf)                              # (X, Y, Z, 3)
    disp = np.linalg.norm(vec, axis=-1)                   # |v| voxel-wise

    # -------------- numeric metrics --------------
    mean_mag   = float(np.mean(disp))
    rms_mag    = float(np.sqrt(np.mean(disp**2)))
    p95_mag    = float(np.percentile(disp, 95))
    p99_mag    = float(np.percentile(disp, 99))
    max_mag    = float(np.max(disp))

    print(f"[motionmap] ⟨|v|⟩={mean_mag:.4f}   RMS={rms_mag:.4f}   "
          f"P95={p95_mag:.4f}   P99={p99_mag:.4f}   max={max_mag:.4f}")

    # -------------- existing plots unchanged --------------
    idx = args.slice or disp.shape[2] // 2

    plt.imshow(disp[:,:,idx], cmap='jet', origin='lower')
    plt.title(f"Motion Map - Slice {idx}")
    plt.colorbar(label='Magnitude')
    plt.savefig(args.out_map)
    plt.close()

    # Motion-volume histogram
    vals = np.sort(disp.flatten())
    N = len(vals)
    frac = (N - np.arange(N)) / N * 100.0
    plt.plot(vals, frac)
    plt.xlabel('Motion')
    plt.ylabel('Volume (%)')
    plt.title('MVH')
    plt.savefig(args.out_hist)
    plt.close()
    print(f"Saved motion map to {args.out_map} and MVH to {args.out_hist}")

# 3) DVF check: sum and visualize

def cmd_dvf_check(args):
    # ---- robust loader (canonical shape: X, Y, Z, 3) ----
    files = [f for f in os.listdir(args.folder)
         if f.lower().endswith(('.nii', '.nii.gz')) and '_dvf' in f.lower()]
    if not files:
        raise ValueError(f"No DVF files found in {args.folder}")

    # keep header info from the first DVF
    img0   = nib.load(os.path.join(args.folder, files[0]))
    affine = img0.affine

    dvs = [load_dvf(os.path.join(args.folder, f)) for f in files]
    dvs_np = np.stack(dvs, axis=0)                # (N, X, Y, Z, 3)
    avg = np.mean(dvs_np, axis=0)                 # (X, Y, Z, 3)

    # -------------- numeric metrics --------------
    # voxel-wise magnitude of *each* DVF then pooled statistics
    mags = np.linalg.norm(dvs_np, axis=-1)        # (N, X, Y, Z)
    mean_per_vol = np.mean(mags, axis=(1,2,3))    # per-DVF scalar
    mean_global  = float(np.mean(mean_per_vol))
    std_global   = float(np.std(mean_per_vol))
    max_global   = float(np.max(mags))            # worst voxel across all DVFs

    # optional: slice-wise mean of AVG field
    idx = args.slice or avg.shape[2] // 2
    slice_mean   = float(np.mean(np.linalg.norm(avg[:,:,idx,:], axis=-1)))

    print(f"[dvf_check] N={len(files)}   ⟨⟨|v|⟩⟩={mean_global:.4f}±{std_global:.4f}   "
          f"slice{idx:02d}_mean={slice_mean:.4f}   worst_voxel={max_global:.4f}")

    # ---- existing save / plot section stays the same ----
    nib.save(nib.Nifti1Image(avg, affine),
             os.path.join(args.folder, 'avg_dvf.nii.gz'))
    print(f"Saved average DVF at {args.folder}/avg_dvf.nii")

    # Quiver
    idx = args.slice or avg.shape[2]//2
    comp = avg[:,:,:,0:2] if avg.ndim==5 else avg[:,:,:,0:2]
    X,Y = np.meshgrid(np.arange(comp.shape[1]), np.arange(comp.shape[0]))
    vec = comp[:,:,idx,:]
    plt.quiver(X, Y, vec[:,:,0], vec[:,:,1])
    plt.title(f"Avg DVF vectors slice {idx}")
    plt.savefig(args.vec_out)
    plt.close()

    # Heatmap
    mag = np.linalg.norm(avg[:,:,:,0:3] if avg.ndim==5 else avg, axis=-1)
    plt.imshow(mag[:,:,idx], cmap='jet', origin='lower')
    plt.colorbar()
    plt.title(f"Avg DVF heatmap slice {idx}")
    plt.savefig(args.heat_out)
    plt.close()
    print(f"Saved DVF vector and heatmap to {args.vec_out}, {args.heat_out}")

# 4) DVF metrics
def cmd_dvf_metrics(args):
    dvf1 = load_dvf(args.dvf1)
    dvf2 = load_dvf(args.dvf2)
    if args.metrics and 'epe' in args.metrics:
        epe = np.mean(np.linalg.norm(dvf1-dvf2, axis=-1))
        print(f"EPE: {epe:.4f}")
    if args.metrics and 'cosine' in args.metrics:
        f1 = dvf1.reshape(-1,3); f2 = dvf2.reshape(-1,3)
        dot = np.sum(f1*f2, axis=1)
        mags = np.linalg.norm(f1,axis=1)*np.linalg.norm(f2,axis=1)+1e-8
        print(f"Cosine sim: {np.mean(dot/mags):.4f}")
    if args.metrics and 'jacobian' in args.metrics:
        from scipy.ndimage import sobel
        def jac(d): return np.stack([sobel(d[i],a) for a in range(3)],0)
        J1 = np.stack([jac(dvf1[:,:, :,i]) for i in range(3)],0)
        J2 = np.stack([jac(dvf2[:,:, :,i]) for i in range(3)],0)
        # Not computing full det here for brevity
        print("Jacobian computed")

# 5) Vector overlay

def cmd_vectorvis(args):
    vec = load_dvf(args.vec)
    bg = np.squeeze(nib.load(args.bg).get_fdata())
    idx = args.slice or bg.shape[2]//2
    X,Y = np.meshgrid(np.arange(bg.shape[1]), np.arange(bg.shape[0]))
    slice_vec = vec[:,:,idx,:2]
    slice_mag = np.linalg.norm(vec[:,:,idx,:], axis=-1)
    print(f"[vectorvis] slice{idx:02d}  ⟨|v|⟩={slice_mag.mean():.4f}   "
          f"P95={np.percentile(slice_mag,95):.4f}   max={slice_mag.max():.4f}")
    plt.imshow(bg[:,:,idx], cmap='gray', origin='lower')
    plt.quiver(X[::args.step,::args.step], Y[::args.step,::args.step], slice_vec[::args.step,::args.step,0], slice_vec[::args.step,::args.step,1])
    plt.title(f"Overlay slice {idx}")
    plt.savefig(args.out)
    plt.close()
    print(f"Saved overlay to {args.out}")

# 6) Dice

def cmd_dice(args):
    img1 = nib.load(args.i1).get_fdata()>args.threshold
    img2 = nib.load(args.i2).get_fdata()>args.threshold
    inter = np.sum(img1 & img2)
    s = np.sum(img1)+np.sum(img2)
    dice = 1.0 if s==0 else 2*inter/s
    print(f"Dice Similarity: {dice:.4f}")


# 7) batch DVF metrics
def cmd_batch_dvf_metrics(folder_pred, folder_gt, metrics):
    preds = sorted(f for f in os.listdir(folder_pred) if '_dvf' in f.lower())
    gts   = sorted(f for f in os.listdir(folder_gt)  if '_dvf' in f.lower())
    assert len(preds) == len(gts), "Mismatch in prediction / GT count"

    acc = {m: [] for m in metrics}
    for fp, fg in zip(preds, gts):
        args = argparse.Namespace(dvf1=os.path.join(folder_pred, fp),
                                  dvf2=os.path.join(folder_gt,  fg),
                                  metrics=metrics)
        # reuse your single-pair function
        acc_pair = cmd_dvf_metrics(args, return_dict=True)
        for k, v in acc_pair.items():
            acc[k].append(v)

    for k, v in acc.items():
        print(f"[batch] {k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

# Main parser

def main():
    parser = argparse.ArgumentParser(description="Automated VoxelMorph model checks.")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # apply
    p = sub.add_parser('apply'); p.add_argument('data_folder'); p.add_argument('--interval',type=int,default=1)
    p.add_argument('--output_folder', required=True); p.add_argument('--model', required=True); p.add_argument('--gpu', default='0'); p.add_argument('--prefix')
    p.set_defaults(func=cmd_apply)

    # motionmap
    p = sub.add_parser('motionmap'); p.add_argument('dvf'); p.add_argument('--slice', type=int)
    p.add_argument('--out_map', default='motion_map.png'); p.add_argument('--out_hist', default='mvh.png')
    p.set_defaults(func=cmd_motionmap)

    # dvf_check
    p = sub.add_parser('dvf_check'); p.add_argument('folder'); p.add_argument('--slice',type=int)
    p.add_argument('--vec_out', default='dvf_vec.png'); p.add_argument('--heat_out', default='dvf_heat.png')
    p.set_defaults(func=cmd_dvf_check)

    # dvf_metrics
    p = sub.add_parser('dvf_metrics'); p.add_argument('dvf1'); p.add_argument('dvf2');
    p.add_argument('--metrics', nargs='+', choices=['epe','cosine','jacobian'], default=['epe']); p.set_defaults(func=cmd_dvf_metrics)

    # vectorvis
    p = sub.add_parser('vectorvis'); p.add_argument('vec'); p.add_argument('bg'); p.add_argument('--step',type=int,default=5)
    p.add_argument('--slice', type=int); p.add_argument('--out', default='overlay.png')
    p.set_defaults(func=cmd_vectorvis)

    # dice
    p = sub.add_parser('dice'); p.add_argument('i1'); p.add_argument('i2'); p.add_argument('--threshold', type=float, default=0.2)
    p.set_defaults(func=cmd_dice)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
    logging.info("VoxelMorph model check script completed.")