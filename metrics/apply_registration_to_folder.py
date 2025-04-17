import os
import sys
import argparse
import torch
import nibabel as nib
import importlib.util

# Dynamically import networks.py
networks_path = r"C:\Users\P096350\OneDrive - Amsterdam UMC\Documenten\voxelmorph\voxelmorph\torch\networks.py"
def import_from_path(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

networks = import_from_path("networks", networks_path)

# Patch default_unet_features for standalone use
import types
networks.default_unet_features = [32, 32, 32, 32, 32, 32]


def run_registration(moving_path, fixed_path, model, device):
    import voxelmorph.py.utils as vxm_utils

    # Load volumes
    moving = vxm_utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=True)
    fixed, fixed_affine = vxm_utils.load_volfile(fixed_path, add_batch_axis=True, add_feat_axis=True, ret_affine=True)

    # Prepare tensors
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

    # Run model
    moved, warp = model(input_moving, input_fixed, registration=True)

    # Convert to numpy
    moved = moved.detach().cpu().numpy().squeeze()
    warp = warp.detach().cpu().numpy().squeeze()

    return moved, warp, fixed_affine


def main(args):
    # Device setup
    if args.gpu != '-1' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    # Load model
    model = networks.VxmDense.load(args.model, device)
    model.to(device)
    model.eval()

    # Get sorted list of NIfTI files
    all_files = sorted([f for f in os.listdir(args.data_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    os.makedirs(args.output_folder, exist_ok=True)

    for i in range(0, len(all_files) - args.interval):
        moving_file = all_files[i]
        fixed_file = all_files[i + args.interval]

        moving_path = os.path.join(args.data_folder, moving_file)
        fixed_path = os.path.join(args.data_folder, fixed_file)

        moved_img, warp_field, affine = run_registration(moving_path, fixed_path, model, device)

        # Output names
        prefix = args.prefix or f"{os.path.splitext(moving_file)[0]}_to_{os.path.splitext(fixed_file)[0]}"
        moved_out = os.path.join(args.output_folder, f"{prefix}_moved.nii.gz")
        warp_out = os.path.join(args.output_folder, f"{prefix}_warp.nii.gz")

        # Save
        import voxelmorph.py.utils as vxm_utils
        vxm_utils.save_volfile(moved_img, moved_out, affine)
        vxm_utils.save_volfile(warp_field, warp_out, affine)

        print(f"[INFO] Registered {moving_file} -> {fixed_file} | Output: {moved_out}, {warp_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', required=True, help='Folder with .nii/.nii.gz files')
    parser.add_argument('--interval', type=int, default=1, help='Index gap between moving and fixed image')
    parser.add_argument('--output-folder', required=True, help='Where to save registered results')
    parser.add_argument('--model', required=True, help='Path to trained VoxelMorph .pt model')
    parser.add_argument('--gpu', default='0', help='GPU ID to use, set to -1 for CPU')
    parser.add_argument('--prefix', help='Prefix for saved output files (default: auto-named from input pair)')

    args = parser.parse_args()
    main(args)
