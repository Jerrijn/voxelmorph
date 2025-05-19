import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.linalg import norm

def preprocess_dvf(dvf, target_shape):
    # Squeeze singleton trailing dim if exists
    if dvf.ndim == 5 and dvf.shape[-1] == 1:
        print("Squeezing singleton trailing dimension...")
        dvf = np.squeeze(dvf, axis=-1)

    # Pad depth (z-dimension) if needed
    if dvf.shape[3] < target_shape[3]:
        pad_before = 7
        pad_after = target_shape[3] - dvf.shape[3] - pad_before
        print(f"Padding depth dimension: before={pad_before}, after={pad_after}")
        dvf = np.pad(dvf, pad_width=((0, 0), (0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
    
    return dvf

def load_dvf(path):
    print(f"Loading DVF: {path}")
    dvf = nib.load(path).get_fdata()
    if dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    elif dvf.shape[0] == 3:
        return dvf
    else:
        raise ValueError("DVF must be in shape (3, H, W, D) or (H, W, D, 3)")


def endpoint_error(dvf1, dvf2):
    print("Computing Endpoint Error (EPE)...")

    return np.mean(norm(dvf1 - dvf2, axis=0))


def cosine_similarity(dvf1, dvf2):
    print("Computing Cosine Similarity...")
    flat1 = dvf1.reshape(3, -1)
    flat2 = dvf2.reshape(3, -1)
    dot_product = np.sum(flat1 * flat2, axis=0)
    norms = norm(flat1, axis=0) * norm(flat2, axis=0)
    cosine = dot_product / (norms + 1e-8)
    return np.mean(cosine)


def jacobian_determinant(dvf):
    print("Computing Jacobian Determinant...")
    from scipy.ndimage import sobel
    J = np.stack([np.stack([sobel(dvf[d], axis=a) for a in range(3)], axis=0) for d in range(3)], axis=0)
    detJ = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) -
            J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) +
            J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
    return detJ


def visualize_quiver_slice(dvf, title, output_path, axis=3, slice_idx=None):
    print(f"Generating quiver plot: {output_path}")
    
    if slice_idx is None:
        slice_idx = dvf.shape[axis] // 2

    # Select the slice along the correct axis
    if axis == 1:  # Sagittal
        u = dvf[0, slice_idx, :, :]
        v = dvf[1, slice_idx, :, :]
    elif axis == 2:  # Coronal
        u = dvf[0, :, slice_idx, :]
        v = dvf[1, :, slice_idx, :]
    elif axis == 3:  # Axial
        u = dvf[0, :, :, slice_idx]
        v = dvf[1, :, :, slice_idx]
    else:
        raise ValueError("Axis must be 1 (sagittal), 2 (coronal), or 3 (axial)")

    plt.figure(figsize=(6, 6))
    plt.quiver(u, v)
    plt.title(f"{title} (axis={axis}, slice={slice_idx})")
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()



def save_dvf(dvf, reference_path, output_path):
    print(f"Saving DVF difference to: {output_path}")
    ref_img = nib.load(reference_path)
    dvf_out = np.moveaxis(dvf, 0, -1)
    nib.save(nib.Nifti1Image(dvf_out, ref_img.affine, ref_img.header), output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dvf1', required=True, help='DVF 1 (e.g. B-spline)')
    parser.add_argument('--dvf2', required=True, help='DVF 2 (e.g. VoxelMorph)')
    parser.add_argument('--metrics', nargs='+', choices=['epe', 'cosine', 'jacobian'], default=['epe'])
    parser.add_argument('--visualize', type=str, default='false')
    parser.add_argument('--output_dir', default='./dvf_comparison_output')
    parser.add_argument('--save_diff_dvf', type=str, help='Path to save the DVF difference (DVF1 - DVF2)')
    args = parser.parse_args()

    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)

    dvf1 = load_dvf(args.dvf1)
    dvf2 = load_dvf(args.dvf2)

    target_shape = dvf1.shape  # Assume dvf1 has the correct shape
    dvf2 = preprocess_dvf(dvf2, target_shape)

    if 'epe' in args.metrics:
        epe = endpoint_error(dvf1, dvf2)
        print(f"Endpoint Error (EPE): {epe:.4f}")

    if 'cosine' in args.metrics:
        cos_sim = cosine_similarity(dvf1, dvf2)
        print(f"Cosine Similarity: {cos_sim:.4f}")

    if 'jacobian' in args.metrics:
        jac1 = jacobian_determinant(dvf1)
        jac2 = jacobian_determinant(dvf2)
        print(f"Jacobian Determinant (DVF1): min={jac1.min():.4f}, max={jac1.max():.4f}, mean={jac1.mean():.4f}")
        print(f"Jacobian Determinant (DVF2): min={jac2.min():.4f}, max={jac2.max():.4f}, mean={jac2.mean():.4f}")

    if args.visualize.lower() == 'true':
        visualize_quiver_slice(dvf1, "DVF1 Quiver", os.path.join(args.output_dir, "dvf1_quiver.png"), axis=3)
        visualize_quiver_slice(dvf2, "DVF2 Quiver", os.path.join(args.output_dir, "dvf2_quiver.png"), axis=3)

    if args.save_diff_dvf:
        print("Calculating DVF difference (DVF1 - DVF2)...")
        diff_dvf = dvf1 - dvf2
        save_dvf(diff_dvf, args.dvf1, args.save_diff_dvf)
        print(f"Saved difference DVF to {args.save_diff_dvf}")
        
        if args.visualize.lower() == 'true':
            print("Visualizing DVF difference (quiver plot)...")
            visualize_quiver_slice(
                diff_dvf,
                "DVF Difference Quiver",
                os.path.join(args.output_dir, "dvf_diff_quiver.png"),
                axis=3
            )



if __name__ == '__main__':
    main()
