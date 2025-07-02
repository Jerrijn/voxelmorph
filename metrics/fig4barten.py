#!/usr/bin/env python

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# === Utility ===
def normalize_image(img):
    img = img.astype(np.float32)
    max_val = np.max(img)
    return img / max_val if max_val > 0 else img

def create_jacobian_colormap():
    colors = [(0.0, 0.0, 0.5), (0.0, 0.5, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("jac_cmap", colors, N=256)
    return cmap

def create_he_colormap():
    colors = [(0.0, 0.0, 0.5), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("he_cmap", colors, N=256)
    return cmap

# === Core Computations ===
def compute_jacobian_determinant(dvf):
    dx = np.gradient(dvf[..., 0], axis=(0, 1, 2))
    dy = np.gradient(dvf[..., 1], axis=(0, 1, 2))
    dz = np.gradient(dvf[..., 2], axis=(0, 1, 2))
    J = np.stack([np.stack(dx, -1), np.stack(dy, -1), np.stack(dz, -1)], axis=-2)
    return np.linalg.det(J)

def compute_harmonic_energy(dvf):
    he = np.zeros(dvf.shape[:3])
    for i in range(3):
        for axis in range(3):
            he += sobel(dvf[..., i], axis=axis) ** 2
    return np.sqrt(he)

# === Visualization ===
def save_overlay_image(base_img, overlay_img, colormap, alpha, title, label, vmin, vmax, output_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(normalize_image(base_img), cmap="gray")
    im = ax.imshow(overlay_img, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✅] Saved: {output_path}")

def save_dir_overlay_image(mri_ref_slice, mri_def_slice, output_path):
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)

    # Overlay image (RGB: green = ref, red = def)
    ax_img = fig.add_subplot(gs[0])
    rgb_overlay = np.stack([
        normalize_image(mri_def_slice),
        normalize_image(mri_ref_slice),
        np.zeros_like(mri_ref_slice)
    ], axis=-1)

    ax_img.imshow(rgb_overlay)
    ax_img.set_title("DIR REGISTRATION")
    ax_img.axis("off")

    # Custom vertical bar
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(0, 1)
    ax_bar.axis("off")
    ax_bar.add_patch(patches.Rectangle((0, 0.5), 1, 0.5, facecolor='green'))
    ax_bar.add_patch(patches.Rectangle((0, 0), 1, 0.5, facecolor='magenta'))
    ax_bar.text(1.1, 0.75, "REFERENCE\nDYNAMIC", va='center', fontsize=8)
    ax_bar.text(1.1, 0.25, "DEFORMED\nIMAGE", va='center', fontsize=8)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✅] Saved: {output_path}")

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to reference image")
    parser.add_argument("--def-img", required=True, help="Path to deformed image")
    parser.add_argument("--dvf", required=True, help="Path to DVF file")
    parser.add_argument("--slice", type=int, required=True, help="Slice index")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ref = nib.load(args.ref).get_fdata()
    deform = nib.load(args.def_img).get_fdata()
    dvf = np.squeeze(nib.load(args.dvf).get_fdata())
    if dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)

    print("[INFO] Computing Jacobian and HE...")
    jac = compute_jacobian_determinant(dvf)
    he = compute_harmonic_energy(dvf)

    jac0 = np.mean(jac < 0)
    jac2 = np.mean(jac > 2)
    mu_he = np.mean(he)
    print(f"[STAT] Jacobian < 0: {jac0:.3f}, Jacobian > 2: {jac2:.3f}, Mean HE: {mu_he:.2f}")

    slice_idx = args.slice
    base = ref[:, :, slice_idx]
    deformed = deform[:, :, slice_idx]
    jac_slice = jac[:, :, slice_idx]
    he_slice = he[:, :, slice_idx]

    # Save visualizations
    save_dir_overlay_image(base, deformed, os.path.join(args.out_dir, "dir_overlay.png"))

    save_overlay_image(
        base, jac_slice,
        create_jacobian_colormap(), alpha=0.4,
        title="JAC", label="Jacobian Determinant",
        vmin=-2, vmax=4, output_path=os.path.join(args.out_dir, "jac_overlay.png")
    )

    save_overlay_image(
        base, he_slice,
        create_he_colormap(), alpha=0.4,
        title="HE", label="Harmonic Energy",
        vmin=0, vmax=7, output_path=os.path.join(args.out_dir, "he_overlay.png")
    )

if __name__ == "__main__":
    main()
