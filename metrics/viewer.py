#!/usr/bin/env python

import argparse
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio

def save_slice_as_png(data, slice_idx, output_file):
    """Save the specified slice as a PNG image."""
    print(f"[INFO] Extracting slice index {slice_idx} from shape {data.shape}")
    if slice_idx < 0 or slice_idx >= data.shape[2]:
        print(f"[ERROR] Slice index {slice_idx} out of bounds (0–{data.shape[2]-1})")
        return
    slice_img = data[:, :, slice_idx]
    plt.imsave(output_file, slice_img, cmap='gray')
    print(f"[INFO] Saved slice {slice_idx} to '{output_file}'")

def save_video(data, start_slice, end_slice, output_file, fps=2):
    """Create a scroll video or GIF through slices."""
    print(f"[INFO] Creating video from slices {start_slice} to {end_slice}")
    if start_slice < 0 or end_slice >= data.shape[2]:
        print(f"[ERROR] Slice range {start_slice}-{end_slice} out of bounds (0–{data.shape[2]-1})")
        return

    frames = []
    for i in range(start_slice, end_slice + 1):
        print(f"[INFO] Processing slice {i}")
        frame = data[:, :, i]
        norm_frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-5)
        rgb_frame = (plt.cm.gray(norm_frame)[:, :, :3] * 255).astype(np.uint8)
        frames.append(rgb_frame)

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".mp4":
        print(f"[INFO] Saving MP4 to '{output_file}'")
        imageio.mimsave(output_file, frames, fps=fps, format='FFMPEG')
    elif ext == ".gif":
        print(f"[INFO] Saving GIF to '{output_file}'")
        imageio.mimsave(output_file, frames, fps=fps)
    else:
        print(f"[ERROR] Unsupported file extension '{ext}'. Use .mp4 or .gif")
        return

    print(f"[INFO] Video saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Extract a slice or scroll video from a Cine-MRI file.")
    parser.add_argument('--input', required=True, help='Path to .nii or .nii.gz file')
    parser.add_argument('--slice', type=int, help='Index of the slice to extract')
    parser.add_argument('--video', nargs=2, type=int, metavar=('START', 'END'),
                        help='Range of slices to include in video')
    parser.add_argument('--output', required=True, help='Output file path (e.g. slice.png or video.mp4)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found.")
        return

    print(f"[INFO] Loading NIfTI file from '{args.input}'...")
    nii = nib.load(args.input)
    data = nii.get_fdata()
    print(f"[INFO] Loaded image shape: {data.shape} (dtype={data.dtype})")

    # If 4D (X,Y,Z,T), assume first timepoint
    if data.ndim == 4:
        print(f"[INFO] Detected 4D image. Using first timepoint (index 0).")
        data = data[..., 0]

    if args.slice is not None:
        print(f"[INFO] Mode: Slice Extraction")
        save_slice_as_png(data, args.slice, args.output)
    elif args.video is not None:
        print(f"[INFO] Mode: Video Generation")
        start, end = args.video
        save_video(data, start, end, args.output)
    else:
        print("[ERROR] You must specify either --slice or --video START END.")

if __name__ == '__main__':
    main()
