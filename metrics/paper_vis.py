import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob


def load_nifti(path):
    return nib.load(path).get_fdata()


def normalize_dvf(dvf):
    dvf = np.squeeze(dvf)
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)
    elif dvf.ndim == 5 and dvf.shape[0] == 1:
        dvf = dvf[0]
    return dvf


def compute_motion_map(dvf_paths):
    dvfs = [normalize_dvf(load_nifti(p)) for p in sorted(dvf_paths)]
    dvfs = np.array(dvfs)
    return np.mean(np.linalg.norm(dvfs, axis=-1), axis=0)


def compute_sorted_motion_values(motion_map):
    motion_vals = motion_map.flatten()
    sorted_vals = np.sort(motion_vals)[::-1]
    cumulative_volume = np.linspace(0, 100, len(sorted_vals))
    return sorted_vals, cumulative_volume


def collect_selected_dvf_folders(root_folder, target_patients=("pt002", "pt022")):
    entries = []
    for patient_id in target_patients:
        p_folder = os.path.join(root_folder, patient_id)
        if not os.path.isdir(p_folder):
            print(f"[WARNING] Folder not found for {patient_id}, skipping.")
            continue
        for i in range(1, 4):
            dvf_folder = os.path.join(p_folder, f"MR{i}", "warp")
            if os.path.isdir(dvf_folder):
                dvf_files = sorted(glob(os.path.join(dvf_folder, "*.nii*")))
                if dvf_files:
                    entries.append((patient_id, f"MR{i}", dvf_files))
                else:
                    print(f"[WARNING] No DVF files in {dvf_folder}")
            else:
                print(f"[SKIP] Folder not found: {dvf_folder}")
    return entries


def plot_all_mvhs(entries, output_path):
    plt.figure(figsize=(10, 6))
    for patient_id, mr_session, dvf_files in entries:
        print(f"[INFO] Processing {patient_id} - {mr_session} with {len(dvf_files)} DVFs...")
        motion_map = compute_motion_map(dvf_files)
        sorted_vals, cumulative_volume = compute_sorted_motion_values(motion_map)
        label = f"{patient_id} - {mr_session}"
        plt.plot(sorted_vals, cumulative_volume, label=label)

    plt.xlabel("Motion Magnitude (mm/frame)")
    plt.ylabel("Volume (%)")
    plt.title("Motion-Volume Histogram (pt002 & pt022)")
    plt.grid(True)
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "all_motion_volume_histograms.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] Plot saved to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True, help="Path to folder containing pt002 and pt022 folders")
    parser.add_argument("--output_path", required=True, help="Path to save the final plot")
    args = parser.parse_args()

    print(f"[START] Searching for DVFs in: {args.root_folder}")
    all_entries = collect_selected_dvf_folders(args.root_folder, target_patients=("pt002", "pt022"))

    if not all_entries:
        print("[ERROR] No DVFs found for pt002 or pt022.")
    else:
        print(f"[INFO] Found {len(all_entries)} DVF sets.")
        plot_all_mvhs(all_entries, args.output_path)
