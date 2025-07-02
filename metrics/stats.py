import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from numpy.linalg import det
import argparse

harmonic_energy_threshold = 150.0

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize_dvf(dvf):
    dvf = np.squeeze(dvf)
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)
    elif dvf.ndim == 5 and dvf.shape[0] == 1:
        dvf = dvf[0]
    return dvf

def compute_jacobian_det(dvf):
    grid = np.stack(np.meshgrid(
        np.arange(dvf.shape[0]),
        np.arange(dvf.shape[1]),
        np.arange(dvf.shape[2]),
        indexing='ij'), axis=-1)
    warped = grid + dvf
    grad = np.gradient(warped, 1.0, axis=(0, 1, 2))
    jacobian = np.zeros(dvf.shape[:3])
    for i in range(dvf.shape[0]):
        for j in range(dvf.shape[1]):
            for k in range(dvf.shape[2]):
                J = np.array([[grad[0][i,j,k,d], grad[1][i,j,k,d], grad[2][i,j,k,d]] for d in range(3)])
                jacobian[i,j,k] = det(J)
    return jacobian

def compute_harmonic_energy(dvf):
    grad = np.gradient(dvf, axis=(0, 1, 2))
    he = sum((g ** 2).sum() for g in grad)
    return he / np.prod(dvf.shape[:3])

def compute_motion_magnitude(dvf):
    return np.linalg.norm(dvf, axis=-1)

def main(root_folder, output_folder):
    patients = ['pt002', 'pt022']
    sessions = ['MR1', 'MR2', 'MR3']

    results = []

    for pid in patients:
        print(f"[INFO] Processing {pid}")
        for sess in sessions:
            warp_dir = os.path.join(root_folder, pid, sess, "warp")
            if not os.path.isdir(warp_dir):
                print(f"[WARNING] Warp folder not found: {warp_dir}")
                continue
            dvf_paths = sorted(glob(os.path.join(warp_dir, "*.nii*")))
            if not dvf_paths:
                continue
            print(f"[INFO] Found {len(dvf_paths)} DVFs in {pid}/{sess}")

            for idx, path in enumerate(dvf_paths):
                filename = os.path.basename(path)
                print(f"[DEBUG] Loading DVF: {filename}")
                try:
                    dvf = normalize_dvf(load_nifti(path))
                    jac = compute_jacobian_det(dvf)
                    he = compute_harmonic_energy(dvf)
                    motion = compute_motion_magnitude(dvf)

                    motion_flat = motion.flatten()
                    motion_mean = np.mean(motion_flat)
                    motion_std = np.std(motion_flat)
                    motion_max = np.max(motion_flat)
                    motion_min = np.min(motion_flat)
                    motion_10 = np.percentile(motion_flat, 10)
                    motion_50 = np.percentile(motion_flat, 50)
                    motion_90 = np.percentile(motion_flat, 90)

                    jac_mean = np.mean(jac)
                    jac_std = np.std(jac)
                    jac_min = np.min(jac)
                    jac_neg_perc = np.mean(jac <= 0)

                    QA_JAC0 = jac_min > 0
                    QA_JAC2 = jac_neg_perc <= 0.02
                    QA_HE = he <= harmonic_energy_threshold

                    result = {
                        "Patient": pid,
                        "Session": sess,
                        "File": filename,
                        "Motion Mean": motion_mean,
                        "Motion Std": motion_std,
                        "Motion Max": motion_max,
                        "Motion Min": motion_min,
                        "Motion P10": motion_10,
                        "Motion P50": motion_50,
                        "Motion P90": motion_90,
                        "Jacobian Mean": jac_mean,
                        "Jacobian Std": jac_std,
                        "Jacobian Min": jac_min,
                        "Jacobian % Negative": jac_neg_perc,
                        "Harmonic Energy": he,
                        "QA_JAC0": QA_JAC0,
                        "QA_JAC2": QA_JAC2,
                        "QA_HE": QA_HE,
                        "QA_Passed": QA_JAC0 and QA_JAC2 and QA_HE
                    }
                    results.append(result)

                    if idx == 0:
                        print("[FIRST DVF STATS]")
                        for k, v in result.items():
                            print(f"{k}: {v}")

                except Exception as e:
                    print(f"[ERROR] Failed to process {path}: {e}")

    os.makedirs(output_folder, exist_ok=True)
    df = pd.DataFrame(results)
    output_csv_path = os.path.join(output_folder, "dvf_statistics.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"[SAVED] Statistics CSV to {output_csv_path}")

    import ace_tools as tools; tools.display_dataframe_to_user(name="DVF QA Statistics", dataframe=df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True, help="Path to folder containing pt002 and pt022 folders")
    parser.add_argument("--output_folder", required=True, help="Path to save the final CSV")
    args = parser.parse_args()

    print("[START] Cine-MRI QA & Motion Statistics")
    main(args.root_folder, args.output_folder)
    print("[DONE]")
