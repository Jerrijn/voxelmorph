import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from numpy.linalg import det
import argparse

# === Barten et al. (2024) thresholds ===
JAC0_THRESHOLD = 4.5   # percent
JAC2_THRESHOLD = 5.0   # percent
MU_HE_THRESHOLD = 4.0  # mean HE

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize_dvf(dvf):
    dvf = np.squeeze(dvf)
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)
    elif dvf.ndim == 5 and dvf.shape[0] == 1:
        dvf = dvf[0]
    dvf = dvf[:, :, 7:57, :]
    return dvf

def compute_jacobian_det(dvf):
    # dvf shape: (X, Y, Z, 3)
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
                # Construct 3x3 spatial Jacobian matrix at voxel (i,j,k)
                J = np.array([[grad[0][i,j,k,d], grad[1][i,j,k,d], grad[2][i,j,k,d]] for d in range(3)])
                jacobian[i,j,k] = det(J)
    return jacobian

def compute_harmonic_energy(dvf):
    # Harmonic energy: mean of sum of squared spatial gradients (over all components/voxels)
    grad = np.gradient(dvf, axis=(0, 1, 2))
    he = sum((g ** 2).sum() for g in grad)
    return he / np.prod(dvf.shape[:3])

def compute_motion_magnitude(dvf):
    return np.linalg.norm(dvf, axis=-1)

def main(root_folder, output_folder):
    # Edit these to your subject/session names if needed!
    patients = ['pt001', 'pt002', 'pt011','pt022']
    sessions = ['MR1', 'MR2', 'MR3','MR4']

    results = []

    for pid in patients:
        print(f"[INFO] Processing {pid}")
        for sess in sessions:
            warp_dir = os.path.join(root_folder, pid, sess, "motility_dynamics", "warp")
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

                    # --- QA metrics per Barten et al. (2024) ---
                    num_vox = np.prod(jac.shape)
                    jac0_pct = 100.0 * np.sum(jac < 0) / num_vox  # % JAC < 0
                    jac2_pct = 100.0 * np.sum(jac > 2) / num_vox  # % JAC > 2
                    mu_he = he  # mean harmonic energy

                    # QA cutoff logic
                    QA_JAC0 = jac0_pct <= JAC0_THRESHOLD
                    QA_JAC2 = jac2_pct <= JAC2_THRESHOLD
                    QA_HE = mu_he <= MU_HE_THRESHOLD
                    QA_Passed = QA_JAC0 and QA_JAC2 and QA_HE

                    # --- Extra stats for exploration ---
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
                    jac_max = np.max(jac)

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
                        "Jacobian Max": jac_max,
                        "JAC0%": jac0_pct,
                        "JAC2%": jac2_pct,
                        "μHE": mu_he,
                        "QA_JAC0": QA_JAC0,
                        "QA_JAC2": QA_JAC2,
                        "QA_HE": QA_HE,
                        "QA_Passed": QA_Passed
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

    # (Optional) If running in notebook/ace_tools
    try:
        import ace_tools as tools; tools.display_dataframe_to_user(name="DVF QA Statistics", dataframe=df)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True, help="Path to folder containing pt002 and pt022 folders")
    parser.add_argument("--output_folder", required=True, help="Path to save the final CSV")
    args = parser.parse_args()

    print("[START] Cine-MRI QA & Motion Statistics")
    main(args.root_folder, args.output_folder)
    print("[DONE]")
