#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import sobel
from glob import glob
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.lines import Line2D

# Barten et al. (2024) QA thresholds
JAC0_THRESHOLD_BARTEN = 4.5     # % negative Jacobian
JAC2_THRESHOLD_BARTEN = 5.0     # % Jacobian > 2
HE_THRESHOLD_BARTEN   = 4.0     # mean harmonic energy

def crop64(img):
    """Crop axes of size 64 to [7:57] (middle 50). Accepts np.ndarray."""
    slices = []
    for axlen in img.shape:
        if axlen == 64:
            slices.append(slice(7, 57))
        else:
            slices.append(slice(0, axlen))
    return img[tuple(slices)]


def normalize_image(img):
    img = img.astype(np.float32)
    max_val = np.max(img)
    return img / max_val if max_val > 0 else img

def create_jacobian_colormap():
    colors = [(0.0, 0.0, 0.5), (0.0, 0.5, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    return LinearSegmentedColormap.from_list("jac_cmap", colors, N=256)

def create_he_colormap():
    colors = [(0.0, 0.0, 0.5), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    return LinearSegmentedColormap.from_list("he_cmap", colors, N=256)

def load_nifti(path):
    print(f"[INFO] Loading NIfTI file: {path}")
    img = nib.load(path).get_fdata()
    img = crop64(img)
    return img

def normalize_dvf(dvf):
    dvf = np.squeeze(dvf)
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        dvf = np.moveaxis(dvf, 0, -1)
    elif dvf.ndim == 5 and dvf.shape[0] == 1:
        dvf = dvf[0]
    return dvf

def compute_jacobian_determinant(dvf):
    """
    Computes the Jacobian determinant for a DVF in voxel units.
    dvf: [H, W, D, 3] - displacement field, last axis = (dx, dy, dz)
    Returns: [H, W, D] Jacobian determinant map
    """
    print(f"[INFO] Computing Jacobian determinant (correct transformation method)")
    # Build the coordinate grid
    shape = dvf.shape[:-1]
    grid = np.stack(np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    ), axis=-1).astype(dvf.dtype)  # [H, W, D, 3]
    mapping = grid + dvf  # [H, W, D, 3]

    # Compute gradients of mapping
    grad = np.empty(mapping.shape + (3,))  # [H, W, D, 3, 3]
    for i in range(3):  # output dim
        for j in range(3):  # input dim
            grad[..., i, j] = np.gradient(mapping[..., i], axis=j)
    jac = np.linalg.det(grad)
    return jac

def compute_harmonic_energy(dvf):
    print(f"[INFO] Computing Harmonic Energy")
    he = np.zeros(dvf.shape[:3], dtype=np.float32)
    for i in range(3):  # displacement components
        for j in range(3):  # spatial derivatives
            grad = np.gradient(dvf[..., i], axis=j)
            he += grad ** 2
    return he

def compute_motion_map(dvf_paths):
    print(f"[INFO] Computing mean motion map from {len(dvf_paths)} DVFs")
    dvfs = [normalize_dvf(load_nifti(p)) for p in sorted(dvf_paths)]
    dvfs = np.array(dvfs)
    return np.mean(np.linalg.norm(dvfs, axis=-1), axis=0)

def save_overlay_image(base_img, overlay_img, colormap, alpha, title, label, vmin, vmax, output_path):
    print(f"[INFO] Saving overlay image: {title} -> {output_path}")
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

def find_all_csvs_in_folder(folder):
    # Recursively find all .csv files in a folder and its subfolders
    return glob(os.path.join(folder, '**', '*.csv'), recursive=True)

def expand_csv_paths(qa_csv_list, model_names):
    # Returns a list of list: one list of CSVs per model
    expanded_csv_paths = []
    for item in qa_csv_list:
        if os.path.isdir(item):
            csvs = find_all_csvs_in_folder(item)
            expanded_csv_paths.append(csvs)
        else:
            expanded_csv_paths.append([item])
    if len(expanded_csv_paths) != len(model_names):
        raise ValueError(
            f"Number of CSV sets found ({len(expanded_csv_paths)}) does not match number of models ({len(model_names)})."
        )
    return expanded_csv_paths

def save_dir_overlay_image(mri_ref_slice, mri_def_slice, output_path):
    print(f"[INFO] Saving DIR overlay: {output_path}")
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax_img = fig.add_subplot(gs[0])
    rgb_overlay = np.stack([
        normalize_image(mri_def_slice),
        normalize_image(mri_ref_slice),
        np.zeros_like(mri_ref_slice)
    ], axis=-1)
    ax_img.imshow(rgb_overlay)
    ax_img.set_title("DIR REGISTRATION")
    ax_img.axis("off")
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

def compute_mvh(motion_map, n_points=500):
    print(f"[INFO] Computing MVH curve ({n_points} thresholds, ECDF method)")
    vals = motion_map.flatten()
    thresholds = np.linspace(vals.min(), vals.max(), n_points)
    ecdf = ECDF(vals)
    mvh = 100 * (1 - ecdf(thresholds))  # percent of volume above threshold
    return thresholds, mvh

def plot_patient_avg_mvhs_multi(
    mvh_data_dirs, model_names, output_path, patient_ids=None
):
    """
    Aggregates and plots MVHs. Colors = model, line styles = patient.
    Legend: model colors first, then patient line styles (black).
    """
    from collections import defaultdict

    assert len(mvh_data_dirs) == len(model_names), "Mismatch between folders and model names!"
    all_curves = defaultdict(list)  # (patient, model) → list of dfs

    # Gather data
    for mvh_dir, model in zip(mvh_data_dirs, model_names):
        mvh_csvs = sorted(glob(os.path.join(mvh_dir, "*_mvh.csv")))
        for mvh_csv in mvh_csvs:
            base = os.path.basename(mvh_csv)
            patient = base.split("_")[0]
            if patient_ids and len(patient_ids) > 0 and patient not in patient_ids:
                continue
            df = pd.read_csv(mvh_csv)
            all_curves[(patient, model)].append(df)

    if not all_curves:
        print("[ERROR] No matching MVH curves found.")
        return

    # Assign colors to models and linestyles to patients
    model_colors = {}
    color_palette = plt.get_cmap('tab10')
    for i, model in enumerate(model_names):
        model_colors[model] = color_palette(i % 10)
    # Find all unique patients (preserve sorted order for line styles)
    patients = sorted({k[0] for k in all_curves.keys()})
    line_styles = ['-', '--', ':', '-.']
    patient_to_style = {p: line_styles[i % len(line_styles)] for i, p in enumerate(patients)}

    plt.figure(figsize=(11, 7))

    # Plot: ordered by model, then patient
    for model in model_names:
        for patient in patients:
            dfs = all_curves.get((patient, model))
            if not dfs:
                continue
            if len(dfs) == 1:
                mean_vol = dfs[0]["volume_percent"].values
                threshold = dfs[0]["threshold"].values
            else:
                mat = np.stack([d["volume_percent"].values for d in dfs])
                mean_vol = mat.mean(axis=0)
                threshold = dfs[0]["threshold"].values
            # Cap to threshold <= 5
            cap_idx = threshold <= 5
            threshold = threshold[cap_idx]
            mean_vol = mean_vol[cap_idx]
            label = f"{patient} - {model}"
            plt.plot(
                threshold, mean_vol,
                label=label,
                color=model_colors[model],
                linestyle=patient_to_style[patient],
                linewidth=2,
            )

    plt.xlabel("Motion Magnitude (mm/frame)")
    plt.ylabel("Volume (%)")
    if patient_ids:
        ids_str = ", ".join(patient_ids)
        title = f"Aggregated MVH Curves for {ids_str}"
    else:
        title = "Aggregated MVH Curves (All Patients)"
    plt.title(title)
    plt.xlim(0, 5)
    plt.grid(True)

    # Build legend: model colors first, then patient line styles
    model_handles = [
        Line2D([0], [0], color=model_colors[m], lw=3, label=m)
        for m in model_names
    ]
    patient_handles = [
        Line2D([0], [0], color='black', linestyle=patient_to_style[p], lw=2, label=p)
        for p in patients
    ]
    plt.legend(handles=model_handles + patient_handles, ncol=2, loc='best', title="Models and Patients")

    os.makedirs(output_path, exist_ok=True)
    fname = (
        f"aggregated_patient_model_mvh_{'_'.join(patient_ids) if patient_ids else 'all'}.png"
    )
    save_path = os.path.join(output_path, fname)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVED] Aggregated patient/model MVHs to: {save_path}")

def plot_harmonic_energy_summaries(df, output_dir):
    import seaborn as sns

    # --- Build label "Model - Patient" ---
    if "Model" not in df.columns:
        raise ValueError("No 'Model' column found in DataFrame!")
    if "Patient" not in df.columns:
        raise ValueError("No 'Patient' column found in DataFrame!")

    df["Model_Patient"] = df["Model"].astype(str) + " - " + df["Patient"].astype(str)

    # Set order: by model, then patient (preserves logical grouping)
    model_order = list(df["Model"].unique())
    patient_order = list(df["Patient"].unique())
    order = []
    for m in model_order:
        for p in patient_order:
            label = f"{m} - {p}"
            if label in df["Model_Patient"].values:
                order.append(label)

    # --- Boxplot: Harmonic Energy (log scaled) per Model - Patient ---
    plt.figure(figsize=(2 + len(order)*0.7, 6))
    sns.boxplot(x="Model_Patient",
                y=pd.to_numeric(df["μHE"], errors='coerce'),
                data=df,
                order=order,
                palette="husl")
    plt.xticks(rotation=45, ha="right")
    plt.title("Harmonic Energy Distribution by Model and Patient")
    plt.ylabel("Harmonic Energy")
    plt.xlabel("Model - Patient")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "harmonic_energy_distribution_by_model_patient.png"))
    plt.close()

    # --- Scatter: Harmonic Energy vs. Mean Motion, colored by Model_Patient ---
    plt.figure(figsize=(8, 6))
    unique_ids = [x for x in order if x in df["Model_Patient"].unique()]
    color_map = dict(zip(unique_ids, sns.color_palette("husl", len(unique_ids))))
    for mpid in unique_ids:
        sub = df[df["Model_Patient"] == mpid]
        plt.scatter(pd.to_numeric(sub["μHE"], errors='coerce'),
                    pd.to_numeric(sub["Motion Mean"], errors='coerce'),
                    label=mpid, color=color_map[mpid], s=20, alpha = 0.5)
    plt.xlabel("Harmonic Energy")
    plt.ylabel("Mean Motion")
    plt.title("Correlation: Harmonic Energy vs Mean Motion (Model + Patient grouped)")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_vs_he_by_model_patient.png"))
    plt.close()
    stats_df = (
        df.groupby("Model_Patient")
          .agg(
              he_mean=("μHE", lambda x: pd.to_numeric(x, errors='coerce').mean()),
              he_std=("μHE", lambda x: pd.to_numeric(x, errors='coerce').std()),
              he_min=("μHE", lambda x: pd.to_numeric(x, errors='coerce').min()),
              he_max=("μHE", lambda x: pd.to_numeric(x, errors='coerce').max()),
              he_n=("μHE", lambda x: pd.to_numeric(x, errors='coerce').count()),
              motion_mean=("Motion Mean", lambda x: pd.to_numeric(x, errors='coerce').mean()),
              motion_std=("Motion Mean", lambda x: pd.to_numeric(x, errors='coerce').std()),
              motion_min=("Motion Mean", lambda x: pd.to_numeric(x, errors='coerce').min()),
              motion_max=("Motion Mean", lambda x: pd.to_numeric(x, errors='coerce').max()),
              motion_n=("Motion Mean", lambda x: pd.to_numeric(x, errors='coerce').count()),
          )
          .reset_index()
    )
    stats_path = os.path.join(output_dir, "harmonic_energy_per_model_patient_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[SAVED] Per-group stats to {stats_path}")



def summarize_dvf_statistics_from_df(df, output_path):
    # Full summary: per model & patient
    summary = (
        df.groupby(['Model', 'Patient'])
        .agg(
            Motion_Mean_mean=("Motion Mean", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))),
            Motion_Mean_std=("Motion Mean", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce"))),
            μHE_mean=("μHE", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))),
            μHE_std=("μHE", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce"))),
            JAC0_mean=("JAC0%", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce")) if "JAC0%" in df.columns else np.nan),
            JAC0_std=("JAC0%", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")) if "JAC0%" in df.columns else np.nan),
            JAC2_mean=("JAC2%", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce")) if "JAC2%" in df.columns else np.nan),
            JAC2_std=("JAC2%", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")) if "JAC2%" in df.columns else np.nan),
            QA_pass_rate=("QA_Passed", lambda x: np.mean(x.astype(bool))),
            n_sessions=("QA_Passed", "count"),
        )
        .reset_index()
    )
    summary["Motion_Mean"] = summary["Motion_Mean_mean"].round(3).astype(str) + " ± " + summary["Motion_Mean_std"].round(3).astype(str)
    summary["μHE"] = summary["μHE_mean"].round(3).astype(str) + " ± " + summary["μHE_std"].round(3).astype(str)
    summary["JAC0%"] = summary["JAC0_mean"].round(3).astype(str) + " ± " + summary["JAC0_std"].round(3).astype(str)
    summary["JAC2%"] = summary["JAC2_mean"].round(3).astype(str) + " ± " + summary["JAC2_std"].round(3).astype(str)
    summary["QA_pass_rate"] = (summary["QA_pass_rate"] * 100).round(1).astype(str) + "%"
    table = summary[["Model", "Patient", "Motion_Mean", "μHE", "JAC0%", "JAC2%", "QA_pass_rate", "n_sessions"]]

    # Save full table (per model & patient)
    csv_out = output_path if output_path.endswith(".csv") else os.path.join(output_path, "summary_table.csv")
    table.to_csv(csv_out, index=False)
    print(f"[SAVED] Summary table (per model & patient) to {csv_out}")

    # Summary per model
    model_summary = (
        df.groupby(['Model'])
        .agg(
            Motion_Mean_mean=("Motion Mean", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))),
            Motion_Mean_std=("Motion Mean", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce"))),
            μHE_mean=("μHE", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))),
            μHE_std=("μHE", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce"))),
            JAC0_mean=("JAC0%", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce")) if "JAC0%" in df.columns else np.nan),
            JAC0_std=("JAC0%", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")) if "JAC0%" in df.columns else np.nan),
            JAC2_mean=("JAC2%", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce")) if "JAC2%" in df.columns else np.nan),
            JAC2_std=("JAC2%", lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")) if "JAC2%" in df.columns else np.nan),
            QA_pass_rate=("QA_Passed", lambda x: np.mean(x.astype(bool))),
            n_sessions=("QA_Passed", "count"),
        )
        .reset_index()
    )
    model_summary["Motion_Mean"] = model_summary["Motion_Mean_mean"].round(3).astype(str) + " ± " + model_summary["Motion_Mean_std"].round(3).astype(str)
    model_summary["μHE"] = model_summary["μHE_mean"].round(3).astype(str) + " ± " + model_summary["μHE_std"].round(3).astype(str)
    model_summary["JAC0%"] = model_summary["JAC0_mean"].round(3).astype(str) + " ± " + model_summary["JAC0_std"].round(3).astype(str)
    model_summary["JAC2%"] = model_summary["JAC2_mean"].round(3).astype(str) + " ± " + model_summary["JAC2_std"].round(3).astype(str)
    model_summary["QA_pass_rate"] = (model_summary["QA_pass_rate"] * 100).round(1).astype(str) + "%"
    model_table = model_summary[["Model", "Motion_Mean", "μHE", "JAC0%", "JAC2%", "QA_pass_rate", "n_sessions"]]
    csv_out_model = os.path.join(os.path.dirname(csv_out), "summary_table_per_model.csv")
    model_table.to_csv(csv_out_model, index=False)
    print(f"[SAVED] Summary table (per model) to {csv_out_model}")

    return table, model_table
def plot_all_mvhs(entries, output_dir, n_points=500, save_png=True):
    """
    entries: list of (patient_id, session_id, dvf_paths) tuples
    output_dir: where to save the csv/png/npy
    n_points: number of thresholds in MVH
    """
    os.makedirs(output_dir, exist_ok=True)
    for patient_id, session_id, dvf_paths in entries:
        label = f"{patient_id}_{session_id}"
        print(f"[INFO] Processing MVH for {label} ({len(dvf_paths)} DVFs)")

        # 1. Compute motion map
        motion_map = compute_motion_map(dvf_paths)

        # 2. Save mean motion map as .npy
        motion_map_path = os.path.join(output_dir, f"{label}_motion_map.npy")
        np.save(motion_map_path, motion_map)
        print(f"[SAVED] Motion map: {motion_map_path}")

        # 3. Compute MVH
        thresholds, mvh = compute_mvh(motion_map, n_points=n_points)

        # 4. Save as CSV
        mvh_df = pd.DataFrame({
            'threshold': thresholds,
            'volume_percent': mvh
        })
        mvh_csv_path = os.path.join(output_dir, f"{label}_mvh.csv")
        mvh_df.to_csv(mvh_csv_path, index=False)
        print(f"[SAVED] MVH CSV: {mvh_csv_path}")

        # 5. Save PNG plot
        if save_png:
            plt.figure(figsize=(7, 5))
            plt.plot(thresholds, mvh, label=label)
            plt.xlabel("Motion Magnitude (mm/frame)")
            plt.ylabel("Volume (%)")
            plt.title(f"Motion-Volume Histogram: {label}")
            plt.grid(True)
            plt.legend()
            png_path = os.path.join(output_dir, f"{label}_mvh.png")
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()
            print(f"[SAVED] MVH PNG: {png_path}")

def plot_qa_rejections_bar_per_patient(df, output_dir, model_names):
    print(f"[INFO] Plotting QA failure reasons grouped bar chart (by model-patient)")
    # Compose Model_Patient label
    df['Model_Patient'] = df['Model'].astype(str) + " - " + df['Patient'].astype(str)
    patient_order = ["pt001", "pt002", "pt011", "pt022"]
    # If not provided, infer order
    if model_names is None:
        model_names = list(df["Model"].unique())
    if patient_order is None:
        patient_order = sorted(df["Patient"].unique())

    # Compose X-labels in desired order: first by model, then by patient
    ordered_xlabels = []
    for model in model_names:
        for patient in patient_order:
            label = f"{model} - {patient}"
            if label in df["Model_Patient"].values:
                ordered_xlabels.append(label)

    # Aggregate failures per model-patient
    bar_data = []
    for label in ordered_xlabels:
        sub = df[df['Model_Patient'] == label]
        n_jac0 = (~sub['QA_JAC0']).sum()
        n_jac2 = (~sub['QA_JAC2']).sum()
        n_he   = (~sub['QA_HE']).sum()
        n_total = len(sub)
        bar_data.append({
            'Model_Patient': label,
            'JAC0%': n_jac0,
            'JAC2%': n_jac2,
            'μHE': n_he,
            'Total DVFs': n_total,
        })
    bar_df = pd.DataFrame(bar_data)
    value_cols = ['JAC0%', 'JAC2%', 'μHE']
    x = np.arange(len(bar_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(18, len(x)*0.9), 7))
    colors = {"JAC0%": "#1f77b4", "JAC2%": "#ff7f0e", "μHE": "#2ca02c"}
    for i, cat in enumerate(value_cols):
        offset = (i - 1) * width
        ax.bar(x + offset, bar_df[cat], width, label=cat, color=colors[cat])

    ax.set_ylabel('Number of Failed DVFs')
    ax.set_title('QA Failure Reasons per Model-Patient')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_df['Model_Patient'], rotation=35, ha='right')
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "qa_failure_reasons_per_model_patient_bar.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVED] QA grouped bar plot (per model-patient) saved to: {out_path}")

    # Save CSV table as well
    bar_csv_path = os.path.join(output_dir, "qa_failure_reasons_per_model_patient_bar.csv")
    bar_df.to_csv(bar_csv_path, index=False)
    print(f"[SAVED] QA failure reasons (per model-patient) table as CSV to: {bar_csv_path}")

def normalize_slice(img_slice):
    min_val = np.min(img_slice)
    max_val = np.max(img_slice)
    img_norm = (img_slice - min_val) / (max_val - min_val + 1e-8)
    return img_norm

def make_multi_model_registration_figure(
    ref_img, def_imgs, dvfs, model_names,
    slice_idx=0, output_path="multi_model_registration.png"
):
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(model_names)
    fig = plt.figure(figsize=(3.5 * n_models + 1, 17))
    gs = plt.GridSpec(5, n_models+1, width_ratios=[1]*n_models+[0.07], wspace=0.0, hspace=0.04)

    jac_cmap = create_jacobian_colormap()
    he_cmap = create_he_colormap()

    row_labels = [
        "REFERENCE AXIAL", "DEFORMED AXIAL", "DIR REGISTRATION", "JAC", "HE"
    ]

    # Store axes to draw black lines after all plotting
    axes = []

    for col in range(n_models):
        def_img = crop64(def_imgs[col])
        dvf = crop64(dvfs[col])
        model_name = model_names[col]
        ref_slice = normalize_slice(crop64(ref_img)[..., slice_idx])
        def_slice = normalize_slice(crop64(def_img)[..., slice_idx])

        # --- Row 0: REFERENCE (fixed) AXIAL ---
        ax = plt.subplot(gs[0, col])
        ax.imshow(ref_slice, cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.annotate(row_labels[0], xy=(-0.19, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', rotation=90, fontweight="bold")
        ax.set_title(model_name, fontsize=13, pad=12)
        axes.append(ax)

        # --- Row 1: DEFORMED AXIAL ---
        ax = plt.subplot(gs[1, col])
        ax.imshow(def_slice, cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.annotate(row_labels[1], xy=(-0.19, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', rotation=90, fontweight="bold")
        axes.append(ax)

        # --- Row 2: DIR REGISTRATION (RGB Overlay) ---
        ax = plt.subplot(gs[2, col])
        rgb_overlay = np.stack([
            np.clip(def_slice, 0, 1),        # Red: deformed
            np.clip(ref_slice, 0, 1),        # Green: reference
            np.zeros_like(ref_slice)         # Blue: nothing
        ], axis=-1)
        ax.imshow(rgb_overlay)
        ax.axis("off")
        if col == 0:
            ax.annotate(row_labels[2], xy=(-0.19, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', rotation=90, fontweight="bold")
        axes.append(ax)

        # --- Row 3: JACOBIAN overlay ---
        ax = plt.subplot(gs[3, col])
        jac = compute_jacobian_determinant(dvf)
        jac_slice = jac[..., slice_idx]
        ax.imshow(def_slice, cmap="gray", alpha=1.0)
        im_jac = ax.imshow(jac_slice, cmap=jac_cmap, vmin=-2, vmax=4, alpha=0.4)
        ax.axis("off")
        if col == 0:
            ax.annotate(row_labels[3], xy=(-0.19, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', rotation=90, fontweight="bold")
        axes.append(ax)

        # --- Row 4: HARMONIC ENERGY overlay ---
        ax = plt.subplot(gs[4, col])
        he = compute_harmonic_energy(dvf)
        he_slice = he[..., slice_idx]
        ax.imshow(def_slice, cmap="gray", alpha=1.0)
        im_he = ax.imshow(he_slice, cmap=he_cmap, vmin=2, vmax=7, alpha=0.4)
        ax.axis("off")
        if col == 0:
            ax.annotate(row_labels[4], xy=(-0.19, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', rotation=90, fontweight="bold")
        axes.append(ax)

    # --- Colorbars (always last col) ---
    ax_jac_bar = plt.subplot(gs[3, -1])
    plt.colorbar(im_jac, cax=ax_jac_bar, orientation="vertical")
    ax_jac_bar.set_ylabel("JAC", fontsize=12)
    ax_jac_bar.yaxis.label.set_rotation(270)
    ax_jac_bar.yaxis.set_label_coords(3,0.5)

    ax_he_bar = plt.subplot(gs[4, -1])
    plt.colorbar(im_he, cax=ax_he_bar, orientation="vertical")
    ax_he_bar.set_ylabel("HE", fontsize=12)
    ax_he_bar.yaxis.label.set_rotation(270)
    ax_he_bar.yaxis.set_label_coords(3,0.5)

    # ------- Tight layout: remove as much whitespace as possible --------
    plt.subplots_adjust(
        left=0.09, right=0.97, top=0.97, bottom=0.06,
        wspace=0.00, hspace=0.06
    )


    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()

def plot_qa_rejections_bar(df, output_dir):
    print(f"[INFO] Plotting QA failure reasons grouped bar chart (by model)")
    data = []
    for model in df['Model'].unique():
        sub = df[df['Model'] == model]
        n_jac0 = (~sub['QA_JAC0']).sum()
        n_jac2 = (~sub['QA_JAC2']).sum()
        n_he   = (~sub['QA_HE']).sum()
        n_total = len(sub)
        data.append({
            'Model': model,
            'JAC0%': n_jac0,
            'JAC2%': n_jac2,
            'μHE': n_he,
            'Total DVFs': n_total,
        })
    bar_df = pd.DataFrame(data)
    value_cols = ['JAC0%', 'JAC2%', 'μHE']
    x = np.arange(len(bar_df))
    width = 0.25  # Width of each bar

    xtick_labels = bar_df['Model']
    fig, ax = plt.subplots(figsize=(max(8, len(x)*1.6), 6))
    colors = {"JAC0%": "#1f77b4", "JAC2%": "#ff7f0e", "μHE": "#2ca02c"}
    for i, cat in enumerate(value_cols):
        offset = (i - 1) * width  # so -width, 0, width for 3 bars
        ax.bar(x + offset, bar_df[cat], width, label=cat, color=colors[cat])

    ax.set_ylabel('Number of Failed DVFs')
    ax.set_title('QA Failure Reasons per Model (all patients)')
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=20, ha='right')
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "qa_failure_reasons_per_model_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] QA grouped bar plot (per model) saved to: {out_path}")

    # Save bar_df as CSV
    bar_csv_path = os.path.join(output_dir, "qa_failure_reasons_per_model_bar.csv")
    bar_df.to_csv(bar_csv_path, index=False)
    print(f"[SAVED] QA failure reasons (per model) table as CSV to: {bar_csv_path}")

def find_dvf_folder(base_folder):
    candidates = [
        os.path.join(base_folder, sub)
        for sub in ["warp", "DVF"]
    ] + [
        os.path.join(base_folder, "motility_dynamics", sub)
        for sub in ["warp", "DVF"]
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            print(f"[INFO] Found DVF folder: {candidate}")
            return candidate
    print(f"[WARNING] No DVF/warp folder found in {base_folder} (tried {candidates})")
    return None

def find_all_csvs_in_folder(folder):
    # Recursively find all dvf_statistics.csv in a folder and its subfolders
    csvs = glob(os.path.join(folder, '**', 'dvf_statistics.csv'), recursive=True)
    return csvs

# ------------ NEW: Visualize registered slices with overlays --------------
def visualize_registered_slices(
    img_folder, seg_folder=None, base_img_folder=None,
    slice_idx=0, output_dir=".", cmap_img="gray", cmap_seg="spring", alpha_img=0.5, alpha_seg=0.3
):
    """
    Visualizes and saves a selected slice from registered images, with optional overlays:
      - base_img_folder: underlying static/dynamic MRI
      - seg_folder: segmentation mask (e.g. bowel mask)
    """
    img_files = sorted(glob(os.path.join(img_folder, "*.nii*")))
    if not img_files:
        print(f"[ERROR] No NIfTI images found in {img_folder}")
        return
    os.makedirs(output_dir, exist_ok=True)
    for img_path in img_files:
        img = nib.load(img_path).get_fdata()
        fname = os.path.splitext(os.path.basename(img_path))[0]
        base_img = None
        seg = None

        # Try to find and load base MRI
        if base_img_folder:
            match_base = sorted(glob(os.path.join(base_img_folder, f"{fname}*.nii*")))
            if match_base:
                base_img = nib.load(match_base[0]).get_fdata()
        # Try to find and load seg mask
        if seg_folder:
            match_seg = sorted(glob(os.path.join(seg_folder, f"{fname}*.nii*")))
            if match_seg:
                seg = nib.load(match_seg[0]).get_fdata()
                # Force binary
                seg = (seg > 0.5).astype(np.float32)

        plt.figure(figsize=(6, 6))
        shown_img = img[..., slice_idx]
        plt.imshow(shown_img, cmap=cmap_img)
        # Overlay base MRI (optional)
        if base_img is not None:
            base_slice = base_img[..., slice_idx]
            plt.imshow(base_slice, cmap="gray", alpha=alpha_img)
        # Overlay segmentation (optional)
        if seg is not None:
            seg_slice = seg[..., slice_idx]
            plt.imshow(np.ma.masked_where(seg_slice < 0.5, seg_slice), cmap=cmap_seg, alpha=alpha_seg)
        plt.title(f"{fname} | Slice {slice_idx}")
        plt.axis("off")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"vis_{fname}_slice{slice_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {save_path}")
# --------------------------------------------------------------------------

def plot_motion_percentiles_by_model(csv_paths, model_names, output_path, ycap=5.0):
    all_data = []
    for csv_list, model in zip(csv_paths, model_names):
        dfs = [pd.read_csv(csv_path) for csv_path in csv_list]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df["Model"] = model
        all_data.append(df)
    if not all_data:
        print("[ERROR] No data loaded for plotting.")
        return
    df_all = pd.concat(all_data, ignore_index=True)
    df_all["QA_Passed"] = df_all["QA_Passed"].astype(bool)

    plot_rows = []
    for (patient, model), group in df_all.groupby(["Patient", "Model"]):
        for perc, col, marker in [(10, "Motion P90", "^"), (50, "Motion P50", "s"), (90, "Motion P10", "o")]:
            val_all = group[col].mean()
            plot_rows.append(dict(patient=patient, model=model, percentile=perc, value=val_all, marker=marker, filled=True))
            passed = group[group["QA_Passed"]]
            if not passed.empty:
                val_pass = passed[col].mean()
                plot_rows.append(dict(patient=patient, model=model, percentile=perc, value=val_pass, marker=marker, filled=False))
            else:
                plot_rows.append(dict(patient=patient, model=model, percentile=perc, value=np.nan, marker=marker, filled=False))

    plot_df = pd.DataFrame(plot_rows)

    # Order x-axis: all patients per model, repeat for each model
    plot_df["xlab"] = plot_df.apply(lambda r: f"{r['model']} - {r['patient']}", axis=1)
    # Maintain unique (patient, model) order for x-axis
    x_labels = plot_df[["xlab", "patient", "model"]].drop_duplicates().sort_values(["model", "patient"])["xlab"].tolist()
    plot_df["x"] = plot_df["xlab"].apply(lambda v: x_labels.index(v))

    colors = {10: "royalblue", 50: "darkorange", 90: "k"}
    markers = {10: "o", 50: "s", 90: "^"}
    percentiles = [10, 50, 90]
    fills = [True, False]

    # Custom legend setup
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    for perc in percentiles:
        for filled in [True, False]:
            face = colors[perc] if filled else "none"
            edge = colors[perc]
            label = f"M{perc}% {'all data' if filled else 'QA filtered'}"
            handle = Line2D([0], [0], marker=markers[perc], color="w",
                            markerfacecolor=face, markeredgecolor=edge,
                            markeredgewidth=2, markersize=12, linestyle="None")
            legend_handles.append(handle)
            legend_labels.append(label)

    fig, ax = plt.subplots(figsize=(max(12, len(x_labels)*0.6), 8))
    for perc in percentiles:
        for filled in [True, False]:
            mask = (plot_df["percentile"] == perc) & (plot_df["filled"] == filled)
            d = plot_df[mask]
            ax.scatter(d["x"], np.clip(d["value"], 0, ycap),
                       marker=markers[perc],
                       edgecolor=colors[perc],
                       facecolor=colors[perc] if filled else "none",
                       s=100, linewidths=2,
                       label=None)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("MOTION (mm)")
    ax.set_xlabel("Model - Patient")
    ax.set_ylim(0, ycap)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.grid(True, axis="y")
    ax.set_title("Motion Percentiles per Patient and Model")
    ax.legend(legend_handles, legend_labels, ncol=3, loc='upper right')
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, "motion_percentiles_per_patient_model_aggregate.png")
    plt.savefig(out_path, dpi=200)
    data_csv_path = os.path.join(output_path, "motion_percentiles_per_patient_model_aggregate.csv")
    plot_df.to_csv(data_csv_path, index=False)
    print(f"[SAVED] Data for motion percentiles plot to: {data_csv_path}")
    plt.close()
    print(f"[SAVED] {out_path}")
    

# ------------- NEW: Revisualize MVH curves and motion maps -----------------
def revisualize_mvhs(mvh_data_dir, output_dir, selected_labels=None):
    """Reloads and replots saved MVH curves and/or motion maps."""
    print(f"[INFO] Revisiting MVH data in {mvh_data_dir}")
    mvh_csvs = sorted(glob(os.path.join(mvh_data_dir, "*_mvh.csv")))
    motion_npy = sorted(glob(os.path.join(mvh_data_dir, "*_motion_map.npy")))

    # If present, load the wide CSV for all-in-one plotting
    all_curves_path = os.path.join(mvh_data_dir, "all_mvh_curves.csv")
    if os.path.exists(all_curves_path):
        print(f"[INFO] Found wide MVH CSV for all-patient plotting: {all_curves_path}")
        df = pd.read_csv(all_curves_path)
        plt.figure(figsize=(10, 6))
        for col in df.columns[1:]:
            if (selected_labels is None) or any(sel in col for sel in selected_labels):
                plt.plot(df['threshold'], df[col], label=col)
        plt.xlabel("Motion Magnitude (mm/frame)")
        plt.ylabel("Volume (%)")
        plt.title("Motion-Volume Histogram (Re-visualized)")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, "all_motion_volume_histograms_revisualized.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[SAVED] MVH plot re-visualized to: {save_path}")
    else:
        # Else, plot each individual
        for csv_path in mvh_csvs:
            label = os.path.basename(csv_path).replace("_mvh.csv", "")
            if (selected_labels is None) or any(sel in label for sel in selected_labels):
                mvh_df = pd.read_csv(csv_path)
                plt.figure()
                plt.plot(mvh_df["threshold"], mvh_df["volume_percent"], label=label)
                plt.xlabel("Motion Magnitude (mm/frame)")
                plt.ylabel("Volume (%)")
                plt.title(f"MVH: {label}")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(output_dir, f"mvh_{label}_revisualized.png")
                plt.savefig(save_path)
                plt.close()
                print(f"[SAVED] {save_path}")

    # Optionally, plot motion maps (heatmaps) for selected sessions
    for npy_path in motion_npy:
        label = os.path.basename(npy_path).replace("_motion_map.npy", "")
        if (selected_labels is None) or any(sel in label for sel in selected_labels):
            motion_map = np.load(npy_path)
            plt.figure()
            # Show mid-axial slice as example
            mid = motion_map.shape[2] // 2
            plt.imshow(motion_map[..., mid], cmap="hot")
            plt.title(f"Motion map {label} (axial mid-slice)")
            plt.colorbar(label="Motion (mm/frame)")
            save_path = os.path.join(output_dir, f"motionmap_{label}_midaxial.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[SAVED] {save_path}")

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_motion_percentiles", action="store_true", help="Plot patient motion percentiles (P10/P50/P90, all vs QA filtered)")
    parser.add_argument("--mvh", action="store_true", help="Plot Motion Volume Histograms")
    parser.add_argument("--overlays", action="store_true", help="Save overlay images (DVF, Jacobian, HE)")
    parser.add_argument("--qa_csv", nargs='+', help="CSV(s) or folders with QA results for barplots & summary")
    parser.add_argument("--model_names", nargs='+', help="Names of models matching the CSVs or folders")
    parser.add_argument("--root_folder", help="Folder with patient DVFs (for MVH)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ref", help="Reference image for overlays")
    parser.add_argument("--def_img", help="Deformed image for overlays")
    parser.add_argument("--dvf", help="DVF file for overlays")
    parser.add_argument("--slice", type=int, default=0, help="Slice index for overlay")

    # ---- New for re-visualization ----
    parser.add_argument("--revisualize_mvh", help="Directory with saved MVH/motion map files")
    parser.add_argument("--select_label", nargs="*", help="Only plot MVHs containing these substrings (optional)")

    # ---- NEW: Visualize registered slices ----
    parser.add_argument("--visualize_registered_slices", help="Folder with registered NIfTI images to visualize")
    parser.add_argument("--visualize_slice", type=int, default=0, help="Slice index to visualize")
    parser.add_argument("--visualize_seg_folder", help="Folder with segmentation NIfTIs for overlay (optional)")
    parser.add_argument("--visualize_base_folder", help="Folder with base/reference MRIs for overlay (optional)")

    parser.add_argument("--plot_patient_avg_mvhs", action="store_true", help="Aggregate and plot MVH curves for each patient, for each model")
    parser.add_argument("--mvh_data_dirs", nargs='+', help="List of folders containing *_mvh.csv (one per model)")
    parser.add_argument("--patient_id", nargs="*", help="Optionally, specific patient(s) to plot (default: all patients)")
    
    parser.add_argument("--multi_model_viz", action="store_true")
    parser.add_argument("--ref_img", help="Reference NIfTI")
    parser.add_argument("--def_imgs", nargs='+', help="List of deformed images (per model)")
    parser.add_argument("--dvfs", nargs='+', help="List of DVFs (per model)")

    parser.add_argument("--plot_harmonic_energy_summaries", action="store_true", help="Plot Harmonic Energy vs Mean Motion and distribution boxplot from QA CSV(s)")
    parser.add_argument("--summarize_dvf_statistics", action="store_true", help="Generate a summary table per model/patient with key metrics (mean ± std, QA rate)")




    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.plot_harmonic_energy_summaries:
        if not args.qa_csv or not args.model_names:
            print("[ERROR] --plot_harmonic_energy_summaries requires --qa_csv and --model_names!")
        else:
            expanded_csv_paths = []
            expanded_model_names = []
            for csv_arg, model_name in zip(args.qa_csv, args.model_names):
                if os.path.isdir(csv_arg):
                    found_csvs = find_all_csvs_in_folder(csv_arg)
                    if not found_csvs:
                        print(f"[WARNING] No dvf_statistics.csv found in folder {csv_arg}")
                    for csv in found_csvs:
                        expanded_csv_paths.append(csv)
                        expanded_model_names.append(model_name)
                else:
                    expanded_csv_paths.append(csv_arg)
                    expanded_model_names.append(model_name)

            if not expanded_csv_paths:
                print("[ERROR] No QA CSVs found for harmonic energy analysis.")
                return
            if len(expanded_csv_paths) != len(expanded_model_names):
                raise ValueError("Number of expanded CSVs does not match number of model names.")

            all_dfs = [pd.read_csv(p) for p in expanded_csv_paths]
            for df, name in zip(all_dfs, expanded_model_names):
                df["Model"] = name
            df = pd.concat(all_dfs, ignore_index=True)
            plot_harmonic_energy_summaries(df, args.output)

    if args.multi_model_viz:
        ref_img = nib.load(args.ref_img).get_fdata()
        def_imgs = [nib.load(f).get_fdata() for f in args.def_imgs]
        dvfs = [normalize_dvf(nib.load(f).get_fdata()) for f in args.dvfs]
        make_multi_model_registration_figure(
            ref_img, def_imgs, dvfs, args.model_names, slice_idx=args.slice,
            output_path=os.path.join(args.output, "multi_model_registration.png")
        )

    if args.plot_motion_percentiles:
        if not args.qa_csv or not args.model_names:
            print("[ERROR] --plot_motion_percentiles requires --qa_csv and --model_names!")
        else:
            model_csvs = expand_csv_paths(args.qa_csv, args.model_names)
            plot_motion_percentiles_by_model(
                csv_paths=model_csvs,  # Pass as a list of lists
                model_names=args.model_names,
                output_path=args.output,
                ycap=5.0,
            )


    if args.plot_patient_avg_mvhs:
        if not args.mvh_data_dirs or not args.model_names:
            print("[ERROR] Must provide --mvh_data_dirs and --model_names!")
        else:
            # None means all, otherwise use list of patients
            patient_ids = args.patient_id if args.patient_id and len(args.patient_id) > 0 else None
            plot_patient_avg_mvhs_multi(
                mvh_data_dirs=args.mvh_data_dirs,
                model_names=args.model_names,
                output_path=args.output,
                patient_ids=patient_ids,
            )


    if args.mvh and args.root_folder:
        print(f"[START] Running MVH plotting on root folder: {args.root_folder}")
        entries = []
        for patient_id in ("pt001", "pt002", "pt011", "pt022", "pt023"):
            p_folder = os.path.join(args.root_folder, patient_id)
            if not os.path.isdir(p_folder):
                print(f"[WARNING] Missing patient folder: {p_folder}")
                continue
            for i in range(1, 4):
                base_folder = os.path.join(p_folder, f"MR{i}\motility_dynamics")
                dvf_folder = find_dvf_folder(base_folder)
                if dvf_folder:
                    dvf_files = sorted(glob(os.path.join(dvf_folder, "**", "*.nii*"), recursive=True))
                    if dvf_files:
                        entries.append((patient_id, f"MR{i}", dvf_files))
                    else:
                        print(f"[WARNING] No DVF files in: {dvf_folder}")

        if entries:
            plot_all_mvhs(entries, args.output)
        else:
            print("[ERROR] No DVFs found for MVH plotting.")

    if args.summarize_dvf_statistics:
        if not args.qa_csv or not args.model_names:
            print("[ERROR] --summarize_dvf_statistics requires --qa_csv and --model_names!")
        else:
            # Gather all csvs for each model, matching previous logic
            expanded_csv_paths_per_model = []
            for csv_arg in args.qa_csv:
                if os.path.isdir(csv_arg):
                    found_csvs = find_all_csvs_in_folder(csv_arg)
                    if not found_csvs:
                        print(f"[WARNING] No dvf_statistics.csv found in folder {csv_arg}")
                    expanded_csv_paths_per_model.append(found_csvs)
                else:
                    expanded_csv_paths_per_model.append([csv_arg])
            if len(expanded_csv_paths_per_model) != len(args.model_names):
                raise ValueError("Number of models does not match number of CSV (or folder) inputs!")
            
            # For each model, combine its CSVs (if multiple) before summary
            all_dfs = []
            for csvs, model_name in zip(expanded_csv_paths_per_model, args.model_names):
                dfs = [pd.read_csv(csv_path) for csv_path in csvs if os.path.isfile(csv_path)]
                if not dfs:
                    continue
                df_model = pd.concat(dfs, ignore_index=True)
                df_model['Model'] = model_name
                all_dfs.append(df_model)
            if not all_dfs:
                print("[ERROR] No data found for summary table.")
                return
            df = pd.concat(all_dfs, ignore_index=True)
            plot_summary = summarize_dvf_statistics_from_df(df, args.output)

    if args.overlays and args.ref and args.def_img and args.dvf:
        print(f"[START] Creating overlays with ref={args.ref}, def={args.def_img}, dvf={args.dvf}")
        ref = nib.load(args.ref).get_fdata()
        deform = nib.load(args.def_img).get_fdata()
        dvf = normalize_dvf(nib.load(args.dvf).get_fdata())
        jac = compute_jacobian_determinant(dvf)
        he = compute_harmonic_energy(dvf)
        slice_idx = args.slice
        base = ref[:, :, slice_idx]
        deformed = deform[:, :, slice_idx]
        jac_slice = jac[:, :, slice_idx]
        he_slice = he[:, :, slice_idx]
        jac0 = 100 * np.mean(jac < 0)
        jac2 = 100 * np.mean(jac > 2)
        mu_he_value = np.mean(he)
        print(f"[STAT] Jacobian < 0%%: {jac0:.2f} | Jacobian > 2%%: {jac2:.2f} | Mean HE: {mu_he_value:.2f}")
        if jac0 > JAC0_THRESHOLD_BARTEN:
            print(f"[WARNING] JAC0% ({jac0:.2f}) exceeds Barten et al. threshold ({JAC0_THRESHOLD_BARTEN})")
        if jac2 > JAC2_THRESHOLD_BARTEN:
            print(f"[WARNING] JAC2% ({jac2:.2f}) exceeds Barten et al. threshold ({JAC2_THRESHOLD_BARTEN})")
        if mu_he_value > HE_THRESHOLD_BARTEN:
            print(f"[WARNING] μHE ({mu_he_value:.2f}) exceeds Barten et al. threshold ({HE_THRESHOLD_BARTEN})")

        save_dir_overlay_image(base, deformed, os.path.join(args.output, "dir_overlay.png"))
        save_overlay_image(base, jac_slice, create_jacobian_colormap(), 0.4, "JAC", "Jacobian Determinant", -2, 4, os.path.join(args.output, "jac_overlay.png"))
        save_overlay_image(base, he_slice, create_he_colormap(), 0.4, "HE", "Harmonic Energy", 0, 7, os.path.join(args.output, "he_overlay.png"))

        # Save overlay slices as npy for reproducibility
        overlay_data_dir = os.path.join(args.output, "overlay_data")
        os.makedirs(overlay_data_dir, exist_ok=True)
        np.save(os.path.join(overlay_data_dir, f"ref_slice{slice_idx}.npy"), base)
        np.save(os.path.join(overlay_data_dir, f"def_slice{slice_idx}.npy"), deformed)
        np.save(os.path.join(overlay_data_dir, f"jac_slice{slice_idx}.npy"), jac_slice)
        np.save(os.path.join(overlay_data_dir, f"he_slice{slice_idx}.npy"), he_slice)
        print(f"[SAVED] Overlay input/metric arrays to: {overlay_data_dir}")

    # ------- UPDATED QA CSV LOGIC --------
    if args.qa_csv and args.model_names:
        print(f"[START] Generating QA plots from CSVs or folders")

        expanded_csv_paths = []
        expanded_model_names = []

        for csv_arg, model_name in zip(args.qa_csv, args.model_names):
            if os.path.isdir(csv_arg):
                found_csvs = find_all_csvs_in_folder(csv_arg)
                if not found_csvs:
                    print(f"[WARNING] No dvf_statistics.csv found in folder {csv_arg}")
                for csv in found_csvs:
                    expanded_csv_paths.append(csv)
                    expanded_model_names.append(model_name)
            else:
                expanded_csv_paths.append(csv_arg)
                expanded_model_names.append(model_name)

        if not expanded_csv_paths:
            print("[ERROR] No QA CSVs found for analysis.")
            return
        if len(expanded_csv_paths) != len(expanded_model_names):
            raise ValueError("Number of expanded CSVs does not match number of model names.")

        all_dfs = [pd.read_csv(p) for p in expanded_csv_paths]
        for df, name in zip(all_dfs, expanded_model_names):
            df["Model"] = name
        df = pd.concat(all_dfs, ignore_index=True)
        plot_qa_rejections_bar_per_patient(df, args.output, model_names = args.model_names)
        # (Add other summary/correlation plots here if needed)

    # ---- New: Revisualize MVH curves/motion maps ----
    if args.revisualize_mvh:
        revisualize_mvhs(args.revisualize_mvh, args.output, args.select_label)

    # ---- NEW: Visualize registered slices ----
    if args.visualize_registered_slices:
        visualize_registered_slices(
            img_folder=args.visualize_registered_slices,
            seg_folder=args.visualize_seg_folder,
            base_img_folder=args.visualize_base_folder,
            slice_idx=args.visualize_slice,
            output_dir=args.output
        )

    print("[DONE] All requested visualizations are complete.")

if __name__ == "__main__":
    main()



# MVH:
# python metrics\master_vis.py --mvh --root_folder PATH_TO_DVF_ROOT --output OUTPUT_DIR

# Overlays:
# python metrics\master_vis.py --overlays --ref REF.nii --def_img DEF.nii --dvf DVF.nii --slice 20 --output OUTPUT_DIR

# QA/Barplots:
# python metrics\master_vis.py --qa_csv qa_model1.csv qa_model2.csv --model_names Model1 Model2 --output OUTPUT_DIR

# Re-visualize saved MVH curves and motion maps:
# python metrics\master_vis.py --revisualize_mvh PATH_TO_MVH_DATA_DIR --output OUTPUT_DIR --select_label Model1 Model2

# Visualize registered slices:
#python metrics/master_vis.py --visualize_registered_slices /path/to/registered/ --visualize_seg_folder /path/to/masks/ --visualize_base_folder /path/to/baseMRI/ --visualize_slice 60 --output /path/to/visualizations/