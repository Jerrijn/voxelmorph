#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# Define a color palette
colors = {
    "JAC0": "#1f77b4",  # blue
    "JAC2": "#ff7f0e",  # orange
    "HE": "#2ca02c",    # green
    "pass_rate": ["#9467bd"],  # purple as list for seaborn compatibility
    "motion_10": "blue",
    "motion_50": "orange",
    "motion_90": "black"
}

def load_and_tag(csv_path, model_name):
    df = pd.read_csv(csv_path)
    df["Model"] = model_name
    df["Patient"] = df["Patient"].str.upper()
    df["Patient_Model"] = df["Patient"] + f" ({model_name})"
    return df

def plot_qa_rejections_bar(df, output_dir):
    # Prepare data: count rejections per patient per model
    data = []
    for model in df['Model'].unique():
        for patient in df[df['Model'] == model]['Patient'].unique():
            sub = df[(df['Model'] == model) & (df['Patient'] == patient)]
            data.append({
                'Model': model,
                'Patient': patient,
                'JAC0%': (~sub['QA_JAC0']).sum(),
                'JAC2%': (~sub['QA_JAC2']).sum(),
                'μHE': (~sub['QA_HE']).sum()
            })
    bar_df = pd.DataFrame(data)

    # Only keep categories with nonzero total rejections
    value_cols = ['JAC0%', 'JAC2%', 'μHE']
    value_cols = [c for c in value_cols if bar_df[c].sum() > 0]

    # Calculate minimum value among all shown data
    min_val = min([bar_df[c].min() for c in value_cols])
    rounded_min = int(np.floor(min_val / 100) * 100)

    # Set up bar plot parameters
    x = np.arange(len(bar_df))
    width = 0.25 if len(value_cols) == 3 else (0.33 if len(value_cols) == 2 else 0.5)

    # X-tick labels are patient-model pairs
    xtick_labels = [f"{row['Patient']} ({row['Model']})" for idx, row in bar_df.iterrows()]

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.7), 6))
    for i, cat in enumerate(value_cols):
        offset = (i - (len(value_cols)-1)/2) * width
        color_key = cat.replace('%','').replace('μ','HE') if cat != 'μHE' else 'HE'
        ax.bar(x + offset, bar_df[cat], width, label=cat, color=colors[color_key])

    ax.set_ylabel('Amount of Rejected DVFs')
    ax.set_title('QA Rejected DVFs per Patient/Model')
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
    ax.set_ylim(bottom=rounded_min)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "qa_rejections_per_patient_bar.png"))
    plt.close()




def main(csv_paths, model_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # === Load and tag all models ===
    all_dfs = [load_and_tag(p, n) for p, n in zip(csv_paths, model_names)]
    df = pd.concat(all_dfs, ignore_index=True)

    # === QA Summary Table (per model) ===
    summary_rows = []
    for model in df['Model'].unique():
        df_model = df[df['Model'] == model]
        summary = {
            "Model": model,
            "Total DVFs": len(df_model),
            "Passed All QA": df_model["QA_Passed"].sum(),
            "Failed JAC0": (~df_model["QA_JAC0"]).sum(),
            "Failed JAC2": (~df_model["QA_JAC2"]).sum(),
            "Failed HE": (~df_model["QA_HE"]).sum(),
            "Average Motion Mean (mm)": pd.to_numeric(df_model["Motion Mean"], errors='coerce').mean(),
            "Average Harmonic Energy": pd.to_numeric(df_model["Harmonic Energy"], errors='coerce').mean(),
            "Average Jacobian % Negative": pd.to_numeric(df_model["Jacobian % Negative"], errors='coerce').mean(),
            "QA Pass Rate (%)": 100 * df_model["QA_Passed"].sum() / len(df_model)
        }
        summary_rows.append(summary)
    overview_df = pd.DataFrame(summary_rows)
    overview_df.to_csv(os.path.join(output_dir, "qa_summary_table.csv"), index=False)

    # === Visual QA Summary Table for Presentation ===
    fig, ax = plt.subplots(figsize=(12, 1 + len(overview_df)*0.6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=np.round(overview_df.select_dtypes(include=[np.number]).values, 3),
                     colLabels=overview_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qa_summary_visual_table.png"))
    plt.close()

    # === QA Pass Rate Bar Chart ===
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Model", y="QA Pass Rate (%)", data=overview_df, color=colors["pass_rate"][0])
    plt.title("QA Pass Rate per Model")
    plt.ylabel("QA Pass Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qa_pass_rate.png"))
    plt.close()

    # === Harmonic Energy vs Motion Correlation Plot (consistent color per Patient_Model) ===
    plt.figure(figsize=(8, 6))
    unique_ids = df['Patient_Model'].unique()
    color_map = dict(zip(unique_ids, sns.color_palette("husl", len(unique_ids))))
    for pid in unique_ids:
        sub = df[df['Patient_Model'] == pid]
        plt.scatter(pd.to_numeric(sub["Harmonic Energy"], errors='coerce'),
                    pd.to_numeric(sub["Motion Mean"], errors='coerce'),
                    label=pid, color=color_map[pid], s=20)
    plt.xlabel("Harmonic Energy")
    plt.ylabel("Motion Mean")
    plt.title("Correlation: Harmonic Energy vs Mean Motion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_vs_he.png"))
    plt.close()

    # === Harmonic Energy Distribution Box Plot (same color scheme as above) ===
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Patient_Model",
                y=np.log1p(pd.to_numeric(df["Harmonic Energy"], errors='coerce')),
                data=df,
                palette=color_map)
    plt.xticks(rotation=45, ha="right")
    plt.title("Harmonic Energy Distribution (log-scaled)")
    plt.ylabel("log(1 + Harmonic Energy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "harmonic_energy_distribution.png"))
    plt.close()

    # === QA Rejections Grouped Bar Plot ===
    plot_qa_rejections_bar(df, output_dir)

    print(f"[DONE] All plots and QA summary saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", nargs='+', required=True, help="List of CSV paths for each model")
    parser.add_argument("--model_names", nargs='+', required=True, help="List of model names in the same order as CSVs")
    parser.add_argument("--output_folder", required=True, help="Directory to save all generated plots")
    args = parser.parse_args()

    if len(args.csv_paths) != len(args.model_names):
        raise ValueError("Number of --csv_paths must match number of --model_names")

    main(args.csv_paths, args.model_names, args.output_folder)
