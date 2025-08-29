#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main(csv_path, output_dir):
    # Load CSV
    df = pd.read_csv(csv_path)
    df["Patient"] = df["Patient"].str.upper()

    os.makedirs(output_dir, exist_ok=True)

    # === 1. QA Rejection Counts (Fig. 6a style) ===
    rejections = df.groupby("Patient")[["QA_JAC0", "QA_JAC2", "QA_HE"]].apply(lambda x: (~x).sum())
    x = np.arange(len(rejections))
    bar_width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width, rejections["QA_JAC0"], width=bar_width, label="JAC0%", color="#D4A017")
    plt.bar(x, rejections["QA_JAC2"], width=bar_width, label="JAC2%", color="#D4A017", hatch='//')
    plt.bar(x + bar_width, rejections["QA_HE"], width=bar_width, label="μHE", color="#D4A017", hatch='..')
    plt.xticks(x, rejections.index)
    plt.ylabel("Amount of Rejected DVFs")
    plt.title("QA Rejected DVFs per Patient")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qa_rejections_bar.png")
    plt.close()

    # === 2. MVH Before/After QA (Fig. 6b style) ===
    mvh = df.groupby("Patient").agg(
        M10_all=("Motion P10", "mean"),
        M50_all=("Motion P50", "mean"),
        M90_all=("Motion P90", "mean")
    )
    mvh_filt = df[df["QA_Passed"]].groupby("Patient").agg(
        M10_filt=("Motion P10", "mean"),
        M50_filt=("Motion P50", "mean"),
        M90_filt=("Motion P90", "mean")
    )

    plt.figure(figsize=(12, 6))
    for i, pid in enumerate(mvh.index):
        if pid in mvh_filt.index:
            plt.plot(i, mvh.loc[pid, "M10_all"], 'o', color='blue')
            plt.plot(i, mvh_filt.loc[pid, "M10_filt"], 'o', markerfacecolor='white', markeredgecolor='blue')
            plt.plot(i, mvh.loc[pid, "M50_all"], 's', color='orange')
            plt.plot(i, mvh_filt.loc[pid, "M50_filt"], 's', markerfacecolor='white', markeredgecolor='orange')
            plt.plot(i, mvh.loc[pid, "M90_all"], '^', color='black')
            plt.plot(i, mvh_filt.loc[pid, "M90_filt"], '^', markerfacecolor='white', markeredgecolor='black')

    plt.xticks(np.arange(len(mvh)), mvh.index)
    plt.ylabel("Motion (mm)")
    plt.title("MVH Parameters Before/After QA Filtering")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mvh_comparison.png")
    plt.close()

    # === 3. Harmonic Energy Distribution ===
    plt.figure(figsize=(12, 6))
    df["Log HE"] = np.log1p(df["Harmonic Energy"])
    sns.boxplot(x="Patient", y="Log HE", data=df)
    plt.title("Harmonic Energy Distribution (log-scaled)")
    plt.ylabel("log(1 + Harmonic Energy)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/harmonic_energy_distribution.png")
    plt.close()

    # === 4. QA Pass Rate per Patient ===
    qa_pass_rate = df.groupby("Patient")["QA_Passed"].mean() * 100
    qa_pass_rate = qa_pass_rate.sort_values()

    plt.figure(figsize=(10, 6))
    qa_pass_rate.plot(kind="barh", color="seagreen")
    plt.xlabel("QA Pass Rate (%)")
    plt.title("Proportion of DVFs Passing QA")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qa_pass_rate.png")
    plt.close()

    # === 5. Motion vs Harmonic Energy Correlation ===
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Harmonic Energy", y="Motion Mean", hue="Patient", data=df)
    plt.title("Correlation: Harmonic Energy vs Mean Motion")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/motion_vs_he.png")
    plt.close()

    print(f"[DONE] All plots saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the dvf_statistics.csv file")
    parser.add_argument("--output_folder", required=True, help="Output folder for the saved plots")
    args = parser.parse_args()

    main(args.csv_path, args.output_folder)
