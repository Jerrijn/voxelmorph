#!/usr/bin/env python

import os
import re
import argparse

def rename_to_timestep_only(folder_path):
    for filename in os.listdir(folder_path):
        match = re.search(r'_(\d{3})_preprocessed(?:_metadata)?(\.nii\.gz|\.npy)$', filename)
        if match:
            timestep = match.group(1)
            extension = match.group(2)

            if '_metadata' in filename:
                new_name = f"{timestep}_metadata{extension}"
            else:
                new_name = f"{timestep}{extension}"

            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} ➝ {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename preprocessed files to timestep-only names.")
    parser.add_argument("folder", help="Path to the folder with files to rename")
    args = parser.parse_args()

    if os.path.isdir(args.folder):
        rename_to_timestep_only(args.folder)
    else:
        print(f"Error: {args.folder} is not a valid directory.")
