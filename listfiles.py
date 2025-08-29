import os

def list_and_save_nii_gz_paths(root_directory, output_file, indent=0):
    """
    Recursively traverses the directory tree from `root_directory`, prints the structure,
    and saves all `.nii.gz` file absolute paths into `output_file`.

    Parameters:
        root_directory (str): The starting point of the directory traversal.
        output_file (str): The file where .nii.gz paths will be stored.
        indent (int): Used internally for pretty-printing the directory tree.
    """
    num_files_saved = 0

    with open(output_file, 'w') as f:
        def traverse(directory, indent):
            nonlocal num_files_saved

            dir_name = os.path.basename(directory) if os.path.basename(directory) else directory
            print("    " * indent + f"[{dir_name}]")

            try:
                items = sorted(os.listdir(directory))
            except PermissionError:
                print("    " * (indent + 1) + "Permission Denied")
                return

            for item in items:
                full_path = os.path.join(directory, item)
                if os.path.isdir(full_path):
                    traverse(full_path, indent + 1)
                elif item.endswith('.nii.gz'):
                    abs_path = os.path.abspath(full_path)
                    print("    " * (indent + 1) + abs_path)
                    f.write(abs_path + "\n")
                    num_files_saved += 1
                    print("    " * (indent + 1) + f"(Saved to file)")

        traverse(root_directory, indent)

    print(f"\n✅ Done: {num_files_saved} .nii.gz file paths saved to '{output_file}'.")


# Example usage
list_and_save_nii_gz_paths(
    r"C:\Users\P096350\Documents\data_prepped",
    "data_paths0111.txt"
)
