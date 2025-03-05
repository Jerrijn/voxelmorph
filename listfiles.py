import os

def list_directory_tree(root_directory, indent=0):
    """
    Recursively lists all directories and .nii.gz files within the given root_directory in a hierarchical structure.
    
    For every .nii.gz file, the function prints the absolute path, ensuring that the full location in the file system is displayed.
    
    Parameters:
        root_directory (str): The absolute or relative path to the root directory.
        indent (int): The current indentation level for pretty-printing (used in recursive calls).
    """
    # Extract the directory name for display; if empty, use the full root_directory.
    dir_name = os.path.basename(root_directory) if os.path.basename(root_directory) else root_directory
    print("    " * indent + f"[{dir_name}]")  # Directories are enclosed in brackets.
    
    try:
        # Obtain a sorted list of directory entries for an orderly (alphabetical) output.
        items = sorted(os.listdir(root_directory))
    except PermissionError:
        print("    " * (indent + 1) + "Permission Denied")
        return
    
    # Process each entry in the current directory.
    for item in items:
        full_path = os.path.join(root_directory, item)
        if os.path.isdir(full_path):
            # If the entry is a directory, recursively traverse its contents.
            list_directory_tree(full_path, indent + 1)
        else:
            # Only process files with the .nii.gz extension.
            if item.endswith('.nii.gz'):
                absolute_path = os.path.abspath(full_path)
                print("    " * (indent + 1) + absolute_path)


def save_file_paths(root_directory, output_file):
    """
    Recursively traverses the directory tree starting at 'root_directory' and writes 
    the absolute path of every .nii.gz file encountered to 'output_file'.

    Parameters:
        root_directory (str): The base directory from which the traversal begins.
        output_file (str): The text file path where the absolute file paths will be saved.
    """
    with open(output_file, 'w') as f:
        # os.walk performs a recursive traversal of the directory tree.
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for filename in filenames:
                # Only include files ending with .nii.gz.
                if filename.endswith('.nii.gz'):
                    abs_path = os.path.abspath(os.path.join(dirpath, filename))
                    f.write(abs_path + "\n")


# Example usage:
# Replace the path below with the actual directory you want to traverse.
list_directory_tree(r"/home/rth/jgmandjes/Voxelmorph/preprocessed")
# Example usage:
# Replace the root_directory below with your target directory.
save_file_paths(r"/home/rth/jgmandjes/Voxelmorph/preprocessed", "file_paths.txt")
