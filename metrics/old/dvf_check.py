import os
import nibabel as nib
import numpy as np
import argparse
import matplotlib.pyplot as plt

def sum_dvfs(dvf_folder):
    # Generate file names for 2 through 100
    dvf_files = [f"DVF_20190822_ZKN__P20_MR1_MMT_25_2_S_F{num-1}_M{num}.nii" for num in range(2, 101)]

    # Initialize sum_DVF with the first available DVF
    first_dvf_path = os.path.join(dvf_folder, dvf_files[0])
    
    if not os.path.exists(first_dvf_path):
        print(f"Error: {first_dvf_path} not found.")
        return

    sum_DVF = nib.load(first_dvf_path).get_fdata()

    # Iterate through the remaining files and sum them
    for file in dvf_files[1:]:
        dvf_path = os.path.join(dvf_folder, file)
        if os.path.exists(dvf_path):
            sum_DVF += nib.load(dvf_path).get_fdata()
        else:
            print(f"Warning: {dvf_path} not found. Skipping.")

    # Save the summed DVF for comparison
    sum_DVF /= 100
    sum_nifti = nib.Nifti1Image(sum_DVF, affine=nib.load(first_dvf_path).affine)
    output_path = os.path.join(dvf_folder, "summed_dvf.nii")
    nib.save(sum_nifti, output_path)
    print(f"Summed DVF saved at {output_path}")

    # Generate and save the vector field image for slice 43
    save_vector_field_image(sum_DVF, dvf_folder)
    # Generate and save the heatmap for slice 43
    save_heatmap(sum_DVF, dvf_folder)

def save_vector_field_image(sum_DVF, dvf_folder):
    print(f"Summed DVF shape: {sum_DVF.shape}")
    slice_43 = sum_DVF[:, :, 43, 0, :2]  # Remove the extra singleton dimension

    if slice_43.shape[-1] != 2:
        print("Error: Expected DVF to have at least two components for visualization.")
        return

    x_component = slice_43[:, :, 0]
    y_component = slice_43[:, :, 1]

    # Create a grid for quiver plot
    X, Y = np.meshgrid(np.arange(x_component.shape[1]), np.arange(x_component.shape[0]))

    # Plot the vector field with green arrows
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, x_component, y_component, angles="xy", scale_units="xy", scale=1, color='g')  # Green vectors
    plt.title("DVF Vector Field - Slice 43")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Match medical imaging orientation
    vector_field_path = os.path.join(dvf_folder, "dvf_slice_43_vectors.png")
    plt.savefig(vector_field_path)
    plt.close()
    print(f"Vector field for slice 43 saved at {vector_field_path}")

def save_heatmap(sum_DVF, dvf_folder):
    print(f"Generating heatmap for slice 43")
    
    # Compute displacement magnitude for slice 43
    slice_43 = sum_DVF[:, :, 43, 0, :3]  # Extract X, Y, Z components
    displacement_magnitude = np.linalg.norm(slice_43, axis=-1)  # Compute magnitude

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(displacement_magnitude, cmap='jet', interpolation='nearest')  # Use 'hot' colormap for heatmap
    plt.colorbar(label="Displacement Magnitude")
    plt.title("DVF Displacement Heatmap - Slice 43")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Match medical imaging orientation
    heatmap_path = os.path.join(dvf_folder, "dvf_slice_43_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap for slice 43 saved at {heatmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum DVF files from 2 to 100 in a given folder, visualize vector field and heatmap for slice 43.")
    parser.add_argument("dvf_folder", type=str, help="Path to the folder containing DVF files.")
    args = parser.parse_args()

    sum_dvfs(args.dvf_folder)
