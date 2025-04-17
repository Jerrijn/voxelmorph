import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def process_vector_field(vector_field_path):
    """
    Load and process the 3D displacement field, computing the motion map.
    """
    # Load data and remove singleton dimensions
    vec_img = nib.load(vector_field_path)
    vec_data = np.array(vec_img.get_fdata())  # Ensure data is in array format
    vec_data = np.squeeze(vec_data)  # Remove singleton dimensions
    print("Vector field shape after squeeze:", vec_data.shape)
    
    # Compute motion map: Euclidean norm over the last dimension
    motion_map = np.linalg.norm(vec_data, axis=-1)
    motion_map = np.squeeze(motion_map)  # Remove any singleton dimensions again
    print("Motion map shape after norm & squeeze:", motion_map.shape)
    
    return motion_map

def save_motion_map(motion_map, output_path="motion_map.png"):
    """
    Save a single slice of the motion map as an image.
    """
    slice_idx = 43  # Fixed slice number
    plt.figure(figsize=(6, 6))
    img_display = plt.imshow(motion_map[:, :, slice_idx], cmap='jet', origin='lower')
    plt.title(f"Motion Map - Slice {slice_idx}")
    cb = plt.colorbar(img_display, fraction=0.046, pad=0.04)
    cb.set_label('Motion magnitude')
    
    plt.savefig(output_path)
    print(f"Motion map slice saved as {output_path}")
    plt.close()

def save_motion_volume_histogram(motion_map, output_path="motion_volume_histogram.png"):
    """
    Save the Motion-Volume Histogram (MVH) as an image.
    """
    motion_vals = motion_map.flatten()
    motion_vals_sorted = np.sort(motion_vals)
    N = len(motion_vals_sorted)
    fraction = (N - np.arange(N)) / N * 100.0  # Fraction in percent
    
    plt.figure(figsize=(6, 6))
    plt.plot(motion_vals_sorted, fraction, color='green', linewidth=2)
    plt.title('Motion-Volume Histogram (MVH)')
    plt.xlabel('Motion (units)')
    plt.ylabel('Volume (%)')
    plt.xlim([0, 10])  # Limit x-axis
    plt.ylim([0, 100])
    
    plt.savefig(output_path)
    print(f"Motion-Volume Histogram saved as {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Motion-Volume Histogram (MVH).")
    parser.add_argument("vector_field_path", type=str, help="Path to the 3D displacement field NIfTI file.")
    args = parser.parse_args()
    
    # Process vector field and save visualizations
    motion_map = process_vector_field(args.vector_field_path)
    save_motion_map(motion_map, output_path="motion_map.png")
    save_motion_volume_histogram(motion_map, output_path="motion_volume_histogram.png")

if __name__ == "__main__":
    main()
