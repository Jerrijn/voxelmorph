#!/usr/bin/env python
import nibabel as nib
import numpy as np
import argparse

def generate_noise_nifti(reference_file, output_file):
    # Load the reference NIfTI file to get the image shape and affine.
    ref_img = nib.load(reference_file)
    data_shape = ref_img.shape  # e.g., (X, Y, Z)
    affine = ref_img.affine
    header = ref_img.header

    # Generate a 3D volume of random noise.
    # Here we use a standard normal distribution. 
    noise_data = np.random.randn(*data_shape)

    # Create a new NIfTI image using the same affine and header as the reference.
    noise_img = nib.Nifti1Image(noise_data, affine, header)

    # Save the noise image to disk.
    nib.save(noise_img, output_file)
    print(f"Noise NIfTI saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a 3D NIfTI MRI scan filled with random noise "
                    "that matches the dimensions of a reference file."
    )
    parser.add_argument('reference', help="Path to the reference NIfTI file")
    parser.add_argument('output', help="Path for the output noise NIfTI file")
    args = parser.parse_args()

    generate_noise_nifti(args.reference, args.output)
