#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data.
All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800, 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# Parse command-line arguments.
parser = argparse.ArgumentParser()

# Data organization parameters.
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='optional atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# Training parameters.
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# Network architecture parameters.
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# Loss hyperparameters.
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                    help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10,
                    help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')

args = parser.parse_args()

# Load training file list.
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# If data is single-channel, add an extra feature axis only if not multichannel.
add_feat_axis = not args.multichannel

# Helper function: convert image to single channel if needed.
def ensure_single_channel(image):
    if image.shape[-1] == 3:
        return np.mean(image, axis=-1, keepdims=True)
    return image

# fixed_generator: wraps an original generator and processes images.
def fixed_generator(original_generator):
    for moving, fixed in original_generator:
        moving = np.stack([ensure_single_channel(np.array(m, dtype=np.float32)) for m in moving], axis=0)
        fixed = np.stack([ensure_single_channel(np.array(f, dtype=np.float32)) for f in fixed], axis=0)
        # Yield a tuple of (inputs, targets). In unsupervised registration, targets can equal inputs.
        yield ([moving, fixed], [moving, fixed])

# data_generator: creates a fresh generator each time it's called.
def data_generator():
    if args.atlas:
        atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                          add_batch_axis=True, add_feat_axis=add_feat_axis)
        orig_gen = vxm.generators.scan_to_atlas(train_files, atlas,
                                                batch_size=args.batch_size,
                                                bidir=args.bidir,
                                                add_feat_axis=add_feat_axis)
    else:
        orig_gen = vxm.generators.scan_to_scan(train_files,
                                               batch_size=args.batch_size,
                                               bidir=args.bidir,
                                               add_feat_axis=add_feat_axis)
    return fixed_generator(orig_gen)

def main():
    # For debugging: get one batch and print shapes.
    gen_debug = data_generator()
    (inputs_debug, targets_debug) = next(gen_debug)
    # print("Moving shapes:", [m.shape for m in inputs_debug[0]])
    # print("Fixed shapes:", [f.shape for f in inputs_debug[1]])
    # print("Combined moving shape:", np.array(inputs_debug[0]).shape)
    # print("Combined fixed shape:", np.array(inputs_debug[1]).shape)

    # Extract input shape and number of features from the first moving image.
    sample_shape = inputs_debug[0][0].shape
    inshape = sample_shape[1:-1]
    nfeats = sample_shape[-1]

    # Prepare model folder.
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # TensorFlow device handling.
    device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
    assert np.mod(args.batch_size, nb_devices) == 0, (
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)
    )

    # Set U-Net architecture parameters.
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    # Define model checkpoint save path.
    save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

    # Build the VoxelMorph model.
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats
    )
    if args.load_weights:
        model.load_weights(args.load_weights)

    # Set up the image similarity loss.
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE(args.image_sigma).loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    if args.bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    if args.use_probs:
        flow_shape = model.outputs[-1].shape[1:-1]
        losses += [vxm.losses.KL(args.kl_lambda, flow_shape).loss]
    else:
        losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.lambda_weight]

    # Multi-GPU support.
    if nb_devices > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            save_filename, save_freq=20 * args.steps_per_epoch)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=losses, loss_weights=weights)
    model.save_weights('my_model_weights.weights.h5')

    # Define the expected output structure.
    output_signature = (
        (tf.TensorSpec(shape=(None, *inshape, nfeats), dtype=tf.float32),
         tf.TensorSpec(shape=(None, *inshape, nfeats), dtype=tf.float32)),
        (tf.TensorSpec(shape=(None, *inshape, nfeats), dtype=tf.float32),
         tf.TensorSpec(shape=(None, *inshape, nfeats), dtype=tf.float32))
    )

    # Create the dataset using data_generator and prefetch a small buffer.
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    ).prefetch(1)

    # Train the model.
    model.fit(dataset,
              initial_epoch=args.initial_epoch,
              epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              callbacks=[save_callback],
              verbose=1)

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
