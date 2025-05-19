#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
import random
import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import voxelmorph as vxm  # nopep8
import re
from collections import defaultdict

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

class ScanToScanDataset(Dataset):
    """
    Dataset that returns pairs of scans from the same patient and same MR sequence.
    Groups images by (patient_id, MR_sequence_id) and samples two distinct scans from each group.
    Shuffle_pairs controls random vs. deterministic pairing.
    """
    def __init__(self,
                 image_paths,
                 seg_paths=None,
                 use_segs=False,
                 add_feat_axis=True,
                 bidir=False,
                 shuffle_pairs=True):
        assert isinstance(image_paths, list) and len(image_paths) > 0, \
            "image_paths must be a non-empty list"
        if use_segs:
            assert seg_paths is not None and len(seg_paths) == len(image_paths), \
                "seg_paths must be provided and match image_paths length when use_segs=True"

        self.image_paths   = image_paths
        self.seg_paths     = seg_paths
        self.use_segs      = use_segs
        self.add_feat_axis = add_feat_axis
        self.bidir         = bidir
        self.shuffle_pairs = shuffle_pairs

        # Build groups by (patient_id, MR_sequence)
        self.grp_to_images = defaultdict(list)
        self.grp_to_segs   = defaultdict(list) if use_segs else None

        for i, path in enumerate(self.image_paths):
            pid = self._extract_patient_id(path)
            seq = self._extract_sequence_id(path)
            key = (pid, seq)
            self.grp_to_images[key].append(path)
            if use_segs:
                self.grp_to_segs[key].append(self.seg_paths[i])

        # Only keep groups with ≥2 scans
        self.valid_keys = [k for k, imgs in self.grp_to_images.items() if len(imgs) > 1]
        assert self.valid_keys, "No patient/sequence groups with ≥2 scans found."

    def _extract_patient_id(self, path):
        match = re.search(r'pt\d+', path)
        assert match, f"Could not extract patient ID from path: {path}"
        return match.group()

    def _extract_sequence_id(self, path):
        match = re.search(r'MR(\d+)', path)
        assert match, f"Could not extract MR sequence ID from path: {path}"
        return match.group(1)

    def __len__(self):
        total = sum(len(imgs) for imgs in self.grp_to_images.values())
        assert total > 0, "Dataset has no images"
        return total

    def __getitem__(self, idx):
        if self.shuffle_pairs:
            # random sampling
            key  = random.choice(self.valid_keys)
            imgs = self.grp_to_images[key]
            moving_path, fixed_path = random.sample(imgs, 2)
        else:
            # deterministic: cycle through groups by idx, take first two sorted scans
            key  = self.valid_keys[idx % len(self.valid_keys)]
            imgs = sorted(self.grp_to_images[key])
            moving_path, fixed_path = imgs[0], imgs[1]

        if self.use_segs:
            segs = self.grp_to_segs[key]
            seg_moving = segs[imgs.index(moving_path)]
            seg_fixed  = segs[imgs.index(fixed_path)]

        # Load volumes
        moving = vxm.py.utils.load_volfile(
            moving_path, add_batch_axis=False, add_feat_axis=self.add_feat_axis)
        fixed = vxm.py.utils.load_volfile(
            fixed_path,  add_batch_axis=False, add_feat_axis=self.add_feat_axis)

        # (rest of loading, checks, tensor conversion, identical to before…)
        inputs = [moving, fixed]
        outputs = [fixed]
        if self.bidir:
            outputs.append(moving)
        if self.use_segs:
            mseg = vxm.py.utils.load_volfile(seg_moving, add_batch_axis=False, add_feat_axis=True)
            fseg = vxm.py.utils.load_volfile(seg_fixed,  add_batch_axis=False, add_feat_axis=True)
            inputs.extend([mseg])
            outputs.extend([fseg])
            if self.bidir:
                outputs.append(mseg)

        inputs = [torch.from_numpy(x).float().permute(3,0,1,2) for x in inputs]
        outputs = [torch.from_numpy(x).float().permute(3,0,1,2) for x in outputs]
        return inputs, outputs


class ScanToAtlasDataset(Dataset):
    def __init__(self, image_paths, atlas_path, seg_paths=None, use_segs=False, add_feat_axis=True):
        self.image_paths = image_paths
        self.atlas = vxm.py.utils.load_volfile(atlas_path, add_batch_axis=False, add_feat_axis=add_feat_axis)
        self.seg_paths = seg_paths
        self.use_segs = use_segs
        self.add_feat_axis = add_feat_axis

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        moving = vxm.py.utils.load_volfile(self.image_paths[idx], add_batch_axis=False, add_feat_axis=self.add_feat_axis)
        inputs = [moving, self.atlas]
        outputs = [self.atlas]

        if self.use_segs:
            moving_seg = vxm.py.utils.load_volfile(self.seg_paths[idx], add_batch_axis=False, add_feat_axis=True)
            inputs.append(moving_seg)
            outputs.append(self.atlas)  # atlas seg is not provided — assume label overlap with intensity?

        return [torch.from_numpy(x).float().permute(3, 0, 1, 2) for x in inputs], \
               [torch.from_numpy(x).float().permute(3, 0, 1, 2) for x in outputs]

def format_params(args):
    components = []

    if args.epochs != 1000:
        components.append(f"ep{args.epochs}")
    if args.batch_size != 1:
        components.append(f"bs{args.batch_size}")
    if args.lr != 1e-4:
        components.append(f"lr{args.lr:.0e}")
    if args.image_loss != 'mse':
        components.append(args.image_loss)
    if args.bidir:
        components.append("bidir")
    if args.use_seg:
        components.append("seg")
    if args.use_mi:
        components.append("mi")
    if args.use_bending:
        components.append("bend")
    if args.use_ic:
        components.append("ic")

    return "_".join(components)


def main():
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    # Add new loss hyperparameters
    parser.add_argument('--use-mi', action='store_true',
                        help='use mutual information loss')
    parser.add_argument('--lambda-mi', type=float, default=0.1,
                        help='weight of mutual information loss (default: 0.1)')
    parser.add_argument('--use-bending', action='store_true',
                        help='use bending energy loss')
    parser.add_argument('--lambda-bending', type=float, default=0.01,
                        help='weight of bending energy loss (default: 0.01)')
    parser.add_argument('--use-ic', action='store_true',
                        help='use inverse consistency loss')
    parser.add_argument('--lambda-ic', type=float, default=0.05,
                        help='weight of inverse consistency loss (default: 0.05)')

    # Add command line arguments for segmentation maps
    parser.add_argument('--seg-list', help='line-separated list of segmentation files')
    parser.add_argument('--seg-prefix', help='optional segmentation file prefix')
    parser.add_argument('--seg-suffix', help='optional segmentation file suffix')
    parser.add_argument('--use-seg', action='store_true', help='use segmentation maps as additional input')
    parser.add_argument('--seg-weight', type=float, default=0.5, 
                        help='weight for segmentation-based loss (default: 0.5)')
    parser.add_argument('--shuffle', action='store_true',
                    help='Only sample image pairs from the same patient (default: False)')
    parser.add_argument(
        '--no-shuffle-pairs', dest='shuffle_pairs', action='store_false',
        help='Disable random sampling of scan-pairs (use deterministic ordering)')
    parser.set_defaults(shuffle_pairs=True)

    args = parser.parse_args()
    assert args.batch_size > 0, "batch-size must be > 0"
    assert args.epochs > 0, "epochs must be > 0"
    assert args.lr > 0, "learning rate must be > 0"

    bidir = args.bidir

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                            suffix=args.img_suffix)
    assert len(train_files) > 0, 'Could not find any training data.'

    # Load segmentation files if specified
    seg_files = None
    if args.use_seg and args.seg_list:
        seg_files = vxm.py.utils.read_file_list(args.seg_list, prefix=args.seg_prefix,
                                            suffix=args.seg_suffix)
        assert len(seg_files) == len(train_files), 'Number of segmentation files must match image files.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel

    from torch.utils.data import DataLoader
    image_paths = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    seg_paths = None
    if args.use_seg and args.seg_list:
        seg_paths = vxm.py.utils.read_file_list(args.seg_list, prefix=args.seg_prefix, suffix=args.seg_suffix)

    if args.atlas:
        dataset = ScanToAtlasDataset(
            image_paths=image_paths,
            atlas_path=args.atlas,
            seg_paths=seg_paths,
            use_segs=args.use_seg,
            add_feat_axis=not args.multichannel
        )
    else:
        dataset = ScanToScanDataset(
            image_paths=image_paths,
            seg_paths=seg_paths,
            use_segs=args.use_seg,
            add_feat_axis=not args.multichannel,
            bidir=args.bidir,
            shuffle_pairs=args.shuffle_pairs
        )

    generator = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # extract shape from sampled input
    sample_input, _ = next(iter(generator))  # One batch
    inshape = sample_input[0].shape[2:] 

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        if args.use_seg:
            # Create a model that accepts segmentation maps as additional input
            model = vxm.networks.VxmDenseSegmentation(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=True,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize,
                seg_weight=args.seg_weight
            )
        else:
            # Standard model without segmentation
            model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize
            )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    conv3d_modules = (model.modules() if not isinstance(model, torch.nn.DataParallel)
                  else model.module.modules())
    for m in conv3d_modules:
        # Voxelmorph’s final 3‐channel Conv3d predicts the displacement field
        if isinstance(m, torch.nn.Conv3d) and m.out_channels == 3:
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.to(device)
    model.train()
    from datetime import datetime

    # Custom base log directory
    log_base = r"C:\Users\P096350\Documents\logs"

    # Timestamp for unique run folders
    param_str = format_params(args)
    timestamp = datetime.now().strftime('%m-%d_%H-%M')
    run_name = f"{timestamp}_{param_str}"
    log_dir = os.path.join(log_base, run_name)

    # Create writer
    writer = SummaryWriter(log_dir=log_dir)
    print("✅ Writer created:", log_dir)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]


    loss_names = ['image_loss']
    if bidir:
        loss_names = ['image_loss_fwd', 'image_loss_bwd']
    loss_names += ['grad_loss']
    if args.use_seg:
        loss_names += ['dice_loss']

    # Define loss weights from command line arguments
    lambda_mi = args.lambda_mi if args.use_mi else 0.0
    lambda_bend = args.lambda_bending if args.use_bending else 0.0
    # lambda_ic = args.lambda_ic if args.use_ic else 0.0

    # NEW: Instantiate new loss functions
    mi_loss = vxm.losses.MutualInformation(bins=32, sigma=0.02, eps=1e-10, device=device) if args.use_mi else None
    bend_loss = vxm.losses.BendingEnergyLoss() if args.use_bending else None
    inv_consistency_loss = vxm.losses.InverseConsistencyLoss() if args.use_ic else None


    # Add to the losses section
    if args.use_seg:
        # Add Dice loss for segmentation maps
        dice_loss = vxm.losses.Dice().loss
        losses += [dice_loss]
        weights += [args.seg_weight]

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            filename = f"{run_name}-ep{epoch}.pt"
            model.save(os.path.join(model_dir, filename))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        
        # Add tracking for additional losses
        epoch_mi_loss = [] if args.use_mi else None
        epoch_bend_loss = [] if args.use_bending else None
        epoch_ic_loss = [] if args.use_ic else None


        for step, (inputs, y_true) in enumerate(generator):
            if step >= args.steps_per_epoch:
                break

            step_start_time = time.time()

            # Send to GPU
            inputs = [d.to(device, non_blocking=True) for d in inputs]
            y_true = [d.to(device, non_blocking=True) for d in y_true]

            # Log image stats for debugging
            # print(f"[{epoch}:{step}] Moving image min/max: {inputs[0].min().item():.4f} / {inputs[0].max().item():.4f}")
            # print(f"[{epoch}:{step}] Fixed image min/max:  {inputs[1].min().item():.4f} / {inputs[1].max().item():.4f}")

            # forward
            y_pred = model(*inputs)
            assert isinstance(y_pred, (list, tuple)), "Model output must be list/tuple"
            if args.bidir:
                expected = 4
                names = ["warped_fwd", "flow_fwd", "warped_bwd", "flow_bwd"]
            else:
                expected = 2
                names = ["warped", "flow"]

            assert len(y_pred) == expected, (
                f"Expected {expected} outputs ({', '.join(names)}), got {len(y_pred)}"
            )

            def normalize_flow(flow, shape):
                # flow: (B, 3, D, H, W), shape=(D,H,W)
                D,H,W = shape
                scale = torch.tensor([W-1, H-1, D-1], device=flow.device)[None,:,None,None,None]
                return flow / (scale * 0.5)

            # assume y_pred = [warped_fwd, flow_fwd, warped_bwd, flow_bwd]
            if bidir:
                assert len(y_pred) == 4, f"Expected 4 outputs in bidir mode, got {len(y_pred)}"
                warped_fwd, flow_fwd_raw, warped_bwd, flow_bwd_raw = y_pred

                flow_fwd = normalize_flow(flow_fwd_raw, inputs[0].shape[2:])
                flow_bwd = normalize_flow(flow_bwd_raw, inputs[0].shape[2:])
                y_pred = [warped_fwd, flow_fwd, warped_bwd, flow_bwd]

            else:
                assert len(y_pred) == 2, f"Expected 2 outputs in unidir mode, got {len(y_pred)}"
                warped, flow_raw = y_pred
                flow = normalize_flow(flow_raw, inputs[0].shape[2:])
                y_pred = [warped, flow]


            # Determine bidirectional mode based on output length
            # is_bidir = len(y_pred) == 4

            # # Extract flows
            # if is_bidir:
            #     flow_fwd = y_pred[2]
            #     flow_bwd = y_pred[3]
            # else:
            #     flow_fwd = y_pred[1]
            #     flow_bwd = None  # not available

            # # Log flow magnitude histograms
            # writer.add_histogram('flow_fwd/magnitude', torch.norm(flow_fwd, dim=1), epoch * args.steps_per_epoch + step)
            # if flow_bwd is not None:
            #     writer.add_histogram('flow_bwd/magnitude', torch.norm(flow_bwd, dim=1), epoch * args.steps_per_epoch + step)

            # Compute and log Jacobian determinants
            # import torch.nn.functional as F

            # def compute_jacobian_determinant(flow):
            #     """
            #     Compute the Jacobian determinant of a displacement field using finite differences.
            #     Assumes input shape: (B, 3, D, H, W)
            #     """
            #     # Compute gradients along each axis
            #     dx = flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1]
            #     dy = flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1]
            #     dz = flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1]

            #     # Construct Jacobian matrix J with shape (B, D-1, H-1, W-1, 3, 3)
            #     J = torch.stack((dx, dy, dz), dim=-1)  # (B, 3, D-1, H-1, W-1, 3)
            #     J = J.permute(0, 2, 3, 4, 1, 5)        # -> (B, D-1, H-1, W-1, 3, 3)

            #     # Compute the determinant of J
            #     jac_det = torch.linalg.det(J)         # (B, D-1, H-1, W-1)

            #     return jac_det


            # jac_fwd = compute_jacobian_determinant(flow_fwd)
            # writer.add_histogram('jacobian_fwd/values', jac_fwd, epoch * args.steps_per_epoch + step)
            # writer.add_scalar('jacobian_fwd/folding_ratio', (jac_fwd < 0).float().mean().item(), epoch * args.steps_per_epoch + step)

            # if flow_bwd is not None:
            #     jac_bwd = compute_jacobian_determinant(flow_bwd)
            #     writer.add_histogram('jacobian_bwd/values', jac_bwd, epoch * args.steps_per_epoch + step)
            #     writer.add_scalar('jacobian_bwd/folding_ratio', (jac_bwd < 0).float().mean().item(), epoch * args.steps_per_epoch + step)


            # compute loss
            loss = 0
            loss_list = []
            # image loss forward
            curr_loss = losses[0](y_true[0], y_pred[0]) * weights[0]
            loss_list.append(curr_loss.item())
            loss += curr_loss

            # image loss backward (only if bidir is enabled)
            if bidir:
                curr_loss = losses[1](y_true[1], y_pred[1]) * weights[1]
                loss_list.append(curr_loss.item())
                loss += curr_loss


            # deformation loss (only uses flow field)
            grad_idx = 2 if bidir else 1  # because in unidir: y_pred = [warped, flow]
            curr_loss = losses[grad_idx](None, y_pred[grad_idx]) * weights[grad_idx]
            loss_list.append(curr_loss.item())
            loss += curr_loss

            # inverse consistency loss
            if args.use_ic and bidir:
                raw_ic = inv_consistency_loss(y_pred[2], y_pred[3])        # sum over voxels
                num_voxels = torch.tensor(y_pred[2].numel(), device=raw_ic.device)
                loss_ic = raw_ic / num_voxels                               # now an average per-voxel
                loss += args.lambda_ic * loss_ic
                epoch_ic_loss.append(loss_ic.item())

            # extra losses
            if args.use_mi:
                loss_mi = mi_loss.loss(y_true[0], y_pred[0])
                loss += lambda_mi * loss_mi
                epoch_mi_loss.append(loss_mi.item())

            if args.use_bending:
                loss_bend = bend_loss(y_pred[1])
                loss += lambda_bend * loss_bend
                epoch_bend_loss.append(loss_bend.item())

            # Visualization: log middle slice images
            if step == 0:
                mid_slice = inputs[0].shape[-1] // 2
                writer.add_image('moving', inputs[0][0, 0, :, :, mid_slice], epoch, dataformats='HW')
                writer.add_image('fixed', inputs[1][0, 0, :, :, mid_slice], epoch, dataformats='HW')
                writer.add_image('warped', y_pred[0][0, 0, :, :, mid_slice], epoch, dataformats='HW')

            assert not torch.isnan(loss), "Loss is NaN"

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        
        # Add additional loss info to the print statement
        additional_losses = []
        if args.use_mi:
            additional_losses.append('MI: %.4e' % np.mean(epoch_mi_loss))
        if args.use_bending:
            additional_losses.append('Bend: %.4e' % np.mean(epoch_bend_loss))
        
        additional_info = ''
        if additional_losses:
            additional_info = '  Additional losses: ' + ', '.join(additional_losses)
        
        loss_info = 'loss: %.4e  (%s)%s' % (np.mean(epoch_total_loss), losses_info, additional_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

        writer.add_scalar('Loss/Total', np.mean(epoch_total_loss), epoch)
        for i, name in enumerate(loss_names):
            writer.add_scalar(f'Loss/{name}', np.mean(epoch_loss, axis=0)[i], epoch)

        if args.use_mi:
            writer.add_scalar('Loss/MutualInformation', np.mean(epoch_mi_loss), epoch)
        if args.use_bending:
            writer.add_scalar('Loss/BendingEnergy', np.mean(epoch_bend_loss), epoch)
        if args.use_ic:
            writer.add_scalar('Loss/InverseConsistency', np.mean(epoch_ic_loss), epoch)


    # final model save
    final_filename = f"{run_name}_final.pt"
    model.save(os.path.join(model_dir, final_filename))
    writer.close() 


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()