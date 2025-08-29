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

from voxelmorph.torch.losses import MSE, NCC, jac0_loss, SurfaceLoss, compute_signed_distance_map

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

class ScanToScanDataset(Dataset):
    """
    Dataset that returns pairs of scans from the same patient and same MR sequence,
    plus optional segmentation‐derived weight masks.
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
        # 1) pick a pair of scans (moving/fixed)
        if self.shuffle_pairs:
            key  = random.choice(self.valid_keys)
            imgs = self.grp_to_images[key]
            moving_path, fixed_path = random.sample(imgs, 2)
        else:
            key  = self.valid_keys[idx % len(self.valid_keys)]
            imgs = sorted(self.grp_to_images[key])
            moving_path, fixed_path = imgs[0], imgs[1]
        #print(f"[Dataset] idx={idx} | key={key} | moving={os.path.basename(moving_path)} | fixed={os.path.basename(fixed_path)}")
        # 2) if using segs, pick the corresponding seg files
        if self.use_segs:
            segs = self.grp_to_segs[key]
            seg_moving = segs[imgs.index(moving_path)]
            seg_fixed  = segs[imgs.index(fixed_path)]

        # 3) load volumes (intensity)  
        moving = vxm.py.utils.load_volfile(
            moving_path, add_batch_axis=False, add_feat_axis=self.add_feat_axis)
        fixed  = vxm.py.utils.load_volfile(
            fixed_path,  add_batch_axis=False, add_feat_axis=self.add_feat_axis)

        # 4) load segmentation masks and build weight masks
        if self.use_segs:
            # Assume seg maps are binary already, threshold at 0.5
            mseg = vxm.py.utils.load_volfile(
                seg_moving, add_batch_axis=False, add_feat_axis=True)
            fseg = vxm.py.utils.load_volfile(
                seg_fixed,  add_batch_axis=False, add_feat_axis=True)

            # Binarize: voxel value >0.5 → 1.0
            mseg_bin = (mseg > 0.5).astype(np.float32)
            fseg_bin = (fseg > 0.5).astype(np.float32)
            
            # Build weight: 1.0 inside segmentation, 0.1 outside
            w_moving = mseg_bin * 1.0 + (1.0 - mseg_bin) * 0.1  # shape: (H,W,D,1)
            w_fixed  = fseg_bin * 1.0 + (1.0 - fseg_bin) * 0.1  # shape: (H,W,D,1)

            # Now convert to torch.Tensor with channel‐first ordering
            # We want shape (1, D, H, W) for each mask
            mseg_bin = np.transpose(mseg_bin, (3, 0, 1, 2))  # (1, D, H, W)
            w_moving = torch.from_numpy(mseg_bin).float()
            fseg_bin = np.transpose(fseg_bin, (3, 0, 1, 2))  # (1, D, H, W)
            w_fixed = torch.from_numpy(fseg_bin).float()

        # 5) build inputs/outputs lists as before
        inputs  = [moving, fixed]
        outputs = [fixed]
        if self.bidir:
            outputs.append(moving)

        # 6) convert intensity volumes to torch.Tensor with channel‐first
        inputs  = [torch.from_numpy(x).float().permute(3, 0, 1, 2) for x in inputs]
        outputs = [torch.from_numpy(x).float().permute(3, 0, 1, 2) for x in outputs]

        if self.use_segs:
            # Return the two images (moving/fixed), the target(s), and weight masks
            # For unidirectional, masks = [w_fixed]; for bidir, masks = [w_fixed, w_moving]
            if self.bidir:
                weights = [w_fixed, w_moving]
            else:
                weights = [w_fixed]
            return inputs, outputs, weights, [mseg_bin, fseg_bin]

        # If not using segmentation, just return inputs and outputs
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

def flow_stats(flow):
    """
    flow : tensor (B, 3, D, H, W)  – raw flow in voxel units
    returns dict with mean, rms, p95, max over the whole batch
    """
    mag = torch.linalg.norm(flow, dim=1)             # (B, D, H, W)
    mean = mag.mean().item()
    rms  = torch.sqrt((mag ** 2).mean()).item()
    p95  = torch.quantile(mag, 0.95).item()
    vmax = mag.max().item()
    return dict(mean=mean, rms=rms, p95=p95, vmax=vmax)

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
    parser.add_argument('--lambda-mi', type=float, default=0.2,
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
    parser.add_argument('--use-jac0', action='store_true', help='Add JAC0 (negative Jacobian) penalty loss')
    parser.add_argument('--lambda-jac0', type=float, default=0.1, help='Weight for JAC0 loss')
    parser.set_defaults(shuffle_pairs=True)

    parser.add_argument('--use-surface', action='store_true',
    help='Use surface loss (boundary loss) for segmentation-guided registration')
    parser.add_argument('--lambda-surface', type=float, default=0.1,
        help='Weight for surface loss (default: 0.1)')


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
    batch0 = next(iter(generator))
    if args.use_seg:
        sample_input = batch0[0]
    else:
        sample_input, _ = batch0
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
    if args.use_seg:
        inshape = next(iter(generator))[0][0].shape[2:]  # (D, H, W)
        seg_transformer = vxm.layers.SpatialTransformer(inshape, mode='nearest').to(device)



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
        image_loss_func = NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = MSE().loss
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
        if args.use_surface:
            surface_loss = SurfaceLoss()
            losses += [surface_loss]
            weights += [args.lambda_surface]
            loss_names += ['surface_loss']

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            filename = f"{run_name}-ep{epoch}.pt"
            model.save(os.path.join(model_dir, filename))

        epoch_loss = []
        epoch_total_loss = []
        epoch_flow = [] 
        epoch_step_time = []
        
        # Add tracking for additional losses
        epoch_mi_loss = [] if args.use_mi else None
        epoch_bend_loss = [] if args.use_bending else None
        epoch_ic_loss = [] if args.use_ic else None


        for step, batch in enumerate(generator):
            if step >= args.steps_per_epoch:
                break

            if args.use_seg:
                inputs, y_true, weight_list, seg_pair = batch
            
                moving_seg = seg_pair[0].to(device, non_blocking=True)  # binary
                fixed_seg  = seg_pair[1].to(device, non_blocking=True)  # binary
            else:
                inputs, y_true = batch
                weight_list = None
            step_start_time = time.time()

            # Send to GPU
            inputs = [d.to(device, non_blocking=True) for d in inputs]
            y_true = [d.to(device, non_blocking=True) for d in y_true]

            if args.use_seg:
                # weight_list[0] = w_fixed, weight_list[1] = w_moving (if bidir)
                weight_list = [m.to(device, non_blocking=True) for m in weight_list]

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

                flow = normalize_flow(flow_fwd_raw, inputs[0].shape[2:])
                flow_bwd = normalize_flow(flow_bwd_raw, inputs[0].shape[2:])
                y_pred = [warped_fwd, flow, warped_bwd, flow_bwd]

            else:
                assert len(y_pred) == 2, f"Expected 2 outputs in unidir mode, got {len(y_pred)}"
                warped, flow_raw = y_pred
                flow = normalize_flow(flow_raw, inputs[0].shape[2:])
                y_pred = [warped, flow]
            
            epoch_flow.append(flow_stats(flow))


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
            loss = 0.0
            loss_list = []

            # 1) Image‐reconstruction loss (weighted if args.use_seg)
            if args.use_seg:
                # w_fixed = weight_list[0], always present if use_seg=True
                w_fixed = weight_list[0]  # shape = (B,1,D,H,W)

                # FORWARD direction:
                # y_true[0] is the fixed image; y_pred[0] is warped→fixed
                if args.image_loss == 'ncc':
                    curr_loss = losses[0](y_true[0], y_pred[0])
                else:
                    curr_loss = losses[0](y_true[0], y_pred[0], w_fixed) * weights[0]
                loss_list.append(curr_loss.item())
                loss += curr_loss

                if bidir:
                    # w_moving = weight_list[1], provided only if bidir & use_seg
                    w_moving = weight_list[1]

                    # y_true[1] is moving image; y_pred[2] is warped←moving
                    curr_loss = losses[1](y_true[1], y_pred[2], w_moving) * weights[1]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

                # ➕ Segmentation-guided registration using Dice loss
                if args.seg_weight > 0:
                    with torch.no_grad():
                        flow_for_seg = y_pred[1]  # (B, 3, D, H, W)
                        flow_for_seg = torch.nn.functional.interpolate(
                            flow_for_seg, size=moving_seg.shape[2:], mode='trilinear', align_corners=True
                        )
                        warped_seg = seg_transformer(moving_seg, flow_for_seg)
                    dice_loss_val = vxm.losses.Dice().loss(fixed_seg, warped_seg) * args.seg_weight
                    loss += dice_loss_val

                    if 'epoch_dice_loss' not in locals():
                        epoch_dice_loss = []
                    epoch_dice_loss.append(dice_loss_val.item())
                
                # ➕ Surface Loss (only if enabled)
                    if args.use_surface:
                        batch_size = fixed_seg.shape[0]
                        surface_loss_val = 0.0

                        for i in range(batch_size):
                            # Get the correct segmentation path for each sample in the batch
                            seg_idx = (step * batch_size + i) % len(seg_paths)
                            fixed_seg_path = seg_paths[seg_idx]

                            base_dir = os.path.dirname(fixed_seg_path)
                            base_name = os.path.splitext(os.path.basename(fixed_seg_path))[0]
                            dist_dir = os.path.join(base_dir, 'distanceMaps')
                            os.makedirs(dist_dir, exist_ok=True)
                            dist_path = os.path.join(dist_dir, f"{base_name}_distanceMap.npy")

                            # Compute or load distance map
                            if os.path.exists(dist_path):
                                dist_map = np.load(dist_path)
                            else:
                                dist_map = compute_signed_distance_map(fixed_seg[i], dist_path)
                            dist_map_tensor = torch.from_numpy(dist_map).unsqueeze(0).unsqueeze(0).to(device)

                            # Use warped_seg for each batch item if it's batched (likely is: (B, 1, D, H, W))
                            surface_loss_val += surface_loss(warped_seg[i:i+1], dist_map_tensor)

                        # Average over batch and multiply by lambda
                        surface_loss_val = (surface_loss_val / batch_size) * args.lambda_surface
                        loss += surface_loss_val

                        if 'epoch_surface_loss' not in locals():
                            epoch_surface_loss = []
                        epoch_surface_loss.append(surface_loss_val.item())

            else:
                # No segmentation weighting → plain MSE or NCC
                curr_loss = losses[0](y_true[0], y_pred[0]) * weights[0]
                loss_list.append(curr_loss.item())
                loss += curr_loss

                if bidir:
                    # In bidir mode, y_pred[2] is warped_bwd
                    curr_loss = losses[1](y_true[1], y_pred[2]) * weights[1]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # 2) Deformation gradient‐loss (unchanged)
            grad_idx = 2 if bidir else 1
            # If bidir: y_pred[3] is flow_bwd_normed; if unidir: y_pred[1] is flow_normed
            flow_for_grad = y_pred[grad_idx]
            curr_loss = losses[grad_idx](None, flow_for_grad) * weights[grad_idx]
            loss_list.append(curr_loss.item())
            loss += curr_loss
            flow_for_jac = y_pred[grad_idx]

            # --- JAC0 Loss (negative Jacobian penalty), batch-compatible ---
            if args.use_jac0 and step == 0:
                with torch.no_grad():
                    # Compute and log negative jacobian fraction for each batch element
                    frac_negs = []
                    for i in range(flow_for_jac.shape[0]):  # batch size loop (e.g., 2)
                        # Convert (3, D, H, W) -> (D, H, W, 3)
                        flow_np = flow_for_jac[i].permute(1, 2, 3, 0).detach().cpu().numpy()
                        jac_det = vxm.py.utils.jacobian_determinant(flow_np)  # (D, H, W)
                        frac_neg = np.mean(jac_det < 0)
                        frac_negs.append(frac_neg)




            # 3) Inverse‐consistency (if requested)
            if args.use_ic and bidir:
                raw_ic = inv_consistency_loss(y_pred[2], y_pred[3])  # y_pred[2]=warped_bwd, y_pred[3]=flow_bwd
                num_voxels = torch.tensor(y_pred[2].numel(), device=raw_ic.device)
                loss_ic = raw_ic / num_voxels
                loss += args.lambda_ic * loss_ic
                epoch_ic_loss.append(loss_ic.item())

            # 4) Mutual‐information and bending if requested (unchanged)
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

            if args.use_jac0 and step == 0:
                with torch.no_grad():
                    # Compute and log negative jacobian fraction for the current batch
                    jac_det = vxm.py.utils.jacobian_determinant(flow_for_jac)  # You might need your own jacobian fn
                    frac_neg = (jac_det < 0).float().mean().item()
                    writer.add_scalar('QA/FractionNegativeJacobian', frac_neg, epoch)

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            epoch_step_time.append(time.time() - step_start_time)

        flow_mean = np.mean([d['mean'] for d in epoch_flow])
        flow_rms  = np.mean([d['rms']  for d in epoch_flow])
        flow_p95  = np.mean([d['p95']  for d in epoch_flow])
        flow_max  = np.max( [d['vmax'] for d in epoch_flow])
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
        metric_info = (f"  ⟨|v|⟩={flow_mean:.3f}"
                   f"  RMS={flow_rms:.3f}"
                   f"  P95={flow_p95:.3f}"
                   f"  max={flow_max:.3f}")
        print(' - '.join((epoch_info, time_info, loss_info, metric_info)), flush=True)

        writer.add_scalar('Loss/Total', np.mean(epoch_total_loss), epoch)
        writer.add_scalar('DVF/Mean', flow_mean, epoch)
        writer.add_scalar('DVF/RMS',  flow_rms,  epoch)
        writer.add_scalar('DVF/P95',  flow_p95,  epoch)
        writer.add_scalar('DVF/Max',  flow_max,  epoch)
        
        avg_losses = np.mean(epoch_loss, axis=0)
        for i, name in enumerate(loss_names[:len(avg_losses)]):
            writer.add_scalar(f'Loss/{name}', avg_losses[i], epoch)

        if args.use_seg and 'epoch_dice_loss' in locals():
            writer.add_scalar('Loss/Dice', np.mean(epoch_dice_loss), epoch)
        if args.use_surface and 'epoch_surface_loss' in locals():
            writer.add_scalar('Loss/Surface', np.mean(epoch_surface_loss), epoch)
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