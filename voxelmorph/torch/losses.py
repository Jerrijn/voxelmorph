import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self,
             y_true: torch.Tensor,
             y_pred: torch.Tensor,
             weight_mask: torch.Tensor = None) -> torch.Tensor:
        # y_true, y_pred: (B, C, D, H, W)
        if weight_mask is None:
            return torch.mean((y_true - y_pred) ** 2)

        # Weighted MSE numerator: sum_x [ w(x) * (error)^2 ]
        sq_error = (y_pred - y_true).pow(2)  # shape: (B, C, D, H, W)

        # Ensure weight_mask is broadcastable to sq_error
        # (we expect weight_mask of shape (B, 1, D, H, W) or (B, C, D, H, W))
        weighted_sq = sq_error * weight_mask

        # Sum over all voxels and channels
        numerator = weighted_sq.sum()

        # Sum of weights over all voxels and channels
        denom = weight_mask.sum()
        # Add a tiny epsilon so we never divide by zero
        return numerator / (denom + 1e-6)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

class MutualInformation:
    def __init__(self, bins=32, sigma=0.02, eps=1e-10, device='cuda'):
        self.bins, self.sigma, self.eps, self.device = bins, sigma, eps, device
        self.bin_centers = torch.linspace(0., 1., bins, device=device).view(1, 1, bins)

    def loss(self, y_true, y_pred):
        B = y_true.size(0)
        y_true = y_true.reshape(B, -1).unsqueeze(-1)  # shape: [B, N, 1]
        y_pred = y_pred.reshape(B, -1).unsqueeze(-1)  # shape: [B, N, 1]

        # Soft assignments using a Gaussian kernel (Parzen window)
        soft_true = torch.exp(-((y_true - self.bin_centers)**2) / (2 * self.sigma**2))
        soft_pred = torch.exp(-((y_pred - self.bin_centers)**2) / (2 * self.sigma**2))
        
        # Joint histogram as a soft outer product, normalized by the voxel count
        joint_hist = torch.matmul(soft_true.transpose(1, 2), soft_pred) / y_true.size(1)
        
        # Marginal distributions
        p_true = joint_hist.sum(dim=2)
        p_pred = joint_hist.sum(dim=1)
        
        # Mutual information computation
        mi = (joint_hist * torch.log((joint_hist + self.eps) / 
              (p_true.unsqueeze(-1) * p_pred.unsqueeze(1) + self.eps))).sum(dim=(-2, -1))
        return -mi.mean()

class InverseConsistencyLoss(nn.Module):
    """
    Inverse Consistency Loss.
    
    Given two displacement fields u_ab and u_ba (from image A→B and B→A),
    we first compute the full transforms:
        phi_ab = id + u_ab   and   phi_ba = id + u_ba.
    Then, we enforce:
        phi_ab(phi_ba(x)) ≈ x    and    phi_ba(phi_ab(x)) ≈ x.
    
    The loss is computed as the mean squared difference over the image domain.
    """
    def __init__(self):
        super(InverseConsistencyLoss, self).__init__()

    def forward(self, u_ab, u_ba):
        # u_ab and u_ba: [B, C, ...] where C is 2 for 2D or 3 for 3D.
        B, ndim = u_ab.shape[:2]
        device = u_ab.device

        if ndim == 2:
            # For 2D, shape: [B, 2, H, W]
            B, C, H, W = u_ab.shape
            # Create an identity grid in voxel coordinates: shape [B, H, W, 2]
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            id_grid = torch.stack((x, y), dim=-1).float()  # [H, W, 2]
            id_grid = id_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Compute full transforms: phi = id + u, where u is re-ordered to [B, H, W, 2]
            u_ab_perm = u_ab.permute(0, 2, 3, 1)  # [B, H, W, 2]
            u_ba_perm = u_ba.permute(0, 2, 3, 1)  # [B, H, W, 2]
            phi_ab = id_grid + u_ab_perm
            phi_ba = id_grid + u_ba_perm

            # Compose phi_ab(phi_ba(x))
            # grid_sample requires normalized coordinates in [-1, 1]:
            norm_x = 2.0 * phi_ba[..., 0] / (W - 1) - 1.0
            norm_y = 2.0 * phi_ba[..., 1] / (H - 1) - 1.0
            grid_norm = torch.stack((norm_x, norm_y), dim=-1)  # [B, H, W, 2]
            phi_ab_for_sampling = phi_ab.permute(0, 3, 1, 2)  # [B, 2, H, W]
            composed_ab = F.grid_sample(phi_ab_for_sampling, grid_norm,
                                         mode='bilinear', align_corners=True)
            composed_ab = composed_ab.permute(0, 2, 3, 1)  # [B, H, W, 2]
            diff_ab = composed_ab - id_grid
            loss_ab = diff_ab.pow(2).mean()

            # Compose phi_ba(phi_ab(x))
            norm_x_b = 2.0 * phi_ab[..., 0] / (W - 1) - 1.0
            norm_y_b = 2.0 * phi_ab[..., 1] / (H - 1) - 1.0
            grid_norm_b = torch.stack((norm_x_b, norm_y_b), dim=-1)
            phi_ba_for_sampling = phi_ba.permute(0, 3, 1, 2)
            composed_ba = F.grid_sample(phi_ba_for_sampling, grid_norm_b,
                                         mode='bilinear', align_corners=True)
            composed_ba = composed_ba.permute(0, 2, 3, 1)
            diff_ba = composed_ba - id_grid
            loss_ba = diff_ba.pow(2).mean()

            loss = loss_ab + loss_ba

        elif ndim == 3:
            # For 3D, shape: [B, 3, D, H, W]
            B, C, D, H, W = u_ab.shape
            # Create identity grid: shape [B, D, H, W, 3]
            z, y, x = torch.meshgrid(torch.arange(D, device=device),
                                     torch.arange(H, device=device),
                                     torch.arange(W, device=device), indexing='ij')
            id_grid = torch.stack((x, y, z), dim=-1).float()  # [D, H, W, 3]
            id_grid = id_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

            u_ab_perm = u_ab.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
            u_ba_perm = u_ba.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
            phi_ab = id_grid + u_ab_perm
            phi_ba = id_grid + u_ba_perm

            # Compose phi_ab(phi_ba(x))
            norm_x = 2.0 * phi_ba[..., 0] / (W - 1) - 1.0
            norm_y = 2.0 * phi_ba[..., 1] / (H - 1) - 1.0
            norm_z = 2.0 * phi_ba[..., 2] / (D - 1) - 1.0
            grid_norm = torch.stack((norm_x, norm_y, norm_z), dim=-1)  # [B, D, H, W, 3]
            phi_ab_for_sampling = phi_ab.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
            composed_ab = F.grid_sample(phi_ab_for_sampling, grid_norm,
                                         mode='bilinear', align_corners=True)
            composed_ab = composed_ab.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
            diff_ab = composed_ab - id_grid
            loss_ab = diff_ab.pow(2).mean()

            # Compose phi_ba(phi_ab(x))
            norm_x_b = 2.0 * phi_ab[..., 0] / (W - 1) - 1.0
            norm_y_b = 2.0 * phi_ab[..., 1] / (H - 1) - 1.0
            norm_z_b = 2.0 * phi_ab[..., 2] / (D - 1) - 1.0
            grid_norm_b = torch.stack((norm_x_b, norm_y_b, norm_z_b), dim=-1)
            phi_ba_for_sampling = phi_ba.permute(0, 4, 1, 2, 3)
            composed_ba = F.grid_sample(phi_ba_for_sampling, grid_norm_b,
                                         mode='bilinear', align_corners=True)
            composed_ba = composed_ba.permute(0, 2, 3, 4, 1)
            diff_ba = composed_ba - id_grid
            loss_ba = diff_ba.pow(2).mean()

            loss = loss_ab + loss_ba

        else:
            raise ValueError("Only 2D or 3D displacement fields are supported.")
        
        return loss


class BendingEnergyLoss(nn.Module):
    """
    Bending Energy Loss.
    
    This loss penalizes rapid changes in the deformation field by approximating
    the second spatial derivatives (both pure and mixed) of the displacement field u.
    
    The loss is defined as:
    
      L_bend = ∑_{p∈Ω} ∑_{i} (∂²u/∂x_i²)² + 2∑_{i<j} (∂²u/∂x_i∂x_j)²
    
    Finite differences (torch.diff) are used to approximate these derivatives.
    """
    def __init__(self):
        super(BendingEnergyLoss, self).__init__()

    def forward(self, u):
        # u: displacement field of shape [B, C, ...] (C should equal 2 for 2D or 3 for 3D)
        dims = u.dim() - 2  # number of spatial dimensions
        loss = 0.0

        # Pure second derivatives along each spatial dimension.
        for d in range(dims):
            # Compute second-order finite differences along dimension d+2 (skip B and channel dims)
            diff2 = torch.diff(u, n=2, dim=d+2)
            loss = loss + diff2.pow(2).mean()

        # Mixed partial derivatives for i < j.
        for i in range(dims):
            for j in range(i+1, dims):
                # First derivative along dimension i
                diff_i = torch.diff(u, n=1, dim=i+2)
                # Then derivative of the result along dimension j
                diff_ij = torch.diff(diff_i, n=1, dim=j+2)
                loss = loss + 2 * diff_ij.pow(2).mean()
        return loss
    
    # Add loss method for consistency with other loss classes
    def loss(self, _, y_pred):
        """
        Wrapper around forward method to maintain consistent API with other loss classes.
        The first argument is ignored (typically y_true) since bending energy only depends on the predicted field.
        """
        return self.forward(y_pred)