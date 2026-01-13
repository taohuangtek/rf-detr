# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    B_heads, head_dim, _ = value.shape
    _, Len_q, LP, _ = sampling_locations.shape
    L = len(value_spatial_shapes)
    P = LP // L

    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=2)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # (B*H, head_dim, H, W)
        value_l_ = value_list[lid_].view(B_heads, head_dim, H, W)

        sampling_grid_l_ = sampling_grids[:, :, lid_*P : (lid_+1)*P, :]

        sampling_grid_l_ = sampling_grid_l_.permute(0, 3, 1, 2)
        sampling_grid_l_ = sampling_grid_l_.permute(0, 2, 3, 1)

        # (B*H, head_dim, Len_q, P)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # (B*H, head_dim, Len_q, LP)
    sampling_value_all = torch.cat(sampling_value_list, dim=-1)

    output = (sampling_value_all * attention_weights).sum(-1)
    return output
