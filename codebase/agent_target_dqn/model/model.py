#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from agent_target_dqn.conf.conf import Config


class Model(nn.Module):
    """
    Basic neural network implemented by PyTorch
    Can choose whether to output softmax in the last layer based on the configuration Config.SOFTMAX

    由pytorch实现的基础神经网络
    可以根据配置Config.SOFTMAX 选择是否在最后一层输出softmax
    """

    def __init__(self, device=None):
        super().__init__()
        action_shape = [Config.DIM_OF_ACTION]
        self.device = device

        modules = []

        all_dims = [Config.DIM_OF_OBSERVATION, 256, 128, 64]

        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                modules.append(nn.LayerNorm(all_dims[i + 1]))
                modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules).to(self.device)

        num_heads = len(action_shape)
        self.heads = nn.ModuleList([nn.Linear(all_dims[-1], np.prod(action_shape[i])) for i in range(num_heads)]).to(
            self.device
        )

    def forward(self, s, state=None, info=None):
        if info is None:
            info = {}

        s = self._prepare_input(s)

        s = self.model(s)

        if Config.SOFTMAX:
            return [F.softmax(head(s), dim=-1) for head in self.heads], state
        else:
            return [head(s) for head in self.heads], state

    def _prepare_input(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(
                self._as_numpy_array(s),
                device=self.device,
                dtype=torch.float32,
            )
        elif self.device is None:
            s = s.to(dtype=torch.float32)
        else:
            s = s.to(device=self.device, dtype=torch.float32)

        if s.dim() == 0:
            s = s.reshape(1, 1)
        elif s.dim() == 1:
            s = s.unsqueeze(0)
        elif s.dim() > 2:
            s = s.reshape(s.shape[0], -1)

        feature_dim = s.shape[-1]
        if feature_dim < Config.DIM_OF_OBSERVATION:
            s = F.pad(s, (0, Config.DIM_OF_OBSERVATION - feature_dim))
        elif feature_dim > Config.DIM_OF_OBSERVATION:
            s = s[:, : Config.DIM_OF_OBSERVATION]
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return s

    def _as_numpy_array(self, s):
        try:
            return np.asarray(s, dtype=np.float32)
        except Exception:
            if not isinstance(s, (list, tuple)):
                return np.zeros((1, 0), dtype=np.float32)
            rows = []
            for item in s:
                try:
                    row = np.asarray(item, dtype=np.float32).reshape(-1)
                except Exception:
                    row = np.zeros(0, dtype=np.float32)
                rows.append(self._fit_numpy_width(row))
            if not rows:
                return np.zeros((1, 0), dtype=np.float32)
            return np.stack(rows)

    def _fit_numpy_width(self, row):
        if row.size < Config.DIM_OF_OBSERVATION:
            return np.pad(row, (0, Config.DIM_OF_OBSERVATION - row.size))
        if row.size > Config.DIM_OF_OBSERVATION:
            return row[: Config.DIM_OF_OBSERVATION]
        return row
