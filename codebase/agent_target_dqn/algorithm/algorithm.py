###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import time
import numpy as np
from agent_target_dqn.conf.conf import Config
from copy import deepcopy
import torch.nn.functional as F


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optim = optimizer
        self.logger = logger
        self.monitor = monitor

        self.num_head = Config.NUMB_HEAD
        self._gamma = Config.GAMMA

        self.target_model = deepcopy(self.model)
        if self.device is not None:
            self.target_model.to(self.device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        # Convert list of SampleData to tensor batch
        # 将 SampleData 数组 转换为 tensor batch
        if not list_sample_data:
            return

        batch_size = len(list_sample_data)
        obs = self._stack_tensor([frame.obs for frame in list_sample_data], dtype=torch.float32).view(batch_size, -1)
        action = self._stack_tensor([frame.act for frame in list_sample_data], dtype=torch.float32).view(batch_size, -1)
        rew = self._stack_tensor([frame.rew for frame in list_sample_data], dtype=torch.float32).view(batch_size, -1)
        _obs = self._stack_tensor([frame._obs for frame in list_sample_data], dtype=torch.float32).view(batch_size, -1)
        not_done = self._stack_tensor([frame.done for frame in list_sample_data], dtype=torch.float32).view(batch_size)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action_indices = self._action_to_head_indices(action)

        # Main implementation of the multi-head output Target_DQN algorithm
        # 多头输出target_dqn算法的主要实现
        self.target_model.eval()

        with torch.no_grad():
            online_next_outputs = self.model(_obs)[0]
            target_outputs = self.target_model(_obs)[0]
            # Calculate the target Q-values for each head
            # 计算各个头的目标q值
            q_targets = []
            for head_idx in range(self.num_head):
                next_action = online_next_outputs[head_idx].argmax(dim=1, keepdim=True)
                next_q_value = target_outputs[head_idx].gather(1, next_action)
                q_targets_head = (
                    rew[:, head_idx].unsqueeze(1)
                    + self._gamma * next_q_value * not_done.unsqueeze(1)
                )
                q_targets.append(q_targets_head)
            q_targets = torch.cat(q_targets, dim=1)

        # Calculate the Q-values for each head
        # 计算各个头的q值
        self.model.train()
        online_outputs = self.model(obs)[0]
        q_values = []
        for head_idx in range(self.num_head):
            q_values_head = online_outputs[head_idx].gather(1, action_indices[:, head_idx].unsqueeze(1))
            q_values.append(q_values_head)
        q_values = torch.cat(q_values, dim=1)

        self.optim.zero_grad()
        loss = F.smooth_l1_loss(q_values.float(), q_targets.float())
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        self.optim.step()
        self.train_step += 1

        if Config.TARGET_UPDATE_FREQ > 0 and self.train_step % Config.TARGET_UPDATE_FREQ == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        target_q_value = q_targets.mean().detach().item()
        q_value = q_values.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "target_q_value": target_q_value,
                "q_value": q_value,
                "model_grad_norm": model_grad_norm,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            if self.logger:
                self.logger.info(
                    f"value_loss: {value_loss}, target_q_value: {target_q_value},\
                                    q_value: {q_value},\
                                    model_grad_norm: {model_grad_norm}"
                )
            self.last_report_monitor_time = now

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def _stack_tensor(self, values, dtype):
        tensors = [self._as_tensor(value, dtype=dtype) for value in values]
        return torch.stack(tensors)

    def _as_tensor(self, value, dtype):
        if self.device is None:
            return torch.as_tensor(value, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _action_to_head_indices(self, action):
        phase_index = action[:, 1].long().clamp(0, Config.DIM_OF_ACTION_PHASE - 1)
        duration_index = (action[:, 2] - Config.MIN_GREEN_DURATION).long()
        duration_index = duration_index.clamp(0, Config.DIM_OF_ACTION_DURATION - 1)
        return torch.stack([phase_index, duration_index], dim=1)
