###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import time
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
        legal_action = self._stack_tensor(
            [frame.legal_action for frame in list_sample_data],
            dtype=torch.float32,
        ).view(batch_size, -1)
        phase_legal_mask = self._phase_legal_mask(legal_action)
        joint_legal_mask = self._joint_legal_mask(phase_legal_mask)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action_indices = self._action_to_joint_index(action)
        total_reward = rew.sum(dim=1, keepdim=True)

        # Main implementation of the joint-action Target_DQN algorithm
        # 联合动作target_dqn算法的主要实现
        self.target_model.eval()

        with torch.no_grad():
            online_next_outputs = self.model(_obs)[0]
            target_outputs = self.target_model(_obs)[0]
            # Calculate the target Q-value for the joint action head
            # 计算联合动作头的目标q值
            next_q_for_action = online_next_outputs[0].masked_fill(~joint_legal_mask, -1e9)
            next_action = next_q_for_action.argmax(dim=1, keepdim=True)
            next_q_value = target_outputs[0].gather(1, next_action)
            q_targets = total_reward + self._gamma * next_q_value * not_done.unsqueeze(1)

        # Calculate the Q-value for the executed joint action
        # 计算实际执行联合动作的q值
        self.model.train()
        online_outputs = self.model(obs)[0]
        q_values = online_outputs[0].gather(1, action_indices)

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

    def _action_to_joint_index(self, action):
        phase_index = action[:, 1].long().clamp(0, Config.DIM_OF_ACTION_PHASE - 1)
        duration_index = (action[:, 2] - Config.MIN_GREEN_DURATION).long()
        duration_index = duration_index.clamp(0, Config.DIM_OF_ACTION_DURATION - 1)
        joint_index = phase_index * Config.DIM_OF_ACTION_DURATION + duration_index
        return joint_index.unsqueeze(1)

    def _phase_legal_mask(self, legal_action):
        if legal_action.dim() == 1:
            legal_action = legal_action.view(-1, 1)

        phase_count = Config.DIM_OF_ACTION_PHASE
        if legal_action.size(1) == 1:
            mask = (legal_action[:, :1] > 0).repeat(1, phase_count)
        else:
            mask = legal_action[:, :phase_count] > 0
            if mask.size(1) < phase_count:
                padding = torch.ones(
                    mask.size(0),
                    phase_count - mask.size(1),
                    dtype=torch.bool,
                    device=mask.device,
                )
                mask = torch.cat([mask, padding], dim=1)

        empty_rows = ~mask.any(dim=1, keepdim=True)
        if empty_rows.any():
            mask = torch.where(empty_rows, torch.ones_like(mask), mask)
        return mask

    def _joint_legal_mask(self, phase_legal_mask):
        joint_mask = phase_legal_mask.repeat_interleave(Config.DIM_OF_ACTION_DURATION, dim=1)
        empty_rows = ~joint_mask.any(dim=1, keepdim=True)
        if empty_rows.any():
            joint_mask = torch.where(empty_rows, torch.ones_like(joint_mask), joint_mask)
        return joint_mask
