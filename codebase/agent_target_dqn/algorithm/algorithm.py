###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import time
import math
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
        list_sample_data = self._sample_batch(list_sample_data)
        if not list_sample_data:
            return

        batch_size = len(list_sample_data)
        obs = self._stack_tensor(
            [self._sample_field(frame, "obs", []) for frame in list_sample_data],
            dtype=torch.float32,
            width=Config.DIM_OF_OBSERVATION,
        ).view(batch_size, -1)
        action = self._stack_tensor(
            [
                self._sample_field(
                    frame,
                    "act",
                    [0, 0, Config.MIN_GREEN_DURATION],
                )
                for frame in list_sample_data
            ],
            dtype=torch.float32,
            width=3,
        ).view(batch_size, -1)
        rew = self._stack_tensor(
            [self._sample_field(frame, "rew", [0.0, 0.0]) for frame in list_sample_data],
            dtype=torch.float32,
            width=2,
        ).view(batch_size, -1)
        _obs = self._stack_tensor(
            [self._sample_field(frame, "_obs", []) for frame in list_sample_data],
            dtype=torch.float32,
            width=Config.DIM_OF_OBSERVATION,
        ).view(batch_size, -1)
        not_done = self._stack_tensor(
            [self._sample_field(frame, "done", 1.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        legal_action = self._stack_tensor(
            [
                self._sample_field(
                    frame,
                    "legal_action",
                    [1] * Config.DIM_OF_ACTION_PHASE,
                )
                for frame in list_sample_data
            ],
            dtype=torch.float32,
            width=Config.DIM_OF_ACTION_PHASE,
        ).view(batch_size, -1)
        obs = self._finite_tensor(obs)
        _obs = self._finite_tensor(_obs)
        rew = self._finite_tensor(rew)
        action = self._finite_tensor(action)
        not_done = self._finite_tensor(not_done).clamp(0.0, 1.0)
        legal_action = self._finite_tensor(legal_action)
        phase_legal_mask = self._phase_legal_mask(legal_action)
        joint_legal_mask = self._joint_legal_mask(phase_legal_mask)
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
            q_targets = self._finite_tensor(q_targets)

        # Calculate the Q-value for the executed joint action
        # 计算实际执行联合动作的q值
        self.model.train()
        online_outputs = self.model(obs)[0]
        q_values = online_outputs[0].gather(1, action_indices)

        self.optim.zero_grad()
        loss = F.smooth_l1_loss(q_values.float(), q_targets.float())
        if not bool(torch.isfinite(loss).item()):
            self._log_info("skip learn step, non-finite loss")
            return
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        try:
            model_grad_norm = float(model_grad_norm.item())
        except AttributeError:
            model_grad_norm = float(model_grad_norm)
        if not math.isfinite(model_grad_norm):
            self.optim.zero_grad()
            self._log_info("skip learn step, non-finite grad norm")
            return
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
            self._put_monitor_data(monitor_data)
            self._log_info(
                f"value_loss: {value_loss}, target_q_value: {target_q_value},\
                                q_value: {q_value},\
                                model_grad_norm: {model_grad_norm}"
            )
            self.last_report_monitor_time = now

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def _put_monitor_data(self, monitor_data):
        if not self.monitor:
            return
        try:
            self.monitor.put_data({os.getpid(): monitor_data})
        except Exception as err:
            self._log_info(f"monitor put_data failed: {err}")

    def _log_info(self, message):
        if not self.logger:
            return
        try:
            self.logger.info(message)
        except Exception:
            pass

    def _stack_tensor(self, values, dtype, width=None):
        tensors = [self._normalize_tensor(value, dtype=dtype, width=width) for value in values]
        return torch.stack(tensors)

    def _sample_batch(self, list_sample_data):
        if list_sample_data is None:
            return []
        try:
            return list(list_sample_data)
        except Exception as err:
            self._log_info(f"skip learn step, sample batch iteration failed: {err}")
            return []

    def _sample_field(self, frame, name, default):
        try:
            return getattr(frame, name, default)
        except Exception:
            return default

    def _normalize_tensor(self, value, dtype, width=None):
        try:
            tensor = self._as_tensor(value, dtype=dtype).view(-1)
        except Exception:
            tensor = self._as_tensor([], dtype=dtype).view(-1)
        tensor = self._finite_tensor(tensor)
        if width is None:
            return tensor
        if tensor.numel() < width:
            tensor = F.pad(tensor, (0, width - tensor.numel()))
        elif tensor.numel() > width:
            tensor = tensor[:width]
        return tensor

    def _finite_tensor(self, tensor):
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _as_tensor(self, value, dtype):
        if self.device is None:
            return torch.as_tensor(value, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _action_to_joint_index(self, action):
        phase_index = action[:, 1].long().clamp(0, Config.DIM_OF_ACTION_PHASE - 1)
        duration_index = torch.round((action[:, 2] - Config.MIN_GREEN_DURATION) / Config.DURATION_STEP).long()
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
