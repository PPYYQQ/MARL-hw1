###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import os
import time
import numpy as np
from agent_ppo.conf.conf import Config
import torch.nn.functional as F
import math


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.num_head = Config.NUMB_HEAD
        self._gamma = Config.GAMMA

        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.m_var_beta = Config.BETA_START
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = self.m_var_beta

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
        legal_action = self._stack_tensor(
            [
                self._sample_field(
                    frame,
                    "legal_action",
                    [1] * (Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1),
                )
                for frame in list_sample_data
            ],
            dtype=torch.float32,
            width=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,
        ).view(batch_size, -1)
        sub_action = self._stack_tensor(
            [self._sample_field(frame, "sub_action", [1] * Config.NUMB_HEAD) for frame in list_sample_data],
            dtype=torch.float32,
            width=Config.NUMB_HEAD,
        ).view(batch_size, -1)
        act = self._stack_tensor(
            [self._sample_field(frame, "act", [0] * Config.NUMB_HEAD) for frame in list_sample_data],
            dtype=torch.float32,
            width=Config.NUMB_HEAD,
        ).view(batch_size, -1)
        prob = self._stack_tensor(
            [
                self._sample_field(
                    frame,
                    "prob",
                    [1.0] * (Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1),
                )
                for frame in list_sample_data
            ],
            dtype=torch.float32,
            width=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,
        ).view(batch_size, -1)
        reward = self._stack_tensor(
            [self._sample_field(frame, "reward", 0.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        reward_sum = self._stack_tensor(
            [self._sample_field(frame, "reward_sum", 0.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        advantage = self._stack_tensor(
            [self._sample_field(frame, "advantage", 0.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        value = self._stack_tensor(
            [self._sample_field(frame, "value", 0.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        next_value = self._stack_tensor(
            [self._sample_field(frame, "next_value", 0.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)
        is_train = self._stack_tensor(
            [self._sample_field(frame, "is_train", 1.0) for frame in list_sample_data],
            dtype=torch.float32,
            width=1,
        ).view(batch_size)

        data_list = [
            self._finite_tensor(obs),
            self._finite_tensor(legal_action),
            self._finite_tensor(sub_action),
            self._finite_tensor(act),
            self._finite_tensor(prob),
            self._finite_tensor(reward),
            self._finite_tensor(reward_sum),
            self._finite_tensor(advantage),
            self._finite_tensor(value),
            self._finite_tensor(next_value),
            self._finite_tensor(is_train),
        ]

        # Configure model before prediction
        # 预测前先对model进行设置
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs)
        total_loss, info_list = self.calculate_loss(data_list, rst_list)
        results = {}

        results["total_loss"] = total_loss.item()

        if not bool(torch.isfinite(total_loss).item()):
            self._log_info("skip PPO learn step, non-finite loss")
            return

        total_loss.backward()

        # Gradient clipping
        # 梯度裁剪
        if Config.USE_GRAD_CLIP:
            model_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
            try:
                model_grad_norm = float(model_grad_norm.item())
            except AttributeError:
                model_grad_norm = float(model_grad_norm)
            if not math.isfinite(model_grad_norm):
                self.optimizer.zero_grad()
                self._log_info("skip PPO learn step, non-finite grad norm")
                return
        else:
            model_grad_norm = 0.0

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            results["model_grad_norm"] = round(model_grad_norm, 4)

            self._log_info(
                f"policy_loss: {round(policy_loss, 2)}, value_loss: {round(value_loss, 2)}, "
                f"entropy_loss: {round(entropy_loss, 2)}, model_grad_norm: {round(model_grad_norm, 4)}"
            )
            self._put_monitor_data(results)

            self.last_report_monitor_time = now

    def calculate_loss(self, list_sample_data, model_output_data):
        (
            obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            value,
            next_value,
            is_train,
        ) = list_sample_data

        train_weight = is_train.view(-1).float().clamp(0.0, 1.0)
        weight_sum = train_weight.sum().clamp_min(1.0)
        value_target = reward_sum.detach()
        legal_action_flag_list = torch.split(legal_action, self.label_size_list, dim=1)
        usq_label_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_label_list.append(act[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index]
                .reshape(-1, 1)
                .long()
                .clamp(0, self.label_size_list[shape_index] - 1)
            )
        # Process probability
        # 处理概率
        sum_ls_list = [sum(self.label_size_list[0:i]) for i in range(len(self.label_size_list))]

        old_label_probability_list = list()
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list.append(
                prob[:, sum_ls_list[shape_index] : sum_ls_list[shape_index] + self.label_size_list[shape_index]]
            )
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.label_size_list[shape_index]
            )
        usq_weight_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list.append(sub_action[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(-1, 1)

        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))

        label_result = model_output_data[:-1]

        value_result = model_output_data[-1]

        # Loss of value network
        # 价值网络损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        value_error = torch.square(value_target - fc2_value_result_squeezed)
        self.value_cost = 0.5 * torch.sum(value_error * train_weight) / weight_sum

        advantage = advantage.detach().view(-1)
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(Config.LOG_EPSILON)
        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=0.0, neginf=0.0)

        policy_losses = []
        entropy_terms = []
        for shape_index, logits in enumerate(label_result):
            legal_mask = legal_action_flag_list[shape_index].float().clamp(0.0, 1.0)
            empty_rows = legal_mask.sum(dim=1, keepdim=True) <= 0.0
            legal_mask = torch.where(empty_rows, torch.ones_like(legal_mask), legal_mask)

            masked_logits = logits.masked_fill(legal_mask <= 0.0, -1e9)
            new_policy = F.softmax(masked_logits, dim=1)
            new_policy = torch.nan_to_num(new_policy, nan=0.0, posinf=0.0, neginf=0.0)
            new_policy = new_policy / new_policy.sum(dim=1, keepdim=True).clamp_min(Config.LOG_EPSILON)
            old_policy = old_label_probability_list[shape_index].float()
            old_policy = torch.nan_to_num(old_policy, nan=0.0, posinf=0.0, neginf=0.0)
            old_policy = old_policy * legal_mask
            old_policy = old_policy / old_policy.sum(dim=1, keepdim=True).clamp_min(Config.LOG_EPSILON)

            label = usq_label_list[shape_index]
            new_action_prob = new_policy.gather(1, label).squeeze(1).clamp_min(Config.MIN_POLICY)
            old_action_prob = old_policy.gather(1, label).squeeze(1).clamp_min(Config.MIN_POLICY)
            ratio = new_action_prob / old_action_prob
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

            head_weight = (weight_list[shape_index].float().clamp(0.0, 1.0) * train_weight).view(-1)
            head_weight_sum = head_weight.sum().clamp_min(1.0)
            policy_losses.append(-torch.sum(surrogate * head_weight) / head_weight_sum)

            entropy = -(new_policy * torch.log(new_policy.clamp_min(Config.LOG_EPSILON))).sum(dim=1)
            entropy_terms.append(torch.sum(entropy * head_weight) / head_weight_sum)

        self.policy_cost = torch.stack(policy_losses).sum()
        self.entropy_cost = torch.stack(entropy_terms).mean()
        self.loss = (
            Config.POLICY_COEF * self.policy_cost
            + Config.VALUE_COEF * self.value_cost
            - Config.ENTROPY_COEF * self.entropy_cost
        )

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def _sample_batch(self, list_sample_data):
        if list_sample_data is None:
            return []
        try:
            return list(list_sample_data)
        except Exception as err:
            self._log_info(f"skip PPO learn step, sample batch iteration failed: {err}")
            return []

    def _sample_field(self, frame, name, default):
        try:
            return getattr(frame, name, default)
        except Exception:
            return default

    def _stack_tensor(self, values, dtype, width=None):
        tensors = [self._normalize_tensor(value, dtype=dtype, width=width) for value in values]
        return torch.stack(tensors)

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
