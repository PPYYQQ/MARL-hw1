#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.traffic_utils import *
from agent_target_dqn.feature.definition import reward_shaping as target_dqn_reward_shaping


ObsData = create_cls("ObsData", feature=None, legal_action=None, sub_action_mask=None)

ActData = create_cls("ActData", junction_id=None, action=None, d_action=None, prob=None, value=None)

# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,
    act=Config.NUMB_HEAD,  # 2
    reward=1,
    reward_sum=1,
    done=1,
    value=1,
    next_value=1,
    advantage=1,
    prob=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,
    sub_action=Config.NUMB_HEAD,  # 2
    is_train=1,
)


def sample_process(list_sample_data):
    if not list_sample_data:
        return []

    samples = []
    for sample in list_sample_data:
        try:
            sample.obs = _fixed_float_list(sample.obs, Config.DIM_OF_OBSERVATION)
            sample.legal_action = _fixed_float_list(
                sample.legal_action,
                Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,
                default=1.0,
            )
            sample.act = _fixed_action_list(sample.act)
            sample.reward = _safe_float(sample.reward)
            sample.reward_sum = _safe_float(sample.reward_sum)
            sample.done = _done_flag(sample.done)
            sample.value = _safe_float(sample.value)
            sample.next_value = _safe_float(sample.next_value)
            sample.advantage = _safe_float(sample.advantage)
            sample.prob = _fixed_probability_list(sample.prob)
            sample.sub_action = _fixed_float_list(sample.sub_action, Config.NUMB_HEAD, default=1.0)
            sample.is_train = 1.0 if _safe_float(sample.is_train, 1.0) != 0.0 else 0.0
        except Exception:
            continue
        samples.append(sample)

    if not samples:
        return []

    for i in range(len(samples) - 1):
        samples[i].next_value = 0.0 if samples[i].done else samples[i + 1].value
    samples[-1].next_value = 0.0

    _calc_reward(samples)

    return samples


def reward_shaping(_obs, act, agent):
    """
    This function is an important function for reward processing, mainly responsible for:
        - Unpacking data, obtaining the data required for reward calculation from _obs
        - Reward calculation, calculating rewards based on the unpacked data
        - Reward concatenation, concatenating all rewards into a list

    Parameters:
        - _obs: The original feature data sent by battlesrv
        - act: The previous act predicted and executed
        - agent: real agent perform action

    Returns:
        - phase reward: The reward corresponding to the action of the phase number
        - duration reward: The reward corresponding to the action of the phase duration
    """
    """
    该函数是奖励处理的重要函数, 主要负责：
        - 数据解包, 从 _obs 获取计算奖励所需要的数据
        - 奖励计算, 根据解包的数据计算奖励
        - 奖励拼接, 将所有的奖励拼接成一个list

    参数：
        - _obs: battlesrv 发送的原始特征数据
        - act: 前一次预测并执行动作
        - agent: 实际执行动作智能体

    返回：
        - phase reward: 对应相位编号动作的奖励
        - duration reward: 对应相位持续时间动作的奖励
    """
    try:
        phase_reward, duration_reward = target_dqn_reward_shaping(_obs, act, agent)
    except Exception:
        return 0.0
    return _clip_reward(phase_reward + duration_reward)


def _calc_reward(list_sample_data):
    """
    Calculate cumulated reward and advantage with GAE.
    reward_sum: used for value loss
    advantage: used for policy loss
    V(s) here is a approximation of target network

    使用 GAE 计算累积奖励和优势函数。
    reward_sum: 用于价值损失
    advantage: 用于策略损失
    V(s) 这里是目标网络的近似值
    """

    gae = 0.0
    gamma, lamda = Config.GAMMA, Config.LAMDA
    for rl_info in reversed(list_sample_data):
        not_done = 1.0 - _done_flag(rl_info.done)
        delta = rl_info.reward + gamma * rl_info.next_value * not_done - rl_info.value
        gae = delta + gamma * lamda * not_done * gae
        rl_info.advantage = gae
        rl_info.reward_sum = gae + rl_info.value


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(value):
        return default
    return float(value)


def _fixed_float_list(value, width, default=0.0):
    try:
        values = np.asarray(value, dtype=np.float32).flatten()
    except Exception:
        values = np.asarray([], dtype=np.float32)
    values = np.nan_to_num(values, nan=default, posinf=default, neginf=default)
    if values.size < width:
        values = np.pad(values, (0, width - values.size), constant_values=default)
    elif values.size > width:
        values = values[:width]
    return values.astype(np.float32).tolist()


def _fixed_probability_list(value):
    width = Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1
    values = _fixed_float_list(value, width, default=0.0)
    phase = np.asarray(values[: Config.DIM_OF_ACTION_PHASE_1], dtype=np.float32)
    duration = np.asarray(values[Config.DIM_OF_ACTION_PHASE_1 :], dtype=np.float32)
    phase = _normalize_probability(phase)
    duration = _normalize_probability(duration)
    return np.concatenate([phase, duration]).astype(np.float32).tolist()


def _normalize_probability(values):
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        values[:] = 1.0 / max(len(values), 1)
        return values
    return values / total


def _fixed_action_list(value):
    action = _fixed_float_list(value, Config.NUMB_HEAD)
    action[0] = float(np.clip(round(action[0]), 0, Config.DIM_OF_ACTION_PHASE_1 - 1))
    action[1] = float(np.clip(round(action[1]), 0, Config.DIM_OF_ACTION_DURATION_1 - 1))
    return action


def _done_flag(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return 1.0
        if normalized in ("false", "0", "no", "n", ""):
            return 0.0
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return 1.0 if value != 0.0 else 0.0


def _clip_reward(value):
    value = _safe_float(value)
    return float(np.clip(value, -Config.REWARD_CLIP, Config.REWARD_CLIP))
