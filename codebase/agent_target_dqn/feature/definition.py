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
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.feature.traffic_utils import *


# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    _obs=Config.DIM_OF_OBSERVATION,
    act=3,
    # [junction_id, phase_index, duration_seconds]
    # [路口id, 相位编号, 持续时间秒数]
    rew=2,
    # [phase_reward, duration_reward]
    # [相位奖励, 持续时间奖励]
    done=1,
    legal_action=4,
    # phase legal actions
    # 相位合法动作
)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", junction_id=None, phase_index=None, duration=None)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


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


def _fixed_action_list(value):
    action = _fixed_float_list(value, 3)
    max_duration = _max_action_duration()
    action[0] = 0.0
    action[1] = float(np.clip(action[1], 0, Config.DIM_OF_ACTION_PHASE - 1))
    action[2] = float(np.clip(action[2], Config.MIN_GREEN_DURATION, max_duration))
    return action


def _max_action_duration():
    return Config.max_action_duration()


def _not_done_flag(value):
    if isinstance(value, bool):
        return 0 if value else 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return 0
        if normalized in ("false", "0", "no", "n", ""):
            return 1
        return 1
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1
    if not np.isfinite(value):
        return 1
    return 1 if value == 0.0 else 0


def sample_process(list_game_data):
    if not list_game_data:
        return []

    sample_datas = []
    for data in list_game_data:
        obs = _safe_getattr(data, "obs", None)
        act = _safe_getattr(data, "act", None)
        if obs is None or act is None:
            continue
        try:
            if len(act) < 3 or any(act[index] is None for index in range(3)):
                continue
        except Exception:
            continue

        obs = _fixed_float_list(obs, Config.DIM_OF_OBSERVATION)
        act = _fixed_action_list(act)
        legal_action = normalize_phase_legal_action(
            _safe_getattr(data, "legal_action", None),
            Config.DIM_OF_ACTION_PHASE,
        )
        reward = _safe_getattr(data, "rew", None)
        reward = reward if reward is not None else (0.0, 0.0)
        reward = _fixed_float_list(reward, 2)
        sample_data = SampleData(
            obs=obs,
            _obs=None,
            act=act,
            rew=reward,
            done=_not_done_flag(_safe_getattr(data, "done", 0)),
            legal_action=legal_action,
        )

        sample_datas.append(sample_data)

    if not sample_datas:
        return []

    for i in range(len(sample_datas) - 1):
        sample_datas[i]._obs = sample_datas[i + 1].obs
        sample_datas[i].legal_action = sample_datas[i + 1].legal_action
    sample_datas[-1]._obs = sample_datas[-1].obs

    if sample_datas[-1].done:
        del sample_datas[-1]

    return sample_datas


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
    if act is None:
        return 0.0, 0.0

    try:
        if len(act) < 3 or act[0] is None or act[1] is None or act[2] is None:
            return 0.0, 0.0
        phase_index = int(np.clip(int(act[1]), 0, Config.DIM_OF_ACTION_PHASE - 1))
        duration = int(act[2])
    except (TypeError, ValueError, IndexError, OverflowError):
        return 0.0, 0.0

    phase_reward, duration_reward = 0.0, 0.0

    frame_state = _obs.get("frame_state") if isinstance(_obs, dict) else None
    if not isinstance(frame_state, dict):
        return 0.0, 0.0
    frame_no = _safe_int(frame_state.get("frame_no", 0))
    vehicles = frame_state.get("vehicles", []) or []
    if not isinstance(vehicles, list):
        vehicles = []
    vehicles = [vehicle for vehicle in vehicles if isinstance(vehicle, dict)]

    phase_pressure, pressure_totals = get_phase_pressure(
        vehicles,
        waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
        phase_count=Config.DIM_OF_ACTION_PHASE,
    )
    enter_vehicle_count = pressure_totals["vehicle_count"]

    if enter_vehicle_count == 0:
        agent.preprocess.old_waiting_time = 0.0
        _mark_phase_served(agent, phase_index, frame_no)
        agent.preprocess.last_phase_index = phase_index
        return 0.0, 0.0

    avg_waiting_time = pressure_totals["waiting_time"] / enter_vehicle_count
    avg_delay = min(pressure_totals["delay"] / enter_vehicle_count, Config.REWARD_DELAY_CAP)
    waiting_delta = agent.preprocess.old_waiting_time - avg_waiting_time
    agent.preprocess.old_waiting_time = avg_waiting_time

    total_pressure = float(np.sum(phase_pressure))
    best_phase = int(np.argmax(phase_pressure))
    selected_pressure = float(phase_pressure[phase_index])
    mean_pressure = total_pressure / Config.DIM_OF_ACTION_PHASE

    if total_pressure > 0:
        phase_reward += (selected_pressure - mean_pressure) / total_pressure
    phase_reward += 0.4 if phase_index == best_phase else -0.2
    phase_reward += 0.03 * float(np.clip(waiting_delta, -20.0, 20.0))
    phase_reward -= 0.02 * pressure_totals["queue"]
    phase_reward -= 0.002 * avg_delay
    phase_reward += _fairness_reward(agent, phase_index, frame_no, phase_pressure)

    target_duration = int(
        np.clip(
            Config.MIN_GREEN_DURATION + selected_pressure,
            Config.MIN_GREEN_DURATION,
            _max_action_duration(),
        )
    )
    duration_reward -= abs(duration - target_duration) / Config.MAX_GREEN_DURATION
    duration_reward += 0.02 * float(np.clip(waiting_delta, -20.0, 20.0))

    if duration < Config.MIN_GREEN_DURATION:
        duration_reward -= 1.0

    last_phase_index = agent.preprocess.last_phase_index
    if last_phase_index is not None and phase_index != last_phase_index and duration < Config.MIN_GREEN_DURATION:
        phase_reward -= 0.5
        duration_reward -= 0.5
    _mark_phase_served(agent, phase_index, frame_no)
    agent.preprocess.last_phase_index = phase_index

    return _clip_reward(phase_reward), _clip_reward(duration_reward)


def _clip_reward(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        value = 0.0
    if not np.isfinite(value):
        value = 0.0
    return float(np.clip(value, -Config.REWARD_CLIP, Config.REWARD_CLIP))


def _fairness_reward(agent, phase_index, frame_no, phase_pressure):
    last_served = _ensure_phase_served(agent)
    phase_ages = np.array(
        [
            max(frame_no - _safe_int(served_frame, frame_no), 0)
            if served_frame is not None
            else 0.0
            for served_frame in last_served
        ],
        dtype=np.float32,
    )
    normalized_age = np.clip(phase_ages / Config.PHASE_AGE_SCALE, 0.0, 1.0)
    fairness_pressure = np.asarray(phase_pressure, dtype=np.float32) * normalized_age
    starved_phase = int(np.argmax(fairness_pressure))
    starved_score = float(fairness_pressure[starved_phase])
    if starved_score <= 0.0:
        return 0.0
    selected_age = float(normalized_age[phase_index])
    if phase_index == starved_phase:
        return Config.FAIRNESS_BONUS_SCALE * selected_age
    return -Config.FAIRNESS_BONUS_SCALE * float(normalized_age[starved_phase])


def _mark_phase_served(agent, phase_index, frame_no):
    last_served = _ensure_phase_served(agent)
    phase_index = int(np.clip(phase_index, 0, Config.DIM_OF_ACTION_PHASE - 1))
    last_served[phase_index] = _safe_int(frame_no)


def _ensure_phase_served(agent):
    last_served = getattr(agent.preprocess, "phase_last_served_frame", None)
    if last_served is None or len(last_served) != Config.DIM_OF_ACTION_PHASE:
        last_served = [None] * Config.DIM_OF_ACTION_PHASE
        agent.preprocess.phase_last_served_frame = last_served
    return last_served
