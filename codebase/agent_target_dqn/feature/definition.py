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
    obs=Config.DIM_OF_OBSERVATION,  # 560
    _obs=Config.DIM_OF_OBSERVATION,  # 560
    act=4,
    # [phase(4 choices)]
    # [相位(4个选择)]
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


def sample_process(list_game_data):
    r_data = np.array(list_game_data).squeeze()

    sample_datas = []
    for data in r_data:
        legal_action = [data.legal_action[0], data.legal_action[0], data.legal_action[0], data.legal_action[0]]
        sample_data = SampleData(
            obs=data.obs,
            _obs=None,
            act=data.act,
            rew=data.rew,
            done=1 if data.done == 0 else 0,
            legal_action=legal_action,
        )

        sample_datas.append(sample_data)

    for i in range(len(sample_datas) - 1):
        sample_datas[i]._obs = sample_datas[i + 1].obs
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
    if not act or act[0] is None:
        return 0.0, 0.0

    phase_index = int(act[1])
    duration = int(act[2])
    phase_reward, duration_reward = 0.0, 0.0

    frame_state = _obs["frame_state"]
    vehicles = frame_state["vehicles"]

    lane_to_phase = {}
    for phase, lanes in get_webster_lane_group().items():
        for lane in lanes:
            lane_to_phase[lane] = int(phase)

    phase_pressure = np.zeros(Config.DIM_OF_ACTION_PHASE, dtype=np.float32)
    total_waiting_time = 0.0
    total_delay = 0.0
    total_queue = 0.0
    enter_vehicle_count = 0

    for vehicle in vehicles:
        if not on_enter_lane(vehicle):
            continue

        lane_phase = lane_to_phase.get(vehicle["lane"])
        if lane_phase is None:
            continue

        speed = float(vehicle.get("speed", 0.0))
        waiting_time = float(vehicle.get("waiting_time", 0.0))
        delay = float(vehicle.get("delay", 0.0))
        is_waiting = 1.0 if speed <= Config.WAITING_SPEED_THRESHOLD else 0.0

        pressure = 1.0 + 2.0 * is_waiting + min(waiting_time, 300.0) / 30.0 + min(delay, 300.0) / 60.0
        phase_pressure[lane_phase] += pressure
        total_waiting_time += waiting_time
        total_delay += delay
        total_queue += is_waiting
        enter_vehicle_count += 1

    if enter_vehicle_count == 0:
        agent.preprocess.old_waiting_time = 0.0
        agent.preprocess.last_phase_index = phase_index
        return 0.0, 0.0

    avg_waiting_time = total_waiting_time / enter_vehicle_count
    avg_delay = total_delay / enter_vehicle_count
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
    phase_reward -= 0.02 * total_queue
    phase_reward -= 0.002 * avg_delay

    target_duration = int(
        np.clip(
            Config.MIN_GREEN_DURATION + selected_pressure,
            Config.MIN_GREEN_DURATION,
            Config.MAX_GREEN_DURATION,
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
    agent.preprocess.last_phase_index = phase_index

    return float(phase_reward), float(duration_reward)
