#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from agent_target_dqn.model.model import Model
from agent_target_dqn.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_target_dqn.conf.conf import Config
from agent_target_dqn.algorithm.algorithm import Algorithm
from agent_target_dqn.feature.preprocessor import FeatureProcess


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.model = Model(device=device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        self._eps = Config.START_EPSILON_GREEDY
        self.end_eps = Config.END_EPSILON_GREEDY
        self.eps_decay = Config.EPSILON_DECAY
        self.head_dim = [
            Config.DIM_OF_ACTION_PHASE,
            Config.DIM_OF_ACTION_DURATION,
        ]
        self.device = device
        self.epsilon = Config.EPSILON
        self.logger = logger
        self.monitor = monitor
        self.preprocess = FeatureProcess(logger)

        self.algorithm = Algorithm(self.model, self.optim, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        self.preprocess.reset()

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        feature = [obs_data.feature for obs_data in list_obs_data]

        model = self.model
        model.eval()

        junction_id = 0

        self._eps = max(self.end_eps, self._eps * self.eps_decay)
        if np.random.rand() >= self._eps or exploit_flag:
            with torch.no_grad():
                list_junction = [
                    junction_id,
                ] * len(list_obs_data)
                res = model(feature)[0]
                list_phase = torch.argmax(res[0], dim=1).cpu().view(-1).tolist()
                list_duration = torch.argmax(res[1], dim=1).cpu().view(-1).tolist()
        else:
            list_junction = [
                junction_id,
            ] * len(list_obs_data)

            random_action = np.random.choice(self.head_dim[0], len(list_obs_data))
            list_phase = random_action

            random_action = np.random.choice(self.head_dim[1], len(list_obs_data))
            list_duration = random_action

        return [
            ActData(
                junction_id=list_junction[i],
                phase_index=list_phase[i],
                duration=list_duration[i],
            )
            for i in range(len(list_obs_data))
        ]

    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    def exploit(self, observation):
        raw_obs = observation["obs"]
        try:
            obs_data = self.observation_process(raw_obs, observation["extra_info"])
            if not obs_data:
                return self.rule_based_action(raw_obs)
            act_data = self.__predict_detail([obs_data], exploit_flag=True)
            return self.action_process(act_data[0])
        except Exception as err:
            if self.logger:
                self.logger.error(f"exploit fallback to rule_based_action: {err}")
            return self.rule_based_action(raw_obs)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict = self.model.state_dict()
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))

        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, raw_obs, extra_info=None):
        """
        This function is an important function for feature processing, mainly responsible for:
            - Parsing raw data from proto data
            - Calculating features through raw data to obtain multiple feature vectors
            - Concatenation of features
            - Labeling of legal actions

        Args:
            - raw_obs: Raw feature data sent by battlesrv

        Returns:
            - ObsData: A variable containing observation and sub_action_mask
        """
        """
            该函数是特征处理的重要函数, 主要负责：
                - 从 proto 数据中解析原始数据
                - 通过原始数据计算特征, 得到多个特征向量
                - 特征的拼接
                - 合法动作的标注

            参数：
                - raw_obs: battlesrv 发送的原始特征数据

            返回：
                - ObsData: 包含 observation 与 sub_action_mask 的变量
        """

        # User-defined section, can record or update traffic information per frame.
        # 用户自定义部分, 可每帧对交通信息进行记录或更新
        self.preprocess.update_traffic_info(raw_obs, extra_info)

        # Note: The unpacking of the following raw data is for example purposes only,
        # please modify according to the actual situation
        # 注意: 以下原始数据的解包为示例, 请根据实际情况修改
        frame_state = raw_obs["frame_state"]

        # Parse frame_state
        # 解析 frame_state
        _, _, vehicles = (
            frame_state["frame_no"],
            frame_state["frame_time"],
            frame_state["vehicles"],
        )

        # Divide the lane into several grids along the lane direction and the vehicle driving direction
        # 沿车道方向和车辆行驶方向将车道划分为数个栅格
        speed_dict = {}
        position_dict = {}
        # The default value of junction_id in a single intersection scenario is 0
        # 单交叉口场景junction_id默认为0
        junction_id = 0
        junction_ids = list(self.preprocess.junction_dict.keys()) or [junction_id]
        for current_junction_id in junction_ids:
            speed_dict[current_junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))
            position_dict[current_junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))

        # Initialize state-related variables to prevent errors when there are no vehicles in the traffic scenario
        # 初始化状态相关变量, 防止交通场景内车辆为空时报错
        position = list(position_dict[junction_id].astype(int).flatten())
        speed = list(speed_dict[junction_id].flatten())

        for vehicle in vehicles:
            # Only count vehicles on the enter lane
            # 仅统计位于进口车道上的车辆信息
            if on_enter_lane(vehicle):
                # Convert the vehicle x,y coordinates to grid coordinates. Here,
                # get_lane_code maps the lane number to integers 0-13, corresponding to 14 import lanes
                # 将车辆x,y坐标转化为栅格坐标, 此处get_lane_code将车道编号映射至整数0-13, 对应14条进口车道
                x_pos = get_lane_code(vehicle)
                if x_pos is None:
                    continue
                y_pos = int(get_lane_position_meters(vehicle) // Config.GRID_LENGTH)

                if y_pos < 0 or y_pos >= Config.GRID_NUM:
                    continue

                vehicle_config = self.preprocess.vehicle_configs_dict.get(vehicle.get("v_config_id"), {})
                max_speed = max(float(vehicle_config.get("max_speed", Config.DEFAULT_MAX_SPEED)), 1.0)
                target_junction = vehicle.get("target_junction", junction_id)
                if target_junction not in speed_dict:
                    target_junction = junction_id
                speed_dict[target_junction][x_pos, y_pos] = float(
                    max(0.0, min(float(vehicle.get("speed", 0.0)) / max_speed, 1.0))
                )
                position_dict[target_junction][x_pos, y_pos] = 1
            else:
                continue

        position = list(position_dict[junction_id].astype(int).flatten())
        speed = list(speed_dict[junction_id].flatten())

        # Integrate all state quantities into the observation
        # 将所有状态量整合在observation中
        observation = position + speed

        return ObsData(feature=observation)

    def action_process(self, act_data):
        junction_id = int(act_data.junction_id)
        phase_index = int(np.clip(act_data.phase_index, 0, Config.DIM_OF_ACTION_PHASE - 1))
        duration_index = int(np.clip(act_data.duration, 0, Config.DIM_OF_ACTION_DURATION - 1))
        duration = int(
            np.clip(
                Config.MIN_GREEN_DURATION + duration_index,
                Config.MIN_GREEN_DURATION,
                Config.MAX_GREEN_DURATION,
            )
        )
        return [junction_id, phase_index, duration]

    def rule_based_action(self, raw_obs):
        frame_state = raw_obs.get("frame_state", {})
        vehicles = frame_state.get("vehicles", [])
        lane_to_phase = {}
        for phase, lanes in get_webster_lane_group().items():
            for lane in lanes:
                lane_to_phase[lane] = int(phase)

        phase_pressure = np.zeros(Config.DIM_OF_ACTION_PHASE, dtype=np.float32)
        for vehicle in vehicles:
            try:
                if not on_enter_lane(vehicle):
                    continue
            except KeyError:
                continue

            lane_phase = lane_to_phase.get(vehicle.get("lane"))
            if lane_phase is None:
                continue

            speed = float(vehicle.get("speed", 0.0))
            waiting_time = float(vehicle.get("waiting_time", 0.0))
            delay = float(vehicle.get("delay", 0.0))
            is_waiting = 1.0 if speed <= Config.WAITING_SPEED_THRESHOLD else 0.0
            phase_pressure[lane_phase] += (
                1.0 + 2.0 * is_waiting + min(waiting_time, 300.0) / 30.0 + min(delay, 300.0) / 60.0
            )

        phase_index = int(np.argmax(phase_pressure))
        duration_index = int(np.clip(round(float(phase_pressure[phase_index])), 0, Config.DIM_OF_ACTION_DURATION - 1))
        return [
            0,
            phase_index,
            int(
                np.clip(
                    Config.MIN_GREEN_DURATION + duration_index,
                    Config.MIN_GREEN_DURATION,
                    Config.MAX_GREEN_DURATION,
                )
            ),
        ]
