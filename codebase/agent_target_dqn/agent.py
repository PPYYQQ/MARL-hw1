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
            Config.DIM_OF_ACTION,
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
        phase_masks = np.asarray(
            [self._phase_action_mask(obs_data.legal_action) for obs_data in list_obs_data],
            dtype=bool,
        )
        joint_masks = self._joint_action_mask(phase_masks)

        model = self.model
        model.eval()

        junction_id = 0

        if not exploit_flag:
            self._eps = max(self.end_eps, self._eps * self.eps_decay)
        if exploit_flag or np.random.rand() >= self._eps:
            with torch.no_grad():
                list_junction = [
                    junction_id,
                ] * len(list_obs_data)
                res = model(feature)[0]
                joint_q = res[0].clone()
                joint_mask_tensor = torch.as_tensor(joint_masks, device=joint_q.device, dtype=torch.bool)
                joint_q = joint_q.masked_fill(~joint_mask_tensor, -1e9)
                list_joint_action = torch.argmax(joint_q, dim=1).cpu().view(-1).tolist()
        else:
            list_junction = [
                junction_id,
            ] * len(list_obs_data)

            list_joint_action = [int(np.random.choice(np.flatnonzero(joint_mask))) for joint_mask in joint_masks]

        list_phase = [action // Config.DIM_OF_ACTION_DURATION for action in list_joint_action]
        list_duration = [action % Config.DIM_OF_ACTION_DURATION for action in list_joint_action]

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
        if isinstance(observation, dict):
            raw_obs = observation.get("obs", observation)
            extra_info = observation.get("extra_info")
        else:
            raw_obs = observation
            extra_info = None
        if not isinstance(raw_obs, dict):
            raw_obs = {}
        try:
            obs_data = self.observation_process(raw_obs, extra_info)
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
        if path is None:
            path = "agent_target_dqn/ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        if path is None:
            path = "agent_target_dqn/ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if str(id) == "latest":
                if self.logger:
                    self.logger.info(f"skip load model, {model_file_path} does not exist yet")
                return
            raise FileNotFoundError(model_file_path)
        model_state = torch.load(model_file_path, map_location=self.model.device)
        if isinstance(model_state, dict) and "state_dict" in model_state:
            model_state = model_state["state_dict"]
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as err:
            if str(id) == "latest":
                if self.logger:
                    self.logger.info(f"skip load model, incompatible checkpoint {model_file_path}: {err}")
                return
            raise
        self.algorithm.update_target_q()

        if self.logger:
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
        if not isinstance(raw_obs, dict):
            raw_obs = {}
        frame_state = raw_obs.get("frame_state", {})
        if not isinstance(frame_state, dict):
            frame_state = {}

        # Parse frame_state
        # 解析 frame_state
        vehicles = frame_state.get("vehicles", []) or []
        if not isinstance(vehicles, list):
            vehicles = []
        vehicles = [vehicle for vehicle in vehicles if isinstance(vehicle, dict)]

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
            try:
                is_enter_lane = on_enter_lane(vehicle)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if is_enter_lane:
                # Convert the vehicle x,y coordinates to grid coordinates. Here,
                # get_lane_code maps the lane number to integers 0-13, corresponding to 14 import lanes
                # 将车辆x,y坐标转化为栅格坐标, 此处get_lane_code将车道编号映射至整数0-13, 对应14条进口车道
                try:
                    x_pos = get_lane_code(vehicle)
                    y_pos = int(get_lane_position_meters(vehicle) // Config.GRID_LENGTH)
                    vehicle_config = self.preprocess.vehicle_configs_dict.get(vehicle.get("v_config_id"), {})
                    max_speed = max(float(vehicle_config.get("max_speed", Config.DEFAULT_MAX_SPEED)), 1.0)
                    speed = float(vehicle.get("speed", 0.0) or 0.0)
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue

                if y_pos < 0 or y_pos >= Config.GRID_NUM:
                    continue

                target_junction = vehicle.get("target_junction", junction_id)
                if target_junction not in speed_dict:
                    target_junction = junction_id
                speed_dict[target_junction][x_pos, y_pos] = float(max(0.0, min(speed / max_speed, 1.0)))
                position_dict[target_junction][x_pos, y_pos] = 1
            else:
                continue

        position = list(position_dict[junction_id].astype(int).flatten())
        speed = list(speed_dict[junction_id].flatten())

        # Integrate all state quantities into the observation
        # 将所有状态量整合在observation中
        phase_feature = self._phase_feature(frame_state)
        phase_age_feature = self._phase_age_feature(frame_state)
        traffic_summary = get_traffic_summary(
            vehicles,
            waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
            phase_count=Config.DIM_OF_ACTION_PHASE,
        )
        traffic_feature = self._traffic_feature(traffic_summary)
        traffic_trend_feature = self._traffic_trend_feature(traffic_summary)
        traffic_history_feature = self._traffic_history_feature(traffic_summary)
        lane_stat_feature = self._lane_stat_feature(vehicles)
        observation = (
            position
            + speed
            + phase_feature
            + phase_age_feature
            + traffic_feature
            + traffic_trend_feature
            + traffic_history_feature
            + lane_stat_feature
        )

        return ObsData(
            feature=observation,
            legal_action=normalize_phase_legal_action(raw_obs.get("legal_action"), Config.DIM_OF_ACTION_PHASE),
        )

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
        if not isinstance(raw_obs, dict):
            raw_obs = {}
        frame_state = raw_obs.get("frame_state", {})
        if not isinstance(frame_state, dict):
            frame_state = {}
        vehicles = frame_state.get("vehicles", []) or []
        if not isinstance(vehicles, list):
            vehicles = []
        vehicles = [vehicle for vehicle in vehicles if isinstance(vehicle, dict)]
        phase_pressure, _ = get_phase_pressure(
            vehicles,
            waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
            phase_count=Config.DIM_OF_ACTION_PHASE,
        )
        phase_age_feature = np.asarray(self._phase_age_feature(frame_state), dtype=np.float32)
        fair_pressure = phase_pressure * (1.0 + Config.FAIRNESS_BONUS_SCALE * phase_age_feature)
        legal_mask = self._phase_action_mask(raw_obs.get("legal_action"))
        masked_pressure = np.where(legal_mask, fair_pressure, -1.0)
        phase_index = int(np.argmax(masked_pressure))
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

    def _phase_feature(self, frame_state):
        phases = frame_state.get("phases", []) if isinstance(frame_state, dict) else []
        if not isinstance(phases, list):
            phases = []
        phase_info = {}
        for candidate in phases:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("s_id", 0) == 0:
                phase_info = candidate
                break
        if not phase_info and phases:
            phase_info = next((candidate for candidate in phases if isinstance(candidate, dict)), {})

        phase_feature = [0.0] * Config.DIM_OF_ACTION_PHASE
        if not phase_info:
            return phase_feature + [0.0, 0.0, 0.0, 0.0]

        try:
            phase_id = int(
                np.clip(
                    phase_info.get("phase_id", phase_info.get("phase_idx", 0)),
                    0,
                    Config.DIM_OF_ACTION_PHASE - 1,
                )
            )
            duration = max(float(phase_info.get("duration", 0.0) or 0.0), 0.0)
            remaining_duration = max(float(phase_info.get("remaining_duration", 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            phase_id = 0
            duration = 0.0
            remaining_duration = 0.0
        elapsed_duration = max(duration - remaining_duration, 0.0)

        phase_feature[phase_id] = 1.0
        return phase_feature + [
            float(np.clip(duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            float(np.clip(remaining_duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            float(np.clip(elapsed_duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            1.0,
        ]

    def _phase_age_feature(self, frame_state):
        try:
            frame_no = int(frame_state.get("frame_no", 0) or 0)
        except (TypeError, ValueError):
            frame_no = 0
        phase_info = self._current_phase_info(frame_state)
        last_served = self.preprocess.phase_last_served_frame
        if len(last_served) != Config.DIM_OF_ACTION_PHASE:
            last_served = [None] * Config.DIM_OF_ACTION_PHASE
            self.preprocess.phase_last_served_frame = last_served
        if all(served_frame is None for served_frame in last_served):
            for phase_idx in range(Config.DIM_OF_ACTION_PHASE):
                last_served[phase_idx] = frame_no
        if phase_info:
            try:
                phase_id = int(
                    np.clip(
                        phase_info.get("phase_id", phase_info.get("phase_idx", 0)),
                        0,
                        Config.DIM_OF_ACTION_PHASE - 1,
                    )
                )
            except (TypeError, ValueError):
                phase_id = 0
            last_served[phase_id] = frame_no
        return [
            float(np.clip((frame_no - (served_frame or frame_no)) / Config.PHASE_AGE_SCALE, 0.0, 1.0))
            for served_frame in last_served
        ]

    def _current_phase_info(self, frame_state):
        phases = frame_state.get("phases", []) if isinstance(frame_state, dict) else []
        if not isinstance(phases, list):
            return {}
        for candidate in phases:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("s_id", 0) == 0:
                return candidate
        for candidate in phases:
            if isinstance(candidate, dict):
                return candidate
        return {}

    def _traffic_feature(self, traffic_summary):
        traffic_feature = [
            float(np.clip(pressure / Config.TRAFFIC_PRESSURE_SCALE, 0.0, 1.0))
            for pressure in traffic_summary["phase_pressure"]
        ]
        traffic_feature.extend(
            [
                float(np.clip(traffic_summary["vehicle_count"] / Config.TRAFFIC_COUNT_SCALE, 0.0, 1.0)),
                float(np.clip(traffic_summary["queue_ratio"], 0.0, 1.0)),
                float(np.clip(traffic_summary["avg_waiting_time"] / Config.TRAFFIC_TIME_SCALE, 0.0, 1.0)),
                float(np.clip(traffic_summary["avg_delay"] / Config.TRAFFIC_TIME_SCALE, 0.0, 1.0)),
            ]
        )
        return traffic_feature

    def _traffic_trend_feature(self, traffic_summary):
        trend_feature = get_traffic_trend(
            traffic_summary,
            self.preprocess.last_traffic_summary,
            pressure_scale=Config.TRAFFIC_PRESSURE_SCALE,
            count_scale=Config.TRAFFIC_COUNT_SCALE,
            time_scale=Config.TRAFFIC_TIME_SCALE,
        )
        self.preprocess.last_traffic_summary = traffic_summary
        return trend_feature

    def _traffic_history_feature(self, traffic_summary):
        history_feature = get_traffic_history_feature(
            self.preprocess.traffic_history,
            phase_count=Config.DIM_OF_ACTION_PHASE,
            pressure_scale=Config.TRAFFIC_PRESSURE_SCALE,
            count_scale=Config.TRAFFIC_COUNT_SCALE,
            time_scale=Config.TRAFFIC_TIME_SCALE,
        )
        self.preprocess.traffic_history.append(traffic_summary)
        if len(self.preprocess.traffic_history) > Config.TRAFFIC_HISTORY_SIZE:
            del self.preprocess.traffic_history[0 : len(self.preprocess.traffic_history) - Config.TRAFFIC_HISTORY_SIZE]
        return history_feature

    def _lane_stat_feature(self, vehicles):
        lane_stats = get_lane_statistics(
            vehicles,
            waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
            lane_count=Config.GRID_WIDTH,
        )
        lane_feature = [
            float(np.clip(count / Config.LANE_COUNT_SCALE, 0.0, 1.0)) for count in lane_stats["counts"]
        ]
        lane_feature.extend(
            float(np.clip(queue / Config.LANE_COUNT_SCALE, 0.0, 1.0)) for queue in lane_stats["queues"]
        )
        lane_feature.extend(
            float(np.clip(wait / Config.TRAFFIC_TIME_SCALE, 0.0, 1.0))
            for wait in lane_stats["avg_waiting_times"]
        )
        return lane_feature

    def _phase_action_mask(self, legal_action):
        mask = np.asarray(
            normalize_phase_legal_action(legal_action, Config.DIM_OF_ACTION_PHASE),
            dtype=bool,
        )
        if not mask.any():
            mask[:] = True
        return mask

    def _joint_action_mask(self, phase_masks):
        phase_masks = np.asarray(phase_masks, dtype=bool)
        if phase_masks.ndim == 1:
            phase_masks = phase_masks.reshape(1, -1)
        joint_masks = np.repeat(phase_masks, Config.DIM_OF_ACTION_DURATION, axis=1)
        empty_rows = ~joint_masks.any(axis=1)
        if np.any(empty_rows):
            joint_masks[empty_rows, :] = True
        return joint_masks
