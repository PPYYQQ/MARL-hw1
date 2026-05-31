#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch


def _configure_torch_threads():
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_configure_torch_threads()

import os
import math
import time
from agent_target_dqn.model.model import Model
from agent_target_dqn.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_target_dqn.conf.conf import Config
from agent_target_dqn.algorithm.algorithm import Algorithm
from agent_target_dqn.feature.preprocessor import FeatureProcess


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _safe_nonnegative_float(value, default=0.0):
    return max(_safe_float(value, default), 0.0)


def _safe_int(value, default=0):
    return int(_safe_float(value, default))


def _safe_optional_int(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return int(value)


def _mapping_key(value, mapping, default=None):
    normalized_value = _safe_optional_int(value, None)
    if normalized_value is not None:
        if normalized_value in mapping:
            return normalized_value
        for key in mapping:
            if _safe_optional_int(key, None) == normalized_value:
                return key
    try:
        if value in mapping:
            return value
    except TypeError:
        pass
    return default


def _safe_mapping_get(mapping, key, default=None):
    if isinstance(mapping, dict):
        try:
            return mapping.get(key, default)
        except Exception:
            return default
    if mapping is None:
        return default
    try:
        return getattr(mapping, key, default)
    except Exception:
        return default


def _is_record(value):
    return value is not None and not isinstance(value, (str, bytes, bool, int, float, complex))


_RECORD_FIELD_KEYS = {
    "v_id",
    "v_config_id",
    "vehicle_config_id",
    "lane",
    "junction",
    "position_in_lane",
    "speed",
    "delay",
    "waiting_time",
    "s_id",
    "signal_id",
    "signal_idx",
    "phase_id",
    "phase_idx",
    "current_phase",
    "current_phase_id",
    "duration",
    "remaining_duration",
    "remaining_time",
    "remain_duration",
    "remain_time",
    "lane_id",
    "v_count",
    "congestion",
    "queue_length",
}


def _first_record_field(record, keys, default=None):
    for key in keys:
        value = _safe_mapping_get(record, key, None)
        if _is_record(value):
            return value
    return default


def _dict_record_items(value):
    try:
        if any(key in value for key in _RECORD_FIELD_KEYS):
            return [value]
        items = list(value.values())
    except Exception:
        return []

    values = []
    for item in items:
        if isinstance(item, (list, tuple)):
            values.extend(item)
        elif isinstance(item, dict):
            values.extend(_dict_record_items(item))
        elif _is_record(item):
            values.append(item)
    return values


def _as_record_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    elif isinstance(value, dict):
        values = _dict_record_items(value)
    elif isinstance(value, (str, bytes)):
        values = []
    else:
        try:
            values = list(value)
        except Exception:
            values = [value] if _is_record(value) else []
    return [item for item in values if _is_record(item)]


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
        list_obs_data = self._obs_batch(list_obs_data)
        list_obs_data = [
            obs_data for obs_data in list_obs_data if self._obs_data_field(obs_data, "feature") is not None
        ]
        if not list_obs_data:
            return []

        feature = [self._obs_data_field(obs_data, "feature") for obs_data in list_obs_data]
        phase_masks = np.asarray(
            [self._phase_action_mask(self._obs_data_field(obs_data, "legal_action")) for obs_data in list_obs_data],
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
        raw_obs = _first_record_field(observation, ("obs", "observation", "_obs"), observation)
        extra_info = _first_record_field(observation, ("extra_info", "_state", "state", "info"), None)
        if raw_obs is None:
            raw_obs = {}
        try:
            obs_data = self.observation_process(raw_obs, extra_info)
            if not obs_data:
                return self._safe_rule_based_action(raw_obs)
            act_data = self.__predict_detail([obs_data], exploit_flag=True)
            if not act_data:
                return self._safe_rule_based_action(raw_obs)
            return self.action_process(act_data[0])
        except Exception as err:
            self._log_error(f"exploit fallback to rule_based_action: {err}")
            return self._safe_rule_based_action(raw_obs)

    def _safe_rule_based_action(self, raw_obs):
        try:
            return self.rule_based_action(raw_obs)
        except Exception as err:
            self._log_error(f"rule_based_action failed, use default action: {err}")
            return [0, 0, Config.MIN_GREEN_DURATION]

    def _obs_batch(self, list_obs_data):
        if list_obs_data is None:
            return []
        try:
            return list(list_obs_data)
        except Exception as err:
            self._log_error(f"predict observation batch failed: {err}")
            return []

    def _obs_data_field(self, obs_data, name, default=None):
        try:
            return getattr(obs_data, name, default)
        except Exception:
            return default

    def learn(self, list_sample_data):
        try:
            return self.algorithm.learn(list_sample_data)
        except Exception as err:
            self._log_error(f"learn failed: {err}")
            return None

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        if path is None:
            path = "agent_target_dqn/ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_tmp_path = f"{model_file_path}.tmp"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        try:
            torch.save(model_state_dict_cpu, model_tmp_path)
            os.replace(model_tmp_path, model_file_path)
        except Exception:
            if os.path.exists(model_tmp_path):
                try:
                    os.remove(model_tmp_path)
                except OSError:
                    pass
            raise

        self._log_info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        if path is None:
            path = "agent_target_dqn/ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if str(id) == "latest":
                self._log_info(f"skip load model, {model_file_path} does not exist yet")
                return
            raise FileNotFoundError(model_file_path)
        try:
            model_state = torch.load(model_file_path, map_location=self.model.device)
        except Exception as err:
            if str(id) == "latest":
                self._log_info(f"skip load model, unreadable checkpoint {model_file_path}: {err}")
                return
            raise
        if isinstance(model_state, dict) and "state_dict" in model_state:
            model_state = model_state["state_dict"]
        if not isinstance(model_state, dict):
            err = TypeError(f"checkpoint payload should be dict, got {type(model_state).__name__}")
            if str(id) == "latest":
                self._log_info(f"skip load model, invalid checkpoint {model_file_path}: {err}")
                return
            raise err
        try:
            self.model.load_state_dict(model_state)
        except (RuntimeError, TypeError, ValueError) as err:
            if str(id) == "latest":
                self._log_info(f"skip load model, incompatible checkpoint {model_file_path}: {err}")
                return
            raise
        self.algorithm.update_target_q()

        self._log_info(f"load model {model_file_path} successfully")

    def _log_info(self, message):
        if not self.logger:
            return
        try:
            self.logger.info(message)
        except Exception:
            pass

    def _log_error(self, message):
        if not self.logger:
            return
        try:
            self.logger.error(message)
        except Exception:
            pass

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
        try:
            self.preprocess.update_traffic_info(raw_obs, extra_info)
        except Exception as err:
            self._log_error(f"traffic info update failed: {err}")

        # Note: The unpacking of the following raw data is for example purposes only,
        # please modify according to the actual situation
        # 注意: 以下原始数据的解包为示例, 请根据实际情况修改
        if raw_obs is None:
            raw_obs = {}
        frame_state = _safe_mapping_get(raw_obs, "frame_state", {})
        if not _is_record(frame_state):
            frame_state = {}

        # Parse frame_state
        # 解析 frame_state
        vehicles = _as_record_list(_safe_mapping_get(frame_state, "vehicles", []))
        lanes = _as_record_list(_safe_mapping_get(frame_state, "lanes", []))

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
        if junction_id not in speed_dict:
            speed_dict[junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))
            position_dict[junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))

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
                    vehicle_config_id = _safe_mapping_get(
                        vehicle,
                        "v_config_id",
                        _safe_mapping_get(vehicle, "vehicle_config_id"),
                    )
                    vehicle_config_key = _mapping_key(
                        vehicle_config_id,
                        self.preprocess.vehicle_configs_dict,
                        vehicle_config_id,
                    )
                    vehicle_config = self.preprocess.vehicle_configs_dict.get(vehicle_config_key, {})
                    max_speed = max(
                        _safe_float(_safe_mapping_get(vehicle_config, "max_speed"), Config.DEFAULT_MAX_SPEED),
                        1.0,
                    )
                    speed = _safe_nonnegative_float(_safe_mapping_get(vehicle, "speed", 0.0))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue

                if y_pos < 0 or y_pos >= Config.GRID_NUM:
                    continue

                target_junction = _mapping_key(
                    _safe_mapping_get(vehicle, "target_junction", junction_id),
                    speed_dict,
                    junction_id,
                )
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
            lanes=lanes,
        )
        traffic_feature = self._traffic_feature(traffic_summary)
        traffic_trend_feature = self._traffic_trend_feature(traffic_summary)
        traffic_history_feature = self._traffic_history_feature(traffic_summary)
        lane_stat_feature = self._lane_stat_feature(vehicles, lanes)
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
        observation = self._sanitize_observation(observation)

        return ObsData(
            feature=observation,
            legal_action=normalize_phase_legal_action(
                _safe_mapping_get(raw_obs, "legal_action"),
                Config.DIM_OF_ACTION_PHASE,
            ),
        )

    def _sanitize_observation(self, observation):
        try:
            values = np.asarray(observation, dtype=np.float32).flatten()
        except (TypeError, ValueError):
            values = np.asarray([], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.size < Config.DIM_OF_OBSERVATION:
            values = np.pad(values, (0, Config.DIM_OF_OBSERVATION - values.size))
        elif values.size > Config.DIM_OF_OBSERVATION:
            values = values[: Config.DIM_OF_OBSERVATION]
        return values.astype(np.float32).tolist()

    def action_process(self, act_data):
        junction_id = 0
        phase_index = self._safe_action_index(
            getattr(act_data, "phase_index", 0),
            Config.DIM_OF_ACTION_PHASE,
        )
        duration_index = self._safe_action_index(
            getattr(act_data, "duration", 0),
            Config.DIM_OF_ACTION_DURATION,
        )
        duration = Config.duration_index_to_seconds(duration_index)
        return [junction_id, phase_index, duration]

    def _safe_action_index(self, value, action_dim):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        return int(np.clip(value, 0, action_dim - 1))

    def rule_based_action(self, raw_obs):
        if raw_obs is None:
            raw_obs = {}
        frame_state = _safe_mapping_get(raw_obs, "frame_state", {})
        if not _is_record(frame_state):
            frame_state = {}
        vehicles = _as_record_list(_safe_mapping_get(frame_state, "vehicles", []))
        lanes = _as_record_list(_safe_mapping_get(frame_state, "lanes", []))
        phase_pressure, _ = get_phase_pressure(
            vehicles,
            waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
            phase_count=Config.DIM_OF_ACTION_PHASE,
        )
        if not np.any(phase_pressure) and lanes:
            phase_pressure, _ = get_lane_observation_phase_pressure(
                lanes,
                phase_count=Config.DIM_OF_ACTION_PHASE,
            )
        phase_age_feature = np.asarray(self._phase_age_feature(frame_state), dtype=np.float32)
        fair_pressure = phase_pressure * (1.0 + Config.FAIRNESS_BONUS_SCALE * phase_age_feature)
        legal_mask = self._phase_action_mask(_safe_mapping_get(raw_obs, "legal_action"))
        masked_pressure = np.where(legal_mask, fair_pressure, -1.0)
        phase_index = int(np.argmax(masked_pressure))
        duration_index = int(np.clip(round(float(phase_pressure[phase_index])), 0, Config.DIM_OF_ACTION_DURATION - 1))
        duration = Config.duration_index_to_seconds(duration_index)
        return [
            0,
            phase_index,
            duration,
        ]

    def _phase_feature(self, frame_state):
        phase_info = self._current_phase_info(frame_state)

        phase_feature = [0.0] * Config.DIM_OF_ACTION_PHASE
        if not phase_info:
            return phase_feature + [0.0, 0.0, 0.0, 0.0]

        phase_id = self._safe_action_index(
            self._phase_record_value(phase_info, ("phase_id", "phase_idx", "current_phase", "current_phase_id"), 0),
            Config.DIM_OF_ACTION_PHASE,
        )
        duration = _safe_nonnegative_float(self._phase_record_value(phase_info, ("duration",), 0.0))
        remaining_duration = _safe_nonnegative_float(
            self._phase_record_value(
                phase_info,
                ("remaining_duration", "remaining_time", "remain_duration", "remain_time"),
                0.0,
            )
        )
        elapsed_duration = max(duration - remaining_duration, 0.0)

        phase_feature[phase_id] = 1.0
        return phase_feature + [
            float(np.clip(duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            float(np.clip(remaining_duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            float(np.clip(elapsed_duration / Config.MAX_GREEN_DURATION, 0.0, 1.0)),
            1.0,
        ]

    def _phase_age_feature(self, frame_state):
        frame_no = _safe_int(_safe_mapping_get(frame_state, "frame_no", 0))
        phase_info = self._current_phase_info(frame_state)
        last_served = self.preprocess.phase_last_served_frame
        if not isinstance(last_served, list) or len(last_served) != Config.DIM_OF_ACTION_PHASE:
            last_served = [None] * Config.DIM_OF_ACTION_PHASE
        last_served = [
            frame_no if served_frame is None else _safe_int(served_frame, frame_no)
            for served_frame in last_served
        ]
        self.preprocess.phase_last_served_frame = last_served
        if phase_info:
            phase_id = self._safe_action_index(
                self._phase_record_value(phase_info, ("phase_id", "phase_idx", "current_phase", "current_phase_id"), 0),
                Config.DIM_OF_ACTION_PHASE,
            )
            last_served[phase_id] = frame_no
        return [
            float(np.clip((frame_no - served_frame) / Config.PHASE_AGE_SCALE, 0.0, 1.0))
            for served_frame in last_served
        ]

    def _current_phase_info(self, frame_state):
        phases = _as_record_list(_safe_mapping_get(frame_state, "phases", []))
        for candidate in phases:
            if self._phase_record_value(candidate, ("s_id", "signal_id", "signal_idx"), 0) == 0:
                return candidate
        for candidate in phases:
            return candidate
        return {}

    def _phase_record_value(self, phase_info, keys, default=None):
        for key in keys:
            value = _safe_mapping_get(phase_info, key, None)
            if value is not None:
                return value
        return default

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

    def _lane_stat_feature(self, vehicles, lanes=None):
        lane_stats = get_lane_statistics(
            vehicles,
            waiting_speed_threshold=Config.WAITING_SPEED_THRESHOLD,
            lane_count=Config.GRID_WIDTH,
        )
        if lanes:
            lane_stats = merge_lane_observation_statistics(
                lane_stats,
                get_lane_observation_statistics(lanes, lane_count=Config.GRID_WIDTH),
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
