#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_target_dqn.feature.traffic_utils import *


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


def _safe_junction_id(value, default=None):
    try:
        junction_id = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(junction_id):
        return default
    return int(junction_id)


def _junction_key(value, junction_ids, default=None):
    missing = object()
    junction_id = _safe_junction_id(value, missing)
    if junction_id is missing:
        return default
    if junction_id in junction_ids:
        return junction_id
    for existing_junction_id in junction_ids:
        if _safe_junction_id(existing_junction_id, None) == junction_id:
            return existing_junction_id
    return junction_id


def _is_hashable(value):
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_record(value):
    return value is not None and not isinstance(value, (str, bytes, bool, int, float, complex))


_RECORD_FIELD_KEYS = {
    "j_id",
    "junction_id",
    "e_id",
    "edge_id",
    "l_id",
    "lane_id",
    "v_id",
    "v_config_id",
    "vehicle_config_id",
    "lane",
    "junction",
    "position_in_lane",
    "speed",
    "delay",
    "waiting_time",
    "v_count",
    "congestion",
    "queue_length",
    "lanes",
    "enter_lanes_on_directions",
    "lane_configs",
    "vehicle_configs",
    "max_speed",
}


def _first_record_value(record, *keys, default=None):
    for key in keys:
        value = record_value(record, key, None)
        if value is not None:
            return value
    return default


def _dict_list_items(value):
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
            values.extend(_dict_list_items(item))
        else:
            values.append(item)
    return values


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return _dict_list_items(value)
    if isinstance(value, (str, bytes)):
        return []
    try:
        return list(value)
    except Exception:
        return [value]


def _safe_position_pair(vehicle):
    position = vehicle_value(vehicle, "position_in_lane", {})
    x_pos = float(record_value(position, "x"))
    y_pos = float(record_value(position, "y"))
    if not math.isfinite(x_pos) or not math.isfinite(y_pos):
        raise ValueError("non-finite vehicle position")
    return x_pos, y_pos


def _default_target_junction(junction_ids):
    junction_zero = _junction_key(0, junction_ids, None)
    if junction_zero is not None:
        return junction_zero
    for junction_id in junction_ids:
        return junction_id
    return None


def _waiting_target_junction(vehicle, junction_ids):
    target_junction = vehicle_value(vehicle, "target_junction", None)
    if target_junction is not None:
        return _junction_key(target_junction, junction_ids, None)
    if on_enter_lane(vehicle):
        return _default_target_junction(junction_ids)
    return None


class FeatureProcess:
    """
    Update traffic information and perform feature processing
    """

    """
    更新交通信息并进行特征处理
    """

    def __init__(self, logger):
        self.logger = logger
        self.reset()

    def reset(self):
        # Store road structure and other relatively fixed dictionary-type variables
        # 存储道路结构等相对固定的字典型变量
        self.junction_dict = {}
        self.edge_dict = {}
        self.lane_dict = {}
        self.l_id_to_index = {}
        self.vehicle_configs_dict = {}

        # Store dictionary-type variables for dynamic traffic information in traffic scenarios,
        # and update data after each communication
        # 存储交通场景中动态交通信息的字典型变量, 在每次通信后进行数据更新
        self.vehicle_prev_junction = {}
        self.vehicle_prev_position = {}
        self.vehicle_distance_store = {}
        self.last_waiting_moment = {}
        self.waiting_time_store = {}
        self.enter_lane_time = {}

        # Stores variables that can be used to calculate rewards
        # 存储可用于计算奖励的变量
        self.lane_volume = {}

        # User-defined variable
        # 用户自定义变量
        self.old_waiting_time = 0
        self.last_phase_index = None
        self.last_traffic_summary = None
        self.traffic_history = []
        self.phase_last_served_frame = [None, None, None, None]

    def init_road_info(self, start_info):
        """
        Obtain fixed variables such as road structure
        """
        """
        获取道路结构等信息固定的变量
        """
        if not _is_record(start_info):
            return
        junctions, signals, edges = (
            _safe_list(record_value(start_info, "junctions", [])),
            _safe_list(record_value(start_info, "signals", [])),
            _safe_list(record_value(start_info, "edges", [])),
        )
        lane_configs, vehicle_configs = (
            _safe_list(record_value(start_info, "lane_configs", [])),
            _safe_list(record_value(start_info, "vehicle_configs", [])),
        )
        # Store road structure information in various variables
        # 将道路结构信息存储到各个变量
        for junction in junctions:
            junction_id = _first_record_value(junction, "j_id", "junction_id")
            if junction_id is None:
                continue
            self.junction_dict[junction_id] = junction
            self.l_id_to_index[junction_id] = {}

            index = 0
            for approaching_edges in _safe_list(record_value(junction, "enter_lanes_on_directions", [])):
                for lane in _safe_list(record_value(approaching_edges, "lanes", [])):
                    self.l_id_to_index[junction_id][lane] = index
                    index += 1

            for edge in edges:
                edge_id = _first_record_value(edge, "e_id", "edge_id")
                if edge_id is not None:
                    self.edge_dict[edge_id] = edge
            for lane in lane_configs:
                lane_id = _first_record_value(lane, "l_id", "lane_id")
                if lane_id is not None:
                    self.lane_dict[lane_id] = lane
            for vehicle_config in vehicle_configs:
                vehicle_config_id = _first_record_value(
                    vehicle_config,
                    "v_config_id",
                    "vehicle_config_id",
                )
                if vehicle_config_id is not None:
                    self.vehicle_configs_dict[vehicle_config_id] = vehicle_config
            for lane in lane_configs:
                lane_id = _first_record_value(lane, "l_id", "lane_id")
                if lane_id is not None:
                    self.lane_volume[lane_id] = []

    def update_traffic_info(self, raw_obs, extra_info):
        """
        Update vehicle history information and calculate various dynamic traffic variables
        """
        """
        更新车辆历史信息, 计算各项动态交通变量
        """
        extra_info = extra_info if _is_record(extra_info) else {}
        frame_state = record_value(raw_obs, "frame_state")
        if not _is_record(frame_state):
            return
        frame_no = _safe_int(record_value(frame_state, "frame_no", 0))
        frame_time = _safe_float(record_value(frame_state, "frame_time", 0))
        vehicles = _safe_list(record_value(frame_state, "vehicles", []))

        if frame_no <= 1:
            # Initial frame loads road structure information
            # 初始帧载入道路结构信息
            game_info = record_value(extra_info, "init_state", {})
            if game_info:
                self.init_road_info(game_info)

        for vehicle in vehicles:
            if not _is_record(vehicle):
                continue
            vehicle_id = vehicle_value(vehicle, "v_id")
            if vehicle_id is None or not _is_hashable(vehicle_id):
                continue
            vehicle_junction = _safe_junction_id(vehicle_value(vehicle, "junction", -1), -1)
            try:
                is_enter_lane = on_enter_lane(vehicle)
            except (KeyError, TypeError, ValueError, AttributeError):
                is_enter_lane = False
            # If the vehicle appears for the first time, initialize the vehicle's historical intersection information
            # 如果车辆第一次出现，则初始化车辆的历史交叉口信息
            if vehicle_id not in self.vehicle_prev_junction:
                self.vehicle_prev_junction[vehicle_id] = vehicle_junction
            # For vehicles that appear for the first time, if they are on the lane, record their appearance time
            # 对于首次出现的车辆, 若在车道上则记录其出现时间
            if (
                self.vehicle_prev_junction[vehicle_id] == -1
                and is_enter_lane
                and vehicle_id not in self.enter_lane_time
            ):
                self.enter_lane_time[vehicle_id] = frame_time
            # When a vehicle enters another entrance lane from the intersection, recalculate its appearance time
            # 当车辆从交叉口驶入另一进口车道时, 重新统计其出现时间
            elif self.vehicle_prev_junction[vehicle_id] != vehicle_junction:
                if self.vehicle_prev_junction[vehicle_id] != -1 and is_enter_lane:
                    self.enter_lane_time[vehicle_id] = frame_time

            try:
                self.cal_waiting_time(frame_time, vehicle)
                self.cal_travel_distance(vehicle)
                self.cal_v_num_in_lane(vehicle)
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                continue

    def cal_waiting_time(self, frame_time, vehicle):
        """
        Calculate the waiting time of the vehicle. When the vehicle is on the enter lane,
        count the accumulated time when its speed is <= 0.1m/s as its waiting time when driving at the intersection
        """
        """
        计算车辆等待时间, 当车辆处于进口车道上时, 统计其车速<=0.1m/s的累计时长作为其在该交叉口行驶时的等待时间
        """
        waiting_time = 0
        # Determine whether the vehicle in the lane approaching the intersection is in a waiting state,
        # and calculate the waiting time
        # 对处于车道驶向交叉口的车辆判断是否处于等待状态, 计算等待时间
        if on_enter_lane(vehicle):
            vehicle_id = vehicle_value(vehicle, "v_id")
            speed = _safe_nonnegative_float(vehicle_value(vehicle, "speed", 0.0))
            # Determine whether the vehicle is in a waiting state.
            # The determination condition is that the vehicle speed is <= 0.1m/s
            # 判断车辆是否处于等待状态, 判定条件为车辆速度<=0.1m/s
            if speed <= 0.1:
                if vehicle_id not in self.last_waiting_moment:
                    # Record the starting moment of each time the vehicle enters the waiting state
                    # 记录车辆在每次进入等待状态的起始时刻
                    self.last_waiting_moment[vehicle_id] = frame_time
                    # When the vehicle is in the waiting state for the first time,
                    # initialize its accumulated waiting time
                    # 车辆首次处于等待状态则初始化车辆累计等待时间
                    if vehicle_id not in self.waiting_time_store:
                        self.waiting_time_store[vehicle_id] = 0
                else:
                    # When a vehicle enters the waiting state on a lane,
                    # waiting_time records the duration of the current waiting state
                    # 车辆在一条道路上进入等待状态, waiting_time记录本次等待状态已持续的时间
                    waiting_time = max(frame_time - self.last_waiting_moment[vehicle_id], 0.0)
                    self.waiting_time_store[vehicle_id] += waiting_time
                    self.last_waiting_moment[vehicle_id] = frame_time
            else:
                if vehicle_id in self.last_waiting_moment:
                    del self.last_waiting_moment[vehicle_id]
        else:
            vehicle_id = vehicle_value(vehicle, "v_id")
            # Prevent repeated del when the vehicle is generated for the first time or at an intersection,
            # v_id is not stored in self.waiting_time_store
            # 防止车辆首次生成或位于交叉口时反复del, v_id未储存在self.waiting_time_store内
            if vehicle_id in self.waiting_time_store:
                del self.waiting_time_store[vehicle_id]
            if vehicle_id in self.last_waiting_moment:
                del self.last_waiting_moment[vehicle_id]

    def cal_travel_distance(self, vehicle):
        """
        Calculate the travel distance. When the vehicle is on the enter lane,
        count the total distance it travels at the intersection
        """
        """
        计算旅行路程, 当车辆处于进口车道上时, 统计其在该交叉口行驶时的总路程
        """
        # When the vehicle is on the lane, calculate the cumulative distance
        # 当车辆处于车道上时, 计算累计路程
        if on_enter_lane(vehicle):
            vehicle_id = vehicle_value(vehicle, "v_id")
            x_pos, y_pos = _safe_position_pair(vehicle)
            # When the vehicle enters the lane from inside the intersection for the second or subsequent time,
            # clear the cumulative distance and prepare to calculate the distance of this entry into the inlane
            # 车辆非首次从交叉口内部驶入车道时, 清空累计路程, 准备计算此次进入进口车道的路程
            if self.vehicle_prev_junction.get(vehicle_id, -1) != -1 and vehicle_id in self.vehicle_distance_store:
                del self.vehicle_distance_store[vehicle_id]
            if vehicle_id not in self.vehicle_distance_store:
                self.vehicle_distance_store[vehicle_id] = 0
                self.vehicle_prev_position[vehicle_id] = [x_pos, y_pos]
            else:
                if vehicle_id in self.vehicle_distance_store and vehicle_id in self.vehicle_prev_position:
                    try:
                        prev_x, prev_y = self.vehicle_prev_position[vehicle_id]
                        # Calculate Euclidean distance
                        # 计算欧氏距离
                        distance = math.sqrt(math.pow(x_pos - float(prev_x), 2) + math.pow(y_pos - float(prev_y), 2))
                        if not math.isfinite(distance):
                            raise ValueError
                        self.vehicle_distance_store[vehicle_id] += distance
                    except Exception:
                        raise ValueError
            # Update the vehicle's historical position after each distance calculation
            # 每次计算距离后更新车辆历史位置
            self.vehicle_prev_position[vehicle_id] = [x_pos, y_pos]
        else:
            # When the vehicle enters the intersection,
            # delete the historical location information to avoid calculating the driving distance
            # based on the last departure position when entering the lane next time
            # 当车辆驶入交叉口, 删除历史位置信息, 避免下次进入车道时按上一次离开路口位置计算行驶距离
            vehicle_id = vehicle_value(vehicle, "v_id")
            if vehicle_id in self.vehicle_prev_position:
                del self.vehicle_prev_position[vehicle_id]

    def cal_v_num_in_lane(self, vehicle):
        """
        Calculate the number of vehicles on the lane.
        When a vehicle is in the import lane, the number of vehicles on the enter lane increases
        """
        """
        计算车道上的车辆数, 当车辆处于进口车道上时, 则该进口车道上车辆数增加
        """
        # Update the number of vehicles on each lane
        # 更新每条车道上的车辆数量
        if on_enter_lane(vehicle):
            lane_id = vehicle_value(vehicle, "lane")
            if not _is_hashable(lane_id):
                return
            if lane_id not in self.lane_volume:
                # Defensive handling: initialize the lane
                # 防御性处理：初始化该车道
                self.lane_volume[lane_id] = []

            vehicle_id = vehicle_value(vehicle, "v_id")
            if vehicle_id not in self.lane_volume[lane_id]:
                self.lane_volume[lane_id].append(vehicle_id)

        # Update the vehicle's historical intersection information
        # 更新车辆的历史所在交叉口信息
        self.vehicle_prev_junction[vehicle_value(vehicle, "v_id")] = _safe_junction_id(
            vehicle_value(vehicle, "junction", -1), -1
        )

    def get_all_junction_waiting_time(self, vehicles: list):
        """
        This function obtain a dict of waiting_time by junction

        Args:
            - vehicles (list): input list of Vehicle
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        Returns:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        """
        此函数获取交叉口车辆等待时间的字典

        参数:
            - vehicles (list): 车辆的输入列表
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        返回:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        res = {}
        v_num = {}
        for junction_id in self.junction_dict:
            res[junction_id] = 0
            v_num[junction_id] = 0
        vehicles = _safe_list(vehicles)
        for vehicle in vehicles:
            if not _is_record(vehicle):
                continue
            target_junction = _waiting_target_junction(vehicle, res)
            junction = _safe_junction_id(vehicle_value(vehicle, "junction", -1), -1)
            if junction != -1 or target_junction == -1 or target_junction not in res:
                continue
            vehicle_id = vehicle_value(vehicle, "v_id")
            if vehicle_id is None or not _is_hashable(vehicle_id):
                continue
            if vehicle_id is not None and _is_hashable(vehicle_id) and vehicle_id in self.waiting_time_store:
                t = _safe_nonnegative_float(self.waiting_time_store[vehicle_id])
            else:
                t = 0
            res[target_junction] += t
            v_num[target_junction] += 1
        # Calculate the average waiting time of all vehicles in the scene
        # 计算场景内所有车辆的平均等待时间
        for junction_id in self.junction_dict:
            if v_num[junction_id] != 0:
                res[junction_id] /= v_num[junction_id]
        return res

    def get_all_junction_waiting_time_by_origin(self, vehicles: list):
        """
        This function obtain a dict of waiting_time by junction from the obs vehicles

        Args:
            - vehicles (list): input list of Vehicle
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        Returns:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        """
        此函数获取交叉口车辆等待时间的字典

        参数:
            - vehicles (list): 车辆的输入列表
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        返回:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        res = {}
        v_num = {}
        for junction_id in self.junction_dict:
            res[junction_id] = 0
            v_num[junction_id] = 0
        vehicles = _safe_list(vehicles)
        for vehicle in vehicles:
            if not _is_record(vehicle):
                continue
            target_junction = _waiting_target_junction(vehicle, res)
            junction = _safe_junction_id(vehicle_value(vehicle, "junction", -1), -1)
            if junction != -1 or target_junction == -1 or target_junction not in res:
                continue
            res[target_junction] += _safe_nonnegative_float(vehicle_value(vehicle, "waiting_time", 0.0))
            v_num[target_junction] += 1
        # Calculate the average waiting time of all vehicles in the scene
        # 计算场景内所有车辆的平均等待时间
        for junction_id in self.junction_dict:
            if v_num[junction_id] != 0:
                res[junction_id] /= v_num[junction_id]
        return res
