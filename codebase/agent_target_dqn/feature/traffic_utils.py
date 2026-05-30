#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


def normalize_phase_legal_action(legal_action, phase_count=4):
    if legal_action is None:
        return [1] * phase_count

    try:
        values = np.asarray(legal_action, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return [1] * phase_count

    if values.size == 0:
        return [1] * phase_count

    if values.size == 1:
        return [1 if values[0] != 0 else 0] * phase_count

    mask = [1 if value > 0 else 0 for value in values[:phase_count]]
    if len(mask) < phase_count:
        mask.extend([1] * (phase_count - len(mask)))
    return mask


def on_enter_lane(vehicle):
    """
    This function determines whether the vehicle is located on the enter lane

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于进口车道上

    参数:
        - vehicle
    """
    lane_id = vehicle["lane"]
    inlane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    if lane_id in inlane_code and vehicle["target_junction"] != -1:
        return True
    else:
        return False


def in_junction(vehicle):
    """
    This function determines whether the vehicle is located in the junction

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于交叉口中

    参数:
        - vehicle
    """
    junction = vehicle["junction"]
    target_junction = vehicle["target_junction"]
    if junction != -1:
        return True
    else:
        return False


def on_depart_lane(vehicle):
    """
    This function determines whether the vehicle is located on the depart lane

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于出口车道上

    参数:
        - vehicle
    """
    junction = vehicle["junction"]
    target_junction = vehicle["target_junction"]
    # Prevent vehicles in the right turn lane from being judged as being in the exit lane
    # 避免车辆在右转车道被判定为在出口车道上
    if (on_enter_lane(vehicle) or in_junction(vehicle)) or (junction == -1 and target_junction != -1):
        return False
    else:
        return True


def get_lane_code(vehicle):
    """
    This function divides each import lane into a different number of grids according to
    different rules and classifies them

    Args:
        - lane_id: The ID of the lane where the vehicle is located

    Returns:
        - lane_code: The number assigned to the lane according to the division rule
    """
    """
    此函数将各进口车道按不同规则划分为不同数量的栅格, 并对其进行分类

    参数:
        - lane_id: 车辆所处车道的id

    返回:
        - lane_code: 根据划分规则分配给该车道的编号
    """
    lane_id = vehicle["lane"]
    lane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    return lane_code.get(lane_id)


def get_lane_position_meters(vehicle):
    y_pos = float(vehicle["position_in_lane"]["y"])
    if abs(y_pos) > 200:
        y_pos /= 1000.0
    return y_pos


def get_lane_statistics(vehicles, waiting_speed_threshold=0.1, lane_count=14):
    counts = np.zeros(lane_count, dtype=np.float32)
    queues = np.zeros(lane_count, dtype=np.float32)
    waiting_times = np.zeros(lane_count, dtype=np.float32)

    for vehicle in vehicles:
        try:
            if not on_enter_lane(vehicle):
                continue
        except KeyError:
            continue

        lane_code = get_lane_code(vehicle)
        if lane_code is None or lane_code < 0 or lane_code >= lane_count:
            continue

        speed = float(vehicle.get("speed", 0.0))
        waiting_time = float(vehicle.get("waiting_time", 0.0))
        is_waiting = 1.0 if speed <= waiting_speed_threshold else 0.0

        counts[lane_code] += 1.0
        queues[lane_code] += is_waiting
        waiting_times[lane_code] += waiting_time

    avg_waiting_times = np.divide(waiting_times, counts, out=np.zeros_like(waiting_times), where=counts > 0)
    return {
        "counts": counts,
        "queues": queues,
        "avg_waiting_times": avg_waiting_times,
    }


def get_traffic_summary(vehicles, waiting_speed_threshold=0.1, phase_count=4):
    phase_pressure, totals = get_phase_pressure(
        vehicles,
        waiting_speed_threshold=waiting_speed_threshold,
        phase_count=phase_count,
    )
    vehicle_count = float(totals["vehicle_count"])
    queue_count = float(totals["queue"])
    avg_waiting_time = totals["waiting_time"] / vehicle_count if vehicle_count > 0 else 0.0
    avg_delay = totals["delay"] / vehicle_count if vehicle_count > 0 else 0.0
    return {
        "phase_pressure": phase_pressure,
        "vehicle_count": vehicle_count,
        "queue_count": queue_count,
        "queue_ratio": queue_count / max(vehicle_count, 1.0),
        "avg_waiting_time": avg_waiting_time,
        "avg_delay": avg_delay,
    }


def get_traffic_trend(
    current_summary,
    previous_summary,
    pressure_scale=50.0,
    count_scale=100.0,
    time_scale=120.0,
):
    current_pressure = np.asarray(current_summary["phase_pressure"], dtype=np.float32)
    if previous_summary is None:
        return [0.0] * (len(current_pressure) + 4)

    previous_pressure = np.asarray(
        previous_summary.get("phase_pressure", np.zeros_like(current_pressure)),
        dtype=np.float32,
    )
    if previous_pressure.size < current_pressure.size:
        previous_pressure = np.pad(previous_pressure, (0, current_pressure.size - previous_pressure.size))
    previous_pressure = previous_pressure[: current_pressure.size]

    trend = [
        float(np.clip((current - previous) / pressure_scale, -1.0, 1.0))
        for current, previous in zip(current_pressure, previous_pressure)
    ]
    trend.extend(
        [
            float(
                np.clip(
                    (current_summary["vehicle_count"] - previous_summary.get("vehicle_count", 0.0)) / count_scale,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    current_summary["queue_ratio"] - previous_summary.get("queue_ratio", 0.0),
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (current_summary["avg_waiting_time"] - previous_summary.get("avg_waiting_time", 0.0))
                    / time_scale,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (current_summary["avg_delay"] - previous_summary.get("avg_delay", 0.0)) / time_scale,
                    -1.0,
                    1.0,
                )
            ),
        ]
    )
    return trend


def get_traffic_history_feature(
    traffic_history,
    phase_count=4,
    pressure_scale=50.0,
    count_scale=100.0,
    time_scale=120.0,
):
    if not traffic_history:
        return [0.0] * (phase_count + 4)

    phase_pressures = [
        np.asarray(summary.get("phase_pressure", np.zeros(phase_count)), dtype=np.float32)[:phase_count]
        for summary in traffic_history
    ]
    avg_phase_pressure = np.mean(phase_pressures, axis=0)
    avg_vehicle_count = float(np.mean([summary.get("vehicle_count", 0.0) for summary in traffic_history]))
    avg_queue_ratio = float(np.mean([summary.get("queue_ratio", 0.0) for summary in traffic_history]))
    avg_waiting_time = float(np.mean([summary.get("avg_waiting_time", 0.0) for summary in traffic_history]))
    avg_delay = float(np.mean([summary.get("avg_delay", 0.0) for summary in traffic_history]))

    history_feature = [
        float(np.clip(pressure / pressure_scale, 0.0, 1.0)) for pressure in avg_phase_pressure
    ]
    history_feature.extend(
        [
            float(np.clip(avg_vehicle_count / count_scale, 0.0, 1.0)),
            float(np.clip(avg_queue_ratio, 0.0, 1.0)),
            float(np.clip(avg_waiting_time / time_scale, 0.0, 1.0)),
            float(np.clip(avg_delay / time_scale, 0.0, 1.0)),
        ]
    )
    return history_feature


def get_webster_lane_group():
    """
    Classify according to the green light phase corresponding to each import lane,
    such as "1" corresponding to the [8, 20] lane corresponding to the north-south left turn phase
    """
    """
    根据各进口车道所对应的绿灯通行相位进行分类, 如"1"对应的[8, 20]车道对应南北左转相位
    """
    lane_group = {
        "0": [11, 10, 9, 23, 22, 21],
        "1": [8, 20],
        "2": [129, 128, 127, 163],
        "3": [126, 162],
    }
    return lane_group


def get_phase_pressure(vehicles, waiting_speed_threshold=0.1, phase_count=4):
    lane_to_phase = {}
    for phase, lanes in get_webster_lane_group().items():
        for lane in lanes:
            lane_to_phase[lane] = int(phase)

    phase_pressure = np.zeros(phase_count, dtype=np.float32)
    totals = {
        "waiting_time": 0.0,
        "delay": 0.0,
        "queue": 0.0,
        "vehicle_count": 0,
    }

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
        is_waiting = 1.0 if speed <= waiting_speed_threshold else 0.0

        pressure = 1.0 + 2.0 * is_waiting + min(waiting_time, 300.0) / 30.0 + min(delay, 300.0) / 60.0
        phase_pressure[lane_phase] += pressure
        totals["waiting_time"] += waiting_time
        totals["delay"] += delay
        totals["queue"] += is_waiting
        totals["vehicle_count"] += 1

    return phase_pressure, totals
