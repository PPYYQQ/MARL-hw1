#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _nonnegative_float(value, default=0.0):
    return max(_finite_float(value, default), 0.0)


def _phase_array(values, phase_count):
    try:
        array = np.asarray(values, dtype=np.float32).flatten()
    except Exception:
        array = np.asarray([], dtype=np.float32)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size < phase_count:
        array = np.pad(array, (0, phase_count - array.size))
    elif array.size > phase_count:
        array = array[:phase_count]
    return array.astype(np.float32)


def normalize_phase_legal_action(legal_action, phase_count=4):
    if legal_action is None:
        return [1] * phase_count

    try:
        values = np.asarray(legal_action, dtype=np.float32).flatten()
    except Exception:
        return [1] * phase_count

    if values.size == 0:
        return [1] * phase_count

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.size == 1:
        return [1 if values[0] != 0 else 0] * phase_count

    mask = [1 if value > 0 else 0 for value in values[:phase_count]]
    if len(mask) < phase_count:
        mask.extend([1] * (phase_count - len(mask)))
    return mask


def record_value(record, key, default=None):
    if isinstance(record, dict):
        try:
            return record.get(key, default)
        except Exception:
            return default
    try:
        return getattr(record, key, default)
    except Exception:
        return default


def vehicle_value(vehicle, key, default=None):
    return record_value(vehicle, key, default)


_vehicle_value = vehicle_value


def lane_value(lane, key, default=None):
    return record_value(lane, key, default)


def _first_lane_value(lane, keys, default=None):
    for key in keys:
        value = lane_value(lane, key, None)
        if value is not None:
            return value
    return default


def _safe_lane_id(value):
    try:
        lane_id = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(lane_id):
        return None
    return int(lane_id)


def _safe_junction_id(value, default=-1):
    try:
        junction_id = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(junction_id):
        return default
    return int(junction_id)


def _lane_id_from_record(lane):
    for key in ("lane_id", "laneId", "lane_idx", "laneIdx", "lane", "l_id", "lId", "id"):
        lane_id = _safe_lane_id(lane_value(lane, key))
        if lane_id is not None:
            return lane_id
    return None


def get_enter_lane_code():
    return {
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


def get_lane_code_by_id(lane_id):
    lane_id = _safe_lane_id(lane_id)
    if lane_id is None:
        return None
    return get_enter_lane_code().get(lane_id)


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
    lane_id = vehicle_value(vehicle, "lane")
    target_junction = vehicle_value(vehicle, "target_junction", None)
    target_junction = 0 if target_junction is None else _safe_junction_id(target_junction, -1)
    if get_lane_code_by_id(lane_id) is not None and target_junction != -1:
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
    junction = _safe_junction_id(vehicle_value(vehicle, "junction", -1), -1)
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
    junction = _safe_junction_id(vehicle_value(vehicle, "junction", -1), -1)
    target_junction = _safe_junction_id(vehicle_value(vehicle, "target_junction", -1), -1)
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
    lane_id = vehicle_value(vehicle, "lane")
    return get_lane_code_by_id(lane_id)


def get_lane_position_meters(vehicle):
    position = vehicle_value(vehicle, "position_in_lane", {})
    y_pos = float(record_value(position, "y"))
    if not np.isfinite(y_pos):
        raise ValueError("non-finite lane position")
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

            lane_code = get_lane_code(vehicle)
            if lane_code is None or lane_code < 0 or lane_code >= lane_count:
                continue

            speed = _nonnegative_float(vehicle_value(vehicle, "speed", 0.0))
            waiting_time = _nonnegative_float(vehicle_value(vehicle, "waiting_time", 0.0))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

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


def get_lane_observation_statistics(lanes, lane_count=14):
    counts = np.zeros(lane_count, dtype=np.float32)
    queues = np.zeros(lane_count, dtype=np.float32)
    avg_waiting_times = np.zeros(lane_count, dtype=np.float32)

    for lane in lanes:
        try:
            lane_code = get_lane_code_by_id(_lane_id_from_record(lane))
            if lane_code is None or lane_code < 0 or lane_code >= lane_count:
                continue

            vehicle_count = _nonnegative_float(
                _first_lane_value(lane, ("v_count", "vCount", "vehicle_count", "vehicleCount"), 0.0)
            )
            queue_length = _nonnegative_float(
                _first_lane_value(lane, ("queue_length", "queueLength", "queue_count", "queueCount", "queue"), 0.0)
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

        counts[lane_code] = max(counts[lane_code], vehicle_count)
        queues[lane_code] = max(queues[lane_code], queue_length)

    return {
        "counts": counts,
        "queues": queues,
        "avg_waiting_times": avg_waiting_times,
    }


def _lane_to_phase_map():
    lane_to_phase = {}
    for phase, lanes in get_webster_lane_group().items():
        for lane in lanes:
            lane_to_phase[lane] = int(phase)
    return lane_to_phase


def get_lane_observation_phase_pressure(lanes, phase_count=4):
    lane_to_phase = _lane_to_phase_map()
    phase_pressure = np.zeros(phase_count, dtype=np.float32)
    totals = {
        "waiting_time": 0.0,
        "delay": 0.0,
        "queue": 0.0,
        "vehicle_count": 0.0,
    }

    for lane in lanes:
        try:
            lane_id = _lane_id_from_record(lane)
            lane_phase = lane_to_phase.get(lane_id)
            if lane_phase is None or lane_phase < 0 or lane_phase >= phase_count:
                continue

            vehicle_count = _nonnegative_float(
                _first_lane_value(lane, ("v_count", "vCount", "vehicle_count", "vehicleCount"), 0.0)
            )
            queue_length = _nonnegative_float(
                _first_lane_value(lane, ("queue_length", "queueLength", "queue_count", "queueCount", "queue"), 0.0)
            )
            congestion = _nonnegative_float(
                _first_lane_value(lane, ("congestion", "congestion_level", "congestionLevel"), 0.0)
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

        pressure = vehicle_count + 2.0 * queue_length + min(congestion, 1.0) * 5.0
        if pressure <= 0.0:
            continue
        estimated_vehicle_count = max(vehicle_count, queue_length)
        if estimated_vehicle_count <= 0.0:
            estimated_vehicle_count = 1.0
        phase_pressure[lane_phase] += pressure
        totals["queue"] += queue_length
        totals["vehicle_count"] += estimated_vehicle_count

    return phase_pressure, totals


def merge_lane_observation_statistics(vehicle_stats, lane_stats):
    if lane_stats is None:
        return vehicle_stats

    merged = {}
    for key in ("counts", "queues", "avg_waiting_times"):
        vehicle_values = _phase_array(vehicle_stats.get(key, []), len(lane_stats[key]))
        lane_values = _phase_array(lane_stats.get(key, []), len(vehicle_values))
        merged[key] = np.maximum(vehicle_values, lane_values)
    return merged


def get_traffic_summary(vehicles, waiting_speed_threshold=0.1, phase_count=4, lanes=None):
    phase_pressure, totals = get_phase_pressure(
        vehicles,
        waiting_speed_threshold=waiting_speed_threshold,
        phase_count=phase_count,
    )
    if _nonnegative_float(totals["vehicle_count"]) <= 0.0 and lanes:
        phase_pressure, totals = get_lane_observation_phase_pressure(lanes, phase_count=phase_count)

    vehicle_count = _nonnegative_float(totals["vehicle_count"])
    queue_count = _nonnegative_float(totals["queue"])
    waiting_time = _nonnegative_float(totals["waiting_time"])
    delay = _nonnegative_float(totals["delay"])
    avg_waiting_time = waiting_time / vehicle_count if vehicle_count > 0 else 0.0
    avg_delay = delay / vehicle_count if vehicle_count > 0 else 0.0
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
    current_pressure_values = current_summary["phase_pressure"]
    current_pressure = _phase_array(current_pressure_values, len(current_pressure_values))
    if previous_summary is None:
        return [0.0] * (len(current_pressure) + 4)

    previous_pressure = _phase_array(
        previous_summary.get("phase_pressure", np.zeros_like(current_pressure)),
        len(current_pressure),
    )
    current_vehicle_count = _finite_float(current_summary.get("vehicle_count", 0.0))
    previous_vehicle_count = _finite_float(previous_summary.get("vehicle_count", 0.0))
    current_queue_ratio = _finite_float(current_summary.get("queue_ratio", 0.0))
    previous_queue_ratio = _finite_float(previous_summary.get("queue_ratio", 0.0))
    current_waiting_time = _finite_float(current_summary.get("avg_waiting_time", 0.0))
    previous_waiting_time = _finite_float(previous_summary.get("avg_waiting_time", 0.0))
    current_delay = _finite_float(current_summary.get("avg_delay", 0.0))
    previous_delay = _finite_float(previous_summary.get("avg_delay", 0.0))

    trend = [
        float(np.clip((current - previous) / pressure_scale, -1.0, 1.0))
        for current, previous in zip(current_pressure, previous_pressure)
    ]
    trend.extend(
        [
            float(
                np.clip(
                    (current_vehicle_count - previous_vehicle_count) / count_scale,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    current_queue_ratio - previous_queue_ratio,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (current_waiting_time - previous_waiting_time) / time_scale,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (current_delay - previous_delay) / time_scale,
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

    traffic_history = [summary for summary in traffic_history if isinstance(summary, dict)]
    if not traffic_history:
        return [0.0] * (phase_count + 4)

    phase_pressures = [
        _phase_array(summary.get("phase_pressure", np.zeros(phase_count)), phase_count) for summary in traffic_history
    ]
    avg_phase_pressure = np.mean(phase_pressures, axis=0)
    avg_vehicle_count = float(
        np.mean([_finite_float(summary.get("vehicle_count", 0.0)) for summary in traffic_history])
    )
    avg_queue_ratio = float(
        np.mean([_finite_float(summary.get("queue_ratio", 0.0)) for summary in traffic_history])
    )
    avg_waiting_time = float(
        np.mean([_finite_float(summary.get("avg_waiting_time", 0.0)) for summary in traffic_history])
    )
    avg_delay = float(np.mean([_finite_float(summary.get("avg_delay", 0.0)) for summary in traffic_history]))

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

            lane_phase = lane_to_phase.get(_safe_lane_id(vehicle_value(vehicle, "lane")))
            if lane_phase is None:
                continue

            speed = _nonnegative_float(vehicle_value(vehicle, "speed", 0.0))
            waiting_time = _nonnegative_float(vehicle_value(vehicle, "waiting_time", 0.0))
            delay = _nonnegative_float(vehicle_value(vehicle, "delay", 0.0))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        is_waiting = 1.0 if speed <= waiting_speed_threshold else 0.0

        pressure = 1.0 + 2.0 * is_waiting + min(waiting_time, 300.0) / 30.0 + min(delay, 300.0) / 60.0
        phase_pressure[lane_phase] += pressure
        totals["waiting_time"] += waiting_time
        totals["delay"] += delay
        totals["queue"] += is_waiting
        totals["vehicle_count"] += 1

    return phase_pressure, totals
