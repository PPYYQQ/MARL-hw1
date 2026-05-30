#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from agent_target_dqn.feature.traffic_utils import (
        get_lane_statistics,
        get_traffic_history_feature,
        get_traffic_summary,
        get_traffic_trend,
        normalize_phase_legal_action,
    )

    assert normalize_phase_legal_action(None) == [1, 1, 1, 1]
    assert normalize_phase_legal_action([]) == [1, 1, 1, 1]
    assert normalize_phase_legal_action([1]) == [1, 1, 1, 1]
    assert normalize_phase_legal_action([0]) == [0, 0, 0, 0]
    assert normalize_phase_legal_action([1, 0, 2, -1]) == [1, 0, 1, 0]
    assert normalize_phase_legal_action([0, 1]) == [0, 1, 1, 1]
    assert normalize_phase_legal_action("bad-input") == [1, 1, 1, 1]

    vehicles = [
        {
            "lane": 11,
            "junction": -1,
            "target_junction": 0,
            "speed": 0.0,
            "waiting_time": 24.0,
        },
        {
            "lane": 129,
            "junction": -1,
            "target_junction": 0,
            "speed": 8.0,
            "waiting_time": 0.0,
        },
        {
            "lane": 999,
            "junction": -1,
            "target_junction": 0,
            "speed": 0.0,
            "waiting_time": 10.0,
        },
    ]
    lane_stats = get_lane_statistics(vehicles)
    assert lane_stats["counts"][0] == 1.0
    assert lane_stats["queues"][0] == 1.0
    assert lane_stats["avg_waiting_times"][0] == 24.0
    assert lane_stats["counts"][4] == 1.0
    assert lane_stats["queues"][4] == 0.0

    traffic_summary = get_traffic_summary(vehicles)
    assert traffic_summary["vehicle_count"] == 2.0
    assert traffic_summary["queue_count"] == 1.0
    assert traffic_summary["queue_ratio"] == 0.5

    zero_trend = get_traffic_trend(traffic_summary, None)
    assert zero_trend == [0.0] * 8

    previous_summary = {
        "phase_pressure": [0.0, 0.0, 0.0, 0.0],
        "vehicle_count": 0.0,
        "queue_ratio": 0.0,
        "avg_waiting_time": 0.0,
        "avg_delay": 0.0,
    }
    traffic_trend = get_traffic_trend(traffic_summary, previous_summary)
    assert traffic_trend[0] > 0.0
    assert traffic_trend[2] > 0.0
    assert traffic_trend[4] == 0.02
    assert traffic_trend[5] == 0.5

    zero_history = get_traffic_history_feature([])
    assert zero_history == [0.0] * 8

    history_feature = get_traffic_history_feature([traffic_summary, traffic_summary])
    assert history_feature[0] > 0.0
    assert history_feature[2] > 0.0
    assert history_feature[4] == 0.02
    assert history_feature[5] == 0.5


if __name__ == "__main__":
    main()
