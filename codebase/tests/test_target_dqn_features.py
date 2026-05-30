#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from agent_target_dqn.feature.traffic_utils import get_lane_statistics, normalize_phase_legal_action

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


if __name__ == "__main__":
    main()
