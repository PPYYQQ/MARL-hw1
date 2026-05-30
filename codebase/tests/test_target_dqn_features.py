#!/usr/bin/env python3

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_common_python_stub():
    common_python = types.ModuleType("common_python")
    common_utils = types.ModuleType("common_python.utils")
    common_func = types.ModuleType("common_python.utils.common_func")

    def create_cls(name, **fields):
        class Data:
            def __init__(self, **kwargs):
                for field_name, default in fields.items():
                    setattr(self, field_name, kwargs.get(field_name, None if default is None else default))

        Data.__name__ = name
        return Data

    common_func.create_cls = create_cls
    common_utils.common_func = common_func
    common_python.utils = common_utils
    sys.modules["common_python"] = common_python
    sys.modules["common_python.utils"] = common_utils
    sys.modules["common_python.utils.common_func"] = common_func


def main():
    install_common_python_stub()

    from agent_target_dqn.conf.conf import Config
    from agent_target_dqn.feature.definition import SampleData, reward_shaping, sample_process
    from agent_target_dqn.feature.traffic_utils import (
        get_lane_statistics,
        get_traffic_history_feature,
        get_traffic_summary,
        get_traffic_trend,
        normalize_phase_legal_action,
    )

    assert normalize_phase_legal_action(None) == [1, 1, 1, 1]
    assert normalize_phase_legal_action([]) == [1, 1, 1, 1]
    assert normalize_phase_legal_action(1) == [1, 1, 1, 1]
    assert normalize_phase_legal_action(0) == [0, 0, 0, 0]
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

    frame_type = type("Frame", (), {})
    assert sample_process([]) == []
    malformed_frame = frame_type()
    malformed_frame.obs = None
    malformed_frame.act = [None, None, None]
    assert sample_process([malformed_frame]) == []
    first = frame_type()
    first.obs = [0.0] * Config.DIM_OF_OBSERVATION
    first.act = [0, 0, Config.MIN_GREEN_DURATION]
    first.rew = None
    first.done = 0
    first.legal_action = None
    second = frame_type()
    second.obs = [1.0] * Config.DIM_OF_OBSERVATION
    second.act = [0, 1, Config.MIN_GREEN_DURATION + 1]
    second.rew = (0.5, 0.25)
    second.done = 1
    second.legal_action = [0, 1, 0, 1]
    samples = sample_process([first, second])
    assert len(samples) == 2
    assert isinstance(samples[0], SampleData)
    assert samples[0].rew == (0.0, 0.0)
    assert samples[0]._obs == second.obs
    assert samples[0].legal_action == [0, 1, 0, 1]
    assert samples[1].done == 0
    assert samples[1]._obs == second.obs

    preprocess_type = type("Preprocess", (), {})
    agent_type = type("Agent", (), {})
    dummy_agent = agent_type()
    dummy_agent.preprocess = preprocess_type()
    dummy_agent.preprocess.old_waiting_time = 0.0
    dummy_agent.preprocess.phase_last_served_frame = [None] * Config.DIM_OF_ACTION_PHASE
    dummy_agent.preprocess.last_phase_index = None
    assert reward_shaping({}, [0, 0, Config.MIN_GREEN_DURATION], dummy_agent) == (0.0, 0.0)
    assert reward_shaping({"frame_state": {"frame_no": "bad", "vehicles": "bad"}}, [0, 0, 8], dummy_agent) == (
        0.0,
        0.0,
    )
    assert reward_shaping({"frame_state": {"frame_no": 1, "vehicles": []}}, [0, "bad", 8], dummy_agent) == (
        0.0,
        0.0,
    )


if __name__ == "__main__":
    main()
