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

    class Frame:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    common_func.create_cls = create_cls
    common_func.Frame = Frame
    common_utils.common_func = common_func
    common_python.utils = common_utils
    sys.modules["common_python"] = common_python
    sys.modules["common_python.utils"] = common_utils
    sys.modules["common_python.utils.common_func"] = common_func


def install_workflow_stubs():
    tools = types.ModuleType("tools")
    train_env_conf_validate = types.ModuleType("tools.train_env_conf_validate")
    metrics_utils = types.ModuleType("tools.metrics_utils")
    workflow_disaster_recovery = types.ModuleType("common_python.utils.workflow_disaster_recovery")

    train_env_conf_validate.read_usr_conf = lambda *args, **kwargs: {}
    metrics_utils.get_training_metrics = lambda: {}
    workflow_disaster_recovery.handle_disaster_recovery = lambda *args, **kwargs: False

    sys.modules["tools"] = tools
    sys.modules["tools.train_env_conf_validate"] = train_env_conf_validate
    sys.modules["tools.metrics_utils"] = metrics_utils
    sys.modules["common_python.utils.workflow_disaster_recovery"] = workflow_disaster_recovery


def main():
    install_common_python_stub()
    install_workflow_stubs()

    from agent_target_dqn.conf.conf import Config
    from agent_target_dqn.feature.definition import SampleData, reward_shaping, sample_process
    from agent_target_dqn.feature.traffic_utils import (
        get_lane_statistics,
        get_traffic_history_feature,
        get_traffic_summary,
        get_traffic_trend,
        normalize_phase_legal_action,
    )
    from agent_target_dqn.workflow.train_workflow import (
        _need_to_predict,
        _reward_components,
        _safe_done_flag,
        _safe_extra_info,
        _safe_frame_no,
        _safe_legal_action,
        _safe_observation,
        _should_log_progress,
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

    assert _reward_components(None) == (0.0, 0.0)
    assert _reward_components((1.5, float("nan"))) == (1.5, 0.0)
    assert _reward_components((float("inf"), 2.0)) == (0.0, 2.0)
    assert _reward_components("bad") == (0.0, 0.0)
    assert _safe_observation({"observation": {"legal_action": 1}}) == {"legal_action": 1}
    assert _safe_observation({"observation": None}) == {}
    assert _safe_extra_info({"extra_info": {"init_state": {}}}) == {"init_state": {}}
    assert _safe_extra_info({"extra_info": None}) == {}
    assert _safe_frame_no({"frame_no": "bad"}) == 0
    assert _safe_frame_no({"frame_no": 7}) == 7
    assert _safe_done_flag({"terminated": 1}, "terminated") is True
    assert _safe_done_flag({}, "terminated") is False
    assert _safe_legal_action({"legal_action": [1, 0, 0, 0]}) == [1, 0, 0, 0]
    assert _safe_legal_action(None) is None
    assert _need_to_predict({"legal_action": 0}) is False
    assert _need_to_predict({"legal_action": [0, 1, 0, 0]}) is True
    assert _should_log_progress(0, False, False) is False
    assert _should_log_progress(20, False, True) is True
    assert _should_log_progress(0, True, False) is True


if __name__ == "__main__":
    main()
