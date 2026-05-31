#!/usr/bin/env python3

import math
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
    from agent_target_dqn.feature.preprocessor import FeatureProcess
    from agent_target_dqn.feature.traffic_utils import (
        get_phase_pressure,
        get_lane_statistics,
        get_traffic_history_feature,
        get_traffic_summary,
        get_traffic_trend,
        normalize_phase_legal_action,
    )
    from agent_target_dqn.workflow.train_workflow import (
        _log_error,
        _log_info,
        _get_training_metrics,
        _need_to_predict,
        _normalize_reset_result,
        _normalize_step_result,
        _predict_action,
        _put_monitor_data,
        _reward_components,
        _safe_done_flag,
        _safe_extra_info,
        _safe_frame_no,
        _safe_legal_action,
        _safe_observation,
        _save_latest_model,
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
    assert normalize_phase_legal_action(float("nan")) == [0, 0, 0, 0]
    assert normalize_phase_legal_action([1, float("nan"), float("inf"), float("-inf")]) == [1, 0, 0, 0]
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

    nonfinite_vehicles = [
        {
            "lane": 11,
            "junction": -1,
            "target_junction": 0,
            "speed": float("nan"),
            "waiting_time": float("inf"),
            "delay": float("nan"),
        },
        {
            "lane": 129,
            "junction": -1,
            "target_junction": 0,
            "speed": float("-inf"),
            "waiting_time": -5.0,
            "delay": float("inf"),
        },
    ]
    nonfinite_lane_stats = get_lane_statistics(nonfinite_vehicles)
    assert all(math.isfinite(float(value)) for value in nonfinite_lane_stats["avg_waiting_times"])
    nonfinite_phase_pressure, nonfinite_totals = get_phase_pressure(nonfinite_vehicles)
    assert all(math.isfinite(float(value)) for value in nonfinite_phase_pressure)
    assert all(math.isfinite(float(value)) for value in nonfinite_totals.values())
    assert nonfinite_totals["vehicle_count"] == 2
    nonfinite_summary = get_traffic_summary(nonfinite_vehicles)
    assert all(math.isfinite(float(value)) for value in nonfinite_summary["phase_pressure"])
    assert math.isfinite(nonfinite_summary["avg_waiting_time"])
    assert math.isfinite(nonfinite_summary["avg_delay"])
    noisy_previous_summary = {
        "phase_pressure": [float("nan"), float("inf")],
        "vehicle_count": float("inf"),
        "queue_ratio": float("nan"),
        "avg_waiting_time": float("-inf"),
        "avg_delay": float("inf"),
    }
    assert all(math.isfinite(value) for value in get_traffic_trend(nonfinite_summary, noisy_previous_summary))
    noisy_history = [nonfinite_summary, noisy_previous_summary, "bad-summary"]
    assert all(math.isfinite(value) for value in get_traffic_history_feature(noisy_history))

    preprocess = FeatureProcess(None)
    preprocess.junction_dict = {0: {}}
    first_vehicle_frame = {
        "frame_state": {
            "frame_no": 1,
            "frame_time": 1.0,
            "vehicles": [
                {
                    "v_id": 1,
                    "lane": 11,
                    "junction": -1,
                    "target_junction": 0,
                    "speed": 0.0,
                    "position_in_lane": {"x": 0.0, "y": 0.0},
                }
            ],
        }
    }
    second_vehicle_frame = {
        "frame_state": {
            "frame_no": 2,
            "frame_time": 3.0,
            "vehicles": [
                {
                    "v_id": 1,
                    "lane": 11,
                    "junction": -1,
                    "target_junction": 0,
                    "speed": 0.0,
                    "position_in_lane": {"x": 3.0, "y": 4.0},
                }
            ],
        }
    }
    preprocess.update_traffic_info(first_vehicle_frame, None)
    preprocess.update_traffic_info(second_vehicle_frame, None)
    assert preprocess.waiting_time_store[1] == 2.0
    assert preprocess.vehicle_distance_store[1] == 5.0
    assert preprocess.lane_volume[11] == [1]
    bad_preprocess_frame = {
        "frame_state": {
            "frame_no": float("inf"),
            "frame_time": float("nan"),
            "vehicles": [
                None,
                {"v_id": [], "lane": 11, "junction": -1, "target_junction": 0},
                {
                    "v_id": 2,
                    "lane": 11,
                    "junction": -1,
                    "target_junction": 0,
                    "speed": float("nan"),
                    "position_in_lane": {"x": float("inf"), "y": 1.0},
                },
            ],
        }
    }
    preprocess.update_traffic_info(bad_preprocess_frame, "bad-extra-info")
    assert all(math.isfinite(float(value)) for value in preprocess.waiting_time_store.values())
    waiting_by_store = preprocess.get_all_junction_waiting_time(
        [
            {"v_id": 1, "junction": -1, "target_junction": 0},
            {"v_id": [], "junction": -1, "target_junction": 0},
            {"bad": "vehicle"},
        ]
    )
    assert waiting_by_store == {0: 2.0}
    waiting_by_origin = preprocess.get_all_junction_waiting_time_by_origin(
        [
            {"junction": -1, "target_junction": 0, "waiting_time": 6.0},
            {"junction": -1, "target_junction": 0, "waiting_time": float("inf")},
            {"bad": "vehicle"},
        ]
    )
    assert waiting_by_origin == {0: 3.0}

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
    assert samples[0].rew == [0.0, 0.0]
    assert samples[0]._obs == second.obs
    assert samples[0].legal_action == [0, 1, 0, 1]
    assert samples[1].done == 0
    assert samples[1]._obs == second.obs

    ragged_first = frame_type()
    ragged_first.obs = [float("nan"), float("inf"), 2.0]
    ragged_first.act = [0, float("nan"), float("inf")]
    ragged_first.rew = (float("nan"), float("inf"), 9.0)
    ragged_first.done = "0"
    ragged_first.legal_action = [1, float("nan"), float("inf"), float("-inf")]
    ragged_second = frame_type()
    ragged_second.obs = [1.0] * (Config.DIM_OF_OBSERVATION + 5)
    ragged_second.act = [7, 3, Config.MIN_GREEN_DURATION + Config.DIM_OF_ACTION_DURATION + 5, 99]
    ragged_second.rew = (0.25,)
    ragged_second.done = "1"
    ragged_second.legal_action = [0, 1]
    ragged_samples = sample_process([ragged_first, ragged_second])
    assert len(ragged_samples) == 2
    assert len(ragged_samples[0].obs) == Config.DIM_OF_OBSERVATION
    assert len(ragged_samples[0]._obs) == Config.DIM_OF_OBSERVATION
    assert ragged_samples[0].obs[:3] == [0.0, 0.0, 2.0]
    assert all(math.isfinite(value) for value in ragged_samples[0].obs)
    assert ragged_samples[0].act == [0.0, 0.0, float(Config.MIN_GREEN_DURATION)]
    assert ragged_samples[0].rew == [0.0, 0.0]
    assert ragged_samples[0].done == 1
    assert ragged_samples[0].legal_action == [0, 1, 1, 1]
    assert ragged_samples[1].act == [
        0.0,
        3.0,
        float(Config.MIN_GREEN_DURATION + Config.DIM_OF_ACTION_DURATION - 1),
    ]
    assert ragged_samples[1].rew == [0.25, 0.0]
    assert ragged_samples[1].done == 0

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
    assert reward_shaping(
        {"frame_state": {"frame_no": 1, "vehicles": []}},
        [0, 0, float("inf")],
        dummy_agent,
    ) == (0.0, 0.0)
    dummy_agent.preprocess.old_waiting_time = 0.0
    dummy_agent.preprocess.phase_last_served_frame = ["bad", float("inf"), None, -5]
    dummy_agent.preprocess.last_phase_index = None
    finite_reward = reward_shaping(
        {
            "frame_state": {
                "frame_no": float("inf"),
                "vehicles": [
                    {
                        "lane": 11,
                        "junction": -1,
                        "target_junction": 0,
                        "speed": 0.0,
                        "waiting_time": 12.0,
                        "delay": 3.0,
                    }
                ],
            }
        },
        [0, 0, Config.MIN_GREEN_DURATION],
        dummy_agent,
    )
    assert all(math.isfinite(value) for value in finite_reward)

    assert _reward_components(None) == (0.0, 0.0)
    assert _reward_components((1.5, float("nan"))) == (1.5, 0.0)
    assert _reward_components((float("inf"), 2.0)) == (0.0, 2.0)
    assert _reward_components("bad") == (0.0, 0.0)
    assert _normalize_reset_result(({"legal_action": 1}, {"init_state": {}})) == {
        "observation": {"legal_action": 1},
        "extra_info": {"init_state": {}},
    }
    assert _normalize_reset_result({"observation": {}}) == {"observation": {}}
    assert _normalize_reset_result(None) == {}
    two_item_reward, two_item_obs = _normalize_step_result((0.5, {"frame_no": 3}))
    assert two_item_reward == 0.5
    assert two_item_obs == {"frame_no": 3}
    six_item_reward, six_item_obs = _normalize_step_result((7, {"legal_action": 1}, 1.25, True, False, {"x": 1}))
    assert six_item_reward == 1.25
    assert six_item_obs["frame_no"] == 7
    assert six_item_obs["observation"] == {"legal_action": 1}
    assert six_item_obs["terminated"] is True
    assert six_item_obs["truncated"] is False
    assert six_item_obs["extra_info"] == {"x": 1}
    assert _normalize_step_result({"frame_no": 4}) == (0.0, {"frame_no": 4})
    assert _normalize_step_result(None) == (0.0, {})
    assert _safe_observation({"observation": {"legal_action": 1}}) == {"legal_action": 1}
    assert _safe_observation({"observation": None}) == {}
    assert _safe_extra_info({"extra_info": {"init_state": {}}}) == {"init_state": {}}
    assert _safe_extra_info({"extra_info": None}) == {}
    assert _safe_frame_no({"frame_no": "bad"}) == 0
    assert _safe_frame_no({"frame_no": float("inf")}) == 0
    assert _safe_frame_no({"frame_no": 3.5}) == 3
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

    class FailingLogger:
        def info(self, message):
            raise RuntimeError("logger info failed")

        def error(self, message):
            raise RuntimeError("logger error failed")

    class RecordingMonitor:
        def __init__(self):
            self.records = []

        def put_data(self, data):
            self.records.append(data)

    class FailingMonitor:
        def put_data(self, data):
            raise RuntimeError("monitor failed")

    _log_info(FailingLogger(), "info")
    _log_error(FailingLogger(), "error")
    recording_monitor = RecordingMonitor()
    assert _put_monitor_data(recording_monitor, {"reward": 1.0}, FailingLogger()) is True
    assert recording_monitor.records
    assert _put_monitor_data(FailingMonitor(), {"reward": 1.0}, FailingLogger()) is False
    assert _put_monitor_data(None, {"reward": 1.0}, FailingLogger()) is False

    import agent_target_dqn.workflow.train_workflow as train_workflow

    original_get_training_metrics = train_workflow.get_training_metrics
    try:
        train_workflow.get_training_metrics = lambda: {"reward": 1.0}
        assert _get_training_metrics(FailingLogger()) == {"reward": 1.0}
        train_workflow.get_training_metrics = lambda: None
        assert _get_training_metrics(FailingLogger()) == {}

        def raise_metrics():
            raise RuntimeError("metrics failed")

        train_workflow.get_training_metrics = raise_metrics
        assert _get_training_metrics(FailingLogger()) == {}
    finally:
        train_workflow.get_training_metrics = original_get_training_metrics

    class PredictingAgent:
        def predict(self, list_obs_data):
            return ["action-data"]

        def action_process(self, act_data):
            return [0, 3, Config.MIN_GREEN_DURATION + 2]

        def rule_based_action(self, obs):
            raise AssertionError("rule fallback should not run")

    class EmptyPredictAgent:
        def predict(self, list_obs_data):
            return []

        def action_process(self, act_data):
            raise AssertionError("empty predictions should not be processed")

        def rule_based_action(self, obs):
            return [0, 2, Config.MIN_GREEN_DURATION + 1]

    class FailingPredictAgent:
        def predict(self, list_obs_data):
            raise RuntimeError("model failed")

        def rule_based_action(self, obs):
            return [0, 1, Config.MIN_GREEN_DURATION]

    class FailingFallbackAgent:
        def predict(self, list_obs_data):
            return []

        def rule_based_action(self, obs):
            raise RuntimeError("rule failed")

    assert _predict_action(PredictingAgent(), object(), {}, None) == [0, 3, Config.MIN_GREEN_DURATION + 2]
    assert _predict_action(EmptyPredictAgent(), object(), {}, None) == [0, 2, Config.MIN_GREEN_DURATION + 1]
    assert _predict_action(FailingPredictAgent(), object(), {}, None) == [0, 1, Config.MIN_GREEN_DURATION]
    assert _predict_action(FailingFallbackAgent(), object(), {}, None) == [0, 0, Config.MIN_GREEN_DURATION]

    class SavingAgent:
        def __init__(self):
            self.saved_ids = []

        def save_model(self, id):
            self.saved_ids.append(id)

    class FailingSaveAgent:
        def save_model(self, id):
            raise RuntimeError("save failed")

    saving_agent = SavingAgent()
    assert _save_latest_model(saving_agent, FailingLogger()) is True
    assert saving_agent.saved_ids == ["latest"]
    assert _save_latest_model(FailingSaveAgent(), FailingLogger()) is False


if __name__ == "__main__":
    main()
