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
    from agent_target_dqn.feature.definition import (
        SampleData,
        _max_action_duration,
        _not_done_flag,
        reward_shaping,
        sample_process,
    )
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
        _default_env_metric_snapshot,
        _env_score_metrics,
        _log_error,
        _log_info,
        _get_training_metrics,
        _handle_disaster_recovery,
        _load_latest_model,
        _need_to_predict,
        _normalize_reset_result,
        _normalize_step_result,
        _obs_feature,
        _predict_action,
        _process_samples,
        _process_observation,
        _put_monitor_data,
        _read_usr_conf,
        _reset_env,
        _reset_agent,
        _reward_components,
        _sample_batch_stats,
        _safe_action,
        _safe_done_flag,
        _safe_extra_info,
        _safe_frame_no,
        _safe_legal_action,
        _safe_observation,
        _save_latest_model,
        _send_sample_data,
        _shape_reward,
        _should_log_progress,
        _step_env,
        _update_traffic_info,
        _update_env_metric_snapshot,
    )

    assert Config.duration_index_to_seconds(0) == Config.MIN_GREEN_DURATION
    assert Config.duration_index_to_seconds(Config.DIM_OF_ACTION_DURATION - 1) == Config.MAX_GREEN_DURATION
    assert Config.duration_seconds_to_index(Config.MIN_GREEN_DURATION) == 0
    assert Config.duration_seconds_to_index(Config.MAX_GREEN_DURATION) == Config.DIM_OF_ACTION_DURATION - 1

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

    class FailingLegalAction:
        def __array__(self, dtype=None):
            raise RuntimeError("legal action conversion failed")

    assert normalize_phase_legal_action(FailingLegalAction()) == [1, 1, 1, 1]

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
    ragged_second.act = [7, 3, Config.MAX_GREEN_DURATION + 5, 99]
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
        float(Config.MAX_GREEN_DURATION),
    ]
    assert ragged_samples[1].rew == [0.25, 0.0]
    assert ragged_samples[1].done == 0
    assert _not_done_flag(False) == 1
    assert _not_done_flag(True) == 0
    assert _not_done_flag("false") == 1
    assert _not_done_flag("true") == 0
    assert _not_done_flag("bad") == 1
    assert _not_done_flag(float("inf")) == 1

    class OptionalFailingFrame:
        obs = [0.5] * Config.DIM_OF_OBSERVATION
        act = [0, 2, Config.MIN_GREEN_DURATION + 2]

        @property
        def rew(self):
            raise RuntimeError("reward read failed")

        @property
        def legal_action(self):
            raise RuntimeError("legal action read failed")

        @property
        def done(self):
            raise RuntimeError("done read failed")

    attr_second = frame_type()
    attr_second.obs = [0.25] * Config.DIM_OF_OBSERVATION
    attr_second.act = [0, 1, Config.MIN_GREEN_DURATION + 1]
    attr_second.rew = (0.1, 0.2)
    attr_second.done = 1
    attr_second.legal_action = [1, 0, 0, 0]
    attr_samples = sample_process([OptionalFailingFrame(), attr_second])
    assert len(attr_samples) == 2
    assert attr_samples[0].rew == [0.0, 0.0]
    assert attr_samples[0].done == 1
    assert attr_samples[0].legal_action == [1, 0, 0, 0]

    class FailingArray:
        def __array__(self, dtype=None):
            raise RuntimeError("array conversion failed")

    class FailingAction:
        def __len__(self):
            raise RuntimeError("action length failed")

    conversion_frame = frame_type()
    conversion_frame.obs = FailingArray()
    conversion_frame.act = [0, 2, Config.MIN_GREEN_DURATION + 2]
    conversion_frame.rew = FailingArray()
    conversion_frame.done = 0
    conversion_frame.legal_action = [1, 0, 0, 0]
    bad_action_frame = frame_type()
    bad_action_frame.obs = [1.0] * Config.DIM_OF_OBSERVATION
    bad_action_frame.act = FailingAction()
    bad_action_frame.rew = (1.0, 1.0)
    bad_action_frame.done = 0
    bad_action_frame.legal_action = [1, 1, 1, 1]
    conversion_samples = sample_process([conversion_frame, bad_action_frame, attr_second])
    assert len(conversion_samples) == 2
    assert conversion_samples[0].obs == [0.0] * Config.DIM_OF_OBSERVATION
    assert conversion_samples[0].rew == [0.0, 0.0]
    assert conversion_samples[0].act == [0.0, 2.0, float(Config.MIN_GREEN_DURATION + 2)]

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
    dummy_agent.preprocess.old_waiting_time = 300.0
    dummy_agent.preprocess.phase_last_served_frame = [0, 0, 0, 0]
    dummy_agent.preprocess.last_phase_index = None
    _, saturated_duration_reward = reward_shaping(
        {
            "frame_state": {
                "frame_no": 20,
                "vehicles": [
                    {
                        "lane": 11,
                        "junction": -1,
                        "target_junction": 0,
                        "speed": 0.0,
                        "waiting_time": 300.0,
                        "delay": 300.0,
                    },
                    {
                        "lane": 10,
                        "junction": -1,
                        "target_junction": 0,
                        "speed": 0.0,
                        "waiting_time": 300.0,
                        "delay": 300.0,
                    },
                ],
            }
        },
        [0, 0, _max_action_duration()],
        dummy_agent,
    )
    assert saturated_duration_reward == 0.0

    assert _reward_components(None) == (0.0, 0.0)
    assert _reward_components((1.5, float("nan"))) == (1.5, 0.0)
    assert _reward_components((float("inf"), 2.0)) == (0.0, 2.0)
    assert _reward_components("bad") == (0.0, 0.0)

    class ScoreObject:
        average_waiting_time = 7.0
        phase_change_penalty = 2.0

    score_metrics = _env_score_metrics(
        {
            "score_info": {
                "total_score": 88.0,
                "avg_delay": 2.5,
                "avg_queue_length": 4.0,
            }
        },
        {"extra_info": {"score_info": ScoreObject()}},
    )
    assert score_metrics["env_score"] == 88.0
    assert score_metrics["avg_delay"] == 2.5
    assert score_metrics["avg_queue_length"] == 4.0
    assert score_metrics["avg_waiting_time"] == 7.0
    assert score_metrics["switch_penalty"] == 2.0
    assert _env_score_metrics(3.5)["env_score"] == 3.5

    env_metric_snapshot = _default_env_metric_snapshot()
    assert env_metric_snapshot["env_score"] == 0.0
    updated_metrics = _update_env_metric_snapshot(
        env_metric_snapshot,
        {"score_info": {"total_score": 12.0, "avg_delay": float("nan")}},
        {"score": {"average_queue_length": 5.0}},
    )
    assert updated_metrics == {"env_score": 12.0, "avg_queue_length": 5.0}
    assert env_metric_snapshot["env_score"] == 12.0
    assert env_metric_snapshot["avg_delay"] == 0.0
    assert env_metric_snapshot["avg_queue_length"] == 5.0
    assert _update_env_metric_snapshot(None, 1.0, {}) == {}

    class RewardData:
        def __init__(self, reward):
            self.rew = reward

    class FailingRewardData:
        @property
        def rew(self):
            raise RuntimeError("reward read failed")

    batch_length, batch_phase_rew, batch_duration_rew = _sample_batch_stats(
        [RewardData((1.0, 2.0)), RewardData((float("nan"), 3.0)), FailingRewardData()],
        None,
    )
    assert batch_length == 3
    assert batch_phase_rew == 1.0
    assert batch_duration_rew == 5.0
    assert _sample_batch_stats([], None) == (0, 0.0, 0.0)

    class FailingLengthBatch:
        def __len__(self):
            raise RuntimeError("length failed")

    assert _sample_batch_stats(FailingLengthBatch(), None) == (0, 0.0, 0.0)

    import agent_target_dqn.workflow.train_workflow as train_workflow

    class EarlyFailingLogger:
        def error(self, message):
            raise RuntimeError("logger error failed")

    original_reward_shaping = train_workflow.reward_shaping
    try:
        train_workflow.reward_shaping = lambda *args, **kwargs: (1.5, float("nan"))
        assert _shape_reward({}, [0, 0, Config.MIN_GREEN_DURATION], dummy_agent, None) == (1.5, 0.0)

        def raise_reward(*args, **kwargs):
            raise RuntimeError("reward failed")

        train_workflow.reward_shaping = raise_reward
        assert _shape_reward({}, [0, 0, Config.MIN_GREEN_DURATION], dummy_agent, EarlyFailingLogger()) == (0.0, 0.0)
    finally:
        train_workflow.reward_shaping = original_reward_shaping

    assert _normalize_reset_result(({"legal_action": 1}, {"init_state": {}})) == {
        "observation": {"legal_action": 1},
        "extra_info": {"init_state": {}},
    }
    assert _normalize_reset_result({"observation": {}}) == {"observation": {}}
    assert _normalize_reset_result(None) == {}

    class ResetEnv:
        def __init__(self):
            self.usr_conf = None

        def reset(self, usr_conf):
            self.usr_conf = usr_conf
            return {"observation": {"legal_action": 1}}

    class FailingResetEnv:
        def reset(self, usr_conf):
            raise RuntimeError("env reset failed")

    reset_env = ResetEnv()
    assert _reset_env(reset_env, {"weather": 0}, EarlyFailingLogger()) == {"observation": {"legal_action": 1}}
    assert reset_env.usr_conf == {"weather": 0}
    assert _reset_env(FailingResetEnv(), {}, EarlyFailingLogger()) is None

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
    five_item_reward, five_item_obs = _normalize_step_result(
        ({"legal_action": 1}, 0.75, False, True, {"frame_no": 11})
    )
    assert five_item_reward == 0.75
    assert five_item_obs["frame_no"] == 11
    assert five_item_obs["observation"] == {"legal_action": 1}
    assert five_item_obs["terminated"] is False
    assert five_item_obs["truncated"] is True
    assert five_item_obs["extra_info"] == {"frame_no": 11}
    four_item_reward, four_item_obs = _normalize_step_result(({"legal_action": 1}, 0.5, True, {"frame_no": 12}))
    assert four_item_reward == 0.5
    assert four_item_obs["frame_no"] == 12
    assert four_item_obs["observation"] == {"legal_action": 1}
    assert four_item_obs["terminated"] is True
    assert four_item_obs["truncated"] is False
    assert _normalize_step_result({"frame_no": 4}) == (0.0, {"frame_no": 4})
    assert _normalize_step_result(None) == (0.0, {})

    class StepEnv:
        def __init__(self):
            self.act = None

        def step(self, act):
            self.act = act
            return 0.25, {"frame_no": 9}

    class FailingStepEnv:
        def step(self, act):
            raise RuntimeError("env step failed")

    step_env = StepEnv()
    assert _step_env(step_env, [0, 1, Config.MIN_GREEN_DURATION], EarlyFailingLogger()) == (0.25, {"frame_no": 9})
    assert step_env.act == [0, 1, Config.MIN_GREEN_DURATION]
    assert _step_env(FailingStepEnv(), [0, 1, Config.MIN_GREEN_DURATION], EarlyFailingLogger()) is None

    assert _safe_observation({"observation": {"legal_action": 1}}) == {"legal_action": 1}
    assert _safe_observation({"observation": None}) == {}
    assert _safe_extra_info({"extra_info": {"init_state": {}}}) == {"init_state": {}}
    assert _safe_extra_info({"extra_info": None}) == {}
    assert _safe_frame_no({"frame_no": "bad"}) == 0
    assert _safe_frame_no({"frame_no": float("inf")}) == 0
    assert _safe_frame_no({"frame_no": 3.5}) == 3
    assert _safe_frame_no({"frame_no": 7}) == 7
    assert _safe_done_flag({"terminated": 1}, "terminated") is True
    assert _safe_done_flag({"terminated": 0}, "terminated") is False
    assert _safe_done_flag({"terminated": "true"}, "terminated") is True
    assert _safe_done_flag({"terminated": "False"}, "terminated") is False
    assert _safe_done_flag({"terminated": "bad"}, "terminated") is False
    assert _safe_done_flag({"terminated": float("inf")}, "terminated") is False
    assert _safe_done_flag({"terminated": object()}, "terminated") is False
    assert _safe_done_flag({}, "terminated") is False
    assert _safe_legal_action({"legal_action": [1, 0, 0, 0]}) == [1, 0, 0, 0]
    assert _safe_legal_action(None) is None

    class FailingEnvObs(dict):
        def get(self, key, default=None):
            raise RuntimeError("mapping read failed")

    failing_env_obs = FailingEnvObs()
    assert _safe_observation(failing_env_obs) == {}
    assert _safe_extra_info(failing_env_obs) == {}
    assert _safe_frame_no(failing_env_obs) == 0
    assert _safe_done_flag(failing_env_obs, "terminated") is False
    assert _safe_legal_action(failing_env_obs) is None
    assert _need_to_predict(failing_env_obs) is True
    assert _need_to_predict({"legal_action": 0}) is False
    assert _need_to_predict({"legal_action": [0, 1, 0, 0]}) is True
    assert _need_to_predict({"legal_action": FailingLegalAction()}) is True
    assert _should_log_progress(0, False, False) is False
    assert _should_log_progress(20, False, True) is True
    assert _should_log_progress(0, True, False) is True
    assert _safe_action([9, 2, Config.MIN_GREEN_DURATION + 3], True, None) == [
        0,
        2,
        Config.MIN_GREEN_DURATION + 3,
    ]
    assert _safe_action([9, -1, 999], True, None) == [
        0,
        0,
        Config.MAX_GREEN_DURATION,
    ]
    assert _safe_action([None, None, None], False, None) == [None, None, None]
    assert _safe_action([0, "bad", 8], True, None) == [0, 0, Config.MIN_GREEN_DURATION]
    assert _safe_action([0, 1, float("nan")], True, None) == [0, 0, Config.MIN_GREEN_DURATION]

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

    class FeatureObsData:
        def __init__(self, feature):
            self.feature = feature

    class ObservationAgent:
        def observation_process(self, obs, extra_info):
            return FeatureObsData([1.0, 2.0])

    class FailingObservationAgent:
        def observation_process(self, obs, extra_info):
            raise RuntimeError("observation failed")

    class RecordingPreprocess:
        def __init__(self):
            self.records = []

        def update_traffic_info(self, obs, extra_info):
            self.records.append((obs, extra_info))

    class TrafficInfoAgent:
        def __init__(self):
            self.preprocess = RecordingPreprocess()

    class FailingPreprocess:
        def update_traffic_info(self, obs, extra_info):
            raise RuntimeError("traffic update failed")

    class FailingTrafficInfoAgent:
        def __init__(self):
            self.preprocess = FailingPreprocess()

    processed_obs = _process_observation(ObservationAgent(), {"legal_action": 1}, {}, FailingLogger())
    assert _obs_feature(processed_obs) == [1.0, 2.0]
    assert _process_observation(FailingObservationAgent(), {"legal_action": 1}, {}, FailingLogger()) is None
    assert len(_obs_feature(None)) == Config.DIM_OF_OBSERVATION
    assert _obs_feature(None)[0] == 0.0

    original_workflow_sample_process = train_workflow.sample_process
    try:
        train_workflow.sample_process = lambda collector: ["sample-data"]
        assert _process_samples([object()], FailingLogger()) == ["sample-data"]
        assert _process_samples([], FailingLogger()) == []
        train_workflow.sample_process = lambda collector: None
        assert _process_samples([object()], FailingLogger()) == []

        def raise_sample_process(collector):
            raise RuntimeError("sample process failed")

        train_workflow.sample_process = raise_sample_process
        assert _process_samples([object()], FailingLogger()) == []
    finally:
        train_workflow.sample_process = original_workflow_sample_process

    class ResettingAgent:
        def __init__(self):
            self.reset_obs = None

        def reset(self, env_obs):
            self.reset_obs = env_obs

    class FailingResetAgent:
        def reset(self, env_obs):
            raise RuntimeError("reset failed")

    resetting_agent = ResettingAgent()
    reset_obs = {"observation": {"legal_action": 1}}
    assert _reset_agent(resetting_agent, reset_obs, FailingLogger()) is True
    assert resetting_agent.reset_obs == reset_obs
    assert _reset_agent(FailingResetAgent(), reset_obs, FailingLogger()) is False

    traffic_info_agent = TrafficInfoAgent()
    assert _update_traffic_info(traffic_info_agent, {"legal_action": 0}, {"init_state": {}}, FailingLogger()) is True
    assert traffic_info_agent.preprocess.records
    assert _update_traffic_info(FailingTrafficInfoAgent(), {"legal_action": 0}, {}, FailingLogger()) is False

    recording_monitor = RecordingMonitor()
    assert _put_monitor_data(recording_monitor, {"reward": 1.0}, FailingLogger()) is True
    assert recording_monitor.records
    assert _put_monitor_data(FailingMonitor(), {"reward": 1.0}, FailingLogger()) is False
    assert _put_monitor_data(None, {"reward": 1.0}, FailingLogger()) is False

    original_read_usr_conf = train_workflow.read_usr_conf
    try:
        train_workflow.read_usr_conf = lambda *args, **kwargs: {"weather": 0}
        assert _read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", FailingLogger()) == {"weather": 0}
        train_workflow.read_usr_conf = lambda *args, **kwargs: "bad-conf"
        assert _read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", FailingLogger()) is None

        def raise_usr_conf(*args, **kwargs):
            raise RuntimeError("read conf failed")

        train_workflow.read_usr_conf = raise_usr_conf
        assert _read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", FailingLogger()) is None
    finally:
        train_workflow.read_usr_conf = original_read_usr_conf

    original_handle_disaster_recovery = train_workflow.handle_disaster_recovery
    try:
        train_workflow.handle_disaster_recovery = lambda *args, **kwargs: True
        assert _handle_disaster_recovery({"observation": {}}, FailingLogger()) is True
        train_workflow.handle_disaster_recovery = lambda *args, **kwargs: "non-empty"
        assert _handle_disaster_recovery({"observation": {}}, FailingLogger()) is True
        train_workflow.handle_disaster_recovery = lambda *args, **kwargs: None
        assert _handle_disaster_recovery({"observation": {}}, FailingLogger()) is False

        def raise_disaster(*args, **kwargs):
            raise RuntimeError("disaster recovery failed")

        train_workflow.handle_disaster_recovery = raise_disaster
        assert _handle_disaster_recovery({"observation": {}}, FailingLogger()) is False
    finally:
        train_workflow.handle_disaster_recovery = original_handle_disaster_recovery

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
    assert _predict_action(EmptyPredictAgent(), None, {}, FailingLogger()) == [0, 2, Config.MIN_GREEN_DURATION + 1]

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

    class LoadingAgent:
        def __init__(self):
            self.loaded_ids = []

        def load_model(self, id):
            self.loaded_ids.append(id)

    class FailingLoadAgent:
        def load_model(self, id):
            raise RuntimeError("load failed")

    loading_agent = LoadingAgent()
    assert _load_latest_model(loading_agent, FailingLogger()) is True
    assert loading_agent.loaded_ids == ["latest"]
    assert _load_latest_model(FailingLoadAgent(), FailingLogger()) is False

    class SendingAgent:
        def __init__(self):
            self.sent_samples = []

        def send_sample_data(self, samples):
            self.sent_samples.append(samples)

    class FailingSendAgent:
        def send_sample_data(self, samples):
            raise RuntimeError("send failed")

    sending_agent = SendingAgent()
    samples_to_send = [object()]
    assert _send_sample_data(sending_agent, samples_to_send, FailingLogger()) is True
    assert sending_agent.sent_samples == [samples_to_send]
    assert _send_sample_data(FailingSendAgent(), samples_to_send, FailingLogger()) is False
    assert _send_sample_data(sending_agent, [], FailingLogger()) is False


if __name__ == "__main__":
    main()
