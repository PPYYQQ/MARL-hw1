#!/usr/bin/env python3

import math
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_framework_stubs():
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

    kaiwudrl = types.ModuleType("kaiwudrl")
    kaiwudrl_interface = types.ModuleType("kaiwudrl.interface")
    kaiwudrl_agent = types.ModuleType("kaiwudrl.interface.agent")

    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass

    kaiwudrl_agent.BaseAgent = BaseAgent
    kaiwudrl_interface.agent = kaiwudrl_agent
    kaiwudrl.interface = kaiwudrl_interface
    sys.modules["kaiwudrl"] = kaiwudrl
    sys.modules["kaiwudrl.interface"] = kaiwudrl_interface
    sys.modules["kaiwudrl.interface.agent"] = kaiwudrl_agent


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_fake_obs():
    return {
        "legal_action": [1, 0, 1, 0],
        "frame_state": {
            "frame_no": 1,
            "frame_time": 0,
            "phases": [{"s_id": 0, "phase_id": 0, "duration": 20, "remaining_duration": 12}],
            "vehicles": [
                {
                    "v_id": 1,
                    "v_config_id": 1,
                    "lane": 11,
                    "junction": -1,
                    "target_junction": 0,
                    "position_in_lane": {"x": 0, "y": 50000},
                    "speed": 0.0,
                    "waiting_time": 20.0,
                    "delay": 30.0,
                },
                {
                    "v_id": 2,
                    "v_config_id": 1,
                    "lane": 129,
                    "junction": -1,
                    "target_junction": 0,
                    "position_in_lane": {"x": 0, "y": 30},
                    "speed": 8.5,
                    "waiting_time": 0.0,
                    "delay": 2.0,
                },
            ],
        },
    }


def make_extra_info():
    return {
        "init_state": {
            "junctions": [
                {
                    "j_id": 0,
                    "enter_lanes_on_directions": [
                        {"lanes": [11, 10, 9, 8]},
                        {"lanes": [129, 128, 127, 126]},
                        {"lanes": [23, 22, 21, 20]},
                        {"lanes": [163, 162]},
                    ],
                }
            ],
            "signals": [],
            "edges": [],
            "lane_configs": [],
            "vehicle_configs": [{"v_config_id": 1, "max_speed": 17.0}],
        }
    }


def assert_no_nan(values):
    assert all(not math.isnan(float(value)) for value in values)


def main():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP: torch is not installed in this Python environment")
        return

    install_framework_stubs()

    from agent_target_dqn.agent import Agent
    from agent_target_dqn.algorithm.algorithm import Algorithm
    from agent_target_dqn.conf.conf import Config
    from agent_target_dqn.feature.definition import ActData, SampleData, reward_shaping, sample_process

    agent = Agent(device="cpu", logger=NullLogger(), monitor=None)
    agent.preprocess.update_traffic_info({}, None)
    agent.preprocess.update_traffic_info({"frame_state": {"frame_no": 1, "frame_time": 0}}, {"init_state": {}})
    agent.preprocess.update_traffic_info(
        {"frame_state": {"frame_no": 2, "frame_time": 0, "vehicles": [{"bad": "vehicle"}]}},
        None,
    )
    obs_data = agent.observation_process(make_fake_obs(), make_extra_info())
    assert len(obs_data.feature) == Config.DIM_OF_OBSERVATION
    assert obs_data.legal_action == [1, 0, 1, 0]
    assert_no_nan(obs_data.feature)
    assert sum(obs_data.feature[: Config.GRID_WIDTH * Config.GRID_NUM]) == 2
    phase_feature_start = Config.GRID_WIDTH * Config.GRID_NUM * 2
    assert obs_data.feature[phase_feature_start] == 1.0
    assert obs_data.feature[phase_feature_start + Config.PHASE_FEATURE_DIM - 1] == 1.0
    phase_age_start = phase_feature_start + Config.PHASE_FEATURE_DIM
    assert (
        sum(
            abs(value)
            for value in obs_data.feature[
                phase_age_start : phase_age_start + Config.PHASE_AGE_FEATURE_DIM
            ]
        )
        == 0.0
    )
    traffic_feature_start = phase_age_start + Config.PHASE_AGE_FEATURE_DIM
    assert obs_data.feature[traffic_feature_start] > 0.0
    assert obs_data.feature[traffic_feature_start + 2] > 0.0
    assert abs(obs_data.feature[traffic_feature_start + 4] - 0.02) < 1e-6
    assert abs(obs_data.feature[traffic_feature_start + 5] - 0.5) < 1e-6
    traffic_trend_start = traffic_feature_start + Config.TRAFFIC_FEATURE_DIM
    assert (
        sum(
            abs(value)
            for value in obs_data.feature[
                traffic_trend_start : traffic_trend_start + Config.TRAFFIC_TREND_FEATURE_DIM
            ]
        )
        == 0.0
    )
    traffic_history_start = traffic_trend_start + Config.TRAFFIC_TREND_FEATURE_DIM
    assert (
        sum(
            abs(value)
            for value in obs_data.feature[
                traffic_history_start : traffic_history_start + Config.TRAFFIC_HISTORY_FEATURE_DIM
            ]
        )
        == 0.0
    )
    lane_stat_start = traffic_history_start + Config.TRAFFIC_HISTORY_FEATURE_DIM
    assert abs(obs_data.feature[lane_stat_start] - 0.05) < 1e-6
    assert abs(obs_data.feature[lane_stat_start + 4] - 0.05) < 1e-6
    assert abs(obs_data.feature[lane_stat_start + Config.GRID_WIDTH] - 0.05) < 1e-6
    assert obs_data.feature[lane_stat_start + Config.GRID_WIDTH + 4] == 0.0
    assert abs(obs_data.feature[lane_stat_start + Config.GRID_WIDTH * 2] - (20.0 / 120.0)) < 1e-6
    obs_data_without_extra = agent.observation_process(make_fake_obs(), None)
    assert len(obs_data_without_extra.feature) == Config.DIM_OF_OBSERVATION
    assert len(agent.preprocess.traffic_history) == 2
    empty_obs_data = agent.observation_process({}, None)
    assert len(empty_obs_data.feature) == Config.DIM_OF_OBSERVATION
    assert empty_obs_data.legal_action == [1, 1, 1, 1]
    assert_no_nan(empty_obs_data.feature)
    malformed_obs = {
        "legal_action": 1,
        "frame_state": {
            "frame_no": "bad",
            "frame_time": 0,
            "phases": [None, {"s_id": 0, "phase_id": "bad", "duration": "bad"}],
            "vehicles": [
                None,
                {"bad": "vehicle"},
                {
                    "v_id": 3,
                    "lane": 11,
                    "target_junction": 0,
                    "position_in_lane": {"y": "bad"},
                    "speed": "bad",
                },
            ],
        },
    }
    malformed_obs_data = agent.observation_process(malformed_obs, None)
    assert len(malformed_obs_data.feature) == Config.DIM_OF_OBSERVATION
    assert malformed_obs_data.legal_action == [1, 1, 1, 1]
    assert_no_nan(malformed_obs_data.feature)

    low_duration_action = agent.action_process(ActData(junction_id=0, phase_index=0, duration=0))
    high_duration_action = agent.action_process(ActData(junction_id=0, phase_index=99, duration=99))
    malformed_action = agent.action_process(ActData(junction_id=99, phase_index="bad", duration=float("nan")))
    assert low_duration_action == [0, 0, Config.MIN_GREEN_DURATION]
    assert high_duration_action == [0, Config.DIM_OF_ACTION_PHASE - 1, Config.MIN_GREEN_DURATION + 19]
    assert malformed_action == [0, 0, Config.MIN_GREEN_DURATION]
    assert agent.rule_based_action(make_fake_obs())[1] == 0
    assert agent.rule_based_action(None) == [0, 0, Config.MIN_GREEN_DURATION]

    agent._eps = 1.0
    assert agent.predict([]) == []
    predictions = agent.predict([obs_data, obs_data])
    assert len(predictions) == 2
    assert all(prediction.phase_index in [0, 2] for prediction in predictions)
    training_eps = agent._eps
    agent.exploit({"obs": make_fake_obs(), "extra_info": make_extra_info()})
    fallback_action = agent.exploit({"extra_info": make_extra_info()})
    assert fallback_action[0] == 0
    assert 0 <= fallback_action[1] < Config.DIM_OF_ACTION_PHASE
    assert Config.MIN_GREEN_DURATION <= fallback_action[2] <= Config.MAX_GREEN_DURATION
    assert agent._eps == training_eps

    phase_reward, duration_reward = reward_shaping(make_fake_obs(), [0, 0, Config.MIN_GREEN_DURATION], agent)
    assert isinstance(phase_reward, float)
    assert isinstance(duration_reward, float)
    assert abs(phase_reward) > 0 or abs(duration_reward) > 0
    assert agent.preprocess.phase_last_served_frame[0] == 1
    assert reward_shaping({}, [0, 0, Config.MIN_GREEN_DURATION], agent) == (0.0, 0.0)
    assert reward_shaping({"frame_state": {"frame_no": 2}}, [0, 0, Config.MIN_GREEN_DURATION], agent) == (
        0.0,
        0.0,
    )
    assert reward_shaping(make_fake_obs(), [0], agent) == (0.0, 0.0)

    algorithm = Algorithm(agent.model, agent.optim, device="cpu", logger=NullLogger(), monitor=None)
    assert algorithm.target_model is not agent.model
    model_outputs = agent.model([obs_data.feature])[0]
    assert len(model_outputs) == Config.NUMB_HEAD
    assert model_outputs[0].shape[-1] == Config.DIM_OF_ACTION
    single_outputs = agent.model(obs_data.feature)[0]
    assert single_outputs[0].shape == (1, Config.DIM_OF_ACTION)
    short_outputs = agent.model([obs_data.feature[:-5]])[0]
    assert short_outputs[0].shape == (1, Config.DIM_OF_ACTION)
    long_outputs = agent.model([obs_data.feature + [0.0] * 5])[0]
    assert long_outputs[0].shape == (1, Config.DIM_OF_ACTION)
    action_tensor = torch.tensor([[0, 2, Config.MIN_GREEN_DURATION + 5], [0, 99, 999]], dtype=torch.float32)
    action_indices = algorithm._action_to_joint_index(action_tensor)
    assert action_indices.tolist() == [[45], [Config.DIM_OF_ACTION - 1]]
    phase_mask = algorithm._phase_legal_mask(torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32))
    assert phase_mask.tolist() == [[True, False, True, False], [True, True, True, True]]
    joint_mask = algorithm._joint_legal_mask(phase_mask)
    assert joint_mask.shape == (2, Config.DIM_OF_ACTION)
    assert joint_mask[0, 0]
    assert not joint_mask[0, Config.DIM_OF_ACTION_DURATION]
    assert joint_mask[0, Config.DIM_OF_ACTION_DURATION * 2]
    assert joint_mask[1].all()
    algorithm.update_target_q()

    frame_type = type("Frame", (), {})
    first = frame_type()
    first.obs = obs_data.feature
    first.act = [0, 0, Config.MIN_GREEN_DURATION]
    first.rew = (1.0, 0.5)
    first.done = 0
    first.legal_action = [1, 0, 1, 0]
    second = frame_type()
    second.obs = obs_data.feature
    second.act = [0, 1, Config.MIN_GREEN_DURATION + 1]
    second.rew = (0.5, 0.25)
    second.done = 1
    second.legal_action = [0, 1, 0, 1]
    samples = sample_process([first, second])
    assert len(samples) == 2
    assert isinstance(samples[0], SampleData)
    assert samples[0].act == [0, 0, Config.MIN_GREEN_DURATION]
    assert samples[0].legal_action == [0, 1, 0, 1]
    assert samples[1].done == 0
    assert samples[1]._obs == samples[1].obs
    algorithm.learn(samples)
    bad_sample = SampleData(
        obs=[float("nan")] * Config.DIM_OF_OBSERVATION,
        _obs=[float("inf")] * Config.DIM_OF_OBSERVATION,
        act=[0, 0, Config.MIN_GREEN_DURATION],
        rew=(float("nan"), float("inf")),
        done=1,
        legal_action=[1, 0, 1, 0],
    )
    algorithm.learn([bad_sample])
    assert all(torch.isfinite(param).all().item() for param in agent.model.parameters())

    agent.load_model(id="latest")
    with tempfile.TemporaryDirectory() as model_dir:
        agent.save_model(path=model_dir, id="smoke")
        assert not (Path(model_dir) / "model.ckpt-smoke.pkl.tmp").exists()
        agent.load_model(path=model_dir, id="smoke")
        agent.save_model(path=model_dir, id="latest")
        assert not (Path(model_dir) / "model.ckpt-latest.pkl.tmp").exists()
        agent.load_model(path=model_dir, id="latest")
        latest_path = Path(model_dir) / "model.ckpt-latest.pkl"
        latest_path.write_text("not a checkpoint", encoding="utf-8")
        agent.load_model(path=model_dir, id="latest")
        torch.save(["bad"], latest_path)
        agent.load_model(path=model_dir, id="latest")


if __name__ == "__main__":
    main()
