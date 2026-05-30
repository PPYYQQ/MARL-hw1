#!/usr/bin/env python3

import math
import sys
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
        "legal_action": [1],
        "frame_state": {
            "frame_no": 1,
            "frame_time": 0,
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
    from agent_target_dqn.feature.definition import ActData, reward_shaping

    agent = Agent(device="cpu", logger=NullLogger(), monitor=None)
    obs_data = agent.observation_process(make_fake_obs(), make_extra_info())
    assert len(obs_data.feature) == Config.DIM_OF_OBSERVATION
    assert_no_nan(obs_data.feature)
    assert sum(obs_data.feature[: Config.GRID_WIDTH * Config.GRID_NUM]) == 2

    low_duration_action = agent.action_process(ActData(junction_id=0, phase_index=0, duration=0))
    high_duration_action = agent.action_process(ActData(junction_id=0, phase_index=99, duration=99))
    assert low_duration_action == [0, 0, Config.MIN_GREEN_DURATION]
    assert high_duration_action == [0, Config.DIM_OF_ACTION_PHASE - 1, Config.MIN_GREEN_DURATION + 19]
    assert agent.rule_based_action(make_fake_obs())[1] == 0

    predictions = agent.predict([obs_data, obs_data])
    assert len(predictions) == 2

    phase_reward, duration_reward = reward_shaping(make_fake_obs(), [0, 0, Config.MIN_GREEN_DURATION], agent)
    assert isinstance(phase_reward, float)
    assert isinstance(duration_reward, float)
    assert abs(phase_reward) > 0 or abs(duration_reward) > 0

    algorithm = Algorithm(agent.model, agent.optim, device="cpu", logger=NullLogger(), monitor=None)
    assert algorithm.target_model is not agent.model
    algorithm.update_target_q()


if __name__ == "__main__":
    main()
