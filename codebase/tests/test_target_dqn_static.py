#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    algorithm = read("agent_target_dqn/algorithm/algorithm.py")
    agent = read("agent_target_dqn/agent.py")
    definition = read("agent_target_dqn/feature/definition.py")
    traffic_utils = read("agent_target_dqn/feature/traffic_utils.py")
    workflow = read("agent_target_dqn/workflow/train_workflow.py")

    require("target_model = self.model" not in algorithm, "target model must not alias online model")
    require("deepcopy(self.model)" in algorithm, "target model should be an independent copy")
    require("def update_target_q" in algorithm, "target network sync method is required")
    require("_action_to_head_indices" in algorithm, "training must map env action to Q-head indices")
    require("Config.MIN_GREEN_DURATION" in algorithm, "duration seconds must be converted to duration index")
    require("if not list_sample_data" in algorithm, "learn should handle empty batches")

    require('act=3' in definition, "SampleData.act should match [junction, phase, duration_seconds]")
    require("return 0, 0" not in definition, "reward_shaping must not return all-zero rewards")
    require("get_phase_pressure" in definition, "reward should use shared phase pressure")
    require("rew is not None" in definition, "sample_process should handle missing rewards")

    require("MIN_GREEN_DURATION + duration_index" in agent, "action_process must map duration index to seconds")
    require("def rule_based_action" in agent, "exploit should have a rule-based fallback")
    require("if not os.path.exists(model_file_path)" in agent, "load_model should handle missing latest model")
    require("self.algorithm.update_target_q()" in agent, "load_model should sync target network")

    require("def get_phase_pressure" in traffic_utils, "shared phase pressure helper is required")
    require("get_lane_position_meters" in traffic_utils, "lane coordinate normalization helper is required")

    require("phase_reward" in workflow and "duration_reward" in workflow, "workflow should monitor reward components")
    require("predict_cnt % 20" in workflow, "workflow should not log every frame")


if __name__ == "__main__":
    main()
