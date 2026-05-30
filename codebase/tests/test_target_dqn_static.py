#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    root = ROOT.parent
    algorithm = read("agent_target_dqn/algorithm/algorithm.py")
    agent = read("agent_target_dqn/agent.py")
    conf = read("agent_target_dqn/conf/conf.py")
    definition = read("agent_target_dqn/feature/definition.py")
    traffic_utils = read("agent_target_dqn/feature/traffic_utils.py")
    workflow = read("agent_target_dqn/workflow/train_workflow.py")
    package_script = root / "scripts" / "package_submission.sh"
    check_script = root / "scripts" / "check_offline.sh"

    require(
        "DIM_OF_OBSERVATION = 576" in conf,
        "observation dim should include 560 grid + 8 phase features + 8 traffic features",
    )
    require("PHASE_FEATURE_DIM = 8" in conf, "phase feature dimension should stay explicit")
    require("TRAFFIC_FEATURE_DIM = 8" in conf, "traffic feature dimension should stay explicit")

    require("target_model = self.model" not in algorithm, "target model must not alias online model")
    require("deepcopy(self.model)" in algorithm, "target model should be an independent copy")
    require("def update_target_q" in algorithm, "target network sync method is required")
    require("online_next_outputs" in algorithm, "Double DQN should select next action with online network")
    require("next_q_value = target_outputs" in algorithm, "target network should evaluate selected next action")
    require("smooth_l1_loss" in algorithm, "TD loss should use Huber loss for stability")
    require("_action_to_head_indices" in algorithm, "training must map env action to Q-head indices")
    require("Config.MIN_GREEN_DURATION" in algorithm, "duration seconds must be converted to duration index")
    require("if not list_sample_data" in algorithm, "learn should handle empty batches")

    require('act=3' in definition, "SampleData.act should match [junction, phase, duration_seconds]")
    require("return 0, 0" not in definition, "reward_shaping must not return all-zero rewards")
    require("get_phase_pressure" in definition, "reward should use shared phase pressure")
    require("rew is not None" in definition, "sample_process should handle missing rewards")

    require("MIN_GREEN_DURATION + duration_index" in agent, "action_process must map duration index to seconds")
    require("def _phase_feature" in agent, "observation should include traffic signal phase features")
    require("def _traffic_feature" in agent, "observation should include traffic pressure features")
    require("phase_feature + traffic_feature" in agent, "observation should append phase and traffic features")
    require("def rule_based_action" in agent, "exploit should have a rule-based fallback")
    require("if not os.path.exists(model_file_path)" in agent, "load_model should handle missing latest model")
    require("self.algorithm.update_target_q()" in agent, "load_model should sync target network")

    require("def get_phase_pressure" in traffic_utils, "shared phase pressure helper is required")
    require("get_lane_position_meters" in traffic_utils, "lane coordinate normalization helper is required")
    require("def normalize_phase_legal_action" in traffic_utils, "phase legal action normalizer is required")
    require("masked_fill" in agent, "phase Q-values should be masked before greedy selection")
    require("np.flatnonzero" in agent, "random exploration should sample only legal phase actions")

    require("phase_reward" in workflow and "duration_reward" in workflow, "workflow should monitor reward components")
    require("predict_cnt % 20" in workflow, "workflow should not log every frame")

    require(package_script.exists(), "submission package script is required")
    require(check_script.exists(), "offline check script is required")
    require(package_script.stat().st_mode & 0o111, "submission package script should be executable")
    require(check_script.stat().st_mode & 0o111, "offline check script should be executable")


if __name__ == "__main__":
    main()
