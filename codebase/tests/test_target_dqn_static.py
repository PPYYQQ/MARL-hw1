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
    model = read("agent_target_dqn/model/model.py")
    preprocessor = read("agent_target_dqn/feature/preprocessor.py")
    traffic_utils = read("agent_target_dqn/feature/traffic_utils.py")
    workflow = read("agent_target_dqn/workflow/train_workflow.py")
    package_script = root / "scripts" / "package_submission.sh"
    check_script = root / "scripts" / "check_offline.sh"

    require(
        "DIM_OF_OBSERVATION = 638" in conf,
        "observation dim should include 560 grid + 8 phase + 4 age + 8 traffic + 8 trend + 8 history + 42 lane stats",
    )
    require("PHASE_FEATURE_DIM = 8" in conf, "phase feature dimension should stay explicit")
    require("PHASE_AGE_FEATURE_DIM = 4" in conf, "phase age feature dimension should stay explicit")
    require("TRAFFIC_FEATURE_DIM = 8" in conf, "traffic feature dimension should stay explicit")
    require("TRAFFIC_TREND_FEATURE_DIM = 8" in conf, "traffic trend feature dimension should stay explicit")
    require("TRAFFIC_HISTORY_FEATURE_DIM = 8" in conf, "traffic history feature dimension should stay explicit")
    require("TRAFFIC_HISTORY_SIZE = 4" in conf, "traffic history window should stay explicit")
    require("LANE_STAT_FEATURE_DIM = 42" in conf, "lane stat feature dimension should stay explicit")
    require("DIM_OF_ACTION = DIM_OF_ACTION_PHASE * DIM_OF_ACTION_DURATION" in conf, "joint action dimension should be explicit")
    require("NUMB_HEAD = 1" in conf, "Target-DQN should use one joint action Q head")
    require("FAIRNESS_BONUS_SCALE" in conf, "fairness reward scale should stay explicit")
    require("action_shape = [Config.DIM_OF_ACTION]" in model, "model should output one joint action Q head")

    require("target_model = self.model" not in algorithm, "target model must not alias online model")
    require("deepcopy(self.model)" in algorithm, "target model should be an independent copy")
    require("def update_target_q" in algorithm, "target network sync method is required")
    require("online_next_outputs" in algorithm, "Double DQN should select next action with online network")
    require("next_q_value = target_outputs" in algorithm, "target network should evaluate selected next action")
    require("smooth_l1_loss" in algorithm, "TD loss should use Huber loss for stability")
    require("_action_to_joint_index" in algorithm, "training must map env action to joint Q index")
    require("def _phase_legal_mask" in algorithm, "training should normalize legal phase masks")
    require("def _joint_legal_mask" in algorithm, "training should expand phase legality to joint actions")
    require("joint_legal_mask" in algorithm, "Double DQN joint target should use legal action mask")
    require("total_reward = rew.sum" in algorithm, "joint Q target should train on total reward")
    require("Config.MIN_GREEN_DURATION" in algorithm, "duration seconds must be converted to duration index")
    require("if not list_sample_data" in algorithm, "learn should handle empty batches")

    require('act=3' in definition, "SampleData.act should match [junction, phase, duration_seconds]")
    require("return 0, 0" not in definition, "reward_shaping must not return all-zero rewards")
    require("get_phase_pressure" in definition, "reward should use shared phase pressure")
    require("rew is not None" in definition, "sample_process should handle missing rewards")
    require("sample_datas[i].legal_action = sample_datas[i + 1].legal_action" in definition, "samples should carry next-state phase legality")
    require('frame_state = _obs.get("frame_state")' in definition, "reward should tolerate missing frame_state")
    require('frame_state.get("vehicles", [])' in definition, "reward should tolerate missing vehicles")
    require("def _fairness_reward" in definition, "reward should include phase fairness term")
    require("def _mark_phase_served" in definition, "reward should update phase service bookkeeping")

    require("MIN_GREEN_DURATION + duration_index" in agent, "action_process must map duration index to seconds")
    require("def _phase_feature" in agent, "observation should include traffic signal phase features")
    require("def _phase_age_feature" in agent, "observation should include phase service age features")
    require("def _traffic_feature" in agent, "observation should include traffic pressure features")
    require("def _traffic_trend_feature" in agent, "observation should include traffic trend features")
    require("def _traffic_history_feature" in agent, "observation should include traffic history features")
    require("def _lane_stat_feature" in agent, "observation should include per-lane statistics")
    require(
        "+ traffic_history_feature" in agent and "+ lane_stat_feature" in agent,
        "observation should append phase, phase-age, traffic, trend, history, and lane-stat features",
    )
    require('raw_obs = observation.get("obs", observation)' in agent, "exploit should tolerate missing obs wrapper")
    require('frame_state = raw_obs.get("frame_state", {})' in agent, "observation should tolerate missing frame_state")
    require('frame_state.get("vehicles", [])' in agent, "observation should tolerate missing vehicles")
    require(
        "vehicles = [vehicle for vehicle in vehicles if isinstance(vehicle, dict)]" in agent,
        "observation should filter malformed vehicle records",
    )
    require("def rule_based_action" in agent, "exploit should have a rule-based fallback")
    require("def _joint_action_mask" in agent, "prediction should expand phase legality to joint actions")
    require("list_joint_action" in agent, "prediction should select joint actions")
    require("if not exploit_flag" in agent, "exploit should not decay training epsilon")
    require("if exploit_flag or np.random.rand() >= self._eps" in agent, "exploit should force greedy prediction")
    require("if not os.path.exists(model_file_path)" in agent, "load_model should handle missing latest model")
    require("except RuntimeError as err" in agent, "load_model should handle incompatible latest checkpoints")
    require("incompatible checkpoint" in agent, "load_model should log incompatible latest checkpoints")
    require("self.algorithm.update_target_q()" in agent, "load_model should sync target network")

    require("def get_phase_pressure" in traffic_utils, "shared phase pressure helper is required")
    require("def get_lane_statistics" in traffic_utils, "shared lane statistics helper is required")
    require("def get_traffic_summary" in traffic_utils, "shared traffic summary helper is required")
    require("def get_traffic_trend" in traffic_utils, "shared traffic trend helper is required")
    require("def get_traffic_history_feature" in traffic_utils, "shared traffic history helper is required")
    require("get_lane_position_meters" in traffic_utils, "lane coordinate normalization helper is required")
    require(
        "except (KeyError, TypeError, ValueError, AttributeError)" in traffic_utils,
        "traffic helpers should skip malformed vehicle records",
    )
    require("def normalize_phase_legal_action" in traffic_utils, "phase legal action normalizer is required")
    require("last_traffic_summary = None" in preprocessor, "traffic trend state should reset each episode")
    require("traffic_history = []" in preprocessor, "traffic history state should reset each episode")
    require("phase_last_served_frame" in preprocessor, "phase service state should reset each episode")
    require('start_info.get("junctions", [])' in preprocessor, "road init should tolerate partial init_state")
    require('frame_state = raw_obs.get("frame_state")' in preprocessor, "traffic update should tolerate missing frame_state")
    require('frame_state.get("vehicles", [])' in preprocessor, "traffic update should tolerate missing vehicles")
    require('vehicle_id = vehicle.get("v_id")' in preprocessor, "traffic update should skip malformed vehicles")
    require("masked_fill" in agent, "joint Q-values should be masked before greedy selection")
    require("np.flatnonzero" in agent, "random exploration should sample only legal phase actions")

    require("phase_reward" in workflow and "duration_reward" in workflow, "workflow should monitor reward components")
    require("def _need_to_predict" in workflow, "workflow should normalize legal_action before prediction gating")
    require('obs["legal_action"][0]' not in workflow, "workflow should not assume legal_action is always a list")
    require("_need_to_predict(obs)" in workflow, "workflow should use the robust prediction gate")
    require("run_episodes error: {e}" in workflow, "workflow should log the original episode exception")
    require("from e" in workflow, "workflow should preserve exception cause for platform debugging")
    require('agent.save_model(id="latest")' in workflow, "workflow should save the checkpoint name it later loads")
    require("agent.send_sample_data(list(g_data))" in workflow, "workflow should not clear the list object sent to learner")
    require("def _should_log_progress" in workflow, "workflow should centralize progress log gating")
    require("need_to_predict and predict_cnt > 0 and predict_cnt % 20 == 0" in workflow, "workflow should not log every non-prediction frame")
    require("_should_log_progress(predict_cnt, done, need_to_predict)" in workflow, "workflow should use robust progress log gating")

    require(package_script.exists(), "submission package script is required")
    require(check_script.exists(), "offline check script is required")
    require(package_script.stat().st_mode & 0o111, "submission package script should be executable")
    require(check_script.stat().st_mode & 0o111, "offline check script should be executable")


if __name__ == "__main__":
    main()
