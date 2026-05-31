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
    agent_entrypoints = [
        ("agent_target_dqn/agent.py", agent),
        ("agent_dqn/agent.py", read("agent_dqn/agent.py")),
        ("agent_ppo/agent.py", read("agent_ppo/agent.py")),
        ("agent_diy/agent.py", read("agent_diy/agent.py")),
    ]
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
    require("def _prepare_input" in model, "model should normalize input tensor shape")
    require("s.unsqueeze(0)" in model, "model should batch one-dimensional observations")
    require("F.pad(s" in model, "model should pad short observation vectors")
    require("s[:, : Config.DIM_OF_OBSERVATION]" in model, "model should truncate long observation vectors")
    require("def _as_numpy_array" in model, "model should tolerate ragged Python observation batches")
    require("np.stack(rows)" in model, "model should stack normalized ragged rows")
    require("def _fit_numpy_width" in model, "model should normalize ragged row width before tensor conversion")
    for agent_path, agent_source in agent_entrypoints:
        require("def _configure_torch_threads" in agent_source, f"{agent_path} should guard torch thread setup")
        require("\n_configure_torch_threads()\n" in agent_source, f"{agent_path} should call guarded torch thread setup")
        require("except RuntimeError" in agent_source, f"{agent_path} should tolerate torch thread setup RuntimeError")

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
    require("def _normalize_tensor" in algorithm, "learn should normalize malformed sample tensors")
    require("F.pad(tensor" in algorithm, "learn should pad short sample tensors")
    require("tensor = tensor[:width]" in algorithm, "learn should truncate long sample tensors")
    require("width=Config.DIM_OF_OBSERVATION" in algorithm, "learn should normalize observation width before stacking")
    require("width=Config.DIM_OF_ACTION_PHASE" in algorithm, "learn should normalize legal action width before stacking")
    require("def _finite_tensor" in algorithm, "learn should sanitize non-finite tensors")
    require("obs = self._finite_tensor(obs)" in algorithm, "learn should sanitize observations")
    require("rew = self._finite_tensor(rew)" in algorithm, "learn should sanitize rewards")
    require("legal_action = self._finite_tensor(legal_action)" in algorithm, "learn should sanitize legal action masks")
    require("q_targets = self._finite_tensor(q_targets)" in algorithm, "learn should sanitize TD targets")
    require("torch.isfinite(loss)" in algorithm, "learn should skip optimizer steps on non-finite loss")
    require("non-finite loss" in algorithm, "learn should log non-finite loss skips")
    require("math.isfinite(model_grad_norm)" in algorithm, "learn should skip optimizer steps on non-finite gradients")
    require("non-finite grad norm" in algorithm, "learn should log non-finite gradient skips")
    require("def _put_monitor_data" in algorithm, "learn monitor reporting should be isolated")
    require("monitor.put_data" in algorithm, "learn should still report monitor data when available")
    require("monitor put_data failed" in algorithm, "learn should log monitor reporting failures")
    require("def _log_info" in algorithm, "learn logger calls should be isolated")
    require("except Exception" in algorithm, "learn monitor/logger helpers should tolerate backend failures")

    require('act=3' in definition, "SampleData.act should match [junction, phase, duration_seconds]")
    require("return 0, 0" not in definition, "reward_shaping must not return all-zero rewards")
    require("get_phase_pressure" in definition, "reward should use shared phase pressure")
    require("if not list_game_data" in definition, "sample_process should handle empty trajectories")
    require("if not sample_datas" in definition, "sample_process should handle all-invalid trajectories")
    require("def _fixed_float_list" in definition, "sample_process should normalize sample field widths")
    require("np.nan_to_num(values" in definition, "sample_process should sanitize non-finite sample fields")
    require("_fixed_float_list(obs, Config.DIM_OF_OBSERVATION)" in definition, "sample_process should normalize obs width")
    require("_fixed_action_list(act)" in definition, "sample_process should normalize action width and bounds")
    require("_not_done_flag" in definition, "sample_process should normalize done flags")
    require('reward = getattr(data, "rew", None)' in definition, "sample_process should handle missing rewards")
    require("sample_datas[i].legal_action = sample_datas[i + 1].legal_action" in definition, "samples should carry next-state phase legality")
    require("def _safe_int" in definition, "reward should sanitize scalar integer fields")
    require("except (TypeError, ValueError, IndexError, OverflowError)" in definition, "reward should catch overflow in malformed actions")
    require("frame_no = _safe_int" in definition, "reward should sanitize frame_no")
    require("_safe_int(served_frame" in definition, "fairness reward should sanitize phase service history")
    require('frame_state = _obs.get("frame_state")' in definition, "reward should tolerate missing frame_state")
    require('frame_state.get("vehicles", [])' in definition, "reward should tolerate missing vehicles")
    require("vehicles = [vehicle for vehicle in vehicles if isinstance(vehicle, dict)]" in definition, "reward should skip malformed vehicles")
    require("phase_index = int(np.clip(int(act[1])" in definition, "reward should clamp malformed phase actions")
    require("def _fairness_reward" in definition, "reward should include phase fairness term")
    require("def _mark_phase_served" in definition, "reward should update phase service bookkeeping")

    require("if not list_obs_data" in agent, "predict should handle empty observation batches")
    require('getattr(obs_data, "legal_action", None)' in agent, "predict should default missing legal_action")
    require("MIN_GREEN_DURATION + duration_index" in agent, "action_process must map duration index to seconds")
    require("def _safe_action_index" in agent, "action_process should sanitize action indices")
    require("junction_id = 0" in agent, "action_process should force single-junction actions")
    require("np.isfinite(value)" in agent, "action_process should reject NaN/Inf indices")
    require("def _safe_float" in agent, "agent observation helpers should sanitize scalar values")
    require("def _safe_int" in agent, "agent observation helpers should sanitize frame numbers")
    require("duration = _safe_nonnegative_float" in agent, "phase features should sanitize duration")
    require("frame_no = _safe_int" in agent, "phase age should sanitize frame_no")
    require("not isinstance(last_served, list)" in agent, "phase age should reset malformed service history")
    require("phase_id = self._safe_action_index" in agent, "phase features should sanitize phase ids")
    require("def _phase_feature" in agent, "observation should include traffic signal phase features")
    require("def _phase_age_feature" in agent, "observation should include phase service age features")
    require("def _traffic_feature" in agent, "observation should include traffic pressure features")
    require("def _traffic_trend_feature" in agent, "observation should include traffic trend features")
    require("def _traffic_history_feature" in agent, "observation should include traffic history features")
    require("def _lane_stat_feature" in agent, "observation should include per-lane statistics")
    require("def _sanitize_observation" in agent, "observation should sanitize final feature vectors")
    require("observation = self._sanitize_observation(observation)" in agent, "observation should run final feature sanitation")
    require("np.nan_to_num(values" in agent, "observation sanitation should remove non-finite values")
    require("Config.DIM_OF_OBSERVATION" in agent, "observation sanitation should enforce configured width")
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
    require("model_tmp_path" in agent, "save_model should write checkpoint through a temporary file")
    require("os.replace(model_tmp_path, model_file_path)" in agent, "save_model should atomically publish checkpoints")
    require("os.remove(model_tmp_path)" in agent, "save_model should clean temporary checkpoint files on failure")
    require("if not os.path.exists(model_file_path)" in agent, "load_model should handle missing latest model")
    require("unreadable checkpoint" in agent, "load_model should skip unreadable latest checkpoints")
    require("invalid checkpoint" in agent, "load_model should skip invalid latest checkpoint payloads")
    require(
        "except (RuntimeError, TypeError, ValueError) as err" in agent,
        "load_model should handle incompatible latest checkpoints",
    )
    require("incompatible checkpoint" in agent, "load_model should log incompatible latest checkpoints")
    require("self.algorithm.update_target_q()" in agent, "load_model should sync target network")

    require("def get_phase_pressure" in traffic_utils, "shared phase pressure helper is required")
    require("def get_lane_statistics" in traffic_utils, "shared lane statistics helper is required")
    require("def get_traffic_summary" in traffic_utils, "shared traffic summary helper is required")
    require("def get_traffic_trend" in traffic_utils, "shared traffic trend helper is required")
    require("def get_traffic_history_feature" in traffic_utils, "shared traffic history helper is required")
    require("get_lane_position_meters" in traffic_utils, "lane coordinate normalization helper is required")
    require("def _finite_float" in traffic_utils, "traffic helpers should sanitize scalar values")
    require("def _nonnegative_float" in traffic_utils, "traffic helpers should clamp non-negative traffic metrics")
    require("def _phase_array" in traffic_utils, "traffic helpers should normalize phase arrays")
    require("if not np.isfinite(y_pos)" in traffic_utils, "lane position helper should reject non-finite positions")
    require("np.nan_to_num(values" in traffic_utils, "legal action normalizer should sanitize non-finite values")
    require("np.nan_to_num(array" in traffic_utils, "traffic helper arrays should sanitize non-finite values")
    require(
        "except (KeyError, TypeError, ValueError, AttributeError)" in traffic_utils,
        "traffic helpers should skip malformed vehicle records",
    )
    require("def normalize_phase_legal_action" in traffic_utils, "phase legal action normalizer is required")
    require("last_traffic_summary = None" in preprocessor, "traffic trend state should reset each episode")
    require("traffic_history = []" in preprocessor, "traffic history state should reset each episode")
    require("phase_last_served_frame" in preprocessor, "phase service state should reset each episode")
    require("def _safe_float" in preprocessor, "traffic preprocessor should sanitize scalar frame values")
    require("def _safe_int" in preprocessor, "traffic preprocessor should sanitize frame numbers")
    require("def _is_hashable" in preprocessor, "traffic preprocessor should skip unhashable ids")
    require("def _safe_position_pair" in preprocessor, "traffic preprocessor should sanitize vehicle positions")
    require('start_info.get("junctions", [])' in preprocessor, "road init should tolerate partial init_state")
    require('frame_state = raw_obs.get("frame_state")' in preprocessor, "traffic update should tolerate missing frame_state")
    require('frame_state.get("vehicles", [])' in preprocessor, "traffic update should tolerate missing vehicles")
    require("frame_no = _safe_int" in preprocessor, "traffic update should sanitize frame_no")
    require("frame_time = _safe_float" in preprocessor, "traffic update should sanitize frame_time")
    require("vehicles = vehicles if isinstance(vehicles, list) else []" in preprocessor, "waiting helpers should tolerate non-list vehicles")
    require('vehicle_id = vehicle.get("v_id")' in preprocessor, "traffic update should skip malformed vehicles")
    require("not _is_hashable(vehicle_id)" in preprocessor, "traffic update should skip unhashable vehicle ids")
    require("except (KeyError, TypeError, ValueError, AttributeError, OverflowError)" in preprocessor, "traffic update should skip malformed dynamic fields")
    require("masked_fill" in agent, "joint Q-values should be masked before greedy selection")
    require("np.flatnonzero" in agent, "random exploration should sample only legal phase actions")

    require("phase_reward" in workflow and "duration_reward" in workflow, "workflow should monitor reward components")
    require("def _need_to_predict" in workflow, "workflow should normalize legal_action before prediction gating")
    require("def _finite_float" in workflow, "workflow reward monitor should sanitize non-finite rewards")
    require("except (TypeError, ValueError, OverflowError)" in workflow, "workflow finite float helper should catch malformed scalars")
    require("math.isfinite" in workflow, "workflow reward monitor should reject NaN/Inf values")
    require('return int(_finite_float(_safe_env_value(env_obs, "frame_no", 0)))' in workflow, "workflow frame_no should use finite scalar sanitation")
    require("def _safe_observation" in workflow, "workflow should tolerate missing observation payloads")
    require("def _safe_extra_info" in workflow, "workflow should tolerate missing extra_info payloads")
    require("def _safe_legal_action" in workflow, "workflow should tolerate missing legal_action in sampled frames")
    require("def _put_monitor_data" in workflow, "workflow monitor reporting should be isolated")
    require("monitor.put_data" in workflow, "workflow should still report monitor data when available")
    require("monitor put_data failed" in workflow, "workflow should log monitor reporting failures")
    require("try:" in workflow and "except Exception" in workflow, "workflow logger/monitor helpers should tolerate backend failures")
    require("def _normalize_reset_result" in workflow, "workflow should normalize tuple reset results")
    require("def _normalize_step_result" in workflow, "workflow should normalize tuple step results")
    require("len(step_result) >= 6" in workflow, "workflow should support documented six-item env.step results")
    require("env_obs = _normalize_reset_result(env.reset" in workflow, "workflow should normalize reset result before use")
    require("env_reward, env_obs = _normalize_step_result(env.step(act))" in workflow, "workflow should normalize step result before use")
    require('obs = _safe_observation(env_obs)' in workflow, "workflow reset path should use safe observation extraction")
    require('legal_action=_safe_legal_action(obs)' in workflow, "workflow sampled frame should use safe legal_action extraction")
    require('obs["legal_action"][0]' not in workflow, "workflow should not assume legal_action is always a list")
    require('obs["legal_action"]' not in workflow, "workflow should not directly index legal_action")
    require("_need_to_predict(obs)" in workflow, "workflow should use the robust prediction gate")
    require("def _predict_action" in workflow, "workflow should centralize robust prediction fallback")
    require("predict returned empty action" in workflow, "workflow should log empty predictions")
    require("fallback to rule_based_action" in workflow, "workflow should fall back when prediction fails")
    require("[0, 0, Config.MIN_GREEN_DURATION]" in workflow, "workflow should have a final legal default action")
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
