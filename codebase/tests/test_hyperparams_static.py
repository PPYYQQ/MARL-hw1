#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    target_conf = read("agent_target_dqn/conf/conf.py")
    configure_app = read("conf/configure_app.toml")
    ppo_conf = read("agent_ppo/conf/conf.py")
    ppo_agent = read("agent_ppo/agent.py")
    ppo_algorithm = read("agent_ppo/algorithm/algorithm.py")
    ppo_definition = read("agent_ppo/feature/definition.py")
    ppo_model = read("agent_ppo/model/model.py")
    ppo_workflow = read("agent_ppo/workflow/train_workflow.py")
    ppo_monitor = read("agent_ppo/conf/monitor_builder.py")

    require("GAMMA = 0.99" in target_conf, "Target-DQN should use a long-horizon discount")
    require("LR = 3e-4" in target_conf, "Target-DQN learning rate should match stable PPO-style baselines")
    require("EPSILON_DECAY = 0.995" in target_conf, "Target-DQN epsilon should not collapse early in 1h runs")
    require("END_EPSILON_GREEDY = 0.2" in target_conf, "Target-DQN should keep enough exploration after E05")
    require("TARGET_UPDATE_FREQ = 10" in target_conf, "Target-DQN target network should sync during 1h smoke runs")
    require("PHASE_AGE_SCALE = 90.0" in target_conf, "Target-DQN should make unserved phase age visible sooner")
    require("FAIRNESS_BONUS_SCALE = 0.5" in target_conf, "Target-DQN should reward serving starved phases after E05")
    require("preload_ratio = 0.03125" in configure_app, "Replay should start learner updates earlier after E05")
    require("train_batch_size = 128" in configure_app, "Replay batch size should fit short 1h platform runs")

    require("INIT_LEARNING_RATE_START = 3e-4" in ppo_conf, "PPO should use Adam-friendly baseline learning rate")
    require("BETA_START = 0.01" in ppo_conf, "PPO entropy coefficient should stay near common baseline")
    require("CLIP_PARAM = 0.2" in ppo_conf, "PPO clip parameter should stay at common clipped PPO default")
    require("GAMMA = 0.99" in ppo_conf, "PPO discount should use common baseline value")
    require("LAMDA = 0.95" in ppo_conf, "PPO GAE lambda should use common baseline value")
    require("PPO_FRAGMENT_SIZE = 32" in ppo_conf, "PPO should flush partial trajectories before episode end")
    require("USE_GRAD_CLIP = True" in ppo_conf, "PPO should enable gradient clipping")
    require("GRAD_CLIP_RANGE = 0.5" in ppo_conf, "PPO grad clip should match common max grad norm")
    require("VALUE_COEF = 1.0" in ppo_conf, "PPO current value target blend should remain full GAE return")
    require("DIM_OF_OBSERVATION = 638" in ppo_conf, "PPO should use the same traffic-rich observation as Target-DQN")
    require("DIM_OF_ACTION_DURATION_1 = 20" in ppo_conf, "PPO should expose 20 duration bins")
    require("MIN_GREEN_DURATION = 8" in ppo_conf, "PPO should enforce the minimum green duration")
    require("def duration_index_to_seconds" in ppo_conf, "PPO should map duration index to environment seconds")
    require("torch.optim.Adam" in ppo_agent, "PPO should use Adam optimizer")
    require(ppo_agent.count("self.optimizer =") == 1, "PPO optimizer should not be initialized twice")
    require("TargetFeatureAgent.observation_process" in ppo_agent, "PPO should reuse the robust traffic feature pipeline")
    require("self._full_action_mask" in ppo_agent, "PPO should expand legal phase masks to phase+duration masks")
    require("Config.duration_index_to_seconds" in ppo_agent, "PPO action_process should output duration seconds")
    require('"agent_ppo/ckpt"' in ppo_agent, "PPO should save checkpoints under its own ckpt directory")
    require("policy_losses.append" in ppo_algorithm, "PPO policy loss should be implemented")
    require("torch.clamp(ratio" in ppo_algorithm, "PPO should use clipped probability ratios")
    require("entropy_terms.append" in ppo_algorithm, "PPO entropy term should be implemented")
    require("Config.ENTROPY_COEF" in ppo_algorithm, "PPO loss should include entropy regularization")
    require("target_dqn_reward_shaping" in ppo_definition, "PPO should use non-zero traffic reward shaping")
    require("gae = delta + gamma * lamda" in ppo_definition, "PPO should compute GAE advantages")
    require("[Config.DIM_OF_OBSERVATION, 256, 128, self.unit_size]" in ppo_model, "PPO model should have a shared MLP backbone")
    require("def _prepare_input" in ppo_model, "PPO model should normalize input tensor shape")
    require("nn.Sequential(layers)" in ppo_model, "PPO MLP should explicitly register its child layers")
    require("parameters = list(self.model.parameters())" in ppo_agent, "PPO optimizer should receive a materialized parameter list")
    require("PPO model has no registered parameters" in ppo_agent, "PPO should fail clearly if model files are stale")
    require("_predict_ppo_action" in ppo_workflow, "PPO workflow should preserve action probability and value for samples")
    require("len(collector) > Config.PPO_FRAGMENT_SIZE" in ppo_workflow, "PPO workflow should periodically send samples")
    require("fragment[-1].next_value = collector[-1].value" in ppo_workflow, "PPO fragment flush should bootstrap from retained transition value")
    require("collector = collector[-1:]" in ppo_workflow, "PPO partial sample flush should retain the latest transition")
    require("if samples[-1].done:" in ppo_definition, "PPO GAE should keep fragment bootstrap value for nonterminal chunks")
    require("_default_env_metric_snapshot" in ppo_workflow, "PPO workflow should report environment metrics")
    require("model_grad_norm" in ppo_monitor, "PPO monitor should expose gradient norm")
    require("phase_switch_rate" in ppo_monitor, "PPO monitor should expose action switching behavior")


if __name__ == "__main__":
    main()
