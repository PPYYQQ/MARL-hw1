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
    ppo_conf = read("agent_ppo/conf/conf.py")
    ppo_agent = read("agent_ppo/agent.py")

    require("GAMMA = 0.99" in target_conf, "Target-DQN should use a long-horizon discount")
    require("LR = 3e-4" in target_conf, "Target-DQN learning rate should match stable PPO-style baselines")
    require("EPSILON_DECAY = 0.97" in target_conf, "Target-DQN epsilon should decay within short platform runs")
    require("END_EPSILON_GREEDY = 0.1" in target_conf, "Target-DQN should keep moderate exploration")
    require("TARGET_UPDATE_FREQ = 20" in target_conf, "Target-DQN target network should sync during 1h smoke runs")

    require("INIT_LEARNING_RATE_START = 3e-4" in ppo_conf, "PPO should use Adam-friendly baseline learning rate")
    require("BETA_START = 0.01" in ppo_conf, "PPO entropy coefficient should stay near common baseline")
    require("CLIP_PARAM = 0.2" in ppo_conf, "PPO clip parameter should stay at common clipped PPO default")
    require("GAMMA = 0.99" in ppo_conf, "PPO discount should use common baseline value")
    require("LAMDA = 0.95" in ppo_conf, "PPO GAE lambda should use common baseline value")
    require("USE_GRAD_CLIP = True" in ppo_conf, "PPO should enable gradient clipping")
    require("GRAD_CLIP_RANGE = 0.5" in ppo_conf, "PPO grad clip should match common max grad norm")
    require("VALUE_COEF = 1.0" in ppo_conf, "PPO current value target blend should remain full GAE return")
    require("torch.optim.Adam" in ppo_agent, "PPO should use Adam optimizer")
    require(ppo_agent.count("self.optimizer =") == 1, "PPO optimizer should not be initialized twice")


if __name__ == "__main__":
    main()
