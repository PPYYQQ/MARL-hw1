#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import math
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.feature.definition import *
from agent_target_dqn.feature.traffic_utils import normalize_phase_legal_action
from common_python.utils.common_func import Frame
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    last_save_model_time = 0

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "phase_reward": 0,
        "duration_reward": 0,
        "data_length": 0,
    }
    last_report_monitor_time = time.time()

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        _log_error(logger, "usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        epoch_phase_rew = 0
        epoch_duration_rew = 0

        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger):
            data_length += len(g_data)
            for data in g_data:
                phase_rew, duration_rew = _reward_components(data.rew)
                epoch_phase_rew += phase_rew
                epoch_duration_rew += duration_rew

            epoch_total_rew = epoch_phase_rew + epoch_duration_rew
            _send_sample_data(agent, list(g_data), logger)
            g_data.clear()

        avg_step_reward = 0.0
        avg_phase_reward = 0.0
        avg_duration_reward = 0.0
        if data_length:
            avg_step_reward = round(epoch_total_rew / data_length, 4)
            avg_phase_reward = round(epoch_phase_rew / data_length, 4)
            avg_duration_reward = round(epoch_duration_rew / data_length, 4)

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 1800:
            _save_latest_model(agent, logger)
            last_save_model_time = now

        # Reporting training progress
        # 上报训练进度
        if now - last_report_monitor_time > 60:
            monitor_data["reward"] = avg_step_reward
            monitor_data["phase_reward"] = avg_phase_reward
            monitor_data["duration_reward"] = avg_duration_reward
            monitor_data["data_length"] = data_length
            if _put_monitor_data(monitor, monitor_data, logger):
                last_report_monitor_time = now

        _log_info(
            logger,
            f"Avg Step Reward: {avg_step_reward}, Avg Phase Reward: {avg_phase_reward}, "
            f"Avg Duration Reward: {avg_duration_reward}, Epoch: {epoch}, Data Length: {data_length}",
        )


def run_episodes(n_episode, env, agent, usr_conf, logger):
    try:
        train_test_quick_stop = os.environ.get("is_train_test", "False").lower() == "true"
        for _ in range(n_episode):
            collector = list()
            predict_cnt = 0

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = _get_training_metrics(logger)
            if training_metrics:
                _log_info(logger, f"training_metrics is {training_metrics}")

            # At the start of each environment, loading the latest model file
            # 每次对局开始时, 加载最新model文件
            _load_latest_model(agent, logger)

            # Reset the environment and get the initial extra_info
            # 重置环境, 并获取初始状态
            env_obs = _normalize_reset_result(env.reset(usr_conf=usr_conf))
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, logger):
                break

            agent.reset(env_obs)
            obs = _safe_observation(env_obs)
            extra_info = _safe_extra_info(env_obs)

            # Record the last_predict_act
            # 记录上次预测的动作
            last_predict_act = None

            done = False
            while not done:
                need_to_predict = _need_to_predict(obs)
                if need_to_predict:
                    if len(collector) > 0:
                        # Calculate reward Rewards
                        # 计算奖励
                        reward = reward_shaping(obs, last_predict_act, agent)
                        collector[-1].rew = reward

                    # Feature processing
                    # 特征处理
                    obs_data = agent.observation_process(obs, extra_info)
                    # Agent makes a prediction to get the next frame's action
                    # Agent 进行推理, 获取下一帧的预测动作
                    act = _predict_action(agent, obs_data, obs, logger)
                    predict_cnt += 1
                else:
                    # No need to predict
                    # 不需要预测的情况
                    agent.preprocess.update_traffic_info(obs, extra_info)
                    act = [None, None, None]

                # Interact with the environment, execute actions, get the next extra_info
                # 与环境交互, 执行动作, 获取下一步的状态, 如果遇到不需要预测的帧，则env.step直到得到需要预测的帧
                env_reward, env_obs = _normalize_step_result(env.step(act))
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, logger):
                    if len(collector) > 10:
                        collector = sample_process(collector)
                        yield collector
                    break

                frame_no = _safe_frame_no(env_obs)
                _obs = _safe_observation(env_obs)
                terminated = _safe_done_flag(env_obs, "terminated")
                truncated = _safe_done_flag(env_obs, "truncated")
                extra_info = _safe_extra_info(env_obs)
                # Determine if the environment is over
                # 判断环境结束
                done = terminated or truncated or (train_test_quick_stop and len(collector) > 1)
                if _should_log_progress(predict_cnt, done, need_to_predict):
                    _log_info(logger, f"current frame_no is {frame_no}, predict_cnt is {predict_cnt}")
                if truncated:
                    _log_info(logger, f"truncated is True, frame_no is {frame_no}, so this episode timeout")
                elif terminated:
                    _log_info(logger, f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

                # Save samples only when predicting
                # 只有预测步才保存样本
                if need_to_predict:
                    # Construct environment frames to prepare for sample construction
                    # 构造环境帧，为构造样本做准备
                    frame = Frame(
                        obs=obs_data.feature,
                        act=act,
                        rew=None,
                        done=0,
                        legal_action=_safe_legal_action(obs),
                    )

                    collector.append(frame)

                # Status update
                # 状态更新
                obs = _obs
                if need_to_predict:
                    last_predict_act = act

                # Perform sample processing and return samples for training
                # 进行样本处理并将样本返回进行训练
                if done:
                    if len(collector) > 1:
                        # Calculate reward Rewards include phase_reward and duration_reward
                        # 奖励有phase_reward和duration_reward
                        reward = reward_shaping(_obs, last_predict_act, agent)
                        collector[-1].done = 1
                        collector[-1].rew = reward
                        collector = sample_process(collector)
                        yield collector
                    break

    except Exception as e:
        _log_error(logger, f"run_episodes error: {e}")
        raise RuntimeError(f"run_episodes error: {e}") from e


def _reward_components(reward):
    if reward is None:
        return 0.0, 0.0
    try:
        phase_reward = _finite_float(reward[0])
        duration_reward = _finite_float(reward[1]) if len(reward) > 1 else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0
    return phase_reward, duration_reward


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _normalize_reset_result(reset_result):
    if isinstance(reset_result, dict):
        return reset_result
    if isinstance(reset_result, tuple) and len(reset_result) >= 2:
        return {
            "observation": reset_result[0],
            "extra_info": reset_result[1],
        }
    return {}


def _normalize_step_result(step_result):
    if isinstance(step_result, tuple):
        if len(step_result) >= 6:
            frame_no, observation, reward, terminated, truncated, extra_info = step_result[:6]
            return reward, {
                "frame_no": frame_no,
                "observation": observation,
                "terminated": terminated,
                "truncated": truncated,
                "extra_info": extra_info,
            }
        if len(step_result) >= 2:
            env_reward, env_obs = step_result[:2]
            return env_reward, env_obs if isinstance(env_obs, dict) else {}
    if isinstance(step_result, dict):
        return 0.0, step_result
    return 0.0, {}


def _safe_env_value(env_obs, key, default):
    if isinstance(env_obs, dict):
        return env_obs.get(key, default)
    return default


def _safe_observation(env_obs):
    observation = _safe_env_value(env_obs, "observation", {})
    if isinstance(observation, dict):
        return observation
    return {}


def _safe_extra_info(env_obs):
    extra_info = _safe_env_value(env_obs, "extra_info", {})
    if isinstance(extra_info, dict):
        return extra_info
    return {}


def _safe_frame_no(env_obs):
    return int(_finite_float(_safe_env_value(env_obs, "frame_no", 0)))


def _safe_done_flag(env_obs, key):
    return bool(_safe_env_value(env_obs, key, False))


def _safe_legal_action(obs):
    if isinstance(obs, dict):
        return obs.get("legal_action")
    return None


def _need_to_predict(obs):
    legal_action = obs.get("legal_action") if isinstance(obs, dict) else None
    phase_mask = normalize_phase_legal_action(legal_action, Config.DIM_OF_ACTION_PHASE)
    return any(phase_mask)


def _should_log_progress(predict_cnt, done, need_to_predict):
    return done or (need_to_predict and predict_cnt > 0 and predict_cnt % 20 == 0)


def _predict_action(agent, obs_data, obs, logger):
    try:
        act_data = agent.predict(list_obs_data=[obs_data])
        if act_data:
            return agent.action_process(act_data[0])
        _log_error(logger, "predict returned empty action, fallback to rule_based_action")
    except Exception as err:
        _log_error(logger, f"predict fallback to rule_based_action: {err}")

    try:
        return agent.rule_based_action(obs)
    except Exception as err:
        _log_error(logger, f"rule_based_action failed, use default action: {err}")
        return [0, 0, Config.MIN_GREEN_DURATION]


def _save_latest_model(agent, logger):
    try:
        agent.save_model(id="latest")
        return True
    except Exception as err:
        _log_error(logger, f"save latest model failed: {err}")
        return False


def _load_latest_model(agent, logger):
    try:
        agent.load_model(id="latest")
        return True
    except Exception as err:
        _log_error(logger, f"load latest model failed: {err}")
        return False


def _send_sample_data(agent, sample_data, logger):
    if not sample_data:
        return False
    try:
        agent.send_sample_data(sample_data)
        return True
    except Exception as err:
        _log_error(logger, f"send sample data failed: {err}")
        return False


def _get_training_metrics(logger):
    try:
        metrics = get_training_metrics()
    except Exception as err:
        _log_error(logger, f"get training metrics failed: {err}")
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _log_info(logger, message):
    if not logger:
        return
    try:
        logger.info(message)
    except Exception:
        pass


def _log_error(logger, message):
    if not logger:
        return
    try:
        logger.error(message)
    except Exception:
        pass


def _put_monitor_data(monitor, monitor_data, logger=None):
    if not monitor:
        return False
    try:
        monitor.put_data({os.getpid(): monitor_data})
        return True
    except Exception as err:
        _log_error(logger, f"monitor put_data failed: {err}")
        return False
