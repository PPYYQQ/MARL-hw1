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


ENV_SCORE_ALIASES = {
    "env_score": ("score", "total_score", "total", "final_score"),
    "avg_delay": ("avg_delay", "average_delay", "avg_junction_delay", "junction_delay", "delay"),
    "avg_queue_length": (
        "avg_queue_length",
        "average_queue_length",
        "avg_queue",
        "queue_length",
        "queue",
    ),
    "avg_waiting_time": (
        "avg_waiting_time",
        "average_waiting_time",
        "avg_wait_time",
        "waiting_time",
    ),
    "switch_penalty": (
        "switch_penalty",
        "phase_change_penalty",
        "signal_switch_penalty",
        "signal_change_penalty",
        "signal_switch_count",
        "phase_change_count",
        "switch_count",
    ),
}


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
    monitor_data.update(_default_env_metric_snapshot())
    last_report_monitor_time = time.time()

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = _read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        _log_error(logger, "usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        epoch_phase_rew = 0
        epoch_duration_rew = 0
        env_metric_snapshot = _default_env_metric_snapshot()

        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, env_metric_snapshot):
            batch_length, batch_phase_rew, batch_duration_rew = _sample_batch_stats(g_data, logger)
            data_length += batch_length
            epoch_phase_rew += batch_phase_rew
            epoch_duration_rew += batch_duration_rew

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
            monitor_data.update(env_metric_snapshot)
            if _put_monitor_data(monitor, monitor_data, logger):
                last_report_monitor_time = now

        _log_info(
            logger,
            f"Avg Step Reward: {avg_step_reward}, Avg Phase Reward: {avg_phase_reward}, "
            f"Avg Duration Reward: {avg_duration_reward}, Env Score: {env_metric_snapshot['env_score']}, "
            f"Avg Delay: {env_metric_snapshot['avg_delay']}, Avg Queue Length: "
            f"{env_metric_snapshot['avg_queue_length']}, Avg Waiting Time: "
            f"{env_metric_snapshot['avg_waiting_time']}, Switch Penalty: "
            f"{env_metric_snapshot['switch_penalty']}, Epoch: {epoch}, Data Length: {data_length}",
        )


def run_episodes(n_episode, env, agent, usr_conf, logger, env_metric_snapshot=None):
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
            env_obs = _reset_env(env, usr_conf, logger)
            if env_obs is None:
                continue
            # Disaster recovery
            # 容灾
            if _handle_disaster_recovery(env_obs, logger):
                break

            if not _reset_agent(agent, env_obs, logger):
                continue
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
                        reward = _shape_reward(obs, last_predict_act, agent, logger)
                        collector[-1].rew = reward

                    # Feature processing
                    # 特征处理
                    obs_data = _process_observation(agent, obs, extra_info, logger)
                    # Agent makes a prediction to get the next frame's action
                    # Agent 进行推理, 获取下一帧的预测动作
                    act = _predict_action(agent, obs_data, obs, logger)
                    predict_cnt += 1
                else:
                    # No need to predict
                    # 不需要预测的情况
                    _update_traffic_info(agent, obs, extra_info, logger)
                    act = [None, None, None]
                act = _safe_action(act, need_to_predict, logger)

                # Interact with the environment, execute actions, get the next extra_info
                # 与环境交互, 执行动作, 获取下一步的状态, 如果遇到不需要预测的帧，则env.step直到得到需要预测的帧
                step_result = _step_env(env, act, logger)
                if step_result is None:
                    break
                env_reward, env_obs = step_result
                _update_env_metric_snapshot(env_metric_snapshot, env_reward, env_obs)
                # Disaster recovery
                # 容灾
                if _handle_disaster_recovery(env_obs, logger):
                    if len(collector) > 10:
                        samples = _process_samples(collector, logger)
                        if samples:
                            yield samples
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
                        obs=_obs_feature(obs_data),
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
                        reward = _shape_reward(_obs, last_predict_act, agent, logger)
                        collector[-1].done = 1
                        collector[-1].rew = reward
                        samples = _process_samples(collector, logger)
                        if samples:
                            yield samples
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


def _default_env_metric_snapshot():
    return {name: 0.0 for name in ENV_SCORE_ALIASES}


def _update_env_metric_snapshot(snapshot, env_reward, env_obs):
    if snapshot is None:
        return {}
    metrics = _env_score_metrics(env_reward, env_obs)
    snapshot.update(metrics)
    return metrics


def _env_score_metrics(env_reward, env_obs=None):
    metrics = {}
    sources = []
    _append_metric_sources(sources, env_reward)
    _append_metric_sources(sources, env_obs)
    _append_metric_sources(sources, _safe_extra_info(env_obs))

    direct_score = _optional_finite_float(env_reward)
    if direct_score is not None:
        metrics["env_score"] = direct_score

    for metric_name, aliases in ENV_SCORE_ALIASES.items():
        for source in sources:
            value = _source_metric_value(source, aliases)
            if value is not None:
                metrics[metric_name] = value
                break
    return metrics


def _append_metric_sources(sources, source):
    if source is None:
        return
    sources.append(source)
    for key in ("score", "score_info", "scoreInfo"):
        nested_source = _source_raw_value(source, key)
        if nested_source is not None and nested_source is not source:
            sources.append(nested_source)


def _source_metric_value(source, aliases):
    for alias in aliases:
        value = _source_raw_value(source, alias)
        finite_value = _optional_finite_float(value)
        if finite_value is not None:
            return finite_value
    return None


def _source_raw_value(source, key):
    if isinstance(source, dict):
        try:
            return source.get(key)
        except Exception:
            return None
    try:
        return getattr(source, key)
    except Exception:
        return None


def _sample_batch_stats(sample_data, logger):
    if sample_data is None:
        return 0, 0.0, 0.0
    try:
        data_length = len(sample_data)
    except Exception as err:
        _log_error(logger, f"sample batch length failed: {err}")
        return 0, 0.0, 0.0
    if data_length == 0:
        return 0, 0.0, 0.0

    phase_rew = 0.0
    duration_rew = 0.0
    try:
        for data in sample_data:
            try:
                reward = getattr(data, "rew", None)
            except Exception as err:
                _log_error(logger, f"sample reward read failed: {err}")
                reward = None
            sample_phase_rew, sample_duration_rew = _reward_components(reward)
            phase_rew += sample_phase_rew
            duration_rew += sample_duration_rew
    except Exception as err:
        _log_error(logger, f"sample batch iteration failed: {err}")
    return data_length, phase_rew, duration_rew


def _shape_reward(obs, act, agent, logger):
    try:
        reward = reward_shaping(obs, act, agent)
    except Exception as err:
        _log_error(logger, f"reward shaping failed: {err}")
        return 0.0, 0.0
    return _reward_components(reward)


def _process_observation(agent, obs, extra_info, logger):
    try:
        return agent.observation_process(obs, extra_info)
    except Exception as err:
        _log_error(logger, f"observation process failed: {err}")
        return None


def _update_traffic_info(agent, obs, extra_info, logger):
    try:
        agent.preprocess.update_traffic_info(obs, extra_info)
        return True
    except Exception as err:
        _log_error(logger, f"traffic info update failed: {err}")
        return False


def _obs_feature(obs_data):
    feature = getattr(obs_data, "feature", None)
    if feature is None:
        return [0.0] * Config.DIM_OF_OBSERVATION
    return feature


def _process_samples(collector, logger):
    if not collector:
        return []
    try:
        samples = sample_process(collector)
    except Exception as err:
        _log_error(logger, f"sample process failed: {err}")
        return []
    return samples if isinstance(samples, list) else []


def _read_usr_conf(path, logger):
    try:
        usr_conf = read_usr_conf(path, logger)
    except Exception as err:
        _log_error(logger, f"read usr conf failed: {err}")
        return None
    return usr_conf if isinstance(usr_conf, dict) else None


def _handle_disaster_recovery(env_obs, logger):
    try:
        return bool(handle_disaster_recovery(env_obs, logger))
    except Exception as err:
        _log_error(logger, f"handle disaster recovery failed: {err}")
        return False


def _reset_agent(agent, env_obs, logger):
    try:
        agent.reset(env_obs)
        return True
    except Exception as err:
        _log_error(logger, f"agent reset failed: {err}")
        return False


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _optional_finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _reset_env(env, usr_conf, logger):
    try:
        return _normalize_reset_result(env.reset(usr_conf=usr_conf))
    except Exception as err:
        _log_error(logger, f"env reset failed: {err}")
        return None


def _step_env(env, act, logger):
    try:
        return _normalize_step_result(env.step(act))
    except Exception as err:
        _log_error(logger, f"env step failed: {err}")
        return None


def _normalize_reset_result(reset_result):
    if isinstance(reset_result, dict):
        return reset_result
    if isinstance(reset_result, tuple) and len(reset_result) >= 2:
        return {
            "observation": reset_result[0],
            "extra_info": reset_result[1],
        }
    if _is_record(reset_result):
        return reset_result
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
        if len(step_result) == 5:
            observation, reward, terminated, truncated, extra_info = step_result
            return reward, {
                "frame_no": _safe_frame_no({"extra_info": extra_info}),
                "observation": observation,
                "terminated": terminated,
                "truncated": truncated,
                "extra_info": extra_info if _is_record(extra_info) else {},
            }
        if len(step_result) == 4:
            observation, reward, done, extra_info = step_result
            return reward, {
                "frame_no": _safe_frame_no({"extra_info": extra_info}),
                "observation": observation,
                "terminated": done,
                "truncated": False,
                "extra_info": extra_info if _is_record(extra_info) else {},
            }
        if len(step_result) >= 2:
            env_reward, env_obs = step_result[:2]
            return env_reward, env_obs if _is_record(env_obs) else {}
    if isinstance(step_result, dict):
        return _normalize_step_record_result(step_result)
    if _is_record(step_result):
        return _normalize_step_record_result(step_result)
    return 0.0, {}


def _normalize_step_record_result(step_result):
    if _looks_like_observation(step_result) or not _looks_like_step_envelope(step_result):
        return 0.0, step_result

    observation = _first_env_value(step_result, ("observation", "obs", "_obs"), {})
    extra_info = _first_env_value(step_result, ("extra_info", "_state", "state", "info"), {})
    reward = _first_env_value(step_result, ("reward", "score", "env_reward"), 0.0)
    terminated = _first_env_value(step_result, ("terminated", "done"), False)
    truncated = _safe_env_value(step_result, "truncated", False)
    frame_no = _safe_frame_no(step_result)

    return reward, {
        "frame_no": frame_no,
        "observation": observation if _is_record(observation) else {},
        "terminated": terminated,
        "truncated": truncated,
        "extra_info": extra_info if _is_record(extra_info) else {},
    }


def _first_env_value(env_obs, keys, default=None):
    for key in keys:
        value = _safe_env_value(env_obs, key, None)
        if value is not None:
            return value
    return default


def _safe_env_value(env_obs, key, default):
    if isinstance(env_obs, dict):
        try:
            return env_obs.get(key, default)
        except Exception:
            return default
    if env_obs is None:
        return default
    try:
        return getattr(env_obs, key, default)
    except Exception:
        return default


def _is_record(value):
    return value is not None and not isinstance(value, (str, bytes, bool, int, float, complex))


def _looks_like_observation(value):
    return _is_record(value) and (
        _safe_env_value(value, "frame_state", None) is not None
        or _safe_env_value(value, "legal_action", None) is not None
    )


def _looks_like_step_envelope(value):
    return _is_record(value) and any(
        _safe_env_value(value, key, None) is not None
        for key in (
            "observation",
            "obs",
            "_obs",
            "reward",
            "score",
            "env_reward",
            "terminated",
            "truncated",
            "done",
            "extra_info",
            "_state",
            "state",
            "info",
        )
    )


def _safe_observation(env_obs):
    for key in ("observation", "obs", "_obs"):
        observation = _first_env_value(env_obs, (key,), None)
        if _is_record(observation):
            return observation
    if _looks_like_observation(env_obs):
        return env_obs
    return {}


def _safe_extra_info(env_obs):
    for key in ("extra_info", "_state", "state", "info"):
        extra_info = _first_env_value(env_obs, (key,), None)
        if _is_record(extra_info):
            return extra_info
    return {}


def _safe_frame_no(env_obs):
    frame_no = _first_env_value(env_obs, ("frame_no", "frameNo"), None)
    if frame_no is None:
        frame_no = _first_env_value(_safe_extra_info(env_obs), ("frame_no", "frameNo"), 0)
    return int(_finite_float(frame_no))


def _safe_done_flag(env_obs, key):
    value = _safe_env_value(env_obs, key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n", ""):
            return False
        return False
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(value):
        return False
    return value != 0.0


def _safe_legal_action(obs):
    return _safe_env_value(obs, "legal_action", None)


def _need_to_predict(obs):
    legal_action = _safe_env_value(obs, "legal_action", None)
    phase_mask = normalize_phase_legal_action(legal_action, Config.DIM_OF_ACTION_PHASE)
    return any(phase_mask)


def _should_log_progress(predict_cnt, done, need_to_predict):
    return done or (need_to_predict and predict_cnt > 0 and predict_cnt % 20 == 0)


def _action_scalar(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError(f"invalid action scalar {value}") from err
    if not math.isfinite(value):
        raise ValueError(f"non-finite action scalar {value}")
    return value


def _safe_action(act, need_to_predict, logger):
    if not need_to_predict:
        return [None, None, None]
    try:
        if len(act) < 3:
            raise ValueError("action should contain junction, phase and duration")
        phase_index = int(_action_scalar(act[1]))
        duration = int(_action_scalar(act[2]))
    except Exception as err:
        _log_error(logger, f"invalid action, use default action: {err}")
        return [0, 0, Config.MIN_GREEN_DURATION]

    max_duration = Config.max_action_duration()
    phase_index = max(0, min(phase_index, Config.DIM_OF_ACTION_PHASE - 1))
    duration = max(Config.MIN_GREEN_DURATION, min(duration, max_duration))
    return [0, phase_index, duration]


def _predict_action(agent, obs_data, obs, logger):
    if obs_data is None:
        _log_error(logger, "observation process returned empty, fallback to rule_based_action")
        try:
            return agent.rule_based_action(obs)
        except Exception as err:
            _log_error(logger, f"rule_based_action failed, use default action: {err}")
            return [0, 0, Config.MIN_GREEN_DURATION]

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
