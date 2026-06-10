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
import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import *
from agent_target_dqn.workflow.train_workflow import (
    _action_metric_snapshot,
    _default_action_metric_snapshot,
    _default_env_metric_snapshot,
    _get_training_metrics,
    _handle_disaster_recovery,
    _load_latest_model,
    _log_error,
    _log_info,
    _need_to_predict,
    _new_action_stats,
    _obs_feature,
    _put_monitor_data,
    _read_usr_conf,
    _reset_agent,
    _reset_env,
    _safe_action,
    _safe_done_flag,
    _safe_extra_info,
    _safe_frame_no,
    _safe_observation,
    _save_latest_model,
    _send_sample_data,
    _should_log_progress,
    _step_env,
    _update_action_stats,
    _update_env_metric_snapshot,
    _update_traffic_info,
)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    last_save_model_time = 0

    monitor_data = {
        "reward": 0,
        "data_length": 0,
    }
    monitor_data.update(_default_env_metric_snapshot())
    monitor_data.update(_default_action_metric_snapshot())
    last_report_monitor_time = time.time()

    usr_conf = _read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        _log_error(logger, "usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0.0
        env_metric_snapshot = _default_env_metric_snapshot()
        action_stats = _new_action_stats()

        data_length = 0
        for g_data in run_episodes(
            episode_num_every_epoch,
            env,
            agent,
            usr_conf,
            logger,
            env_metric_snapshot,
            action_stats,
        ):
            batch_length, batch_reward = _sample_batch_stats(g_data, logger)
            data_length += batch_length
            epoch_total_rew += batch_reward
            _send_sample_data(agent, list(g_data), logger)
            g_data.clear()

        avg_step_reward = round(epoch_total_rew / data_length, 4) if data_length else 0.0

        now = time.time()
        if now - last_save_model_time >= 1800:
            _save_latest_model(agent, logger)
            last_save_model_time = now

        if now - last_report_monitor_time > 60:
            monitor_data["reward"] = avg_step_reward
            monitor_data["data_length"] = data_length
            monitor_data.update(env_metric_snapshot)
            monitor_data.update(_action_metric_snapshot(action_stats))
            if _put_monitor_data(monitor, monitor_data, logger):
                last_report_monitor_time = now

        action_metric_snapshot = _action_metric_snapshot(action_stats)
        _log_info(
            logger,
            f"PPO Avg Step Reward: {avg_step_reward}, Env Score: {env_metric_snapshot['env_score']}, "
            f"Avg Delay: {env_metric_snapshot['avg_delay']}, Avg Queue Length: "
            f"{env_metric_snapshot['avg_queue_length']}, Avg Waiting Time: "
            f"{env_metric_snapshot['avg_waiting_time']}, Switch Penalty: "
            f"{env_metric_snapshot['switch_penalty']}, Action Count: {action_metric_snapshot['action_count']}, "
            f"Avg Duration: {action_metric_snapshot['avg_duration']}, Phase Switch Count: "
            f"{action_metric_snapshot['phase_switch_cnt']}, Same Phase Ratio: "
            f"{action_metric_snapshot['same_phase_ratio']}, Epoch: {epoch}, Data Length: {data_length}",
        )


def run_episodes(n_episode, env, agent, usr_conf, logger, env_metric_snapshot=None, action_stats=None):
    try:
        train_test_quick_stop = os.environ.get("is_train_test", "False").lower() == "true"
        for _ in range(n_episode):
            collector = []
            predict_cnt = 0

            training_metrics = _get_training_metrics(logger)
            if training_metrics:
                _log_info(logger, f"training_metrics is {training_metrics}")

            _load_latest_model(agent, logger)

            env_obs = _reset_env(env, usr_conf, logger)
            if env_obs is None:
                continue
            if _handle_disaster_recovery(env_obs, logger):
                break
            if not _reset_agent(agent, env_obs, logger):
                continue

            obs = _safe_observation(env_obs)
            extra_info = _safe_extra_info(env_obs)
            last_predict_act = None
            done = False

            while not done:
                need_to_predict = _need_to_predict(obs)
                if need_to_predict:
                    if collector:
                        collector[-1].reward = _shape_reward(obs, last_predict_act, agent, logger)

                    obs_data = _process_observation(agent, obs, extra_info, logger)
                    act_data, act = _predict_ppo_action(agent, obs_data, obs, logger)
                    predict_cnt += 1
                else:
                    _update_traffic_info(agent, obs, extra_info, logger)
                    obs_data, act_data = None, None
                    act = [None, None, None]

                act = _safe_action(act, need_to_predict, logger)
                if need_to_predict:
                    _update_action_stats(action_stats, act)

                step_result = _step_env(env, act, logger)
                if step_result is None:
                    break
                env_reward, env_obs = step_result
                _update_env_metric_snapshot(env_metric_snapshot, env_reward, env_obs)

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
                done = terminated or truncated or (train_test_quick_stop and len(collector) > 1)

                if _should_log_progress(predict_cnt, done, need_to_predict):
                    _log_info(logger, f"current frame_no is {frame_no}, predict_cnt is {predict_cnt}")
                if truncated:
                    _log_info(logger, f"truncated is True, frame_no is {frame_no}, so this episode timeout")
                elif terminated:
                    _log_info(logger, f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

                if need_to_predict:
                    collector.append(_build_sample(obs_data, act_data))
                    if len(collector) > Config.PPO_FRAGMENT_SIZE:
                        fragment = collector[:-1]
                        fragment[-1].next_value = collector[-1].value
                        samples = _process_samples(fragment, logger)
                        if samples:
                            yield samples
                        collector = collector[-1:]

                obs = _obs
                if need_to_predict:
                    last_predict_act = act

                if done:
                    if len(collector) > 1:
                        collector[-1].done = 1
                        collector[-1].reward = _shape_reward(_obs, last_predict_act, agent, logger)
                        samples = _process_samples(collector, logger)
                        if samples:
                            yield samples
                    break

    except Exception as err:
        _log_error(logger, f"run_episodes error: {err}")
        raise RuntimeError(f"run_episodes error: {err}") from err


def _process_observation(agent, obs, extra_info, logger):
    try:
        return agent.observation_process(obs, extra_info)
    except Exception as err:
        _log_error(logger, f"observation process failed: {err}")
        return None


def _predict_ppo_action(agent, obs_data, obs, logger):
    if obs_data is None:
        return _fallback_act_data(agent, obs, logger)

    try:
        act_data_list = agent.predict(list_obs_data=[obs_data])
        if act_data_list:
            act_data = act_data_list[0]
            return act_data, agent.action_process(act_data)
        _log_error(logger, "predict returned empty action, fallback to rule_based_action")
    except Exception as err:
        _log_error(logger, f"predict fallback to rule_based_action: {err}")
    return _fallback_act_data(agent, obs, logger)


def _fallback_act_data(agent, obs, logger):
    try:
        act = agent.rule_based_action(obs)
    except Exception as err:
        _log_error(logger, f"rule_based_action failed, use default action: {err}")
        act = [0, 0, Config.MIN_GREEN_DURATION]

    try:
        act = list(act)
    except Exception:
        act = [0, 0, Config.MIN_GREEN_DURATION]

    phase_index = _safe_phase_index(act[1] if len(act) > 1 else 0)
    duration_index = Config.duration_seconds_to_index(act[2] if len(act) > 2 else Config.MIN_GREEN_DURATION)
    action = [phase_index, duration_index]
    return (
        ActData(
            junction_id=0,
            action=action,
            d_action=action,
            prob=_uniform_prob(),
            value=np.array([0.0], dtype=np.float32),
        ),
        [0, phase_index, Config.duration_index_to_seconds(duration_index)],
    )


def _build_sample(obs_data, act_data):
    return SampleData(
        obs=np.array(_obs_feature(obs_data), dtype=np.float32),
        legal_action=np.array(getattr(obs_data, "legal_action", _uniform_legal_action()), dtype=np.float32),
        act=np.array(getattr(act_data, "action", [0, 0]), dtype=np.float32),
        reward=0.0,
        done=0,
        reward_sum=0.0,
        value=_first_float(getattr(act_data, "value", 0.0)),
        next_value=0.0,
        advantage=0.0,
        prob=np.array(getattr(act_data, "prob", _uniform_prob()), dtype=np.float32),
        sub_action=np.array(getattr(obs_data, "sub_action_mask", [1] * Config.NUMB_HEAD), dtype=np.float32),
        is_train=1,
    )


def _shape_reward(obs, act, agent, logger):
    try:
        reward = reward_shaping(obs, act, agent)
    except Exception as err:
        _log_error(logger, f"reward shaping failed: {err}")
        return 0.0
    return _safe_reward(reward)


def _process_samples(collector, logger):
    if not collector:
        return []
    try:
        samples = sample_process(collector)
    except Exception as err:
        _log_error(logger, f"sample process failed: {err}")
        return []
    return samples if isinstance(samples, list) else []


def _sample_batch_stats(sample_data, logger):
    if sample_data is None:
        return 0, 0.0
    try:
        data_length = len(sample_data)
    except Exception as err:
        _log_error(logger, f"sample batch length failed: {err}")
        return 0, 0.0
    total_reward = 0.0
    try:
        for data in sample_data:
            total_reward += _safe_reward(getattr(data, "reward", 0.0))
    except Exception as err:
        _log_error(logger, f"sample batch iteration failed: {err}")
    return data_length, total_reward


def _safe_phase_index(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        value = 0.0
    if not np.isfinite(value):
        value = 0.0
    return int(np.clip(round(value), 0, Config.DIM_OF_ACTION_PHASE_1 - 1))


def _first_float(value, default=0.0):
    try:
        values = np.asarray(value, dtype=np.float32).flatten()
    except Exception:
        values = np.asarray([], dtype=np.float32)
    if values.size == 0:
        return default
    return _safe_reward(values[0], default=default)


def _safe_reward(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(value):
        return default
    return float(value)


def _uniform_prob():
    phase_prob = [1.0 / Config.DIM_OF_ACTION_PHASE_1] * Config.DIM_OF_ACTION_PHASE_1
    duration_prob = [1.0 / Config.DIM_OF_ACTION_DURATION_1] * Config.DIM_OF_ACTION_DURATION_1
    return phase_prob + duration_prob


def _uniform_legal_action():
    return [1] * (Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1)
