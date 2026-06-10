#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import faulthandler
import os
import sys
import traceback

try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass

import torch


PPO_DIAG_PREFIX = "[PPO_DIAG]"


def _diag_log(logger, message):
    text = f"{PPO_DIAG_PREFIX} {message}"
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass
    if logger is None:
        return
    try:
        logger.info(text)
    except Exception:
        pass


def _configure_torch_threads():
    try:
        torch.set_num_threads(1)
        _diag_log(None, "torch.set_num_threads ok")
    except RuntimeError:
        _diag_log(None, "torch.set_num_threads skipped")
    try:
        torch.set_num_interop_threads(1)
        _diag_log(None, "torch.set_num_interop_threads ok")
    except RuntimeError:
        _diag_log(None, "torch.set_num_interop_threads skipped")


_configure_torch_threads()
_diag_log(None, "agent_ppo.agent module imported")

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
from agent_ppo.conf.conf import Config
from agent_ppo.algorithm.algorithm import Algorithm
from agent_target_dqn.agent import Agent as TargetFeatureAgent
from agent_target_dqn.agent import EXTRA_INFO_KEYS, _first_record_field
from agent_target_dqn.feature.preprocessor import FeatureProcess
from agent_target_dqn.feature.traffic_utils import normalize_phase_legal_action


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        _diag_log(logger, f"Agent.__init__ start agent_type={agent_type}, device={device}")
        try:
            torch.manual_seed(0)
            _diag_log(logger, "torch.manual_seed done")

            self.device = device
            self.model = Model(device)
            _diag_log(logger, "Model constructed")
            if self.device is not None:
                self.model = self.model.to(self.device)
                _diag_log(logger, f"Model moved to device={self.device}")
            else:
                _diag_log(logger, "Model stays on default device")

            initial_lr = Config.INIT_LEARNING_RATE_START
            parameters = list(self.model.parameters())
            parameter_count = sum(parameter.numel() for parameter in parameters)
            _diag_log(logger, f"Model parameter_count={parameter_count}, tensor_count={len(parameters)}")
            if not parameters:
                raise RuntimeError("PPO model has no registered parameters; check agent_ppo/model/model.py was uploaded")

            self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr)
            _diag_log(logger, f"Optimizer constructed lr={initial_lr}")
            self.label_size_list = Config.LABEL_SIZE_LIST
            self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
            self.logger = logger
            self.monitor = monitor
            self.preprocess = FeatureProcess(logger)
            _diag_log(logger, "FeatureProcess constructed")
            self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
            _diag_log(logger, "Algorithm constructed")
            super().__init__(agent_type, device, logger, monitor)
            _diag_log(logger, "BaseAgent initialized")
        except Exception as err:
            _diag_log(logger, f"Agent.__init__ failed: {type(err).__name__}: {err}")
            _diag_log(logger, traceback.format_exc())
            raise

    def reset(self, env_obs):
        self.preprocess.reset()

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        list_obs_data = self._obs_batch(list_obs_data)
        list_obs_data = [
            obs_data for obs_data in list_obs_data if self._obs_data_field(obs_data, "feature") is not None
        ]
        if not list_obs_data:
            return []

        feature = [self._obs_data_field(obs_data, "feature") for obs_data in list_obs_data]
        legal_action = [
            self._full_action_mask(self._obs_data_field(obs_data, "legal_action")) for obs_data in list_obs_data
        ]
        self.model.set_eval_mode()

        with torch.no_grad():
            output_list = self.model(feature, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.detach().cpu().numpy())

        logits, value = np_output[:2]

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(
                logits[i],
                np.array(legal_action[i], dtype=np.float32),
                use_stochastic=not exploit_flag,
            )
            list_act_data.append(
                ActData(
                    junction_id=0,
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=np.array([value[i].reshape(-1)[0]], dtype=np.float32),
                )
            )
        return list_act_data

    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    def exploit(self, observation):
        raw_obs = _first_record_field(observation, ("obs", "observation", "_obs"), observation)
        extra_info = _first_record_field(observation, EXTRA_INFO_KEYS, None)
        if raw_obs is None:
            raw_obs = {}
        try:
            obs_data = self.observation_process(raw_obs, extra_info)
            if not obs_data:
                return self._safe_rule_based_action(raw_obs)
            act_data = self.__predict_detail([obs_data], exploit_flag=True)
            if not act_data:
                return self._safe_rule_based_action(raw_obs)
            return self.action_process(act_data[0], False)
        except Exception as err:
            self._log_error(f"exploit fallback to rule_based_action: {err}")
            return self._safe_rule_based_action(raw_obs)

    def _safe_rule_based_action(self, raw_obs):
        try:
            return self.rule_based_action(raw_obs)
        except Exception as err:
            self._log_error(f"rule_based_action failed, use default action: {err}")
            return [0, 0, Config.MIN_GREEN_DURATION]

    def _obs_batch(self, list_obs_data):
        if list_obs_data is None:
            return []
        try:
            return list(list_obs_data)
        except Exception as err:
            self._log_error(f"predict observation batch failed: {err}")
            return []

    def _obs_data_field(self, obs_data, name, default=None):
        try:
            return getattr(obs_data, name, default)
        except Exception:
            return default

    def learn(self, list_sample_data):
        try:
            return self.algorithm.learn(list_sample_data)
        except Exception as err:
            self._log_error(f"learn failed: {err}")
            return None

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        if path is None:
            path = "agent_ppo/ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_tmp_path = f"{model_file_path}.tmp"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        try:
            torch.save(model_state_dict_cpu, model_tmp_path)
            os.replace(model_tmp_path, model_file_path)
        except Exception:
            if os.path.exists(model_tmp_path):
                try:
                    os.remove(model_tmp_path)
                except OSError:
                    pass
            raise

        self._log_info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        if path is None:
            path = "agent_ppo/ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if str(id) == "latest":
                self._log_info(f"skip load model, {model_file_path} does not exist yet")
                return
            raise FileNotFoundError(model_file_path)
        try:
            model_state = torch.load(model_file_path, map_location=self.model.device)
        except Exception as err:
            if str(id) == "latest":
                self._log_info(f"skip load model, unreadable checkpoint {model_file_path}: {err}")
                return
            raise
        if isinstance(model_state, dict) and "state_dict" in model_state:
            model_state = model_state["state_dict"]
        if not isinstance(model_state, dict):
            err = TypeError(f"checkpoint payload should be dict, got {type(model_state).__name__}")
            if str(id) == "latest":
                self._log_info(f"skip load model, invalid checkpoint {model_file_path}: {err}")
                return
            raise err
        try:
            self.model.load_state_dict(model_state)
        except (RuntimeError, TypeError, ValueError) as err:
            if str(id) == "latest":
                self._log_info(f"skip load model, incompatible checkpoint {model_file_path}: {err}")
                return
            raise

        self._log_info(f"load model {model_file_path} successfully")

    def _log_info(self, message):
        if not self.logger:
            return
        try:
            self.logger.info(message)
        except Exception:
            pass

    def _log_error(self, message):
        if not self.logger:
            return
        try:
            self.logger.error(message)
        except Exception:
            pass

    def observation_process(self, raw_obs, extra_info=None):
        """
        This function is an important function for feature processing, mainly responsible for:
            - Parsing raw data from proto data
            - Calculating features through raw data to obtain multiple feature vectors
            - Concatenation of features
            - Labeling of legal actions

        Args:
            - raw_obs: Raw feature data sent by battlesrv

        Returns:
            - ObsData: A variable containing observation, legal_action and sub_action_mask
        """
        """
            该函数是特征处理的重要函数, 主要负责：
                - 从 proto 数据中解析原始数据
                - 通过原始数据计算特征, 得到多个特征向量
                - 特征的拼接
                - 合法动作的标注

            参数：
                - raw_obs: battlesrv 发送的原始特征数据

            返回：
                - ObsData: 包含 observation, legal_action 与 sub_action_mask 的变量
        """

        target_obs_data = TargetFeatureAgent.observation_process(self, raw_obs, extra_info)
        return ObsData(
            feature=getattr(target_obs_data, "feature", [0.0] * Config.DIM_OF_OBSERVATION),
            legal_action=self._full_action_mask(getattr(target_obs_data, "legal_action", None)),
            sub_action_mask=[1] * Config.NUMB_HEAD,
        )

    def action_process(self, act_data, is_stochastic=True):
        junction_id = 0
        if is_stochastic:
            action = act_data.action
        else:
            action = act_data.d_action

        action_p = self._safe_action_index(action[0], Config.DIM_OF_ACTION_PHASE_1)
        action_d = Config.duration_index_to_seconds(
            self._safe_action_index(action[1], Config.DIM_OF_ACTION_DURATION_1)
        )

        return [junction_id, action_p, action_d]

    def _safe_action_index(self, value, action_dim):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        return int(np.clip(round(value), 0, action_dim - 1))

    def _sample_masked_action(self, logits, legal_action, use_stochastic=True):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []

        从预测的logits和合法动作中采样动作
        返回: 概率、随机动作和确定性动作（包含额外的[]）
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list)):
            # Count non-zero elements. If all actions are illegal, then True
            # 统计非0元素数量，若全部动作都非法，则True
            if np.count_nonzero(legal_actions[index]) == 0:  # np.sum(~np.isnan(probs)) == 0:
                probs = [
                    0,
                ] * self.label_size_list[index]
                sample_action = 0
                d_action = 0
            else:
                probs = self._legal_soft_max(logits_split[index], legal_actions[index])
                sample_action = self._legal_sample(probs, use_max=not use_stochastic)
                d_action = self._legal_sample(probs, use_max=True)
            action_list.append(sample_action)
            d_action_list.append(d_action)
            prob_list += list(probs)

        return prob_list, action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        # Large and small constants for numerical stability
        # 用于数值稳定性的大小常量
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        input_hidden = np.nan_to_num(input_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        legal_action = np.nan_to_num(legal_action, nan=0.0, posinf=0.0, neginf=0.0)
        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        # 最大值裁剪1不是必需的
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        prob_sum = np.sum(tmp, keepdims=True)
        if not np.all(np.isfinite(prob_sum)) or np.any(prob_sum <= 0.0):
            legal_count = max(int(np.count_nonzero(legal_action)), 1)
            probs = legal_action / legal_count
        else:
            probs = tmp / prob_sum
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array

        按概率采样，输入的probs应该是一维数组
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def _full_action_mask(self, legal_action):
        try:
            values = np.asarray(legal_action if legal_action is not None else [], dtype=np.float32).flatten()
        except Exception:
            values = np.asarray([], dtype=np.float32)
        if values.size == Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1:
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            phase_mask = [1 if value > 0 else 0 for value in values[: Config.DIM_OF_ACTION_PHASE_1]]
        else:
            phase_mask = normalize_phase_legal_action(legal_action, Config.DIM_OF_ACTION_PHASE_1)
        if not any(phase_mask):
            phase_mask = [1] * Config.DIM_OF_ACTION_PHASE_1
        return phase_mask + [1] * Config.DIM_OF_ACTION_DURATION_1

    def _phase_action_mask(self, legal_action):
        return np.asarray(self._full_action_mask(legal_action)[: Config.DIM_OF_ACTION_PHASE_1], dtype=bool)

    _sanitize_observation = TargetFeatureAgent._sanitize_observation
    rule_based_action = TargetFeatureAgent.rule_based_action
    _phase_legal_action = TargetFeatureAgent._phase_legal_action
    _phase_feature = TargetFeatureAgent._phase_feature
    _phase_age_feature = TargetFeatureAgent._phase_age_feature
    _current_phase_info = TargetFeatureAgent._current_phase_info
    _phase_record_value = TargetFeatureAgent._phase_record_value
    _traffic_feature = TargetFeatureAgent._traffic_feature
    _traffic_trend_feature = TargetFeatureAgent._traffic_trend_feature
    _traffic_history_feature = TargetFeatureAgent._traffic_history_feature
    _lane_stat_feature = TargetFeatureAgent._lane_stat_feature
