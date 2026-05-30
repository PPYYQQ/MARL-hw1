#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch


def _configure_torch_threads():
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_configure_torch_threads()

from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.feature.definition import *
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.algorithm = Algorithm(logger)
        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        pass

    def exploit(self, list_obs_data):
        pass

    def learn(self, list_sample_data):
        pass

    def save_model(self, path=None, id="1"):
        pass

    def load_model(self, path=None, id="1"):
        pass

    def observation_process(self, raw_obs, traffic_handler, extra_info=None):
        pass

    def action_process(self, act_data):
        pass
