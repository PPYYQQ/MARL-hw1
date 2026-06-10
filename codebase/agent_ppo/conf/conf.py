#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math


class Config:

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = 638
    DIM_OF_ACTION_PHASE_1 = 4
    DIM_OF_ACTION_DURATION_1 = 20
    DIM_SUB_ACTION_MASK = 24

    # Number of output heads for policy, 2 heads if including phase and time
    # 策略输出几个头，如果包含相位和时间，则为2个头
    NUMB_HEAD = 2

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    PHASE_FEATURE_DIM = 8
    PHASE_AGE_FEATURE_DIM = 4
    TRAFFIC_FEATURE_DIM = 8
    TRAFFIC_TREND_FEATURE_DIM = 8
    TRAFFIC_HISTORY_FEATURE_DIM = 8
    TRAFFIC_HISTORY_SIZE = 4
    LANE_STAT_FEATURE_DIM = 42
    PHASE_AGE_SCALE = 120.0
    FAIRNESS_BONUS_SCALE = 0.2
    LANE_COUNT_SCALE = 20.0
    TRAFFIC_PRESSURE_SCALE = 50.0
    TRAFFIC_COUNT_SCALE = 100.0
    TRAFFIC_TIME_SCALE = 120.0
    REWARD_DELAY_CAP = 300.0
    REWARD_CLIP = 5.0
    DEFAULT_MAX_SPEED = 17.0
    WAITING_SPEED_THRESHOLD = 0.1
    MIN_GREEN_DURATION = 8
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
    DURATION_STEP = (MAX_GREEN_DURATION - MIN_GREEN_DURATION) / max(DIM_OF_ACTION_DURATION_1 - 1, 1)

    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.01
    LOG_EPSILON = 1e-6

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    LABEL_SIZE_LIST = [DIM_OF_ACTION_PHASE_1, DIM_OF_ACTION_DURATION_1]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    IS_REINFORCE_TASK_LIST = [
        True,
    ] * NUMB_HEAD

    EVAL_FREQ = 5
    PPO_FRAGMENT_SIZE = 32
    GAMMA = 0.99
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    VALUE_COEF = 1.0
    POLICY_COEF = 1.0
    ENTROPY_COEF = BETA_START

    @classmethod
    def max_action_duration(cls):
        return cls.MAX_GREEN_DURATION

    @classmethod
    def duration_index_to_seconds(cls, duration_index):
        try:
            duration_index = float(duration_index)
        except (TypeError, ValueError, OverflowError):
            duration_index = 0.0
        if not math.isfinite(duration_index):
            duration_index = 0.0
        duration_index = round(max(0.0, min(duration_index, cls.DIM_OF_ACTION_DURATION_1 - 1)))
        duration = cls.MIN_GREEN_DURATION + duration_index * cls.DURATION_STEP
        return int(round(max(cls.MIN_GREEN_DURATION, min(duration, cls.MAX_GREEN_DURATION))))

    @classmethod
    def duration_seconds_to_index(cls, duration_seconds):
        try:
            duration_seconds = float(duration_seconds)
        except (TypeError, ValueError, OverflowError):
            duration_seconds = cls.MIN_GREEN_DURATION
        if not math.isfinite(duration_seconds):
            duration_seconds = cls.MIN_GREEN_DURATION
        duration_seconds = max(cls.MIN_GREEN_DURATION, min(duration_seconds, cls.MAX_GREEN_DURATION))
        duration_index = round((duration_seconds - cls.MIN_GREEN_DURATION) / cls.DURATION_STEP)
        return int(max(0, min(duration_index, cls.DIM_OF_ACTION_DURATION_1 - 1)))
