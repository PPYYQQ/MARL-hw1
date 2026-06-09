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
    DIM_OF_ACTION_PHASE = 4
    DIM_OF_ACTION_DURATION = 20
    DIM_OF_ACTION = DIM_OF_ACTION_PHASE * DIM_OF_ACTION_DURATION
    DIM_SUB_ACTION_MASK = 24

    SOFTMAX = False

    # Algorithm Config
    # 算法的配置
    GAMMA = 0.99
    EPSILON = 0.1

    LR = 3e-4

    START_EPSILON_GREEDY = 1.0
    END_EPSILON_GREEDY = 0.2
    EPSILON_DECAY = 0.995
    LAMBDA = 0.95
    NUMB_HEAD = 1
    TARGET_UPDATE_FREQ = 10

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
    PHASE_AGE_SCALE = 90.0
    FAIRNESS_BONUS_SCALE = 0.5
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
    DURATION_STEP = (MAX_GREEN_DURATION - MIN_GREEN_DURATION) / max(DIM_OF_ACTION_DURATION - 1, 1)

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
        duration_index = round(max(0.0, min(duration_index, cls.DIM_OF_ACTION_DURATION - 1)))
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
        return int(max(0, min(duration_index, cls.DIM_OF_ACTION_DURATION - 1)))
