#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = 630
    DIM_OF_ACTION_PHASE = 4
    DIM_OF_ACTION_DURATION = 20
    DIM_SUB_ACTION_MASK = 24

    SOFTMAX = False

    # Algorithm Config
    # 算法的配置
    GAMMA = 0.9
    EPSILON = 0.1

    LR = 5e-4

    START_EPSILON_GREEDY = 1.0
    END_EPSILON_GREEDY = 0.05
    EPSILON_DECAY = 0.999
    LAMBDA = 0.75
    NUMB_HEAD = 2
    TARGET_UPDATE_FREQ = 500

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    PHASE_FEATURE_DIM = 8
    PHASE_AGE_FEATURE_DIM = 4
    TRAFFIC_FEATURE_DIM = 8
    TRAFFIC_TREND_FEATURE_DIM = 8
    LANE_STAT_FEATURE_DIM = 42
    PHASE_AGE_SCALE = 120.0
    FAIRNESS_BONUS_SCALE = 0.2
    LANE_COUNT_SCALE = 20.0
    TRAFFIC_PRESSURE_SCALE = 50.0
    TRAFFIC_COUNT_SCALE = 100.0
    TRAFFIC_TIME_SCALE = 120.0
    DEFAULT_MAX_SPEED = 17.0
    WAITING_SPEED_THRESHOLD = 0.1
    MIN_GREEN_DURATION = 8
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
