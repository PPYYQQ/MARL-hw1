#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能交通信号灯调度")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="Q值估计",
            name_en="q_value",
            type="line",
        )
        .add_metric(
            metrics_name="q_value",
            expr="avg(q_value{})",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="动作指标",
            group_name_en="action",
        )
        .add_panel(
            name="相位选择次数",
            name_en="phase_counts",
            type="line",
        )
        .add_metric(
            metrics_name="phase_0_cnt",
            expr="avg(phase_0_cnt{})",
        )
        .add_metric(
            metrics_name="phase_1_cnt",
            expr="avg(phase_1_cnt{})",
        )
        .add_metric(
            metrics_name="phase_2_cnt",
            expr="avg(phase_2_cnt{})",
        )
        .add_metric(
            metrics_name="phase_3_cnt",
            expr="avg(phase_3_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="平均动作时长",
            name_en="avg_duration",
            type="line",
        )
        .add_metric(
            metrics_name="avg_duration",
            expr="avg(avg_duration{})",
        )
        .end_panel()
        .add_panel(
            name="切相次数",
            name_en="phase_switch_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="phase_switch_cnt",
            expr="avg(phase_switch_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="同相位比例",
            name_en="same_phase_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="same_phase_ratio",
            expr="avg(same_phase_ratio{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
