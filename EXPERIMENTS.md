# 实验台账

本文件用于记录每次平台训练、评估和调参结果。`PROGRESS.md` 记录开发步骤；本文件记录实验数据。

## 当前基线

- 主线算法：`target_dqn`
- 代码目录：`codebase/agent_target_dqn`
- 当前最新代码：以 GitHub `main` 分支为准
- 提交包命令：`./scripts/package_submission.sh`
- 平台入口：`codebase/train_test.py`

## 实验总览

| 编号 | 日期 | Commit | 环境配置 | 训练时长 | 模型 ID | 评估得分 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E00 | 待补 | 待补 | 简单环境 | 待补 | 待补 | 待补 | 首次平台 smoke |

## 实验模板

### E00 - 首次平台 smoke

- 状态：待运行
- Commit：
- 代码包：
- 环境配置：
  - `weather = 0`
  - `rush_hour = 0`
  - `speeding_cars_rate = 0`
  - `traffic_accidents.random_count = 0`
  - `traffic_control.random_count = 0`
- 训练命令：
- 训练时长：
- 模型 ID：
- 关键监控：
  - `reward`：
  - `phase_reward`：
  - `duration_reward`：
  - `env_score`：
  - `avg_delay`：
  - `avg_queue_length`：
  - `avg_waiting_time`：
  - `switch_penalty`：
  - `value_loss`：
  - `q_value`：
  - `target_q_value`：
  - `model_grad_norm`：
- 平台评分：
  - 总分：
  - 平均延误：
  - 平均排队长度：
  - 平均等待时间：
  - 信号切换惩罚：
- 错误日志：
- 结论：
- 下一步：

## 调参记录建议

- 每次只改变 1-2 个关键因素，避免无法归因。
- 优先顺序：
  1. 确认 `train_test.py` 通过。
  2. 确认简单环境短训练 reward/loss 不异常。
  3. 调整 reward 权重。
  4. 调整 `LR`、`EPSILON_DECAY`、`TARGET_UPDATE_FREQ`。
  5. 增加环境复杂度。
  6. 调整 duration 分桶、reward 权重或训练环境难度。
- 每次实验结束后，将结果同步回 `REPORT_DRAFT.md` 的平台实验表。
