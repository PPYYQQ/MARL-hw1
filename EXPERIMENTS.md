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
| E01 | 2026-06-01 | 平台包为旧模板，非当前 `main` | 平台默认/未记录 | 1h | 任务 ID `194038` | 约 `1200` 训练 score 快照，非正式评估 | 训练链路跑通，但旧包 reward 固定为 0，不能代表当前代码效果 |

## 实验记录

### E01 - Target DQN 一小时平台训练

- 状态：已完成，平台任务自动释放
- 平台任务：
  - 任务名：`dqn`
  - 任务 ID：`194038`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：平台截图未显示；下载的 `dqn1/code-intelligent_traffic_lights-IDE-73.1.1.zip` 已确认为旧模板代码，不是当前 `main`。
- 代码包：`dqn1/code-intelligent_traffic_lights-IDE-73.1.1.zip`。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-01 16:47:23 到 2026-06-01 17:48:03。
- 训练时长：1h。
- 模型 ID：未记录。
- 截图证据：
  - `dqn1/微信图片_20260608185906_183_14.png`
  - `dqn1/微信图片_20260608185916_184_14.png`
  - `dqn1/微信图片_20260608185922_185_14.png`
- 代码包复核：
  - `agent_target_dqn/conf/conf.py` 中 `DIM_OF_OBSERVATION = 560`、`NUMB_HEAD = 2`，仍是旧双头动作结构；当前主线为 638 维观测和 80 维联合动作 Q head。
  - `agent_target_dqn/feature/definition.py` 中 `reward_shaping()` 末尾仍直接 `return 0, 0`，可解释平台 reward 图基本为 0。
  - `agent_target_dqn/algorithm/algorithm.py` 中 `self.target_model = self.model`，目标网络仍别名在线网络，未使用独立 target network。
  - `agent_target_dqn/workflow/train_workflow.py` 仍直接读取 `env_obs["observation"]`、`obs["legal_action"][0]` 和固定二元 step 返回，缺少当前主线的大量平台返回容错。
  - 该平台包缺少当前主线已完成的字段 alias、reward、sample 归一化、checkpoint 容错、score 监控和离线测试体系。
- 关键监控：
  - `train_global_step`：约 `118`。
  - `predict_succ_cnt`：约 `4300+`，预测链路正常。
  - `episode_cnt`：约 `52`。
  - `load_model_succ_cnt`：约 `52`。
  - `sample_receive_cnt`：约 `4000+`。
  - `sample_production_and_consumption_ratio`：约 `7`，样本生产明显高于消费，后续长训需关注 learner 消费能力。
  - `reward`：平台图中基本为 `0`，说明当前 reward 监控或 reward 信号仍需重点排查。
  - `value_loss`：从约 `1.1` 上升到约 `2.0` 后逐步降到接近 `0`，未见 NaN/爆炸。
  - `q_value`：从 `0` 上升到约 `6`，末段约 `5.5`，数值未爆炸。
- 平台评分：
  - 总分/score：约 `1400` 开始，最低约 `1000`，末段约 `1200`，没有形成稳定上升趋势。
  - 平均车辆延误：约 `11-24` 波动，末段约 `16`。
  - 平均车辆等待时间：约 `10-22` 波动，末段约 `15`。
  - 平均排队长度：约 `6-10` 波动，末段约 `8.5`。
  - 平均信号变化频率/惩罚：约 `5-16` 波动，末段约 `6`。
- 错误日志：截图未提供训练日志，未看到崩溃信号。
- 结论：
  - 这是一次有效的 smoke/短训：环境交互、预测、样本接收、模型加载和 learner 更新都在运行。
  - 训练效果不能代表当前代码：平台下载包是旧模板，`reward_shaping()` 固定返回 0，且 Target-DQN 目标网络和动作结构都落后于当前主线。
  - reward 图基本为 0 的主因已经确认是旧包代码问题，而不是当前最新 `main` 的直接证据。
- 下一步：
  - 用当前最新 `main` 重新打包后跑 30-60 分钟短训，对比 reward 是否仍为 0。
  - 新包上传前确认平台包内 `agent_target_dqn/feature/definition.py` 不再包含 `return 0, 0`，`conf.py` 中 `DIM_OF_OBSERVATION = 638`，`NUMB_HEAD = 1`。
  - 若最新包 reward 仍为 0，再在平台训练日志中打印或上报 `phase_reward`、`duration_reward`、`data_length` 和若干 observation 字段命中情况。
  - 若 reward 正常但 score 仍不升，再调 reward 权重、学习率或样本生产/消费配置。

## 实验模板

### EXX - 简短名称

- 状态：
- Commit：
- 代码包：
- 环境配置：
- 训练命令：
- 训练时长：
- 模型 ID：
- 关键监控：
- 平台评分：
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
