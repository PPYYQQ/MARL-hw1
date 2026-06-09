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
| E02 | 2026-06-08 | 上传包未随结果提供；按当前 `main` `dd1fbcc` 记录 | 平台默认/未记录 | 1h | 任务 ID `206699` | 约 `750-780` 训练 score 快照，非正式评估 | 最新包训练链路跑通且 reward 非零，但 score 明显偏低，平均信号变化惩罚为 0，疑似策略过少切相或过度保守 |
| E03 | 2026-06-09 | E03 调参包；按当前 `main` `7d10a9a` 记录 | 平台默认/未记录 | 2h | 任务 ID `206775` | 约 `740-770` 训练 score 快照，非正式评估 | PPO-style 参数调优让 learner step 增加，但 score、延误、等待和排队基本未改善，下一步应看真实动作分布而不是继续只调标量超参 |

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

### E02 - 更新包 Target DQN 一小时平台训练

- 状态：已完成，平台任务自动释放。
- 平台任务：
  - 任务名：`target dqn`
  - 任务 ID：`206699`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：`dqn2/` 只包含监控截图，没有平台下载代码包；按上传前仓库最新 `main` `dd1fbcc` 记录，但不能像 E01 一样逐文件复核平台包内容。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-08 21:41:46 到 2026-06-08 22:42:25。
- 训练时长：1h。
- 模型 ID：截图未显示模型 ID；平台任务 ID 为 `206699`。
- 截图证据：
  - `dqn2/截屏2026-06-08 23.53.17.png`
  - `dqn2/截屏2026-06-08 23.53.25.png`
  - `dqn2/截屏2026-06-08 23.53.30.png`
- 关键监控：
  - `train_global_step`：约 `40`。
  - `predict_succ_cnt`：约 `1650-1700`，预测链路正常。
  - `episode_cnt`：约 `56`。
  - `load_model_succ_cnt`：约 `56`。
  - `sample_receive_cnt`：约 `1600`。
  - `sample_production_and_consumption_ratio`：末段约 `6.3`，样本生产仍高于消费，但没有失控。
  - `reward`：约 `-2.8` 到 `-3.0`，截图悬浮点为 2026-06-08 22:38:06 的 `-2.832675`；这说明 E01 的固定零 reward 问题在本次结果中已不再出现。
  - `value_loss`：约从 `2.1` 逐步下降到 `0.3`，未见 NaN/爆炸。
  - `q_value`：从接近 `0` 下降到约 `-2.5`，与负 reward 方向一致。
- 平台评分：
  - 总分/score：约 `750-780`，整体较平，未形成提升趋势。
  - 平均车辆延误：约 `55-58`，明显高于 E01 截图区间。
  - 平均车辆等待时间：约 `26-28`。
  - 平均排队长度：约 `9-10`。
  - 平均信号变化频率/惩罚：全程接近 `0`。
- 与 E01 对比：
  - E01 分数更高但已确认是旧模板包，reward 固定为 0，因此不能作为当前代码的有效成绩基线。
  - E02 更能代表当前主线：reward 已非零、loss 在下降、learner 在更新。
  - E02 的核心问题不是训练崩溃，而是策略质量差；信号变化惩罚为 0 与高延误同时出现，优先怀疑相位切换过少、动作分布塌缩或 reward/动作时长权重过度保守。
- 错误日志：截图未提供训练日志，未看到崩溃信号。
- 结论：
  - 最新包平台训练链路已经跑通：预测、采样、加载模型、学习更新、非零 reward 监控都正常出现。
  - 当前 1h 训练结果不能直接用于最终提交，score 约 `750-780`，车辆延误和等待偏高。
  - 下一轮不建议盲目加长训练；应先确认动作分布，尤其是四个 phase 的选择次数、平均 duration 和实际 phase switch 次数。
- 下一步：
  - 从平台下载 E02 代码包或模型产物，确认上传包确实包含 `DIM_OF_OBSERVATION = 638`、`NUMB_HEAD = 1` 和非零 `reward_shaping()`。
  - 在 workflow 监控中加入动作分布指标，例如 `phase_0_cnt` 到 `phase_3_cnt`、`avg_duration`、`phase_switch_cnt`。
  - 如果动作确实长期不切相，优先调低切换惩罚权重、增强高压相位公平性奖励，并检查 `duration` 映射是否导致绿灯过长。
  - 如果动作分布正常但 score 仍低，再调 reward 权重、学习率和样本消费配置。

### E03 - PPO-style 调参后两小时平台训练

- 状态：已完成，平台任务自动释放。
- 平台任务：
  - 任务名：`target_dqn3`
  - 任务 ID：`206775`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：按上传前仓库最新 `main` `7d10a9a` 记录；`dqn3/` 只包含监控截图，没有平台下载代码包。
- 关键变更：E03 参数基线使用 `GAMMA=0.99`、`LR=3e-4`、`EPSILON_DECAY=0.97`、`END_EPSILON_GREEDY=0.1`、`TARGET_UPDATE_FREQ=20`。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-09 00:51:34 到 2026-06-09 02:52:14。
- 训练时长：2h。
- 模型 ID：截图未显示模型 ID；平台任务 ID 为 `206775`。
- 截图证据：
  - `dqn3/截屏2026-06-09 09.58.49.png`
  - `dqn3/截屏2026-06-09 09.58.55.png`
  - `dqn3/截屏2026-06-09 09.58.57.png`
- 关键监控：
  - `train_global_step`：约 `130`，相比 E02 的约 `40` 明显增加。
  - `predict_succ_cnt`：约 `4600`。
  - `episode_cnt`：约 `100`。
  - `load_model_succ_cnt`：约 `100`。
  - `sample_receive_cnt`：约 `4500`。
  - `sample_production_and_consumption_ratio`：末段约 `7.3`。
  - `reward`：前半段约 `-2.3` 到 `-2.5`，后半段下降到约 `-3.1` 到 `-3.2`。
  - `value_loss`：从约 `1.8` 下降到 `0.1-0.2`，后段回升到约 `0.45-0.6`，未见 NaN/爆炸。
  - `q_value`：从接近 `0` 逐步下降到约 `-3.5`。
- 平台评分：
  - 总分/score：约 `740-770`，整体较平，未形成提升趋势。
  - 平均车辆延误：约 `55-58`。
  - 平均车辆等待时间：约 `27-30`。
  - 平均排队长度：约 `9-10`。
  - 平均信号变化频率/惩罚：全程接近 `0`。
- 与 E02 对比：
  - E03 的 learner 更新次数更多，说明 `EPSILON_DECAY` 和 `TARGET_UPDATE_FREQ` 调整没有破坏训练链路。
  - score、平均延误、等待和排队与 E02 基本同级，未看到有效改善。
  - E03 的 reward 后半段更负，说明当前策略或 reward 目标仍没有导向平台评分改善。
  - 平台的信号变化惩罚为 `0` 只能说明没有过快切灯惩罚；由于动作最短 duration 已限制为 8 秒，它不能单独证明策略完全不切相。
- 结论：
  - 单纯调学习率、折扣、epsilon 和 target sync 不足以提升该任务成绩。
  - 下一轮不应继续只做标量超参搜索，应先补动作分布监控或下载平台日志，确认相位选择、duration 分布和真实切相次数。
- 下一步：
  - 在 workflow 监控中加入 `phase_0_cnt` 到 `phase_3_cnt`、`avg_duration`、`min_duration`、`max_duration`、`phase_switch_cnt` 和 `same_phase_ratio`。
  - 如果相位分布塌缩，优先修 reward 的相位公平性/压力项；如果 duration 长期偏大或偏小，再调 duration reward。
  - 如果动作分布正常但平台指标仍差，优先调整 reward 权重，让平均延误和等待惩罚更贴近平台 score。

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
