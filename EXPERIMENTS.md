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
| E04 | 2026-06-09 | E04 动作监控包；按修复前 `main` `de6a2b4` 记录 | 平台默认/未记录 | 截图时约 3min | 任务 ID `207146` | 约 `750-770` 训练 score 快照，非正式评估 | 动作监控确认相位塌缩：`phase_0_cnt≈30`、其他相位约 0、`phase_switch_cnt=0`；根因是通用 `legal_action` 被误当相位 mask |
| E05 | 2026-06-09 | E05 legal_action 修复包；按当前 `main` `1607528` 记录 | 平台默认/未记录 | 1h | 任务 ID `207778` | 约 `760-820` 训练 score 快照，非正式评估 | gate 修复有效：已不再 phase 0 锁死，但后半段转为 phase 2 偏置，`same_phase_ratio` 约 `0.75-0.85`，需要增强公平性和短训更新效率 |
| E06 | 2026-06-09 | E06 phase-bias 调参包；按当前 `main` `a0d8efe` 记录 | 平台默认/未记录 | 1h | 任务 ID `208300` | 末段约 `1100` 训练 score 快照，非正式评估 | E06 是当前最佳基线：`train_global_step≈87`，平均延误末段约 `20`、等待约 `10`，phase 2 仍偏高但其他相位恢复参与；建议先同包长训或评估 |
| P01 | 2026-06-10 | PPO 首次平台切换；按用户平台操作记录 | 平台默认/未记录 | 约 3min 后失败 | 任务 ID `209360` | 训练未启动 | learner 创建 agent 失败：`optimizer got an empty parameter list`；已修复 PPO MLP 显式注册和 optimizer 参数检查 |
| P02 | 2026-06-10 | PPO 空参数修复后重跑；`ppo2/code.zip` 确认为 `ecc03cf` 代码包 | 平台默认/未记录 | 约 15min 后失败 | 任务 ID `209365` | 训练未启动 | 已越过空参数错误，但 trainer 5 秒后 `signal_killed`，15 分钟内样本发送 `succ_cnt=0`；已加入 PPO 周期性片段样本发送 |
| P03 | 2026-06-10 | PPO 片段样本修复后重跑；`ppo3` 代码包与当前 `agent_ppo/` 一致 | 平台默认/未记录 | 约 15min 后失败 | 任务 ID `210257` | 训练未启动 | 最新 PPO 包仍在 learner trainer 启动约 5 秒后 `signal_killed`，随后 aisrv workflow 退出；需要 trainer ERROR 日志定位 |

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

### E04 - 动作分布监控短跑

- 状态：截图时任务仍在进行中，已足够定位动作塌缩问题。
- 平台任务：
  - 任务名：`targetdqn4`
  - 任务 ID：`207146`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：E04 动作监控包，按修复前 `main` `de6a2b4` 记录。
- 环境配置：平台默认/未记录。
- 截图时间窗：2026-06-09 11:29:42 到 2026-06-09 11:40:37；页面显示任务运行约 3min。
- 截图证据：
  - `dqn4/截屏2026-06-09 11.41.11.png`
  - `dqn4/截屏2026-06-09 11.41.17.png`
  - `dqn4/截屏2026-06-09 11.41.21.png`
- 关键监控：
  - `train_global_step` 仍为 0，说明截图时 learner 尚未完成一次参数更新。
  - `predict_succ_cnt` 约 `300+`，`episode_cnt` / `load_model_succ_cnt` 约 `8`，`sample_receive_cnt` 约 `250`。
  - `reward` 约 `-2.9`，`value_loss` 和 `q_value` 暂无数据，符合训练尚未更新的状态。
- 平台评分：
  - score 约 `750-770`。
  - 平均车辆延误约 `55-58`。
  - 平均车辆等待时间约 `27-28`。
  - 平均排队长度约 `9.5-10`。
  - 平均信号变化惩罚约 `0`。
- 动作分布：
  - `phase_0_cnt` 约 `30-32`。
  - `phase_1_cnt`、`phase_2_cnt`、`phase_3_cnt` 基本为 `0`。
  - `avg_duration` 约 `20-23` 秒。
  - `phase_switch_cnt` 为 `0`。
- 结论：
  - 用户判断正确，当前动作几乎一直是同一个相位，且没有真实切相。
  - 这不是训练收敛到 phase 0；截图时 `train_global_step=0`，模型还没开始学习。
  - 根因更可能是平台通用 `legal_action` / `legalAction` 是“是否需要决策”的门控向量，例如 `[1,0,0,0]`，而修复前代码把它误当成相位级 mask，只允许 phase 0。
- 后续修复：
  - 通用 `legal_action`、`legalAction`、`actionMask` 改为只判断是否需要预测；只要其中任一值非零，就对四个相位全部放开。
  - 只有显式 `phase_legal_action`、`phaseLegalAction`、`phase_mask`、`phaseMask` 才作为相位级合法动作 mask。
  - 下一轮应上传该修复后包，若动作正常，`phase_1_cnt/phase_2_cnt/phase_3_cnt` 应不再长期为 0，`phase_switch_cnt` 应明显大于 0。

### E05 - legal_action 修复后一小时平台训练

- 状态：已完成，平台任务自动释放。
- 平台任务：
  - 任务名：`target_dqn5`
  - 任务 ID：`207778`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：E05 legal_action 修复包，按上传前仓库最新 `main` `1607528` 记录；`dqn5/` 只包含监控截图，没有平台下载代码包。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-09 15:33:54 到 2026-06-09 16:34:33。
- 训练时长：1h。
- 模型 ID：截图未显示模型 ID；平台任务 ID 为 `207778`。
- 截图证据：
  - `dqn5/截屏2026-06-09 16.37.27.png`
  - `dqn5/截屏2026-06-09 16.37.31.png`
  - `dqn5/截屏2026-06-09 16.37.34.png`
  - `dqn5/截屏2026-06-09 16.37.35.png`
- 关键监控：
  - `train_global_step`：约 `19`。
  - `predict_succ_cnt`：约 `1000`。
  - `episode_cnt` / `load_model_succ_cnt`：约 `45`。
  - `sample_receive_cnt`：约 `950`。
  - `sample_production_and_consumption_ratio`：末段约 `5.1`。
  - `reward`：约从 `-2.0` 降到 `-3.0` 到 `-3.3` 后基本稳定。
  - `value_loss`：约从 `1.8` 降到 `0.9`。
  - `q_value`：约从 `0.3` 降到 `-1.8`。
- 平台评分：
  - 总分/score：初期约 `1280`，随后降到约 `760-820` 并基本横盘。
  - 平均车辆延误：约 `44-49`，优于 E02/E03/E04 的约 `55-58`。
  - 平均车辆等待时间：约 `21-24`，优于 E02/E03/E04 的约 `27-30`。
  - 平均排队长度：约 `11-12`，略高于 E02/E03/E04 的约 `9-10`。
  - 平均信号变化惩罚：末段约 `1`，说明真实切相已出现。
- 动作分布：
  - `phase_2_cnt` 后半段约 `17-20`，成为明显主导相位。
  - `phase_0_cnt`、`phase_1_cnt`、`phase_3_cnt` 后半段多在 `0-2` 附近。
  - `avg_duration` 后半段约 `30-33` 秒。
  - `phase_switch_cnt` 从初期约 `19` 降到末段约 `3-5`。
  - `same_phase_ratio` 从约 `0.3` 升到约 `0.75-0.85`。
- 结论：
  - E04 的 legal_action 门控修复有效：本次已不再出现 phase 0 全锁死，`phase_switch_cnt` 也不再为 0。
  - 新瓶颈是修复后策略快速偏向 phase 2，且绿灯 duration 偏长，连续同相位比例偏高。
  - 一小时内 `train_global_step≈19`，旧 `TARGET_UPDATE_FREQ=20` 基本等于短训末段才首次同步，短训学习效率偏低。
- 后续修复：
  - 将 `train_batch_size` 从 `256` 降到 `128`，`preload_ratio` 从 `0.0625` 降到 `0.03125`，提高一小时短训内 learner 更新次数。
  - 将 `TARGET_UPDATE_FREQ` 从 `20` 降到 `10`，保证短训期间多次同步 target network。
  - 将 `EPSILON_DECAY` 从 `0.97` 放慢到 `0.995`，`END_EPSILON_GREEDY` 从 `0.1` 提到 `0.2`，避免前几百次预测后过早固定到单一相位。
  - 将 `PHASE_AGE_SCALE` 从 `120` 降到 `90`，`FAIRNESS_BONUS_SCALE` 从 `0.2` 提到 `0.5`，增强长期未服务相位的奖励信号。
  - 下一轮 E06 重点看 phase 2 占比是否下降、`same_phase_ratio` 是否低于 `0.7`、平均延误是否继续低于 `45`。

### E06 - phase-bias 调参后一小时平台训练

- 状态：已完成，平台任务自动释放。
- 平台任务：
  - 任务名：`dqn6`
  - 任务 ID：`208300`
  - 实验版本：`V73.1.1`
  - 算法：`Target DQN`
  - 训练模式：分布式
- Commit：E06 phase-bias 调参包，按上传前仓库 `main` `a0d8efe` 记录；`dqn6/` 只包含监控截图，没有平台下载代码包。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-09 20:56:44 到 2026-06-09 21:57:24。
- 训练时长：1h。
- 模型 ID：截图未显示模型 ID；平台任务 ID 为 `208300`。
- 截图证据：
  - `dqn6/截屏2026-06-10 11.02.58.png`
  - `dqn6/截屏2026-06-10 11.03.04.png`
  - `dqn6/截屏2026-06-10 11.03.08.png`
- 关键监控：
  - `train_global_step`：约 `87`，相比 E05 的约 `19` 明显提升。
  - `predict_succ_cnt`：约 `1600`。
  - `episode_cnt` / `load_model_succ_cnt`：约 `58`。
  - `sample_receive_cnt`：约 `1550`。
  - `sample_production_and_consumption_ratio`：末段约 `7.0`。
  - `reward`：大多在约 `-2.4` 到 `-2.9` 间波动，末段约 `-2.6`。
  - `value_loss`：约从 `1.6` 降到 `0.3`。
  - `q_value`：约从 `0` 降到 `-2.3`。
- 平台评分：
  - 总分/score：最高约 `1300`，中段最低约 `900`，末段约 `1100`。
  - 平均车辆延误：末段约 `20`，显著优于 E05 的约 `44-49`。
  - 平均车辆等待时间：末段约 `10`，显著优于 E05 的约 `21-24`。
  - 平均排队长度：末段约 `9`，优于 E05 的约 `11-12`。
  - 平均信号变化惩罚：末段约 `1`，仍在可接受范围。
- 动作分布：
  - `phase_2_cnt` 仍为主导相位，峰值约 `22`，末段约 `14-16`。
  - `phase_0_cnt` 末段约 `6-7`，`phase_1_cnt` 约 `2-4`，`phase_3_cnt` 约 `3-4`；其他相位已经恢复参与。
  - `avg_duration` 约 `22-25` 秒，比 E05 的约 `30-33` 更合理。
  - `phase_switch_cnt` 多数时间在约 `10-19`，末段约 `15`，切相明显恢复。
- 结论：
  - E06 参数是当前最好的 Target-DQN 基线，说明降低 batch、提前 preload、加快 target sync、保留探索和增强公平性有效。
  - phase 2 仍偏高，但不再像 E05 那样几乎压制其他相位；平台核心指标已经显著改善。
  - 当前不建议继续立即改代码，否则难以归因且可能破坏已经有效的基线。
- 下一步：
  - 用同一 E06 代码包跑 2-3h 长训或正式评估，确认末段 score 是否稳定在 `1100+`，平均延误是否能维持在 `20-25`。
  - 若长训后 phase 2 仍长期高于其他相位 2 倍以上，再考虑下载真实 observation 样例，检查四个相位压力字段和车道映射。
  - 若长训稳定但评估分不足，再做小步 reward 调整；优先调排队/等待权重，不再先改 legal_action 或 replay 结构。

### P01 - PPO 首次平台切换失败

- 状态：失败，平台任务已停止。
- 平台任务：
  - 任务名：`ppo1`
  - 任务 ID：`209360`
  - 实验版本：`V73.1.1`
  - 算法：`PPO`
  - 训练模式：分布式
- Commit：按用户平台切换操作记录；平台截图未包含上传代码包，无法逐文件复核。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-10 12:56:35 到 12:59:16 附近。
- 训练时长：页面显示约 3min 后失败。
- 截图证据：
  - `ppo1/截屏2026-06-10 13.02.19.png`
  - `ppo1/截屏2026-06-10 13.03.19.png`
- 错误日志：
  - learner：`failed to run create_standard_agent_wrapper. exit. Error is: optimizer got an empty parameter list`
  - aisrv：`kaiwu_rl_helper workflow() Exception list index out of range`
- 结论：
  - 首个错误是 learner 创建 PPO agent 时 optimizer 拿到空参数列表；aisrv 的 `list index out of range` 是 agent wrapper 创建失败后的连带异常。
  - 当前本地代码已将 PPO MLP 改为显式 `OrderedDict -> nn.Sequential` 注册层，并让 optimizer 使用 `list(self.model.parameters())`，同时增加空参数断言，便于平台确认是否同步了 `agent_ppo/model/model.py`。
- 下一步：
  - 平台重新试 PPO 时必须同步整个 `agent_ppo/` 目录，尤其是 `agent_ppo/model/model.py` 和 `agent_ppo/agent.py`，并确认 `conf/app_conf_intelligent_traffic_lights.toml` 中 `algo = "ppo"`。
  - 如果仍报同一错误，说明平台包仍未使用最新 PPO model 文件，应下载平台代码包复核。

### P02 - PPO 空参数修复后十五分钟失败

- 状态：失败，平台任务已停止。
- 平台任务：
  - 任务名：`ppo2`
  - 任务 ID：`209365`
  - 实验版本：`V73.1.1`
  - 算法：`PPO`
  - 训练模式：分布式
- Commit：P01 修复后重跑；用户后补 `ppo2/code.zip`，确认平台上传包的 `agent_ppo/` 与仓库 `ecc03cf` 完全一致。
- 环境配置：平台默认/未记录。
- 训练时间：2026-06-10 13:11:23 到 13:26:37。
- 训练时长：页面显示约 15min 后失败。
- 截图证据：
  - `ppo2/截屏2026-06-10 13.30.00.png`
  - `ppo2/截屏2026-06-10 13.30.51.png`
  - `ppo2/截屏2026-06-10 13.30.55.png`
  - `ppo2/截屏2026-06-10 13.30.59.png`
- 代码包证据：
  - `ppo2/code.zip`
  - `conf/app_conf_intelligent_traffic_lights.toml` 中 `algo = "ppo"`，说明平台入口已正确切到 PPO。
  - `agent_ppo/model/model.py` 已包含 `nn.Sequential(layers)`，`agent_ppo/agent.py` 已包含 `parameters = list(self.model.parameters())`，说明 P01 空参数修复确实已上传。
  - 相对当前 `main` `99401cb`，该平台包缺少 `agent_ppo/conf/conf.py`、`agent_ppo/workflow/train_workflow.py` 和 `agent_ppo/feature/definition.py` 中的 P02 片段样本发送修复。
- 错误日志：
  - learner：`ProcessHealthMonitor detected process exit: name=trainer, pid=338, type=trainer, exit_reason=signal_killed, exit_code=-6, runtime=5.0s`
  - learner：`learner_init subprocess exited: name=trainer, pid=338, reason=signal_killed`
  - aisrv：`learner_proxy send sample stat, succ_cnt is 0, error_cnt is 0`
  - aisrv：`model_file_sync self.model_pool_apis.pull_keys() is None or empty, so return`
- 结论：
  - P02 已越过 P01 的空参数错误，但 learner trainer 仍在启动后 5 秒退出；平台截图未提供 trainer Python traceback。
  - 可观测的后续问题是 15 分钟内没有成功发送样本，`succ_cnt` 一直为 0，PPO workflow 原先只在 episode 结束时发送样本，遇到长局或异常局会让 learner 长时间拿不到数据。
  - 后补代码包确认 P02 跑的不是旧模板，也不是漏传 P01；失败发生在 P01 修复后的 PPO workflow 版本上。
- 后续修复：
  - 增加 `Config.PPO_FRAGMENT_SIZE = 32`。
  - PPO workflow 每累计超过 32 个决策 transition 就先处理并发送 `collector[:-1]`，保留最后一个 transition 继续接后续 reward，并用保留 transition 的 value 为片段末尾 bootstrap。
  - 下一轮 P03 重点看 `learner_proxy send sample stat` 的 `succ_cnt` 是否大于 0；若 trainer 仍 `signal_killed`，需要补充平台 `trainer` 文件的 ERROR 日志。

### P03 - PPO 最新包仍 trainer 早退

- 状态：失败，平台任务已停止。
- 平台任务：
  - 任务名：`ppo3`
  - 任务 ID：`210257`
  - 实验版本：`V73.1.1`
  - 算法：`PPO`
  - 训练模式：分布式
- 代码包：`ppo3/code-intelligent_traffic_lights-IDE-73.1.1 (3).zip`。
- 环境配置：平台默认/未记录。
- 提交时间：2026-06-10 20:51:47。
- 训练时间：2026-06-10 20:51:47 到 21:07:00。
- 训练时长：页面显示约 15min 后失败。
- 截图证据：
  - `ppo3/截屏2026-06-10 21.14.02.png`
  - `ppo3/截屏2026-06-10 21.14.43.png`
- 代码包证据：
  - `conf/app_conf_intelligent_traffic_lights.toml` 中 `algo = "ppo"`，说明平台入口已正确切到 PPO。
  - `agent_ppo/conf/conf.py` 已包含 `PPO_FRAGMENT_SIZE = 32`。
  - `agent_ppo/workflow/train_workflow.py` 已包含片段样本发送和末尾 value bootstrap。
  - `agent_ppo/feature/definition.py` 已保留非终局片段末尾已有 `next_value`。
  - 展开包内 `agent_ppo/` 与当前本地 `codebase/agent_ppo/` 无差异，排除 PPO 主线漏传。
- 错误日志：
  - learner：`learner_init subprocess exited: name=trainer, pid=339, reason=signal_killed`
  - learner：`ProcessHealthMonitor detected process exit: name=trainer, pid=339, type=trainer, exit_reason=signal_killed, exit_code=-6, runtime=5.0s`
  - aisrv：`ProcessHealthMonitor detected process exit: name=workflow_0, pid=290, type=workflow, exit_reason=unknown, exit_code=None, runtime=3.4s`
  - aisrv：`workflow_1`、`workflow_2`、`workflow_3` 也在约 3.6 到 3.9 秒后 `exit_reason=unknown`。
  - 平台卡片提示：`learner exited, please check the log for details`。
- 结论：
  - P03 已确认不是 P01 空参数问题，也不是 P02 漏传片段样本修复。
  - trainer 在样本产生前已退出，因此继续调 PPO workflow 的样本发送节奏无法解释当前首个失败点。
  - 当前截图仍没有 trainer Python traceback；下一步必须获取平台 `trainer` 文件的 ERROR/ALL 日志，否则只能看到进程被杀，无法确认是平台资源、Reverb/replay 初始化、Torch 运行时还是 PPO agent 初始化中的具体异常。
- 后续修复：
  - 用户确认平台日志筛选只有 `aisrv`、`learner` 和 `env`，无法直接选择 trainer 文件。
  - 已在 PPO 启动路径增加 `[PPO_DIAG]` 面包屑和 `faulthandler`，下一轮 P04 在 `learner` 日志中搜索 `[PPO_DIAG]` 和 `Fatal Python error`。

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
