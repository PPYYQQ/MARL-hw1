# 平台运行与调参记录指南

本文件记录如何在腾讯开悟/KaiwuDRL 环境中验证当前 `target_dqn` 主线，以及平台运行后应回填哪些信息。

## 当前主线

- 代码目录：`codebase/agent_target_dqn`
- 平台算法名：`target_dqn`
- 入口配置：`codebase/conf/app_conf_intelligent_traffic_lights.toml`
- 本地基础入口：`codebase/train_test.py`
- 进度记录：`PROGRESS.md`
- 实验台账：`EXPERIMENTS.md`

## 本地可运行检查

在当前普通 Python 环境中可运行：

```bash
cd codebase
python -m compileall agent_target_dqn tests
python tests/test_target_dqn_static.py
python tests/test_target_dqn_smoke.py
```

也可以在仓库根目录运行一键检查：

```bash
./scripts/check_offline.sh
```

说明：

- `test_target_dqn_static.py` 不依赖 `torch` 和 `kaiwudrl`，用于检查关键源码约束。
- `test_target_dqn_smoke.py` 需要 `torch`；当前本地未安装时会输出 skip。
- `check_offline.sh` 会同时运行编译、静态检查、smoke、空白检查和提交包内容检查。
- `python train_test.py` 需要 KaiwuDRL 平台依赖，当前本地会因缺少 `kaiwudrl` 失败。

## 平台验证步骤

1. 上传或同步 `codebase/` 到腾讯开悟作业环境。需要压缩包时运行：
   ```bash
   ./scripts/package_submission.sh
   ```
   默认产物为 `dist/marl_hw1_codebase.zip`，只包含 `codebase/` 目录内容，不包含日志、checkpoint、截图和本地报告资料。
2. 确认 `codebase/conf/app_conf_intelligent_traffic_lights.toml` 中 `algo = "target_dqn"`。
3. 先运行 `python train_test.py`，确认基础接口、环境 reset/step、样本发送和模型保存不崩溃。
4. 如果 `train_test.py` 通过，再启动短训练任务，建议先使用默认简单环境：
   - `weather = 0`
   - `rush_hour = 0`
   - `speeding_cars_rate = 0`
   - `traffic_accidents.random_count = 0`
   - `traffic_control.random_count = 0`
5. 观察 10-30 分钟监控，确认样本生产、loss、reward 和模型保存正常。
6. 短训练稳定后，再逐步打开随机天气、高峰、事故或管制配置。

## 重点监控指标

- `reward`：总平均 step reward，应避免长期固定为 0。
- `phase_reward`：相位选择奖励，观察是否有方向性改善。
- `duration_reward`：持续时间奖励，观察是否长期强负。
- `env_score`：平台返回的最新总分或 score 快照。
- `avg_delay` / `avg_queue_length` / `avg_waiting_time`：平台评分三项，优先用于比较实验结果。
- `switch_penalty`：平台信号切换惩罚或切换次数快照。
- `value_loss`：DQN TD loss，先看是否有限值且不爆炸。
- `q_value` / `target_q_value`：观察 Q 值是否数值稳定。
- `model_grad_norm`：梯度范数，频繁过大说明奖励或学习率可能不稳。
- 平台评分项：若平台监控字段为空，以评估任务页面中的平均延误、平均排队长度、平均等待时间、信号切换惩罚为准。

## 每次实验回填格式

将每次平台运行追加到 `EXPERIMENTS.md`，关键开发变更仍记录到 `PROGRESS.md`：

```text
### Experiment N - 简短名称

- 状态：
- Commit：
- 环境配置：
- 训练时长：
- 模型 ID：
- 关键指标：
- 评估得分：
- 结论：
- 下一步：
```

## 常见问题

- `ModuleNotFoundError: kaiwudrl`：需要在腾讯开悟/KaiwuDRL 环境运行，普通本地 Python 不包含该依赖。
- `ModuleNotFoundError: torch`：安装 PyTorch 或使用平台镜像。
- `read usr conf failed`：当前 workflow 会记录配置读取/校验异常并退出训练入口；检查 `agent_target_dqn/conf/train_env_conf.toml` 格式、路径和平台配置校验工具。
- `handle disaster recovery failed`：当前 workflow 会记录容灾 helper 异常并按无容灾信号继续；如果反复出现，检查平台容灾 SDK 或 env_obs 格式。
- `load latest model` 找不到文件：当前代码会跳过首次 `latest` 加载，属于从零训练的正常情况。
- `load latest model failed`：当前 workflow 会记录错误并继续使用当前模型参数；如果长期出现，检查 checkpoint 文件权限、路径挂载或模型文件是否被平台并发写坏。
- 训练中恢复 `latest`：当前 workflow 周期保存 `model.ckpt-latest.pkl`，后续 episode 会加载同名 checkpoint。
- `save latest model failed`：当前 workflow 会记录错误并继续训练，同时节流到下一保存周期再重试；需要检查 checkpoint 路径权限、磁盘空间或平台模型目录挂载。
- `get training metrics failed`：当前 workflow 会忽略本轮平台训练指标读取失败并继续 episode；如果长期出现，检查平台 metrics 服务或 SDK 状态。
- `send sample data failed`：当前 workflow 会丢弃本批未发送成功的样本并继续后续 episode；如果长期出现，检查样本池、learner 服务和平台通信状态。
- `sample process failed`：当前 workflow 会丢弃当前 episode 或容灾 collector 并继续后续 episode；如果长期出现，保存原始 collector、最后两帧 observation/action/reward 来定位样本转换输入。
- Frame 属性读取失败：当前 `sample_process()` 会隔离 `obs`、`act`、`rew`、`done`、`legal_action` 的属性访问异常；`obs` / `act` 失败会跳过当前帧，其他字段失败会使用默认 reward、done 或 legal action。
- Frame 字段转换失败：异常 array-like 的 `obs` / `rew` 会按零向量处理；`act` 对象长度或索引异常会跳过当前帧，避免整段 collector 转换失败。
- `legal_action` 转换失败：当前归一化 helper 会把异常 array-like mask 回退为四个相位都可选；如果平台语义明确表示不可决策，需要用真实 observation 日志确认是否应在 workflow 层继续按 `0` 门控。
- `sample reward read failed` / `sample batch length failed`：当前 workflow 会把异常样本批次按可读部分或零 reward 统计，训练发送路径仍单独处理；如果反复出现，检查 `sample_process()` 返回对象是否为 `SampleData` 列表。
- `learn failed`：当前 `Agent.learn()` 会跳过当前 batch 并保留 learner 进程；如果连续出现，优先保存样本池中的原始 batch，检查字段 shape、dtype 和 `Algorithm.learn()` traceback。
- `sample batch iteration failed`：当前 learner 会丢弃无法迭代的异常 batch 容器，generator 式 batch 会先安全转成 list；如果反复出现，优先检查样本池传给 learner 的 batch 类型。
- `latest` 模型结构不兼容：当前联合动作模型会跳过不兼容的旧 `latest` checkpoint，并从当前参数继续训练；若要强制加载指定模型 ID，结构不兼容仍会抛错。
- `legal_action` 是标量而不是列表：当前 workflow 会先归一化为 4 维相位 mask，再判断是否需要决策；若平台提供相位级 mask，也会沿用相位约束。
- 空合法动作 mask 或 `ValueError: 'a' cannot be empty`：当前 Agent 推理侧会把全零相位 mask 和空 joint mask 行回退为可选全集；如果仍出现，优先检查是否有新代码绕过了 `_phase_action_mask()` / `_joint_action_mask()`。
- `predict observation batch failed`：当前 `predict()` 会丢弃无法迭代的异常 ObsData batch，generator 式 batch 会先转成 list；如果反复出现，检查平台传入 `Agent.predict()` 的数据类型。
- `invalid action, use default action`：当前 workflow 会在进入 `env.step()` 前把异常动作回退为 `[0, 0, MIN_GREEN_DURATION]`；如果频繁出现，检查 `predict()`、`action_process()` 或规则兜底返回值。
- `rule_based_action failed, use default action`：评估 `exploit()` 的最终规则兜底失败时会返回默认动作；如果反复出现，保存评估 observation 并检查规则策略输入结构。
- 评估 observation 字段读取失败：当前 `Agent.exploit()`、`observation_process()` 和 `rule_based_action()` 会对 dict 或属性对象 observation 使用安全读取并进入规则/默认兜底；如果反复出现，需要保存原始 observation 类型和 repr。
- repeated 字段是 dict 形态：当前 `vehicles`、`phases`、`lanes`、路网初始化和 reward 解析会兼容单条 dict 记录和 dict-of-records；若平台使用 protobuf repeated wrapper 或其它特殊容器，保存原始类型、repr 和可迭代行为后继续扩展。
- `env reset failed`：当前 workflow 会跳过当前 episode 并在下一 epoch 重试；如果持续出现，检查环境配置、平台任务状态和 reset 返回协议。
- `env step failed`：当前 workflow 会中止当前 episode 并丢弃未完成 collector；如果持续出现，优先确认动作合法性、平台环境状态和前一帧 observation。
- `env.step()` 返回形态不一致：当前 workflow 支持对象式返回、dict/object step envelope、二元封装返回、Gym 四元返回、Gymnasium 五元返回和作业文档六元返回，并会保留对象式 `extra_info`；如果平台返回其他结构，需要保存原始返回值再扩展 `_normalize_step_result()`。
- env_obs/obs 字段读取失败：当前 workflow 会兼容 dict 与属性对象字段读取，嵌套 `observation` / `obs` / `_obs` 和直接包含 `frame_state` / `legal_action` 的裸 observation 都会保留，`extra_info` / `_state` / `state` 都会作为额外信息读取；标量 observation / extra_info 会回退为空对象。如果频繁出现，需要保存原始环境返回类型，确认平台封装是否已经损坏。
- 评估入口 observation 包装不同：当前 `Agent.exploit()` 会兼容 `obs`、`observation`、`_obs` 以及 `extra_info`、`_state`、`state`，如果评估仍固定输出默认动作，优先保存评估入口传入对象的类型和 repr。
- `terminated` / `truncated` 字段异常：当前 workflow 只把 bool true、非零有限数值或明确 true 字符串视为结束；未知字符串、NaN/Inf 和异常对象按 False 处理。
- 样本 `done` 字段异常：当前 `sample_process()` 会把 bool、有限数值和 true/false 字符串统一转成 not_done 标记；未知字符串、NaN/Inf 和异常对象按非终局处理。
- `agent reset failed`：当前 workflow 会跳过本局 episode，避免使用半初始化 agent 状态继续采样；如果反复出现，优先检查 `FeatureProcess.reset()` 和 agent 初始化状态。
- `run_episodes error: ...`：优先看冒号后的原始异常信息和 Python chained traceback，当前 workflow 不再只抛通用错误。
- 日志或监控异常：当前 workflow、learner 和 `Agent` checkpoint/评估兜底日志失败不会中断训练；如果平台看不到指标，先查 monitor 后端或日志权限。
- 日志过多：当前进度日志只在 episode 结束或每 20 次真实预测后打印；若平台日志仍过密，优先检查是否有异常反复重启。
- `reward shaping failed`：当前 workflow 会将该步奖励回退为 `(0.0, 0.0)` 并继续 episode；需要保存对应 observation、action 和 agent 状态定位奖励函数异常。
- duration reward 长期为负：当前 20 个 duration 桶覆盖 `8-40` 秒，reward 目标也限制在同一范围；如果仍长期强负，优先检查压力尺度、`DIM_OF_ACTION_DURATION` 和实际平台动作秒数是否一致。
- `observation process failed` / `traffic info update failed`：当前 workflow 会回退到规则动作、零特征样本或跳过非决策帧预处理并继续；需要保留原始 observation 和 extra_info 定位特征处理异常。
- 观测里有异常 frame 或车辆字段：当前预处理器会兼容 dict、属性对象和单个对象式列表字段，并清洗 frame、车辆 ID、车速和位置；标量坏字段会跳过。若仍异常，优先保存原始 observation 样例和 repr。
- 观测里 `vehicles` 为空但 `lanes` 有值：当前观测、规则兜底和 reward 会用 lanes 的 `lane_id`、`v_count`、`queue_length`、`congestion` 聚合压力；若平台字段名或单位不同，保存 `frame_state.lanes` 原始样例后扩展字段映射。
- 车辆字段缺少 `target_junction`：当前进口车道判断和交叉口等待时间统计会在车辆可识别为进口车道时按单路口目标路口处理；没有车辆 ID 或无法识别进口车道的畸形 targetless 记录会跳过。若平台实际上用其它字段区分车辆目标，需要保存原始车辆样例再扩展映射。
- `junction` / `target_junction` 是字符串：当前会先转换成有限整数再判断进口车道、路口内状态和等待时间归属，`"0"` 会匹配单路口，`"-1"` 会按无目标或不在路口处理；若平台提供其它哨兵值，保存原始车辆样例后扩展清洗规则。
- 路网或车辆 ID 是字符串：当前会清洗 `junction_id`、`edge_id`、`lane_id`、`vehicle_config_id`、车辆 `lane` 和 `v_config_id`，避免路网 key 与车辆字段一个是 `"11"`、一个是 `11` 时相位压力、车道统计或 max speed 查找失效。
- init_state 字段命名差异：当前路网初始化兼容 `j_id/e_id/l_id/v_config_id` 与 `junction_id/edge_id/lane_id/vehicle_config_id`；如果平台还有其它字段名，保存 `extra_info.init_state` 原始样例后补充别名。
- 观测里有异常相位字段或帧号字段：当前相位 ID、duration、remaining duration 和相位年龄都会清洗为有限值；workflow 会从顶层 `frame_no` / `frameNo` 或嵌套 `extra_info` / `_state` / `state` 中回退读取帧号并清洗。若仍异常，优先保留原始 `frame_state.phases` 和 env_obs 返回样例。
- 模型输入含 NaN/Inf 或异常 array-like：当前 `Model._prepare_input()` 会把非有限值归零，ragged 或转换失败的 Python observation 会补零或截断；如果仍出现非有限 Q 值，优先保留进入模型前的 feature。
- reward 长期为 0：检查 `reward_shaping()` 是否收到真实车辆字段或 lane 字段，`vehicles` 和 `lanes` 是否同时为空，以及相位压力是否一直为 0。
- loss 爆炸或 NaN：当前特征、样本和 learner 都会清洗 NaN/Inf，reward 还会限制极端延误项并裁剪到 `[-Config.REWARD_CLIP, Config.REWARD_CLIP]`；若仍出现，优先降低 `Config.LR`，再缩小 reward 权重并保留异常 observation 样例。
- 样本 shape 不一致或 `torch.stack` 报错：当前 `sample_process()` 和 `Algorithm.learn()` 都会定宽归一化样本字段；若仍出现，优先保留一局原始 collector 日志来定位平台返回的异常字段。
- learner 样本属性读取失败：当前 `Algorithm.learn()` 会对 `obs`、`act`、`rew`、`_obs`、`done` 和 `legal_action` 使用安全读取；坏字段按默认值训练，不会直接跳过整批。
