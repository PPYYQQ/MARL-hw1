# 智能交通信号灯调度报告草稿

本文档用于承接最终作业报告内容。正式排版时可参考 `icml2022.zip` 模板，将本文内容迁移到 LaTeX。

## 摘要

本作业面向单交叉口智能交通信号灯调度问题，目标是在不同交通流、天气和事件配置下，通过强化学习智能体选择信号灯相位和绿灯持续时间，降低车辆平均延误、平均排队长度和平均等待时间，并减少过于频繁的信号切换。当前实现以 Target-DQN 为主线，结合规则基线兜底策略，完成了观测网格化、动作映射、奖励设计、目标网络同步、训练样本处理、模型保存加载和本地验证脚本。

## 问题定义

环境为单个十字路口，智能体控制一个信号灯。每次需要决策：

- `junction_id`：固定为 `0`。
- `phase_idx`：四个相位之一，包括南北直行、南北左转、东西直行、东西左转。
- `duration`：当前相位持续时间，单位为秒。

优化指标包括：

- 平均车辆延误。
- 平均排队长度。
- 平均等待时间。
- 信号切换惩罚。

## 方法概述

当前主线为 `agent_target_dqn`。选择 Target-DQN 的原因是动作空间离散，且经验样本可以复用，相比普通 DQN 更稳定，相比 PPO 更容易在有限时间内调通训练闭环。`agent_ppo` 已补成可训练备选线，可用于后续对比实验，但当前尚无平台短训成绩，报告结论仍以 Target-DQN 为主。

实现包含：

- 在线 Q 网络和独立目标 Q 网络。
- Double DQN 风格 TD 目标：在线网络选择下一动作，目标网络评估该动作的 Q 值。
- Huber loss，降低异常 TD error 对训练的冲击。
- 80 维联合动作 Q head，直接评估相位和持续时间组合价值。
- epsilon-greedy 训练探索。
- 评估接口 `exploit()` 强制使用贪心动作，不衰减训练探索率。
- `legal_action` 相位 mask，在贪心预测、随机探索、规则兜底和 Double DQN 下一动作选择中避免选择明确非法的相位。
- 固定频率目标网络同步。
- 规则策略兜底，保证评估阶段即使模型异常也能输出合法动作。

## 状态特征

当前观测维度为 `638`：

```text
14 条进口车道 * 20 个距离网格 * 2 类特征 + 8 维相位时间特征 + 4 维相位年龄特征 + 8 维交通统计特征 + 8 维交通趋势特征 + 8 维滚动历史特征 + 42 维逐车道统计特征 = 638
```

两类特征为：

- 位置占用：车辆所在进口车道与距离格子置为 `1`。
- 归一化速度：车辆速度除以车辆配置最大速度，并截断到 `[0, 1]`。

相位时间特征包括：

- 当前相位 one-hot，4 维。
- 当前相位配置持续时间归一化值。
- 当前相位剩余时间归一化值。
- 当前相位已执行时间归一化值。
- 相位信息是否存在的标志位。

相位年龄特征包括：

- 四个相位距上次服务的归一化帧数。

交通统计特征包括：

- 四个相位对应车道组的归一化压力。
- 当前进口车辆数归一化值。
- 当前排队车辆比例。
- 当前平均等待时间归一化值。
- 当前平均延误归一化值。

交通趋势特征包括：

- 四个相位压力相对上一帧的归一化变化。
- 进口车辆数相对上一帧的归一化变化。
- 排队车辆比例相对上一帧的变化。
- 平均等待时间和平均延误相对上一帧的归一化变化。

滚动历史特征包括：

- 最近 4 帧相位压力、进口车辆数、排队比例、平均等待时间和平均延误的归一化平均值。

逐车道统计特征包括：

- 14 条进口车道的归一化车辆数。
- 14 条进口车道的归一化排队车辆数。
- 14 条进口车道的归一化平均等待时间。

如果 `vehicles` 列表为空或无法推导有效进口车道压力，代码会使用 `frame_state.lanes` 中的 `lane_id` / `laneId`、`v_count` / `vCount` / `vehicle_count`、`queue_length` / `queueLength` / `queue_count` 和 `congestion` / `congestionLevel` 作为 fallback，聚合四个相位压力，并把 lanes 聚合统计与逐车道统计特征取最大值合并。这保持了 638 维输入不变，同时避免真实平台只提供 lane 级聚合信息时交通压力长期为 0。

代码中对 `position_in_lane["y"]` 做了单位兼容：

- 当绝对值大于 `200` 时，按毫米处理并除以 `1000`。
- 否则按米处理。

特征预处理和观测编码对缺失 `frame_state` / `frameState`、缺失 `vehicles`、缺失 `obs` 包装、畸形车辆记录和异常相位字段采用保守跳过或默认值策略，避免异常帧中断训练循环。字段读取同时兼容普通 dict 和作业协议对象属性，覆盖 Observation、FrameState、Vehicle、Phase、Lane、ExtraInfo 等消息形态；若平台把单个 Vehicle/Phase/Lane 直接作为 dict 或对象返回，或把 repeated 字段编码成 dict-of-records，也会按有效记录处理，而标量坏字段会被视为无效输入。
作业文档列出的车辆字段不一定包含 `target_junction`，因此进口车道判断和交叉口等待时间统计会在该字段缺失、且车辆可识别为进口车道时按单路口目标路口处理，避免真实 observation 只包含 `lane` 和 `junction` 时把进口车辆全部忽略。车辆字段读取也兼容常见驼峰别名，例如 `vehicleId`、`vConfigId`、`laneId`、`junctionId`、`targetJunction`、`positionInLane` 和 `waitingTime`。

预处理器会清洗 `frame_no` / `frameNo`、`frame_time` / `frameTime`、车辆 ID、车速、车道位置和路口 ID；`junction` / `junctionId` / `target_junction` / `targetJunction` 即使以字符串形式返回，也会先转换成有限整数，避免字符串 `"-1"` 被误判为有效目标路口或路口内车辆。路网初始化会把 `junction_id` / `junctionId`、`edge_id` / `edgeId`、`lane_id` / `laneId`、`vehicle_config_id` / `vehicleConfigId` 等数字 ID 归一化为整数，车辆 `lane` / `laneId` 和 `v_config_id` / `vConfigId` / `vehicleConfigId` 也会在相位压力、车道统计和车辆配置查找前清洗，车辆配置速度字段兼容 `max_speed` / `maxSpeed`，避免平台 JSON 或 protobuf 转换后出现字符串 key 导致统计信号丢失。等待时间、行驶距离、车道车辆数和交叉口等待时间统计遇到异常车辆字段时跳过单车或按 0 处理，避免一个异常车辆破坏跨帧状态。路网初始化、车辆统计、reward 和 workflow 环境返回解析都通过安全字段读取处理 dict / 属性对象差异，并明确排除 int、float、bool 等标量伪记录。

路网初始化会同时兼容 `init_state` / `initState`，模板字段 `j_id/e_id/l_id/v_config_id`，文档式字段 `junction_id/edge_id/lane_id/vehicle_config_id`，以及 `junctionId/edgeId/laneId/vehicleConfigId`、`laneConfigs`、`vehicleConfigs`、`enterLanesOnDirections` 等驼峰字段，避免真实平台 init_state 命名差异导致路口、边、车道或车辆配置未写入预处理器。

交通统计工具会清洗车辆速度、等待时间、延误、相位压力、交通趋势和历史统计中的 NaN/Inf；最终 observation 返回前也会统一补齐或截断到 `DIM_OF_OBSERVATION`，并把非有限特征归零。这样即使平台返回少量异常车辆字段，也不会把 NaN/Inf 送入模型推理或 reward 计算。

相位时间特征、相位服务年龄、reward 公平性项和 workflow 帧号读取也使用有限值清洗；异常 `phase_id`、`duration`、`remaining_duration`、`frame_no` / `frameNo` 或旧相位服务记录会回退为保守默认值。相位记录读取会兼容 `s_id` / `signal_id` / `signalId`、`phase_id` / `phase_idx` / `phase` / `current_phase` / `currentPhase`、`remaining_duration` / `remaining_time` / `remainingTime` 等字段别名，适配平台对象或封装层命名差异。workflow 帧号会优先读取顶层字段，也会从 `extra_info` / `extraInfo` / `_state` / `state` / `info` 中回退读取，适配平台把帧号放在额外信息对象里的返回形态。

如果后续平台评估显示当前状态表达不足，可增加更长时间窗口或更细粒度的车道趋势。

增加特征时必须同步修改 `Config.DIM_OF_OBSERVATION` 和模型输入层。
模型前向传播会把单条一维 observation 统一成 `[1, DIM_OF_OBSERVATION]` batch；如果输入长度短于配置维度则补零，长于配置维度则截断。对于 Python list 形式的 ragged batch，会先逐行补齐或截断再堆叠，以避免平台封装或测试入口的形状差异直接触发线性层错误。模型入口还会把 NaN/Inf 清零，异常 array-like 转换失败时使用零向量兜底，避免直接模型调用产生非有限 Q 值。

## 动作设计

模型输出一个 80 维联合动作 Q head：

```text
action_id = phase_idx * 20 + duration_idx
```

其中 `phase_idx` 对应四个相位，`duration_idx` 对应 20 个持续时间索引。

环境动作使用秒数，因此代码将 duration index 映射为：

```text
duration_seconds = duration_index_to_seconds(duration_index)
```

当前 `MIN_GREEN_DURATION = 8`、`MAX_GREEN_DURATION = 40`，20 个 duration 桶会覆盖 `8-40` 秒。训练时，算法会将环境动作中的 `[phase_idx, duration_seconds]` 按同一分桶规则反向转换为联合动作索引，避免直接用秒数索引 Q head。

动作输出阶段会固定 `junction_id=0`，并对相位索引和持续时间索引做数值清洗与裁剪；当模型或外部调用传入空预测批次、缺失字段、NaN/Inf 或非数值动作时，代码会返回空预测或保守合法动作，避免向环境提交非法控制指令。

`legal_action` 会被归一化为 4 维相位 mask；读取时同时兼容 `legalAction`、`phaseLegalAction`、`actionMask` 和 `phaseMask` 等常见字段别名：

- 如果平台只提供标量门控，非零值表示四个相位都可选。
- 如果平台提供相位级 mask，则预测和随机探索只在合法相位对应的联合动作中选择。
- 训练 workflow 也使用同一归一化逻辑判断当前帧是否需要调用 `predict()`，避免平台给标量 `int32` 时因下标访问崩溃。
- 训练样本中的 `legal_action` 保存 `_obs` 对应的下一状态 mask，用于 TD target 的下一联合动作选择。
- mask 中的 NaN/Inf 会先归零，避免非有限值被误判为合法相位。
- 如果 `legal_action` 是异常 array-like 对象导致转换失败，归一化 helper 会回退为四个相位都可选，避免 workflow 决策门控、Agent 推理和样本转换被坏 mask 中断。
- 如果 mask 全零，workflow 仍按不需要决策处理；推理侧被直接调用时会回退为四个相位都可选，避免无可选动作导致随机采样或 argmax 路径崩溃。

## 奖励设计

奖励函数返回两个分量：

- `phase_reward`
- `duration_reward`

共享相位压力计算：

```text
pressure = 1
         + 2 * 是否低速等待
         + min(waiting_time, 300) / 30
         + min(delay, 300) / 60
```

phase reward 主要考虑：

- 当前相位压力是否高于平均压力。
- 当前相位是否为压力最大相位。
- 平均等待时间是否下降。
- 当前排队车辆数量。
- 平均延误。
- 高压力相位是否已经较长时间未被服务。

duration reward 主要考虑：

- 当前 duration 与压力估计目标 duration 的差距。
- 平均等待时间变化。
- 过短绿灯惩罚。

奖励设计目标是避免全零奖励，并让策略倾向于服务高压力方向，同时抑制不合理的持续时间。
为降低 DQN 训练中的 TD target 爆炸风险，延误惩罚会先限制到 `REWARD_DELAY_CAP=300`，最终 `phase_reward` 和 `duration_reward` 都会裁剪到 `[-REWARD_CLIP, REWARD_CLIP]`。这能避免少量异常车辆延误或平台字段尖峰主导整批梯度。
相位服务年龄会在每个 episode 重置，避免不同局之间的状态泄漏。
终局或异常帧如果缺少 `frame_state` / `frameState` 或 `vehicles`，奖励函数会保守返回零奖励，避免训练循环因字段缺失中断。
workflow 调用奖励函数时还有一层兜底：如果奖励计算因异常 observation、动作或 agent 状态抛错，本步奖励会回退为 `(0.0, 0.0)` 并记录错误，episode 继续运行。

## 规则兜底策略

`exploit()` 中加入 `rule_based_action()`：

- 统计四个相位对应进口车道组的压力。
- 当车辆明细不可用时，使用 lanes 聚合压力作为兜底。
- 在合法相位中选择压力最高的相位。
- 根据压力大小设置绿灯持续时间。

用途：

- 模型推理异常时避免评估直接崩溃。
- 模型尚未训练充分时提供保守可用策略。
- 作为强化学习策略的 baseline。

## 训练流程

训练 workflow 完成：

1. 读取 `agent_target_dqn/conf/train_env_conf.toml`。
2. 每局开始尝试加载 `latest` 模型；若首次训练无模型、checkpoint 不可读、payload 非 dict 或旧模型结构不兼容，则跳过。
3. 环境 reset 后进行特征处理。
4. 需要决策时调用 `predict()`。
5. 将模型动作转换为环境动作。
6. 调用 `env.step()` 推进环境。
7. 使用 `reward_shaping()` 计算训练奖励。
8. episode 结束后调用 `sample_process()`。
9. 发送样本浅拷贝到训练组件，再清理本地 collector。
10. 定期保存 `latest` checkpoint，供后续 episode 或训练进程恢复。

训练配置读取通过安全 helper 包装；如果平台配置校验工具抛错或返回非字典结果，workflow 会记录配置读取失败并退出入口，避免带着未知配置进入环境 reset。

checkpoint 保存采用临时文件加 `os.replace()` 的方式发布，避免训练进程中断时把半写文件暴露为 `latest`；workflow 的周期性 `latest` 保存失败会记录错误并在下一保存周期重试，不会中断当前训练循环。加载侧会继续跳过缺失、不可读或结构不兼容的 `latest`；如果平台文件系统或模型目录出现未预期加载异常，workflow 会记录错误并继续使用当前模型参数，保证后续 episode 可以继续启动。

workflow 会先通过安全 helper 调用 `env.reset()` 和 `env.step()`，再归一化返回值：既兼容对象式环境返回和当前封装中的二元返回，也兼容 dict/object step envelope、Gym 四元、Gymnasium 五元和作业文档给出的六元 `env.step()` 返回。归一化阶段会保留对象式 env_obs、对象式 `extra_info` / `extraInfo`，以及直接包含 `frame_state` / `frameState` 或合法动作别名的裸 observation，避免平台 score、全局状态或 observation payload 在进入安全读取 helper 前丢失。dict/object step envelope 会读取常见的 `reward` / `score` / `env_reward` / `envReward`、`terminated` / `done` / `is_done` / `terminal`、`truncated` / `timeout` / `is_truncated`、观测和额外信息字段。随后会兼容 `observation` / `obs` / `_obs` 三类观测字段，以及 `extra_info` / `extraInfo` / `_state` / `state` / `info` 四类额外信息字段，贴合作业文档和 Gym 风格封装的别名写法。如果 reset 抛出平台异常，则跳过当前 episode 并等待下一轮重试；如果 step 抛出异常，则中止当前 episode，避免缺失下一状态时继续构造错误 transition。随后对观测、额外信息、顶层或嵌套的 `frame_no` / `frameNo`、结束标记和采样帧 `legal_action` / `legalAction` / `phaseLegalAction` / `actionMask` / `phaseMask` 使用安全读取；字段缺失、类型异常或 dict-like 对象字段读取抛错时按空 observation、空 extra info 或默认结束状态处理，避免不完整平台响应直接中断 episode。结束和截断字段会显式兼容 bool、有限数值和 true/false 字符串，未知字符串或 NaN/Inf 不会被误判为结束。每局开始时的 `agent.reset()` 也通过安全 helper 执行，失败时跳过当前 episode，防止半初始化的交通统计状态继续进入采样路径。

平台容灾检测通过安全 helper 执行；如果 `handle_disaster_recovery()` 本身抛错，workflow 会记录错误并按无容灾信号继续，避免容灾 SDK 临时异常反向中断训练循环。

训练 workflow 中的预测动作也有兜底链路：模型预测返回空或抛错时回退到 `rule_based_action()`；如果规则策略也异常，则输出 `[0, 0, MIN_GREEN_DURATION]`，避免单次推理异常直接结束整局训练。`predict()` 本身会先把 ObsData batch 安全归一化为 list，并隔离异常 ObsData 属性，避免直接预测调用被坏 batch 或坏属性中断。评估入口 `exploit()` 也复用同样思路，并对外层 observation、`obs` / `observation` / `_obs` 包装、`extra_info` / `extraInfo` / `_state` / `state` / `info` 包装、相位字段和合法动作字段使用安全映射读取；异常 dict-like observation 不会在兜底逻辑之前直接抛出。规则策略兜底失败时返回默认动作，防止评估 episode 因最终 fallback 抛错而失败。进入 `env.step()` 前，workflow 还会对最终动作做一次统一清洗：非决策帧保持 `[None, None, None]`，决策帧强制 `junction_id=0`，相位裁剪到 `0-3`，duration 裁剪到当前联合动作空间可表达的秒数范围。
观测特征处理也通过安全 helper 执行；如果 `observation_process()` 抛错，本次决策会回退到规则动作，采样帧使用零特征占位。非决策帧的交通状态更新失败时只记录错误并继续推进环境，避免单帧异常动态字段中断 episode。

平台训练指标读取也通过容错 helper 执行；`get_training_metrics()` 临时失败或返回非字典结果时只记录错误并按空指标处理，不影响后续模型加载、环境交互和样本生成。

样本处理会保留终局 transition，并对空轨迹、全无效轨迹、缺失 reward、缺失合法动作和无效动作帧做防护；读取 Frame 属性时会隔离 `obs`、`act`、`rew`、`done`、`legal_action` 的属性访问异常，避免单个坏字段导致整段 collector 转换失败。字段定宽转换也会隔离异常 array-like 对象，异常 observation/reward 按零向量处理，畸形 action 对象跳过当前帧。创建 `SampleData` 前会将 `obs`、`act`、`rew` 和 `done` 归一化为固定宽度，清洗 NaN/Inf，并把动作裁剪到单路口、合法相位和模型可表达的 duration 秒数。`done` 会显式兼容 bool、有限数值和 true/false 字符串，终局样本在训练张量中表示 `not_done=0`，Double DQN target 不再引入下一状态 Q 值。

duration reward 的目标时长也与模型动作空间保持一致。当前联合动作头仍保持 80 维，但 20 个 duration 桶覆盖 `8-40` 秒，因此 reward 中的目标 duration 和动作输出使用同一秒数范围，避免训练目标与可输出动作不一致。

workflow 在终局和容灾路径都会通过安全 helper 调用 `sample_process()`；如果样本转换遇到异常输入或平台对象不兼容导致抛错，当前 collector 会被丢弃并记录错误，后续 episode 继续运行。workflow 发送样本时会先复制当前批次，再清空本地 collector，避免异步消费引用被清空；如果样本池或 learner 通道临时抛错，当前批次会记录失败并丢弃，后续 episode 继续运行，便于平台服务恢复后重新产生样本。

workflow 当前监控：

- 总 reward。
- phase reward。
- duration reward。
- data length。
- 平台 score 快照、平均延误、平均排队长度、平均等待时间和信号切换惩罚。

平台 score 监控会同时尝试从 `env.step()` 返回的 `score`、`env_obs.score`、`extra_info.score_info`、`scoreInfo`、`metrics`、`env_info`、`info` 和对象属性中读取常见字段，并兼容 `totalScore`、`avgDelay`、`avgQueueLength`、`avgWaitingTime`、`switchPenalty` 等驼峰字段；读取过程使用有界递归遍历已知指标容器，字段读取失败、循环引用或字段缺失时保留默认值，不影响训练样本发送和模型更新。

workflow、算法训练以及 `Agent` 的模型保存/加载和评估兜底日志都通过容错 helper 执行；如果平台日志后端或 `monitor.put_data()` 临时失败，训练路径会继续运行，避免非核心观测上报故障中断 episode。

所有 agent 入口的 Torch 线程配置都通过容错 helper 执行；如果平台进程已经启动过 Torch 并行运行，`set_num_threads()` 或 `set_num_interop_threads()` 抛出 `RuntimeError` 时会跳过该设置，避免训练或评估在导入阶段失败。

算法监控：

- value loss。
- q value。
- target q value。
- gradient norm。

训练张量进入 `Algorithm.learn()` 后会先把样本批次安全归一化为 list，兼容 generator 式 batch，并在异常 batch 容器无法迭代时记录后跳过当前更新。随后 learner 安全读取 `obs`、`_obs`、`rew`、`act`、`done` 和 `legal_action` 字段，字段属性访问失败时使用默认 observation/action/reward/not_done/legal_action，避免单个异常样本属性导致整批学习跳过。随后 learner 会再次按字段定宽补齐或截断，并统一清洗 NaN/Inf：`obs`、`_obs`、`rew`、`act`、`done`、`legal_action` 和 TD target 中的非有限值都会归零，`done` 还会裁剪到 `[0, 1]`。如果 loss 或梯度范数仍为 NaN/Inf，本次 optimizer step 会被跳过，避免把异常数值写入模型参数。`Agent.learn()` 外层会记录并隔离未预期学习异常，让单个坏 batch 不会直接终止 learner 进程。workflow 统计 reward 分量时同样会把 NaN/Inf 归零，并隔离异常样本对象、异常 `rew` 属性和异常 batch 长度，避免监控聚合本身中断训练。

## 本地验证

当前普通本地环境缺少 `torch`、`kaiwudrl` 和 `common_python`，因此真实训练入口无法运行。但已提供两类本地检查：

```bash
cd codebase
python -m compileall agent_target_dqn tests
python tests/test_target_dqn_static.py
python tests/test_target_dqn_smoke.py
```

当前状态：

- 编译检查通过。
- 无平台依赖特征工具测试通过。
- 静态约束检查通过。
- smoke 脚本在缺少 `torch` 时明确 skip。
- `python train_test.py` 需要 KaiwuDRL 环境。

## 平台实验记录

详细实验流水记录在 `EXPERIMENTS.md`。正式报告中只保留代表性实验和结论。

| 实验 | Commit | 环境配置 | 训练时长 | 模型 ID | 评估得分 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | 平台包为旧模板，非当前 `main` | 平台默认/未记录 | 1h | 任务 ID `194038` | 训练 score 末段约 `1200`，非正式评估 | 训练链路跑通，但旧包 `reward_shaping()` 固定返回 0，不能代表当前代码效果 |
| E02 | 上传包未随结果提供；按当前 `main` `dd1fbcc` 记录 | 平台默认/未记录 | 1h | 任务 ID `206699` | 训练 score 约 `750-780`，非正式评估 | reward 已非零且 loss 下降，但平均延误约 `55-58`、信号变化惩罚为 0，疑似策略过少切相或过度保守 |
| E03 | E03 调参包；按当前 `main` `7d10a9a` 记录 | 平台默认/未记录 | 2h | 任务 ID `206775` | 训练 score 约 `740-770`，非正式评估 | learner step 增至约 `130`，但平均延误、等待和排队基本未改善，单纯标量超参调优无效 |
| E04 | E04 动作监控包；按修复前 `main` `de6a2b4` 记录 | 平台默认/未记录 | 截图时约 3min | 任务 ID `207146` | 训练 score 约 `750-770`，非正式评估 | 动作监控确认 `phase_0_cnt≈30`、其他相位约 0、`phase_switch_cnt=0`，根因是通用 `legal_action` 被误当相位 mask |
| E05 | E05 legal_action 修复包；按当前 `main` `1607528` 记录 | 平台默认/未记录 | 1h | 任务 ID `207778` | 训练 score 约 `760-820`，非正式评估 | gate 修复有效，平均延误降到约 `44-49`，但后半段 phase 2 明显偏置且 `same_phase_ratio` 约 `0.75-0.85` |
| E06 | E06 phase-bias 调参包；按当前 `main` `a0d8efe` 记录 | 平台默认/未记录 | 1h | 任务 ID `208300` | 训练 score 末段约 `1100`，非正式评估 | 当前最佳 Target-DQN 基线，`train_global_step≈87`，平均延误末段约 `20`、等待约 `10`，建议先同包长训或评估 |
| P01 | PPO 首次切换；按平台操作记录 | 平台默认/未记录 | 约 3min | 任务 ID `209360` | 训练未启动 | learner 创建 agent 失败，首错为 `optimizer got an empty parameter list`，已补 PPO model 显式参数注册和参数检查 |
| P02 | PPO 空参数修复后重跑；按 `main` `ecc03cf` 记录 | 平台默认/未记录 | 约 15min | 任务 ID `209365` | 训练未启动 | 已越过空参数错误，但 trainer 5 秒后 `signal_killed`，15 分钟内样本发送 `succ_cnt=0`；已补 PPO 周期性片段样本发送和片段 bootstrap |

E02 相比 E01 的关键进展是 `reward` 不再固定为 0，说明当前 reward 链路已进入 learner；但训练 score 明显偏低，平均延误和等待时间偏高。下一轮应先增加动作分布监控，确认四个相位选择次数、平均 duration 和实际切相次数，再决定是否调 reward 权重或训练更久。

基于 E02 的短训步数较少这一现象，后续 E03 基线已参考常见 PPO 稳定参数做保守调参：折扣因子调为 `0.99`，学习率降到 `3e-4`，epsilon 衰减从 `0.999` 加快到 `0.97`，最终探索率保持 `0.1`，target network 同步间隔从 `500` 调到 `20`。这样一小时短训内既能保持探索，也能更快进入非纯随机策略，并至少完成数次 target 同步。

E03 两小时结果显示，上述超参调整提高了 learner 更新次数，但没有改善平台 score。下一步应补充动作分布监控，直接判断相位选择、duration 和真实切相行为，再改 reward 或动作策略。PPO 备选线已具备短训条件，若后续需要算法对比，可将平台入口切到 `ppo` 跑 10-30 分钟 smoke，再比较 score、loss、entropy 和动作分布。

为定位 E02/E03 的成绩瓶颈，workflow 已补充动作分布监控：每个上报周期统计四个相位选择次数、动作总数、平均/最小/最大 duration、切相次数、切相率和连续同相位比例。E04 短跑确认动作塌缩在 phase 0，且截图时 `train_global_step=0`，因此这不是训练收敛结果，而是合法动作语义处理错误。当前修复将通用 `legal_action` 只作为决策门控，只有显式 `phaseMask` / `phaseLegalAction` 才限制相位；下一轮应先验证四个 phase 计数是否恢复，再继续调 reward。

E05 一小时结果显示 legal_action 修复有效：动作不再固定 phase 0，`phase_switch_cnt` 末段约 `3-5`，平均车辆延误从 E02/E03/E04 的约 `55-58` 降到约 `44-49`。但策略后半段明显偏向 phase 2，`phase_2_cnt` 约 `17-20`，其他 phase 多在 `0-2`，`avg_duration` 约 `30-33`，`same_phase_ratio` 升到约 `0.75-0.85`。由于本次 1h 只有约 `19` 次 learner 更新，后续 E06 基线将 batch 从 `256` 降到 `128`、target 同步间隔从 `20` 降到 `10`，同时放慢 epsilon 衰减并增强相位服务年龄奖励，用于验证 phase 2 偏置能否降低。

E06 一小时结果显示上述调参有效：`train_global_step` 提升到约 `87`，`value_loss` 从约 `1.6` 降到 `0.3`，score 末段约 `1100`，平均车辆延误末段约 `20`，等待约 `10`，排队约 `9`。phase 2 仍是主导相位，但 phase 0/1/3 已恢复参与，平均 duration 降到约 `22-25`，`phase_switch_cnt` 末段约 `15`。因此当前不宜继续立即改代码，下一步应使用同一 E06 包跑 2-3 小时长训或正式评估，先确认稳定性。

PPO 首次平台切换 P01 未进入训练，learner 在创建 agent wrapper 时提示 optimizer 参数列表为空。该问题属于启动阶段错误，不代表 PPO 策略效果。当前已将 PPO MLP 层改为显式注册，并让 optimizer 使用 materialized 参数列表；重跑 PPO 前需要确认平台同步了完整 `agent_ppo/` 目录。

PPO 第二次平台重跑 P02 已越过空参数错误，但 learner trainer 子进程在启动约 5 秒后 `signal_killed`，aisrv 在 15 分钟内没有成功发送样本。后补代码包确认 P02 的 `agent_ppo/` 与 `ecc03cf` 一致，已包含 P01 空参数修复，但尚未包含 P02 之后的片段发送修复。由于截图没有 trainer Python traceback，当前只能先修复可观测的样本饥饿风险：PPO workflow 改为每 32 个决策 transition 发送一次片段样本，并用保留 transition 的 value 为片段末尾 bootstrap。下一轮 P03 应重点确认 `succ_cnt` 是否变为正数；若 trainer 仍被杀，需要补充平台 trainer ERROR 日志。

建议先使用简单环境：

- 晴天。
- 非高峰。
- 无事故。
- 无交通管制。
- 无超速车辆。

简单环境稳定后，再逐步增加随机天气、高峰、事故、管制和超速车辆比例。

## 已知限制

- 当前历史特征只使用 4 帧滚动摘要，尚未包含更长时域的拥堵趋势。
- 奖励权重尚未经过平台训练调优。
- 本地无法验证真实环境交互，因为缺少 KaiwuDRL 依赖。
- 最终成绩仍依赖平台训练、评估和多轮超参数调整。

## 后续改进方向

- 增加更长时域的历史趋势特征。
- 对比 80 维联合 Q 输出和规则基线的评估表现。
- 根据平台评分调整奖励权重。
- 对比规则基线、DQN 和 Target-DQN。
- 保存每次实验配置、模型 ID、评估得分和关键监控曲线。
