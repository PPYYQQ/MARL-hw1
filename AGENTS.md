# AGENTS.md

本文件适用于当前目录及其子目录。后续 coding agent 在本作业目录中工作时，应优先遵守这里的说明。

github 目录：https://github.com/PPYYQQ/MARL-hw1.git
如果遇到hugging face的网络问题考虑挂：https://hf-mirror.com
如果遇到网络问题需要代理，可以考虑看batshrc里的setp
写作使用模板：`/home/yongqian/Documents/yongqian/MARL大作业/hw1/icml2022.zip`

## 交流约定

- 每次回复用户必须以 `Harry` 开头。
- 默认用中文沟通，除非用户明确要求英文。
- 先读 `智能交通信号灯调度作业文档.md`，再修改代码。
- 不要改动 `screenshot/`、`icml2022.zip` 和 `codebase/log/`，除非用户明确要求。
- 当前目录已经是 git 仓库，远端为 `git@github.com:PPYYQQ/MARL-hw1.git`；关键修改必须 commit 并 push。
- 代码包实际位于 `codebase/`，运行训练检查时应先进入 `codebase/`。
- 每个关键步骤必须更新 `PROGRESS.md`，平台运行方法记录在 `RUNBOOK.md`，报告草稿记录在 `REPORT_DRAFT.md`。

## 总体概述

本作业是腾讯开悟智能交通信号灯调度任务：开发强化学习智能体，控制单个十字路口的信号灯相位和持续时间，优化车辆通过效率。

核心目标：

- 从环境观测中提取车辆、车道、相位、排队、等待、延误等状态。
- 输出合法动作：`junction_id=0`、`phase_idx in {0,1,2,3}`、合理 `duration`。
- 优化平均车辆延误、平均排队长度、平均等待时间，并减少过于频繁的信号切换。
- 完成 `observation_process()`、`action_process()`、`reward_shaping()`、`sample_process()`、模型、算法、训练 workflow、模型保存与评估逻辑。

当前代码包提供四套 agent：`agent_target_dqn`、`agent_dqn`、`agent_ppo`、`agent_diy`。默认配置选择 `target_dqn`，建议优先完成 `agent_target_dqn`，除非用户明确要求切换算法。

## 当前代码包结构

实际目录结构：

```text
.
├── AGENTS.md
├── 智能交通信号灯调度作业文档.md
├── icml2022.zip
├── screenshot/
└── codebase
    ├── kaiwu.json
    ├── train_test.py
    ├── tests/
    ├── conf
    │   ├── app_conf_intelligent_traffic_lights.toml
    │   ├── algo_conf_intelligent_traffic_lights.toml
    │   └── configure_app.toml
    ├── agent_target_dqn/
    ├── agent_dqn/
    ├── agent_ppo/
    ├── agent_diy/
    └── log/
```

辅助资产：

- `scripts/check_offline.sh`：根目录一键离线检查脚本，会进入 `codebase/` 编译、运行无平台依赖测试、运行 smoke、检查空白并打包。
- `scripts/package_submission.sh`：根目录打包脚本，输出 `dist/marl_hw1_codebase.zip`，只打包 `codebase/`，排除日志、`__pycache__`、checkpoint 和模型 pkl。
- `codebase/tests/test_target_dqn_features.py`：无 `torch` / KaiwuDRL 依赖的功能边界测试。
- `codebase/tests/test_target_dqn_static.py`：无 `torch` / KaiwuDRL 依赖的源码锚点测试。
- `codebase/tests/test_target_dqn_smoke.py`：需要 `torch` 的模型和 Agent smoke 测试；当前普通本地环境缺少 `torch` 时会明确 skip。
- `RUNBOOK.md`：平台运行、打包、调参、常见错误和实验回填指南。
- `EXPERIMENTS.md`：平台训练/评估实验台账；平台跑完后要把真实指标写入这里。
- `REPORT_DRAFT.md`：报告草稿；平台实验后补齐真实分数、曲线和问题分析。
- `PROGRESS.md`：开发过程记录；关键修改、验证和 commit 必须追加记录。

算法入口：

- `codebase/train_test.py`：本地基础检查入口，默认 `algorithm_name = "target_dqn"`。
- `codebase/conf/app_conf_intelligent_traffic_lights.toml`：平台训练默认 `algo = "target_dqn"`。
- `codebase/conf/algo_conf_intelligent_traffic_lights.toml`：四套算法到 agent/workflow 的映射。
- `codebase/conf/configure_app.toml`：样本池、batch size、模型保存和同步配置。

Target-DQN 关键文件：

- `codebase/agent_target_dqn/agent.py`：预测、评估、特征处理、动作转换、保存加载。
- `codebase/agent_target_dqn/algorithm/algorithm.py`：Target-DQN 学习逻辑。
- `codebase/agent_target_dqn/model/model.py`：PyTorch Q 网络。
- `codebase/agent_target_dqn/feature/definition.py`：样本结构、奖励、样本处理。
- `codebase/agent_target_dqn/feature/preprocessor.py`：路网初始化、等待时间、行驶距离、车道车辆统计。
- `codebase/agent_target_dqn/feature/traffic_utils.py`：进口车道、出口车道、相位车道组工具。
- `codebase/agent_target_dqn/workflow/train_workflow.py`：episode 循环、样本发送、监控、模型保存。

## 代码审阅结论

当前代码包来自官方教学模板。主线已集中在 `agent_target_dqn`，并完成了一批从模板到可训练闭环所需的基础修复。后续仍应优先维护 `agent_target_dqn`，除非用户明确要求切换算法。

本次完整代码包复核结论：

- 仓库现在包含平台完整 `codebase/`、本地 `tests/`、打包/检查脚本和文档台账；`codebase/log/` 是运行产物，不应纳入开发修改范围。
- `codebase/conf/app_conf_intelligent_traffic_lights.toml` 和 `codebase/train_test.py` 都默认选择 `target_dqn`，当前平台入口与主线一致。
- `agent_target_dqn` 是唯一被系统性修复和测试覆盖的算法包；`agent_dqn`、`agent_ppo`、`agent_diy` 仍应视为平台模板/备选代码，除非用户明确切换算法，不要把后续改动扩散过去。
- `tests/test_target_dqn_features.py` 和 `tests/test_target_dqn_static.py` 是当前普通本地环境最可靠的回归检查；`tests/test_target_dqn_smoke.py` 依赖 `torch`，平台或安装 PyTorch 后再补跑。
- `tests/test_hyperparams_static.py` 用于锁定 E03 调参基线，防止 Target-DQN 和 PPO 参数退回平台模板值。
- `scripts/package_submission.sh` 当前只打包 `codebase/`；根目录文档、截图、日志、checkpoint、模型 pkl 和 `icml2022.zip` 不会进入提交包。
- `.gitignore` 已忽略 `__pycache__`、`.pyc`、`dist/`、`codebase/log/` 和 checkpoint；如果本地看到缓存文件，不要手动纳入提交。
- 普通本地环境仍无法运行真实 `python train_test.py`，因为缺少 `kaiwudrl`、`common_python` 和 `torch`；真实 reset/step、样本池、模型池和评分必须在腾讯开悟/KaiwuDRL 环境确认。

已处理的高优先级问题：

- Target-DQN 使用独立 target network，并按 `TARGET_UPDATE_FREQ` 同步。
- `reward_shaping()` 已返回非零奖励，基于相位压力、等待时间变化、排队和延误。
- `reward_shaping()` 对终局或异常 observation 缺失 `frame_state` / `frameState` / `vehicles` 的情况保守返回零奖励，避免训练循环崩溃。
- `reward_shaping()` 会将极端延误惩罚限制到 `REWARD_DELAY_CAP=300`，并将每个 reward 分量裁剪到 `[-REWARD_CLIP, REWARD_CLIP]`，降低 TD target 爆炸风险。
- 训练 workflow 调用 reward shaping 时会隔离异常，奖励计算失败会记录错误并使用 `(0.0, 0.0)`。
- `FeatureProcess.update_traffic_info()` 对缺失 `frame_state` / `frameState`、缺失 `vehicles` 或畸形车辆记录会保守跳过，避免异常帧中断特征处理，并会兼容 `init_state` / `initState` 初始化路网。
- `FeatureProcess` 会清洗 `frame_no`、`frame_time`、车辆 ID、车速和车道位置，等待时间/行驶距离/车道计数统计遇到异常动态字段会跳过单车而不是中断整帧。
- `FeatureProcess.init_road_info()` 已兼容模板字段 `j_id/e_id/l_id/v_config_id`、文档式字段 `junction_id/edge_id/lane_id/vehicle_config_id` 以及常见驼峰字段 `junctionId/edgeId/laneId/vehicleConfigId`、`laneConfigs`、`vehicleConfigs`、`enterLanesOnDirections`，避免真实 init_state 使用不同命名时路网配置被漏载。
- 训练 workflow 调用 `observation_process()` 和非决策帧 `update_traffic_info()` 时会隔离异常，特征处理失败会回退到规则动作和零特征样本。
- `observation_process()`、`rule_based_action()` 和共享交通统计工具会保守处理缺失 `frame_state` / `frameState`、缺失 `vehicles`、缺失 `obs` 包装和畸形车辆/相位记录。
- workflow、Agent、reward、preprocessor 和交通统计 helper 的关键协议字段读取已兼容普通 dict 与属性对象，贴合作业文档中的 Observation / FrameState / Vehicle / Phase / Lane 消息形态；标量字段不会被误当成协议对象，单个 dict 记录、dict-of-records 和单个非 dict 对象式 vehicles/phases/lanes 容器都会按有效记录处理。
- 共享交通统计工具不再要求车辆记录必须包含 `target_junction` / `targetJunction`；缺失时会按单路口进口车道车辆处理，等待时间统计也会复用同一默认目标路口，贴合作业文档字段列表。
- 车辆 `junction` / `junctionId` / `target_junction` / `targetJunction` 会统一清洗为有限整数，兼容字符串形式的 `"0"` / `"-1"`，避免字符串 `"-1"` 被误判为有效目标路口或路口内车辆。
- 路网初始化、相位压力、车道车辆统计和 observation 中的车辆配置查找会清洗数字 ID 字段，兼容 `junction_id` / `junctionId`、`edge_id` / `edgeId`、`lane_id` / `laneId`、`vehicle_config_id` / `vehicleConfigId`、车辆 `lane` / `laneId` 和 `v_config_id` / `vConfigId` / `vehicleConfigId` 以字符串形式返回，并兼容车辆配置速度字段 `max_speed` / `maxSpeed`。
- `frame_state.lanes` 已作为车辆列表缺失或稀疏时的 fallback 信号；观测交通统计、逐车道统计、规则兜底和 reward 都能复用 `lane_id` / `laneId`、`v_count` / `vCount` / `vehicle_count`、`queue_length` / `queueLength` / `queue_count`、`congestion` / `congestionLevel` 聚合出的相位压力。
- `action_process()` 已将 duration index 映射为实际秒数。
- `predict()` 对空 observation batch 会返回空列表；`action_process()` 会固定 `junction_id=0` 并清洗异常相位/时长索引，确保输出合法动作。
- `predict()` 会先把 ObsData batch 安全归一化为 list，并隔离异常 ObsData 属性，避免直接预测调用被坏 batch 或坏属性中断。
- 训练时已将 `[phase_idx, duration_seconds]` 转换为 80 维联合动作索引，避免 Q head gather 越界。
- 观测处理已兼容 `position_in_lane["y"]` / `positionInLane["y"]` 的米/毫米单位。
- 交通统计工具会清洗车辆 `speed`、`waiting_time` / `waitingTime`、`delay`、历史趋势和相位压力中的 NaN/Inf，避免异常车辆字段污染 reward、规则兜底和观测统计。
- 相位时间特征、相位年龄、reward 公平性项和 workflow `frame_no` 会清洗 NaN/Inf/Overflow；相位特征会兼容 `s_id` / `signal_id` / `signalId`、`phase_id` / `phase_idx` / `phase` / `current_phase` / `currentPhase`、`remaining_duration` / `remaining_time` / `remainingTime` 等别名，workflow 帧号会从顶层 `frame_no` / `frameNo` 或嵌套 `extra_info` / `extraInfo` / `_state` / `state` / `info` 中回退读取，避免异常相位字段或帧号中断推理和奖励计算。
- `observation_process()` 会在返回前统一清洗最终特征向量，保证长度为 `Config.DIM_OF_OBSERVATION` 且非有限值归零。
- `Model.forward()` 会把单条一维 observation 转成 batch，并对异常长度或 ragged Python batch observation 做补零或截断，避免输入形状差异直接触发线性层错误。
- `Model.forward()` 的 `_prepare_input()` 会统一清洗 NaN/Inf，异常 array-like observation 转换失败会补零，避免直接模型调用产生非有限 Q 值。
- optimizer 重复初始化已清理，当前使用 Adam。
- `exploit()` 已有规则基线兜底。
- `save_model()` / `load_model()` 已支持默认 checkpoint 路径和首次训练无 latest 模型的情况。
- 训练 workflow 定期保存 `model.ckpt-latest.pkl`，与每局开始的 `load_model(id="latest")` 保持一致。
- `save_model()` 会先写临时 checkpoint，再用 `os.replace()` 原子发布，降低中断时留下半写 `latest` 的风险。
- `load_model(id="latest")` 会跳过缺失、不可读、payload 非 dict 或结构不兼容的 checkpoint，避免坏 `latest` 卡住每局加载。
- workflow 周期性保存 `latest` checkpoint 失败时只记录错误并节流重试，不会中断当前训练循环。
- workflow 每局加载 `latest` checkpoint 时会隔离异常，意外加载失败只记录错误并继续使用当前模型参数。
- 四个 agent 入口已对 `torch.set_num_threads()` 和 `torch.set_num_interop_threads()` 的 `RuntimeError` 做容错，避免平台预先启动 Torch 并行后导入 agent 崩溃。
- workflow 读取平台训练指标时会隔离异常，`get_training_metrics()` 临时失败只记录错误并返回空指标。
- workflow 发送训练样本时会隔离 `send_sample_data()` 异常，样本通道临时失败只记录错误并继续后续训练循环。
- workflow 进行终局或容灾样本转换时会隔离 `sample_process()` 异常，转换失败只丢弃当前 collector，不再中断后续 episode。
- Target-DQN 已将 `legal_action` / `legalAction` / `phaseLegalAction` / `actionMask` / `phaseMask` 等字段归一化为 4 维相位 mask，用于贪心预测、随机探索和规则兜底选相位。
- Agent 推理侧对全零相位 mask 会回退为四个相位都可选，joint action mask 的空行也会回退为全动作可选，避免直接调用 `predict()` / `exploit()` 时无可采样动作导致崩溃。
- 训练 workflow 已用同一归一化逻辑判断是否需要决策，兼容平台文档中的 `int32` 标量门控和 4 维相位 mask。
- 训练 workflow 会归一化 `env.reset()` 的对象式返回、二元 tuple 返回和 `env.step()` 的对象式返回、dict/object step envelope、二元、Gym 四元、Gymnasium 五元、作业文档六元 tuple 返回，兼容当前封装、常见环境封装与作业文档形式。
- 训练 workflow 会隔离 `env.reset()` 和 `env.step()` 抛出的平台异常；reset 失败跳过当前 episode，step 失败中止当前 episode。
- 训练 workflow 对 reset/step 返回的 `observation` / `obs` / `_obs`、`extra_info` / `extraInfo` / `_state` / `state` / `info`、顶层或嵌套的 `frame_no` / `frameNo`、结束标记和采样帧 `legal_action` / `legalAction` / `phaseLegalAction` / `actionMask` / `phaseMask` 会安全读取；如果平台直接返回带 `frame_state` / `frameState` 和合法动作别名的裸 observation dict 或对象，也会按原始 observation 处理，避免被误归一化为空观测。
- step/reset 归一化阶段会保留对象式 env_obs 和对象式 `extra_info` / `extraInfo`，并兼容 step envelope 的 `envReward`，避免先前字段读取兼容逻辑在进入安全 helper 前丢失平台 score 或 observation payload。
- 训练 workflow 对 env_obs/obs 映射读取异常会统一回退默认值，避免异常 dict-like 返回对象中断预测门控和状态解析。
- 训练 workflow 会显式解析 `terminated` / `done` / `is_done` / `terminal` 与 `truncated` / `timeout` / `is_truncated` 等结束或截断别名，并兼容 bool、数值和字符串形式，避免 `"False"` 这类非空字符串被误判为结束。
- 训练 workflow 调用 `agent.reset(env_obs)` 时会隔离异常，reset 失败会记录错误并跳过当前 episode，避免半初始化状态继续采样。
- 训练 workflow 读取 `train_env_conf.toml` 时会隔离配置校验工具异常，读取失败会记录错误并清晰退出入口。
- 训练 workflow 调用平台容灾 helper 时会隔离异常，容灾检测失败会记录错误并按无容灾信号继续。
- 训练 workflow 的预测动作通过 `_predict_action()` 统一处理，模型预测返回空或异常时会回退到规则策略，规则策略再失败则输出 `[0, 0, MIN_GREEN_DURATION]`。
- 训练 workflow 在进入 `env.step()` 前会统一清洗最终动作：非决策帧保持 `[None, None, None]`，决策帧强制 `junction_id=0`、相位裁剪到 `0-3`、duration 裁剪到模型动作空间可表达秒数。
- 训练 workflow 抛错时会保留原始异常信息和异常链，便于平台日志定位真实崩溃点。
- workflow 和 `Algorithm.learn()` 的日志/监控上报已隔离异常，`logger` 或 `monitor.put_data()` 后端失败不会中断训练。
- `Agent` 的 `exploit()`、`save_model()` 和 `load_model()` 日志调用已隔离异常，日志后端失败不会打断评估兜底或 checkpoint 流程。
- `Agent.exploit()` 的规则策略兜底也会隔离异常；规则策略失败时返回 `[0, 0, MIN_GREEN_DURATION]`，避免评估入口因最终兜底失败崩溃。
- `Agent.exploit()`、`observation_process()` 和 `rule_based_action()` 的关键映射读取会走安全 helper，异常 dict-like observation 不会绕过评估兜底。
- `Agent.exploit()` 会兼容评估入口传入的 `obs`、`observation`、`_obs` 观测包装和 `extra_info`、`extraInfo`、`_state`、`state`、`info` 额外信息包装，避免平台字段别名导致评估只看到空 observation。
- 训练 workflow 发送样本时会传递 `g_data` 的浅拷贝，再清理本地列表，避免异步消费时引用被清空。
- 训练 workflow 的进度日志只在 episode 结束或真实预测计数达到间隔时打印，避免无决策帧刷屏。
- 训练 workflow 会安全提取平台 `score`、`score_info`、`scoreInfo`、`metrics`、`env_info`、`info` 或对象属性里的 `env_score`、`avg_delay`、`avg_queue_length`、`avg_waiting_time` 和 `switch_penalty`，并兼容 `totalScore`、`avgDelay`、`avgQueueLength`、`avgWaitingTime`、`switchPenalty` 等驼峰评分字段；读取过程会用有界递归遍历嵌套指标容器，用于平台监控和调参记录。
- `sample_process()` 会把训练样本中的 `legal_action` 设置为下一状态相位 mask，供 Double DQN target 选择下一相位时使用。
- `sample_process()` 对空轨迹、全无效轨迹、缺失 reward 和无效动作帧会保守跳过或补零，避免样本转换边界崩溃。
- `sample_process()` 读取 Frame 的 `obs`、`act`、`rew`、`done`、`legal_action` 属性时会隔离属性访问异常，坏属性只影响当前字段或当前帧，不再导致整段 collector 转换失败。
- `sample_process()` 会显式解析 `done` 的 bool、数值和 true/false 字符串形式，避免字符串终局标记误写为非终局 TD target。
- `sample_process()` 的字段定宽转换会隔离异常 array-like 对象，异常 observation/reward 按默认零向量处理；畸形 action 对象会跳过当前帧。
- `sample_process()` 在创建 `SampleData` 前会对 `obs`、`act`、`rew` 和 `done` 做定宽归一化、NaN/Inf 清洗和动作边界裁剪，避免畸形轨迹污染样本池。
- `normalize_phase_legal_action()` 会将合法动作里的 NaN/Inf 显式归零，避免非有限值被误判为可选相位。
- `normalize_phase_legal_action()` 会隔离异常 array-like 合法动作输入；转换失败时按保守全相位可选处理，避免 workflow 决策门控、Agent 推理和样本转换被坏 mask 中断。
- workflow 聚合样本批次 reward 监控时会隔离异常样本、异常 `rew` 字段和异常 batch 长度；坏样本只按零 reward 统计。
- `Algorithm.learn()` 会清洗 observation、reward、action、not_done、legal_action 和 TD target 中的 NaN/Inf，workflow reward 监控也会把非有限值归零。
- `Algorithm.learn()` 会先把样本批次安全归一化为 list，兼容 generator 式批次，并隔离异常 batch 容器。
- `Algorithm.learn()` 读取样本字段时会隔离属性访问异常，坏字段使用默认 observation/action/reward/not_done/legal_action，避免单个异常样本属性导致整批学习跳过。
- `Algorithm.learn()` 会在 `torch.stack()` 前对 obs、_obs、action、reward、done 和 legal_action 做定宽补齐/截断，避免畸形样本长度不一致时训练崩溃。
- `Algorithm.learn()` 遇到非有限 loss 或梯度范数时会跳过本次 optimizer step，避免 NaN/Inf 参数污染模型。
- `Agent.learn()` 会隔离 `Algorithm.learn()` 未预期异常，记录 `learn failed` 并跳过当前 batch，避免 learner 因单批异常样本退出。
- `exploit()` 强制走贪心推理且不衰减训练用 `_eps`，避免评估调用改变训练探索状态。
- 观测和 reward 已加入相位服务年龄，用于降低高压相位长期不被服务的风险。
- Target-DQN 已从 phase/duration 双头改为 80 维联合动作 Q 头，可表达相位和时长组合价值。
- E02 平台短训结果位于 `dqn2/`：任务 ID `206699`，2026-06-08 21:41:46 到 22:42:25 跑满 1h，reward 约 `-2.8` 到 `-3.0`，`value_loss` 从约 `2.1` 降到 `0.3`，说明最新包的非零 reward 和 learner 更新链路已跑通。
- E03 参数基线已参考常见 PPO 稳定配置调整：Target-DQN 使用 `GAMMA=0.99`、`LR=3e-4`、`EPSILON_DECAY=0.97`、`END_EPSILON_GREEDY=0.1`、`TARGET_UPDATE_FREQ=20`，PPO 模板使用 `lr=3e-4`、`gamma=0.99`、`lambda=0.95`、`clip=0.2`、`entropy=0.01`、`grad_clip=0.5`。
- E03 平台结果位于 `dqn3/`：任务 ID `206775`，2026-06-09 00:51:34 到 02:52:14 跑满 2h，`train_global_step` 约 `130`，score 约 `740-770`，平均延误约 `55-58`，等待约 `27-30`，排队约 `9-10`；超参调整增加了 learner step，但没有改善成绩。

仍需关注的问题：

- E02/E03 训练 score 均只有约 `740-780`，平均延误约 `55-58`、等待约 `26-30`；后续优先检查动作分布、phase switch 次数和 duration 分布，不要继续只做标量超参搜索。
- 平台的平均信号变化惩罚为 `0` 不能单独证明策略完全不切相，因为当前动作最短 duration 已限制为 8 秒；必须额外上报真实 `phase_switch_cnt` 和 `same_phase_ratio`。
- 平台文档中 `legal_action` 更像是否需要决策的标量门控；当前 Target-DQN 已保守兼容标量门控、4 维相位 mask 和常见合法动作字段别名，但仍需在真实平台 observation 上确认是否存在相位级 mask 语义。
- `agent_dqn`、`agent_ppo`、`agent_diy` 仍基本保留模板状态，不是当前主线；`agent_ppo` 虽已调参并切到 Adam，但 reward、policy loss 和 entropy loss 仍未完整实现。
- 当前状态特征包含占用/速度网格、当前相位、相位服务年龄、持续时间、剩余时间、相位压力、全局等待/延误统计、一帧交通趋势、4 帧滚动交通历史和逐车道车辆/排队/等待统计。
- 作业文档定义了 `frame_state.lanes` 的 `lane_id`、`v_count`、`congestion`、`queue_length` 字段；当前已接入 lanes fallback，并兼容常见驼峰/别名字段，但仍需在真实 observation 上确认字段单位和是否存在 protobuf repeated wrapper 等特殊容器形态。
- 当前列表字段解析支持 list、tuple、iterable、单条 dict 记录、dict-of-records 和单个非 dict 协议对象；dict 解析只在出现已知协议字段或值明显是记录/列表时展开，避免盲目把任意 dict 当作有效车辆。
- reward 权重尚未经过平台训练调优。
- 本地普通 Python 缺少 `torch`、`kaiwudrl`、`common_python`，真实 `train_test.py` 仍需在平台环境验证。

## 完成期望预估

coding agent 通常可以自动完成：

- 在 `codebase/agent_target_dqn/` 内补齐 Target-DQN 模型、目标网络、奖励、动作映射和评估逻辑。
- 设计固定维度特征，包括 14 条进口车道的占用/速度网格、相位、相位服务年龄、剩余时间、相位压力、交通统计、交通趋势、滚动历史和逐车道统计。
- 实现规则基线兜底：按相位对应车道组的排队/等待压力选择相位和时长。
- 编写本地 fake observation 测试、模型 forward 测试、动作合法性测试和样本转换测试。
- 在平台环境可用时跑 `python train_test.py` 和短回合 smoke test。
- 根据日志和监控做有限超参数调优。

coding agent 无法单独保证：

- 在没有 KaiwuDRL SDK 或腾讯开悟平台环境时完成真实训练。
- 在本地离线环境中得到平台最终评分。
- 访问平台、创建评估任务、提交模型、读取云端监控，除非用户提供账号、环境和命令。
- 保证一次训练达到高分；强化学习成绩需要多轮训练、评估和调参。

合理完成目标分层：

- 第一阶段：`target_dqn` 导入、模型 forward、动作转换、样本处理都通过本地检查。
- 第二阶段：规则兜底策略能稳定评估，不出现非法动作或频繁崩溃。
- 第三阶段：Target-DQN 能训练并保存模型，reward/loss 指标可监控。
- 第四阶段：在平台上通过多轮训练和评估，逐步优化最终成绩。

## 计算资源预估

本地开发与测试：

- CPU：2-4 核即可。
- 内存：8 GB 可用，16 GB 更稳。
- GPU：不需要。
- 用途：静态检查、导入测试、特征和动作单元测试、模型 forward、短回合 smoke test。

最小训练调通：

- CPU：4-8 核。
- 内存：16 GB。
- GPU：可选；小型 MLP 的 DQN/Target-DQN 对 GPU 依赖不强。
- 时间：30 分钟到 3 小时，用于确认 reward、loss、模型保存和评估流程能跑通。

成绩调优训练：

- 推荐使用腾讯开悟平台或等价分布式训练环境。
- CPU：8-16 核。
- 内存：16-32 GB。
- GPU：可选 4-8 GB 显存；如果模型较小，瓶颈更可能在环境交互而不是神经网络。
- 时间：6-24 小时，通常需要多组环境配置和奖励权重对比。

## 需要用户或平台支持

- KaiwuDRL/腾讯开悟运行环境、依赖安装方式和平台入口。
- 训练/评估命令、提交要求、模型包格式和评分反馈。
- 允许使用的算法范围、训练时长、是否允许预训练模型。
- 平台监控截图或日志，尤其是 reward、loss、平均延误、排队长度、等待时间、切换惩罚、CPU/GPU/内存。
- 真实 observation 样例最好保留一份脱敏 JSON，便于本地写 fake 测试和确认坐标单位。

## 推荐项目实施架构

主线开发目录为 `codebase/agent_target_dqn/`。

模块职责：

- `feature/traffic_utils.py`：维护车道 ID、相位到车道组映射、进口/出口/路口判断。
- `feature/preprocessor.py`：维护跨帧统计，例如等待时间、车辆行驶距离、车道车辆数、上一相位和上一动作。
- `feature/definition.py`：实现 reward 和 sample 处理；不要在这里塞大量观测解析代码。
- `agent.py`：只做 agent 接口编排，包括 `observation_process()`、`predict()`、`exploit()`、`action_process()`。
- `model/model.py`：只定义网络结构，保持输入输出形状清晰。
- `algorithm/algorithm.py`：只负责 TD target、loss、反向传播、目标网络同步和监控指标。
- `workflow/train_workflow.py`：只负责环境交互、样本发送、模型保存和训练监控。

动作设计：

- 当前使用 80 维联合动作 Q 头：`action_id = phase_idx * 20 + duration_idx`。
- `phase_idx` 必须落在 `0-3`。
- `duration_idx` 必须通过 `Config.duration_index_to_seconds()` 映射为环境接受的秒数；当前 20 个 duration 桶覆盖 `MIN_GREEN_DURATION=8` 到 `MAX_GREEN_DURATION=40`。
- `legal_action` / `legalAction` / `phaseLegalAction` / `actionMask` / `phaseMask` 需要先通过 `normalize_phase_legal_action()` 转为 4 维相位 mask；如果真实平台只给标量门控，则非零表示四个相位都可选。
- 训练 workflow 判断是否调用 `predict()` 时也必须走 `normalize_phase_legal_action()`，不要直接写 `legal_action[0]`。
- 训练样本里的 `legal_action` 代表 `_obs` 对应的下一状态相位 mask，不是当前已执行动作的 mask。
- 相位级 mask 会展开成 80 维 joint mask，训练和预测都只在合法相位对应的动作组合中选动作。
- 全零 `legal_action` 在 workflow 中仍表示当前帧不需要决策；如果外部直接调用 `Agent.predict()` / `Agent.exploit()` 绕过 workflow，Agent 内部会把空 mask 兜底为可选全集，避免随机探索的空集合采样或贪心推理的无效 mask。
- `exploit()` 必须使用贪心或规则兜底，不应使用随机探索，也不应修改训练 epsilon 状态。

特征设计：

- 当前 `Config.DIM_OF_OBSERVATION = 638`，其中 560 维为 14 条进口车道乘 20 个格子的占用和速度，8 维为信号相位与持续时间特征，4 维为相位服务年龄，8 维为相位压力与全局交通统计，8 维为一帧交通趋势，8 维为 4 帧滚动交通历史，42 维为逐车道统计。
- 相位服务年龄特征表示四个相位距上次服务的归一化帧数。
- 交通统计特征包括 4 个归一化相位压力、进口车辆数、排队比例、平均等待时间和平均延误。
- 趋势特征包括 4 个相位压力变化、进口车辆数变化、排队比例变化、平均等待时间变化和平均延误变化。
- 历史特征包括最近 4 帧交通摘要的平均相位压力、进口车辆数、排队比例、平均等待时间和平均延误。
- 逐车道统计特征包括 14 条进口车道的车辆数、排队数和平均等待时间。
- 如继续增加更长历史窗口等特征，必须同步修改 `Config.DIM_OF_OBSERVATION` 和模型输入层。
- 坐标单位必须通过真实 observation 确认；若 `position_in_lane["y"]` / `positionInLane["y"]` 是毫米，应使用 `/ 1000` 后再按 `GRID_LENGTH` 分桶。
- 原始 observation、`frame_state` / `frameState`、vehicle、phase、`extra_info` / `extraInfo` 可能是 dict，也可能是平台协议对象；新增解析逻辑必须使用安全字段读取 helper，不要直接假设 `.get()` 或下标访问一定可用。注意区分协议对象和标量，避免把 `1`、`False` 这类坏字段当成有效 observation。
- `frame_state.lanes` 是文档中明确存在的车道聚合字段；当前会在 `vehicles` 推导压力为空时，用 lanes 聚合相位压力，并将 lanes 统计与逐车道统计特征取最大值合并，支持 `lane_id` / `laneId`、`v_count` / `vCount`、`queue_length` / `queueLength` 和 `congestion` / `congestionLevel`，保持 638 维特征总长不变。
- 车辆记录可能没有 `target_junction` / `targetJunction`，进口车道判断和交叉口等待时间统计不能强依赖该字段；单路口场景下缺失且能识别为进口车道时按目标路口 0 处理，字符串形式的路口、车道和车辆配置 ID 要先清洗为整数，并兼容 `vehicleId`、`laneId`、`junctionId`、`positionInLane`、`waitingTime` 等常见驼峰字段，无法识别目标的畸形记录应跳过。
- 所有特征必须固定长度、无 NaN、范围稳定；空车辆和缺失字段要有默认值。

奖励设计：

- 主要负项：平均延误、平均排队长度、平均等待时间、拥堵。
- 变化项：排队长度下降、等待时间下降可给小额正奖励。
- 切换惩罚：相位变化过快或持续时间小于 8 秒时惩罚。
- 公平性惩罚：某方向长期排队但未放行时惩罚。
- 当前公平性实现会跟踪四个相位上次服务帧，高压且长时间未服务的相位被选中时给小额正奖励，否则给小额负项。
- duration reward 的目标时长必须限制在模型动作空间可表达范围内；当前 20 个 duration 桶覆盖 `MIN_GREEN_DURATION` 到 `MAX_GREEN_DURATION`。
- 奖励量级应归一化，当前通过 `REWARD_DELAY_CAP` 限制极端延误项，并通过 `REWARD_CLIP` 裁剪每个 reward 分量。

## 实施计划

1. 每次修改前确认 `codebase/conf/app_conf_intelligent_traffic_lights.toml` 和 `codebase/train_test.py` 仍指向 `target_dqn`。
2. 在 `codebase/` 运行局部检查：`python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py`、`python tests/test_target_dqn_static.py`。
3. 在根目录运行 `./scripts/check_offline.sh`，确认编译、静态/功能测试、smoke skip、空白检查和打包内容检查通过。
4. 有 `torch` 时补跑 `cd codebase && python tests/test_target_dqn_smoke.py`。
5. 在腾讯开悟/KaiwuDRL 环境运行 `cd codebase && python train_test.py`。
6. 通过 `./scripts/package_submission.sh` 生成平台上传包，并确认压缩包里只有平台需要的 `codebase/` 内容。
7. 使用平台监控调参，记录每次配置、模型 ID、训练时长和评估得分。
8. 将真实实验结果回填到 `EXPERIMENTS.md`、`PROGRESS.md` 和 `REPORT_DRAFT.md`。
9. 如基础 Target-DQN 稳定，再根据平台指标优先处理 reward 权重、duration 分桶和训练配置调参。

## 测试计划

基础检查：

- 普通本地环境优先运行 `./scripts/check_offline.sh`。
- 平台环境再在 `codebase/` 运行 `python train_test.py`。
- 所有新增模块能正常 import。
- `Model.forward()` 对随机 batch 返回一个 joint Q head，形状为 `[batch, 80]`。
- 无平台依赖时运行 `python tests/test_target_dqn_features.py`，覆盖特征工具和合法动作归一化。

单元测试：

- 构造 fake observation，验证 `observation_process()` 输出固定长度、无 NaN、数值范围合理。
- 构造边界 observation：无车辆但 lanes 有值、全拥堵、只存在部分车道、相位缺失、`extra_info` 无 `init_state`。
- 验证 `action_process()` 对任意模型输出都能产生合法 `phase_idx` 和合法秒数 duration。
- 验证 `reward_shaping()` 对排队/等待下降给出更高奖励，对频繁切灯给出惩罚。
- 验证 `sample_process()` 中 `_obs`、`done`、`rew`、`act` 的 tensor 形状与 `Algorithm.learn()` 一致。
- 验证 `sample_process()` 对短/长 observation、短 reward、NaN/Inf 和字符串 done 的归一化结果。

集成测试：

- 使用真实环境执行一次 `reset()`、一次 `predict()`、一次 `action_process()`、一次 `env.step()`。
- 跑 1-3 个短 episode，确认不会因非法动作、shape 错误、保存路径错误中断。
- 测试 `save_model()` 后再 `load_model()`，同一 observation 的 `exploit()` 输出应稳定。
- 在训练模式确认 epsilon 探索存在，在评估模式确认 `exploit()` 不随机。

训练与评估检查：

- 监控 `reward`、`value_loss`、`q_value`、`target_q_value`、平均延误、平均排队长度、平均等待时间、切换惩罚。
- 检查 `sample_production_and_consumption_ratio`，避免样本生产和训练严重失衡。
- 关注 CPU、内存和 GPU 使用率，内存接近上限时降低 batch size 或 replay buffer。
- 对比规则基线、DQN/Target-DQN、不同奖励权重和不同 duration 映射。
- 保存每次有效实验的配置、模型 ID、训练时长、评估得分和关键日志。

## 完成标准

- 本地基础测试通过。
- 短回合环境交互不崩溃。
- 训练能产生样本、更新模型、保存 checkpoint。
- Target-DQN 目标网络是真正独立网络，并按 `TARGET_UPDATE_FREQ` 同步。
- `reward_shaping()` 不再返回全零奖励。
- `action_process()` 输出合法 phase 和实际可接受 duration。
- `exploit()` 能加载 checkpoint 并输出稳定合法动作。
- 至少保留一套规则基线作为强化学习模型异常时的兜底策略。
- 最终说明文档包含方法、特征、奖励、训练配置、评估结果和问题分析。
