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

已处理的高优先级问题：

- Target-DQN 使用独立 target network，并按 `TARGET_UPDATE_FREQ` 同步。
- `reward_shaping()` 已返回非零奖励，基于相位压力、等待时间变化、排队和延误。
- `action_process()` 已将 duration index 映射为实际秒数。
- 训练时已将 duration 秒数转换回 duration head 索引，避免 Q head gather 越界。
- 观测处理已兼容 `position_in_lane["y"]` 的米/毫米单位。
- optimizer 重复初始化已清理，当前使用 Adam。
- `exploit()` 已有规则基线兜底。
- `save_model()` / `load_model()` 已支持默认 checkpoint 路径和首次训练无 latest 模型的情况。
- Target-DQN 已将 `legal_action` 归一化为 4 维相位 mask，用于贪心预测、随机探索和规则兜底选相位。
- `sample_process()` 会把训练样本中的 `legal_action` 设置为下一状态相位 mask，供 Double DQN target 选择下一相位时使用。

仍需关注的问题：

- 平台文档中 `legal_action` 更像是否需要决策的标量门控；当前 Target-DQN 已保守兼容标量门控和 4 维相位 mask，但仍需在真实平台 observation 上确认格式。
- `agent_dqn`、`agent_ppo`、`agent_diy` 仍基本保留模板状态，不是当前主线。
- Target-DQN 使用 phase head 和 duration head 两个独立 Q 输出，不能表达相位和时长的联合组合价值；先调通可以保留，追分时可改为 80 维联合动作。
- 当前状态特征包含占用/速度网格、当前相位、持续时间、剩余时间、相位压力和全局等待/延误统计；仍未显式包含更细的逐车道统计。
- reward 权重尚未经过平台训练调优。
- 本地普通 Python 缺少 `torch`、`kaiwudrl`、`common_python`，真实 `train_test.py` 仍需在平台环境验证。

## 完成期望预估

coding agent 通常可以自动完成：

- 在 `codebase/agent_target_dqn/` 内补齐 Target-DQN 模型、目标网络、奖励、动作映射和评估逻辑。
- 设计固定维度特征，包括 14 条进口车道的占用/速度网格、相位、剩余时间、相位压力和交通统计。
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

- 短期可保留双头输出：phase head 为 4 维，duration head 为 20 维。
- `phase_idx` 必须落在 `0-3`。
- `duration_idx` 必须映射为环境接受的秒数；保守建议从 `8 + duration_idx` 或离散档位开始，避免低于 8 秒的切灯惩罚。
- `legal_action` 需要先通过 `normalize_phase_legal_action()` 转为 4 维相位 mask；如果真实平台只给标量门控，则非零表示四个相位都可选。
- 训练样本里的 `legal_action` 代表 `_obs` 对应的下一状态相位 mask，不是当前已执行动作的 mask。
- 如果改为联合动作，使用 `action_id = phase_idx * 20 + duration_idx`，输出 80 维 Q 值。
- `exploit()` 必须使用贪心或规则兜底，不应使用随机探索。

特征设计：

- 当前 `Config.DIM_OF_OBSERVATION = 576`，其中 560 维为 14 条进口车道乘 20 个格子的占用和速度，8 维为信号相位与持续时间特征，8 维为相位压力与全局交通统计。
- 交通统计特征包括 4 个归一化相位压力、进口车辆数、排队比例、平均等待时间和平均延误。
- 如增加逐车道统计、历史趋势等特征，必须同步修改 `Config.DIM_OF_OBSERVATION` 和模型输入层。
- 坐标单位必须通过真实 observation 确认；若 `position_in_lane["y"]` 是毫米，应使用 `/ 1000` 后再按 `GRID_LENGTH` 分桶。
- 所有特征必须固定长度、无 NaN、范围稳定；空车辆和缺失字段要有默认值。

奖励设计：

- 主要负项：平均延误、平均排队长度、平均等待时间、拥堵。
- 变化项：排队长度下降、等待时间下降可给小额正奖励。
- 切换惩罚：相位变化过快或持续时间小于 8 秒时惩罚。
- 公平性惩罚：某方向长期排队但未放行时惩罚。
- 奖励量级应归一化，避免单项过大导致训练不稳定。

## 实施计划

1. 在 `codebase/` 运行离线检查：`python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_static.py`。
2. 有 `torch` 时运行 `python tests/test_target_dqn_smoke.py`。
3. 在腾讯开悟/KaiwuDRL 环境运行 `python train_test.py`。
4. 通过 `./scripts/package_submission.sh` 生成平台上传包。
5. 使用平台监控调参，记录每次配置、模型 ID、训练时长和评估得分。
6. 将真实实验结果回填到 `PROGRESS.md` 和 `REPORT_DRAFT.md`。
7. 如基础 Target-DQN 稳定，再考虑增加逐车道统计、历史趋势特征或改成 80 维联合动作。

## 测试计划

基础检查：

- 在 `codebase/` 运行 `python train_test.py`。
- 所有新增模块能正常 import。
- `Model.forward()` 对随机 batch 返回两个 head，形状分别为 `[batch, 4]` 和 `[batch, 20]`。
- 无平台依赖时运行 `python tests/test_target_dqn_features.py`，覆盖特征工具和合法动作归一化。

单元测试：

- 构造 fake observation，验证 `observation_process()` 输出固定长度、无 NaN、数值范围合理。
- 构造边界 observation：无车辆、全拥堵、只存在部分车道、相位缺失、`extra_info` 无 `init_state`。
- 验证 `action_process()` 对任意模型输出都能产生合法 `phase_idx` 和合法秒数 duration。
- 验证 `reward_shaping()` 对排队/等待下降给出更高奖励，对频繁切灯给出惩罚。
- 验证 `sample_process()` 中 `_obs`、`done`、`rew`、`act` 的 tensor 形状与 `Algorithm.learn()` 一致。

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
