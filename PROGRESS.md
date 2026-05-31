# 作业进度记录

本文档用于记录智能交通信号灯调度作业的关键修改、验证结果和后续计划。每个关键代码修改应配套一个 git commit。

## 当前目标

- 主线算法：`codebase/agent_target_dqn`
- 优先完成：Target-DQN 可训练闭环、有效奖励、合法动作、基础测试、实验记录
- 暂缓范围：PPO 完整实现、DIY 从零实现、大规模模型重构

## 2026-05-30

### Step 0 - 仓库和进度追踪初始化

- 状态：完成
- Commit：`9f3964f Initialize traffic light assignment workspace`
- 内容：
  - 确认当前目录原先不是 git 仓库。
  - 确认 GitHub 远端 `https://github.com/PPYYQQ/MARL-hw1.git` 可访问但未返回 HEAD。
  - SSH 认证可用，已将远端切换为 `git@github.com:PPYYQQ/MARL-hw1.git`。
  - 增加 `.gitignore`，避免提交运行日志、缓存、checkpoint、截图和本地论文模板压缩包。
  - 增加 `PROGRESS.md`，后续每个关键修改都记录目的、文件、验证和 commit。
- 验证：
  - 已执行 `git init -b main`。
  - 已执行首次 commit。
  - 已执行 `git push -u origin main`，GitHub `main` 分支已创建。
- 下一步：
  - 修复 `agent_target_dqn` 的目标网络、奖励、动作时长映射和基础健壮性。

### Step 1 - Target-DQN 目标网络和模型骨干

- 状态：完成
- Commit：`b324e4a Implement Target-DQN target network backbone`
- 内容：
  - 将 `agent_target_dqn` 的目标网络改为独立 `deepcopy`，避免与在线网络共享参数对象。
  - 增加 `update_target_q()`，按 `TARGET_UPDATE_FREQ` 定期同步目标网络。
  - 将 Q 网络骨干从过小的 `560-16-32-16` 调整为 `560-256-128-64`。
  - 将优化器改为单个 Adam，去掉重复初始化。
  - 将学习率降到 `5e-4`，epsilon 下限降到 `0.05`，放慢 epsilon 衰减。
  - 增加 `monitor` / `logger` 空值防御，方便本地测试。
  - 修正 `.gitignore`，避免误忽略平台源码目录 `agent_*/model/`。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn`，语法编译通过。
- 下一步：
  - 修复动作 duration 映射和观测坐标单位。

### Step 2 - 动作映射、观测健壮性和奖励函数

- 状态：完成
- Commit：`cbf0f19 Add Target-DQN action mapping and rewards`
- 内容：
  - 修正 Target-DQN batch 预测，只取第一条结果的问题。
  - 将 duration head 的 `0-19` 输出映射为 `8-27` 秒区间内的实际绿灯持续时间。
  - 增加 `MIN_GREEN_DURATION`、默认最大速度和等待速度阈值配置。
  - 增加 `get_lane_position_meters()`，兼容 `position_in_lane["y"]` 以毫米或米表示的情况。
  - 观测处理在无 `init_state` 时默认初始化单路口网格，并防御缺失车辆配置。
  - 实现基于相位压力、等待时间变化、排队、延误和 duration 匹配的非零奖励。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn`，语法编译通过。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 补充可脱离 KaiwuDRL 的本地验证脚本。

### Step 3 - Target-DQN 本地 smoke 测试

- 状态：完成
- Commit：`1ef3342 Add Target-DQN smoke validation script`
- 内容：
  - 新增 `codebase/tests/test_target_dqn_smoke.py`。
  - 通过 stub `common_python` 和 `kaiwudrl`，让测试可在没有平台 SDK 的本地环境运行。
  - 覆盖观测特征长度、NaN 检查、占用网格、duration 映射、batch predict、非零奖励和目标网络独立性。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 当前本地 Python 缺少 `torch`，`python tests/test_target_dqn_smoke.py` 会明确输出 skip；平台或安装 PyTorch 后可执行完整 smoke。
- 下一步：
  - 根据 smoke 测试结果修正剩余接口问题。

### Step 4 - 本地依赖和平台入口检查

- 状态：完成
- Commit：`f81d8ca Record local dependency check`
- 内容：
  - 运行 `python train_test.py` 检查平台训练入口。
  - 运行 `python -m pip show torch kaiwudrl common_python` 检查本地依赖。
- 验证：
  - `python train_test.py` 当前失败于 `ModuleNotFoundError: No module named 'kaiwudrl'`。
  - 当前本地 Python 未安装 `torch`、`kaiwudrl`、`common_python`。
- 结论：
  - 本地只能做语法编译和 stub smoke；真实训练检查需要腾讯开悟/KaiwuDRL 环境或用户提供依赖安装方式。

### Step 5 - 规则基线兜底策略

- 状态：完成
- Commit：`a72df32 Add Target-DQN rule-based fallback`
- 内容：
  - 在 `agent_target_dqn.Agent` 中新增 `rule_based_action()`。
  - 规则策略按四个相位对应进口车道的车辆数、低速排队、等待时间和延误估计压力。
  - `exploit()` 在模型推理或特征处理异常时回退到规则动作，避免评估阶段直接崩溃。
  - smoke 测试增加规则动作断言。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 考虑把 reward 和规则基线的压力计算抽为共享工具，减少后续调参分叉。

### Step 6 - 相位压力计算共享化

- 状态：完成
- Commit：`97bf0f0 Share Target-DQN phase pressure utility`
- 内容：
  - 在 `traffic_utils.py` 中新增 `get_phase_pressure()`。
  - `reward_shaping()` 和 `rule_based_action()` 复用同一套相位压力估计。
  - 减少奖励函数和规则基线之间的重复逻辑，便于后续统一调参。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 检查训练 workflow 和 sample 结构是否还存在明显形状风险。

### Step 7 - 模型 forward 设备一致性

- 状态：完成
- Commit：`c073b85 Align Target-DQN model input device`
- 内容：
  - 修正 `Model.forward()` 中输入已经是 tensor 时只转 dtype、不转 device 的问题。
  - 移除未使用的 `batch` 局部变量。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 继续检查 workflow/sample 的训练形状风险。

### Step 8 - 训练动作索引和样本形状修复

- 状态：完成
- Commit：`024b55b Fix Target-DQN training action indices`
- 内容：
  - 将 `SampleData.act` 从 4 维修正为实际使用的 3 维 `[junction_id, phase_index, duration_seconds]`。
  - 简化 `sample_process()`，直接遍历轨迹列表，避免 `np.array(...).squeeze()` 在边界长度下改变结构。
  - 在 `Algorithm.learn()` 中统一用 `torch.as_tensor()` 堆叠样本，兼容 list、numpy 和 tensor。
  - 将环境动作中的 duration 秒数转换回 duration head 索引，避免 20 维 Q head gather 越界。
  - smoke 测试增加动作索引转换和 sample_process 断言。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 检查是否需要在训练 workflow 中记录更多 reward/score 指标。

### Step 9 - 模型保存加载默认路径修复

- 状态：完成
- Commit：`e29d7f9 Make Target-DQN model loading robust`
- 内容：
  - 为 `save_model()` 和 `load_model()` 增加默认 checkpoint 目录 `agent_target_dqn/ckpt`。
  - `save_model()` 自动创建 checkpoint 目录，并防御 logger 为空。
  - `load_model(id="latest")` 在初始无模型时跳过加载，避免从零训练第一局崩溃。
  - 成功加载模型后同步 Target-DQN 目标网络。
  - smoke 测试增加保存/加载路径检查。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 再检查训练 workflow 指标记录和异常日志。

### Step 10 - 训练 reward 统计和样本容灾

- 状态：完成
- Commit：`7d6779e Improve Target-DQN training reward metrics`
- 内容：
  - `sample_process()` 对 `rew=None` 的容灾轨迹补零奖励，避免样本转换后训练崩溃。
  - `sample_process()` 对缺失 `legal_action` 做保守默认值。
  - `train_workflow` 同时统计 phase reward、duration reward 和总 reward。
  - 监控上报使用数值类型，不再把平均 reward 格式化成字符串。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 检查 `train_workflow` 是否还需要对 `logger` 为空做防御。

### Step 11 - workflow 日志限流和 logger 防御

- 状态：完成
- Commit：`a303b63 Reduce Target-DQN workflow log noise`
- 内容：
  - 为 `train_workflow` 增加 `_log_info()` 和 `_log_error()`，避免本地或测试环境 `logger=None` 时崩溃。
  - 将逐帧日志改为每 20 次预测或 episode 结束时记录一次，降低平台日志限流风险。
  - 保留训练指标、截断和终止日志。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 做一次 Target-DQN 当前 TODO/风险复扫。

### Step 12 - `extra_info` 缺失健壮性

- 状态：完成
- Commit：`8606453 Handle missing Target-DQN extra info`
- 内容：
  - `FeatureProcess.update_traffic_info()` 支持 `extra_info=None`，避免本地测试或部分环境响应缺少 `init_state` 时崩溃。
  - smoke 测试增加无 `extra_info` 的观测处理断言。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 汇总当前剩余受阻项：真实 KaiwuDRL 训练验证需要平台依赖。

### Step 13 - 离线静态验证和平台运行手册

- 状态：完成
- Commit：`3b873da Add offline validation and platform runbook`
- 内容：
  - `Algorithm.learn()` 对空 batch 直接返回，避免边界输入崩溃。
  - `_stack_tensor()` 不再向 `torch.as_tensor()` 传入 `device=None`，兼容不同 PyTorch 版本。
  - 新增 `codebase/tests/test_target_dqn_static.py`，在无 `torch` / `kaiwudrl` 环境中检查关键源码约束。
  - 新增 `RUNBOOK.md`，记录本地检查、平台验证步骤、监控指标和实验回填格式。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 平台环境可用后运行 `python train_test.py` 并回填真实结果。

### Step 14 - 报告草稿整理

- 状态：完成
- Commit：`a8495ee Add Target-DQN report draft`
- 内容：
  - 新增 `REPORT_DRAFT.md`，整理问题定义、Target-DQN 方法、状态特征、动作映射、奖励设计、规则兜底、训练流程、本地验证、平台实验表和后续改进方向。
  - 报告草稿可迁移到 `icml2022.zip` 模板继续排版。
- 验证：
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `git diff --check`，未发现空白错误。
  - 已检查 `REPORT_DRAFT.md`、`RUNBOOK.md`、`PROGRESS.md` 行数。
- 下一步：
  - 平台环境可用后补真实实验结果和曲线。

### Step 15 - 平台提交打包脚本

- 状态：完成
- Commit：`d399f83 Add platform submission package script`
- 内容：
  - 新增 `scripts/package_submission.sh`，将 `codebase/` 打包为平台上传压缩包。
  - 打包时排除日志、`__pycache__`、checkpoint、模型 pkl 和其他运行产物。
  - `.gitignore` 增加 `dist/`，避免提交生成的压缩包。
  - `RUNBOOK.md` 增加打包命令和默认产物说明。
- 验证：
  - 已运行 `bash scripts/package_submission.sh`，生成 `dist/marl_hw1_codebase.zip`。
  - 已运行 `zipinfo` 内容检查，确认压缩包未包含日志、checkpoint、`.pkl`、截图、报告和根目录追踪文档。
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 更新 `AGENTS.md` 中已经过期的仓库状态和风险描述。

### Step 16 - 更新 AGENTS 当前状态

- 状态：完成
- Commit：`a596f96 Refresh agent instructions for current state`
- 内容：
  - 修正 `AGENTS.md` 中“当前目录不是 git 仓库”的过期描述。
  - 将早期代码审阅风险拆分为“已处理问题”和“仍需关注问题”。
  - 更新后续实施计划，指向当前实际的离线检查、平台验证、打包脚本和实验回填流程。
- 验证：
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 继续补平台运行后的实验回填内容。

### Step 17 - 相位时间特征并入观测

- 状态：完成
- Commit：`a7b610b Add signal phase features to Target-DQN observations`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `560` 扩展为 `568`。
  - 在占用/速度网格后追加 8 维相位时间特征：phase one-hot、duration、remaining duration、elapsed duration、phase present flag。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md` 的特征说明。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `python tests/test_target_dqn_smoke.py`，当前本地缺少 `torch`，脚本明确 skip。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 平台验证后根据训练表现决定是否加入车道统计特征或 80 维联合动作。

### Step 18 - 实验台账模板

- 状态：完成
- Commit：`d986957 Add platform experiment ledger`
- 内容：
  - 新增 `EXPERIMENTS.md`，将平台实验记录从开发进度中拆出来。
  - 增加首次平台 smoke 的实验模板，覆盖 commit、代码包、环境配置、监控、评分和错误日志。
  - 更新 `RUNBOOK.md` 和 `REPORT_DRAFT.md`，指向独立实验台账。
- 验证：
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `python -m compileall agent_target_dqn tests`，语法编译通过。
  - 已运行 `git diff --check`，未发现空白错误。
- 下一步：
  - 平台环境可用后填写 E00 首次 smoke 结果。

### Step 19 - 一键离线检查脚本

- 状态：完成
- Commit：`27ccce5 Add offline check script`
- 内容：
  - 新增 `scripts/check_offline.sh`，串联编译检查、静态约束检查、smoke、空白检查、打包和压缩包内容检查。
  - 修正 `scripts/package_submission.sh` 可执行权限，保证 `RUNBOOK.md` 中的 `./scripts/package_submission.sh` 可直接运行。
  - 更新 `RUNBOOK.md` 的本地检查说明。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，编译、静态检查、smoke skip、空白检查、打包和压缩包内容检查完成。
- 下一步：
  - 继续维护平台实验台账。

### Step 20 - 静态约束补强

- 状态：完成
- Commit：`964825f Strengthen offline static checks`
- 内容：
  - `test_target_dqn_static.py` 增加 `DIM_OF_OBSERVATION = 568` 和 `PHASE_FEATURE_DIM = 8` 约束检查。
  - 静态测试检查 `package_submission.sh` 和 `check_offline.sh` 存在且可执行。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后运行真实 `train_test.py`。

### Step 21 - Double DQN 目标和 Huber loss

- 状态：完成
- Commit：`e3c7f00 Use Double DQN target calculation`
- 内容：
  - TD target 改为 Double DQN 风格：在线网络选择下一动作，目标网络评估该动作 Q 值。
  - value loss 从 MSE 改为 Huber loss，降低异常 TD error 的影响。
  - 静态检查增加 Double DQN 目标和 Huber loss 约束。
  - 报告草稿同步说明目标值计算方式。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察 `value_loss`、`q_value`、`target_q_value` 是否更稳定。

### Step 22 - 交通压力特征并入观测

- 状态：完成
- Commit：`fbf6110 Add traffic pressure features to Target-DQN observations`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `568` 扩展为 `576`。
  - 在占用/速度网格和相位时间特征后追加 8 维交通统计：4 个相位压力、进口车辆数、排队比例、平均等待时间和平均延误。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md` 的特征说明。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察新增交通统计是否改善 `phase_reward` 和平均等待指标。

### Step 23 - 合法动作相位掩码

- 状态：完成
- Commit：`fcf6836 Apply phase legal action masks`
- 内容：
  - 新增 `normalize_phase_legal_action()`，兼容平台标量门控和 4 维相位 mask 两种格式。
  - Target-DQN 贪心预测在 phase Q 值上应用 mask，随机探索只从合法相位采样。
  - 规则兜底策略在合法相位中选择压力最高相位。
  - `sample_process()` 保留归一化后的 4 维合法相位信息。
  - 新增无 `torch` 依赖的 `tests/test_target_dqn_features.py`，并纳入 `./scripts/check_offline.sh`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；新增无平台依赖特征工具测试通过，smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后用真实 observation 确认 `legal_action` 的实际格式。

### Step 24 - Double DQN 目标动作掩码

- 状态：完成
- Commit：`04f5ce1 Mask Double DQN target phase actions`
- 内容：
  - `sample_process()` 将训练样本里的 `legal_action` 更新为 `_obs` 对应的下一状态相位 mask。
  - `Algorithm.learn()` 在 Double DQN 选择下一相位时应用合法相位 mask。
  - 新增 `_phase_legal_mask()`，兼容标量门控、短 mask 和全零 mask。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认真实 `legal_action` 是否可作为相位级 mask；若只是标量门控，则当前逻辑等价于全相位可选。

### Step 25 - 逐车道统计特征并入观测

- 状态：完成
- Commit：`91d08ae Add per-lane traffic statistics features`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `576` 扩展为 `618`。
  - 新增 `get_lane_statistics()`，统计 14 条进口车道的车辆数、排队数和平均等待时间。
  - 在观测末尾追加 42 维逐车道统计特征。
  - 更新无平台依赖特征测试、静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；新增无平台依赖逐车道统计测试通过，smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察逐车道统计是否改善排队和等待指标。

### Step 26 - 一帧交通趋势特征

- 状态：完成
- Commit：`4cf403d Add traffic trend observation features`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `618` 扩展为 `626`。
  - 新增 `get_traffic_summary()` 和 `get_traffic_trend()`，统计当前交通摘要并计算相对上一帧的变化。
  - 在观测中追加 8 维趋势特征：4 个相位压力变化、进口车辆数变化、排队比例变化、平均等待时间变化和平均延误变化。
  - `FeatureProcess.reset()` 清空 `last_traffic_summary`，避免跨 episode 泄漏趋势状态。
  - 更新无平台依赖特征测试、静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；新增无平台依赖交通趋势测试通过，smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察趋势特征是否改善拥堵缓解方向的学习。

### Step 27 - 相位服务年龄和公平性奖励

- 状态：完成
- Commit：`b181a8c Add phase fairness reward and age features`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `626` 扩展为 `630`。
  - `FeatureProcess.reset()` 初始化四个相位的上次服务帧。
  - 观测中追加 4 维相位服务年龄特征，表示四个相位距上次服务的归一化帧数。
  - `reward_shaping()` 增加公平性奖励项，鼓励服务高压力且较久未被服务的相位。
  - 规则兜底策略在压力相近时用相位年龄做小幅加权。
  - 更新 smoke/static 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察是否降低单方向长期排队的问题。

### Step 28 - 80 维联合动作 Q 头

- 状态：完成
- Commit：`eb6cdb5 Use joint action Q head for Target-DQN`
- 内容：
  - Target-DQN 从 phase/duration 双头输出改为单个 80 维联合动作 Q 头。
  - `predict()` 贪心和随机探索都先选择 joint action，再解码为 `phase_idx` 和 `duration_idx`。
  - `Algorithm.learn()` 将环境动作映射为 `phase_idx * 20 + duration_idx` 的联合索引。
  - Double DQN target 使用 80 维 joint mask，避免选择非法相位对应的动作组合。
  - 训练目标使用 `phase_reward + duration_reward` 的总奖励，同时 workflow 仍保留分量监控。
  - 更新 smoke/static 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后对比联合动作模型和历史双头模型的训练稳定性。

### Step 29 - 联合动作 checkpoint 迁移容灾

- 状态：完成
- Commit：`df2bec5 Handle incompatible latest checkpoints`
- 内容：
  - `load_model(id="latest")` 现在会捕获结构不兼容的 checkpoint，并跳过加载继续从当前参数训练。
  - 保留显式模型 ID 的严格加载行为，避免用户手动指定模型时静默忽略错误。
  - 清理 `Algorithm.learn()` 中旧双头注释和未使用的 `_action_to_head_indices()`。
  - 更新静态测试、`AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认旧双头 `latest` checkpoint 不会阻断新联合动作模型训练。

### Step 30 - 4 帧滚动交通历史特征

- 状态：完成
- Commit：`49240a7`
- 内容：
  - 将 `agent_target_dqn` 观测维度从 `630` 扩展为 `638`。
  - 新增 `get_traffic_history_feature()`，把最近 4 帧交通摘要压缩为 8 维历史特征。
  - `FeatureProcess.reset()` 清空 `traffic_history`，避免跨 episode 泄漏。
  - 观测中追加最近 4 帧平均相位压力、进口车辆数、排队比例、平均等待时间和平均延误。
  - 更新无平台依赖特征测试、静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；新增无平台依赖滚动历史测试通过，smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察历史摘要是否改善对持续拥堵趋势的响应。

### Step 31 - workflow 合法动作门控兼容

- 状态：完成
- Commit：`d33bd28`
- 内容：
  - 将训练 workflow 的是否决策判断从直接读取 `legal_action[0]` 改为 `_need_to_predict()`。
  - `_need_to_predict()` 复用 `normalize_phase_legal_action()`，兼容平台文档中的 `int32` 标量门控和相位级 mask。
  - 为 `normalize_phase_legal_action()` 补充标量 `0/1` 离线测试。
  - 更新静态测试、`AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；新增标量 `legal_action` 测试通过，smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后用真实 observation 日志确认 `legal_action` 是否包含相位级约束。

### Step 32 - 平台异常信息保留

- 状态：完成
- Commit：`1d36834`
- 内容：
  - `run_episodes()` 捕获异常时记录原始异常信息，不再只输出通用 `run_episodes error`。
  - 使用 `raise ... from e` 保留 Python 异常链，便于平台日志定位真实崩溃点。
  - 更新静态测试、`AGENTS.md` 和 `RUNBOOK.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后如仍崩溃，优先根据 chained traceback 修复首个真实异常。

### Step 33 - latest checkpoint 保存加载闭环

- 状态：完成
- Commit：`c87815a`
- 内容：
  - 将 Target-DQN workflow 的周期保存从默认 `model.ckpt-1.pkl` 改为 `model.ckpt-latest.pkl`。
  - 保持每局开始的 `load_model(id="latest")` 与周期保存文件名一致，避免跨 episode 或进程恢复时一直找不到刚保存的模型。
  - 更新静态测试、smoke 保存加载测试、`AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 `agent_target_dqn/ckpt/model.ckpt-latest.pkl` 能被后续 episode 正常加载。

### Step 34 - 样本发送列表引用隔离

- 状态：完成
- Commit：`8790612`
- 内容：
  - 将 `agent.send_sample_data(g_data)` 改为 `agent.send_sample_data(list(g_data))`。
  - 保留后续 `g_data.clear()` 释放本地列表，但避免平台发送实现异步持有原列表引用时样本被清空。
  - 更新静态测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察 `data_length` 与样本生产消费比是否正常。

### Step 35 - 训练进度日志门控

- 状态：完成
- Commit：`e54b6c0`
- 内容：
  - 将 `predict_cnt % 20 == 0 or done` 改为 `_should_log_progress(predict_cnt, done, need_to_predict)`。
  - 进度日志只在 episode 结束或真实预测计数达到 20 的倍数时打印，避免无决策帧里 `predict_cnt == 0` 导致每帧刷屏。
  - 更新静态测试、`AGENTS.md` 和 `RUNBOOK.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察训练日志体量和关键指标是否更容易读取。

### Step 36 - 评估推理不衰减 epsilon

- 状态：完成
- Commit：`39ef749`
- 内容：
  - 将 `_eps` 衰减限制在训练 `predict()` 路径，`exploit()` 不再改变训练探索率。
  - `exploit_flag=True` 时直接进入贪心动作选择，避免多余随机分支判断。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认评估调用不会影响后续训练探索曲线。

### Step 37 - reward 终局缺字段容错

- 状态：完成
- Commit：`8fd7443`
- 内容：
  - `reward_shaping()` 对缺失 `frame_state` 的 observation 保守返回 `(0.0, 0.0)`。
  - `reward_shaping()` 对缺失 `vehicles` 的帧按空车流处理，避免 episode 结束或异常帧字段不完整时崩溃。
  - 对不完整动作列表增加零奖励保护。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认终局 reward 计算不再因缺字段中断训练循环。

### Step 38 - 特征预处理缺字段容错

- 状态：完成
- Commit：`821913b`
- 内容：
  - `FeatureProcess.init_road_info()` 对缺失路网字段或非 dict `init_state` 保守跳过。
  - `FeatureProcess.update_traffic_info()` 对缺失 `frame_state` 或 `vehicles` 的 observation 保守返回。
  - 预处理车辆历史时跳过缺失 `v_id` 或字段不完整的畸形车辆记录，避免异常帧中断训练循环。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认特征预处理不会因偶发不完整 observation 中断。

### Step 39 - 观测编码异常帧容错

- 状态：完成
- Commit：`f1c2f8a`
- 内容：
  - `Agent.exploit()` 对缺失 `obs` 包装的评估 observation 使用原始 dict 兜底，避免 fallback 之前直接崩溃。
  - `Agent.observation_process()` 对缺失 `frame_state`、缺失 `vehicles`、非 list 车辆字段和畸形车辆记录使用空特征或跳过策略。
  - 相位特征解析对非 dict 相位、异常 `phase_id` / `duration` 字段使用默认值，规则基线也能处理空 observation。
  - 共享交通统计工具跳过缺字段、非 dict 或数值异常的车辆记录，降低 reward、规则基线和车道统计的同类崩溃风险。
  - 更新静态测试、smoke 测试、`AGENTS.md` 和 `REPORT_DRAFT.md`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后用真实异常 observation 日志确认观测编码不再中断 episode。

### Step 40 - 样本转换边界容错

- 状态：完成
- Commit：`fd8a391`
- 内容：
  - `sample_process()` 对空轨迹、全无效轨迹、缺失 observation/action 的帧保守返回空样本或跳过，避免边界输入触发 `IndexError`。
  - `sample_process()` 对缺失 reward 补零，并继续把下一状态合法相位 mask 写入当前 transition。
  - `reward_shaping()` 对异常动作、异常 `frame_no`、非 list `vehicles` 和畸形车辆记录保守返回零奖励或跳过。
  - 将 `tests/test_target_dqn_features.py` 扩展为无 `torch` 也会执行的样本转换和 reward 边界测试。
  - 修正 smoke 中终局 transition 的测试期望：终局样本会保留，训练时 `done=0` 表示 `not_done=0`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本生产消费比稳定，且终局 transition 不再触发 shape 或空列表异常。

### Step 41 - 训练非有限数值容错

- 状态：完成
- Commit：`a3b9d1f`
- 内容：
  - `Algorithm.learn()` 统一清洗 `obs`、`_obs`、`rew`、`act`、`not_done`、`legal_action` 和 TD target 中的 NaN/Inf。
  - `not_done` 清洗后裁剪到 `[0, 1]`，避免异常 done 标记放大 target。
  - workflow 的 `_reward_components()` 使用 `_finite_float()`，监控 reward 分量遇到 NaN/Inf 或畸形 reward 时归零。
  - 将 workflow helper 纳入无平台依赖测试，覆盖 `_reward_components()`、`_need_to_predict()` 和 `_should_log_progress()`。
  - 更新静态测试，防止训练张量清洗和 reward 监控清洗回退。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察 `value_loss`、`q_value`、`target_q_value` 是否仍出现 NaN/Inf。

### Step 42 - workflow 环境响应安全读取

- 状态：完成
- Commit：`8c5ff73`
- 内容：
  - `run_episodes()` 在 reset 后通过 `_safe_observation()` 和 `_safe_extra_info()` 读取 observation 与 extra info，避免字段缺失直接 `KeyError`。
  - step 返回后通过 `_safe_frame_no()`、`_safe_observation()`、`_safe_done_flag()` 和 `_safe_extra_info()` 解析状态，字段异常时使用保守默认值。
  - 构造训练 Frame 时通过 `_safe_legal_action()` 读取当前 observation 的合法动作，缺失时交给后续归一化逻辑默认处理。
  - 将 workflow 安全读取 helper 纳入无平台依赖测试，并加入静态锚点防止回退到直接索引。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认异常 env response 不再导致 `KeyError: observation`、`extra_info` 或 `legal_action`。

### Step 43 - workflow 返回结构归一化

- 状态：完成
- Commit：`11c3f4a`
- 内容：
  - 增加 `_normalize_reset_result()`，兼容平台封装 dict 返回和作业文档中的 `(observation, extra_info)` 二元 reset 返回。
  - 增加 `_normalize_step_result()`，兼容当前封装 `(reward, env_obs)` 二元返回和文档中的 `(frame_no, observation, score, terminated, truncated, extra_info)` 六元返回。
  - `run_episodes()` 在 disaster recovery、特征处理和 sample 构造前统一使用归一化后的 `env_obs`。
  - 将 reset/step 返回结构归一化 helper 纳入无平台依赖测试，并加入静态锚点防止回退。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认真实 `env.reset()` / `env.step()` 返回形态均能进入同一训练路径。

### Step 44 - agent 动作输出兜底

- 状态：完成
- Commit：`f6670de`
- 内容：
  - `predict()` 对空 observation batch 或缺失 feature 的条目直接返回空列表，避免空 batch 进入模型推理。
  - `predict()` 读取 `legal_action` 时使用缺失默认值，交给合法动作归一化逻辑兜底。
  - `exploit()` 在贪心预测返回空动作时回退到规则基线。
  - `action_process()` 固定单路口 `junction_id=0`，并通过 `_safe_action_index()` 清洗异常 phase/duration index。
  - smoke 与静态测试增加空预测、畸形动作、NaN/Inf 动作索引和单路口动作输出覆盖。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认动作输出不再出现非法 `junction_id`、越界相位或低于最小绿灯时长。

### Step 45 - latest checkpoint 坏文件容错

- 状态：完成
- Commit：`a3c8a3a`
- 内容：
  - `load_model(id="latest")` 现在会跳过不可读或反序列化失败的 checkpoint，避免坏文件阻断每局开始加载。
  - 对 checkpoint payload 非 dict 的情况增加显式校验，`latest` 路径记录日志后跳过，指定模型加载仍抛错暴露问题。
  - `load_state_dict()` 的容错从 `RuntimeError` 扩展到 `RuntimeError`、`TypeError` 和 `ValueError`，覆盖更多坏结构场景。
  - smoke 测试增加损坏文本 checkpoint 和非 dict checkpoint 覆盖，静态测试增加对应锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认损坏或旧格式 `agent_target_dqn/ckpt/model.ckpt-latest.pkl` 不再阻断后续 episode。

### Step 46 - Torch 线程设置导入容错

- 状态：完成
- Commit：`63dd4da`
- 内容：
  - 四个 agent 入口都将 `torch.set_num_threads(1)` 和 `torch.set_num_interop_threads(1)` 包进 `_configure_torch_threads()`。
  - 当平台进程已经启动过 Torch 并行运行、线程设置抛出 `RuntimeError` 时，agent 导入会跳过该设置继续执行。
  - 静态测试增加四个 agent 入口的容错 helper、实际调用和 `RuntimeError` 捕获锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
  - 已额外运行 `python -m compileall agent_dqn agent_ppo agent_diy`，确认非主线模板 agent 仍能编译。
- 下一步：
  - 平台环境可用后确认 `train_test.py` 或框架算法列表校验不会因 Torch 线程设置在导入阶段失败。

### Step 47 - Target-DQN checkpoint 原子保存

- 状态：完成
- Commit：`44031f1`
- 内容：
  - `Agent.save_model()` 先写 `model.ckpt-*.pkl.tmp` 临时文件，再通过 `os.replace()` 原子替换正式 checkpoint。
  - 保存失败时清理临时 checkpoint，避免失败残留进入后续打包或误判。
  - 静态测试增加临时文件写入、原子发布和失败清理锚点；smoke 测试增加保存后 `.tmp` 文件不存在的断言。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认长时间训练或中断恢复时不会产生半写 `model.ckpt-latest.pkl`。

### Step 48 - 非有限 loss/梯度更新保护

- 状态：完成
- Commit：`81b5dbf`
- 内容：
  - `Algorithm.learn()` 在计算 Huber loss 后检查 `torch.isfinite(loss)`，遇到 NaN/Inf 时记录日志并跳过本次更新。
  - 梯度裁剪后将 grad norm 转成 float，并用 `math.isfinite()` 检查；非有限时清空梯度并跳过 `optimizer.step()`。
  - 静态测试增加非有限 loss、非有限梯度日志和跳过更新锚点；smoke 测试增加一次正常 learn 和一次 NaN/Inf 样本 learn 后参数仍有限的断言。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察 `value_loss`、`model_grad_norm` 和 checkpoint 参数是否仍出现 NaN/Inf。

### Step 49 - workflow 预测动作兜底

- 状态：完成
- Commit：`7443d0d`
- 内容：
  - 新增 `_predict_action()`，集中处理训练 workflow 中的 `predict()`、`action_process()` 和兜底动作生成。
  - 当模型预测返回空列表或抛出异常时，workflow 会记录日志并回退到 `rule_based_action()`。
  - 如果规则策略也异常，workflow 使用 `[0, 0, Config.MIN_GREEN_DURATION]` 作为最终合法动作，避免单次推理异常中断 episode。
  - 无平台依赖测试覆盖正常预测、空预测、预测异常和规则兜底异常四种路径；静态测试增加对应锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认训练 episode 不再因偶发空预测或模型推理异常直接失败。

### Step 50 - 模型输入形状容错

- 状态：完成
- Commit：`5bfc1a5`
- 内容：
  - `Model.forward()` 通过 `_prepare_input()` 统一输入张量形状。
  - 单条一维 observation 会自动转成 batch 维度，保持 Q head 输出形状为 `[1, DIM_OF_ACTION]`。
  - 短 observation 会补零到 `Config.DIM_OF_OBSERVATION`，长 observation 会截断到配置维度，避免平台封装传入异常长度时线性层直接报错。
  - smoke 测试覆盖正常 batch、单条一维、短向量和长向量输入；静态测试增加输入形状归一化锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认模型推理入口不会因 observation 是否带 batch 维度而失败。

### Step 51 - DQN 样本张量定宽归一化

- 状态：完成
- Commit：`a63529c`
- 内容：
  - `Algorithm.learn()` 在 `torch.stack()` 前通过 `_normalize_tensor()` 统一每个样本字段的张量形状。
  - `obs` / `_obs` 补齐或截断到 `Config.DIM_OF_OBSERVATION`，`act`、`rew`、`done` 和 `legal_action` 分别归一化到固定宽度。
  - 畸形值转换失败时使用空张量再补零，避免单条坏样本导致整个 batch 堆叠失败。
  - smoke 测试增加短 obs、长 _obs、短 action、短 reward 和短 legal_action 的 ragged sample learn 覆盖；静态测试增加定宽归一化锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后观察 learner 是否仍因样本 shape 不一致报错。

### Step 52 - 模型 ragged batch 输入容错

- 状态：完成
- Commit：`e00679f`
- 内容：
  - `Model._prepare_input()` 的非 Tensor 输入现在先经过 `_as_numpy_array()`，避免 ragged Python list 在 `np.asarray(..., dtype=float32)` 阶段直接抛错。
  - 对 ragged batch 逐行调用 `_fit_numpy_width()`，按 `Config.DIM_OF_OBSERVATION` 补齐或截断后再 `np.stack()`。
  - 对单行非法值使用零宽数组并补齐为全零 observation，保证模型输出仍保持 batch 形状。
  - smoke 测试覆盖短/长混合 ragged batch 和包含非法行的 batch；静态测试增加 ragged batch 归一化锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认模型入口不会因 Python batch 内各行 observation 长度不一致而失败。

### Step 53 - 样本入口字段规范化

- 状态：完成
- Commit：`70fc04b`
- 内容：
  - `sample_process()` 在创建 `SampleData` 前对 `obs`、`act`、`rew` 和 `done` 做固定宽度归一化。
  - 样本字段中的 NaN/Inf 会归零，短字段补零、长字段截断，动作会裁剪到单路口、合法相位和模型 20 档 duration 可表达范围。
  - `normalize_phase_legal_action()` 会将合法动作 mask 中的 NaN/Inf 显式归零，避免非有限值被误判为可选相位。
  - 无平台依赖测试增加 ragged observation、短 reward、非有限样本值、字符串 done 和非有限 legal_action 覆盖；静态测试增加样本归一化锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本池中 `obs`、`_obs`、`act`、`rew`、`done` 和 `legal_action` 不再因原始轨迹异常产生 shape 不一致。

### Step 54 - 交通统计非有限值清洗

- 状态：完成
- Commit：`6eced8c`
- 内容：
  - `traffic_utils` 增加 `_finite_float()`、`_nonnegative_float()` 和 `_phase_array()`，统一清洗标量交通指标和相位数组。
  - `get_phase_pressure()`、`get_lane_statistics()`、`get_traffic_summary()`、`get_traffic_trend()` 和 `get_traffic_history_feature()` 会将 NaN/Inf 归零，并对等待、延误等非负指标做下界裁剪。
  - `get_lane_position_meters()` 对非有限车道位置显式抛出 `ValueError`，由调用侧跳过异常车辆。
  - `observation_process()` 返回前通过 `_sanitize_observation()` 保证最终特征固定为 `Config.DIM_OF_OBSERVATION`，且非有限值归零。
  - 无平台依赖测试覆盖非有限车辆 speed、waiting_time、delay、趋势和历史统计；静态测试增加交通数值清洗和最终 observation 清洗锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 reward、规则兜底和 observation 不再因单条异常车辆字段产生 NaN/Inf。

### Step 55 - 预处理动态字段容错

- 状态：完成
- Commit：`660143f`
- 内容：
  - `FeatureProcess` 增加 `_safe_float()`、`_safe_int()`、`_is_hashable()` 和 `_safe_position_pair()`，统一清洗 frame、车辆 ID 和车辆位置。
  - `update_traffic_info()` 现在会跳过不可哈希车辆 ID、非 list 车辆集合和异常动态字段，避免单条坏车辆中断整帧预处理。
  - `cal_waiting_time()` 对车速和 frame_time 做非有限值清洗，等待增量不允许为负。
  - `cal_travel_distance()` 将位置转为有限浮点数后再累计距离，遇到 NaN/Inf 或畸形 position 会跳过该车。
  - 交叉口等待时间聚合函数会跳过畸形车辆记录，等待时间按非负有限值累计。
  - 无平台依赖测试覆盖正常等待/距离累计、非有限 frame、不可哈希车辆 ID、异常位置和等待时间聚合；静态测试增加预处理容错锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认异常 frame_time、frame_no、vehicle id 或 position 不再导致 episode 在特征预处理阶段失败。

### Step 56 - 相位时间标量清洗

- 状态：完成
- Commit：`695d5d6`
- 内容：
  - `Agent` 增加 `_safe_float()`、`_safe_nonnegative_float()` 和 `_safe_int()`，用于清洗观测编码中的标量字段。
  - `_phase_feature()` 使用动作索引清洗逻辑处理异常 `phase_id`，并对 `duration`、`remaining_duration` 的 NaN/Inf/Overflow 做非负有限值归零。
  - `_phase_age_feature()` 会清洗 `frame_no` 和历史 `phase_last_served_frame`，异常相位服务记录回退为当前帧，避免相位年龄特征计算崩溃。
  - `reward_shaping()` 对动作 duration overflow、异常 `frame_no` 和畸形相位服务历史做容错，公平性 reward 保持有限值。
  - workflow 的 `_finite_float()` 和 `_safe_frame_no()` 覆盖 malformed/NaN/Inf/Overflow 标量，避免日志和结束路径读取帧号失败。
  - 无平台依赖测试增加 infinity duration、infinity frame_no、畸形 phase service history 和 workflow frame_no 覆盖；静态测试增加相位标量清洗锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认异常 `phase_id`、`duration`、`remaining_duration`、`frame_no` 或旧相位服务状态不再导致推理、reward 或 workflow 日志路径崩溃。

### Step 57 - 日志和监控上报隔离

- 状态：完成
- Commit：`6fd6f40`
- 内容：
  - `Algorithm.learn()` 增加 `_put_monitor_data()` 和 `_log_info()`，隔离 logger/monitor 后端异常。
  - 非有限 loss、非有限梯度和训练指标日志改走安全 logger helper，日志后端失败不会中断 learner。
  - workflow 增加 `_put_monitor_data()`，监控上报成功时才更新时间戳，上报失败时只记录错误并继续训练。
  - workflow 的 `_log_info()` 和 `_log_error()` 捕获 logger 后端异常，避免平台日志服务异常反向打断 episode。
  - 无平台依赖测试覆盖 logger.info/error 抛错、monitor.put_data 成功/失败/缺失路径；静态测试增加算法和 workflow 上报隔离锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认日志或 monitor 服务短暂失败时训练 episode 仍能继续运行。
