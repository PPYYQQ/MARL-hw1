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

### Step 58 - Agent 日志异常隔离

- 状态：完成
- Commit：`cf0cf1c`
- 内容：
  - `Agent` 增加 `_log_info()` 和 `_log_error()`，统一捕获 logger 后端异常。
  - `exploit()` 的规则兜底日志改走安全 error logger，日志失败不再阻止评估返回规则动作。
  - `save_model()` 成功日志和 `load_model()` 的 latest 缺失、不可读、非法 payload、结构不兼容及加载成功日志都改走安全 info logger。
  - 静态测试增加 Agent 日志 helper 和禁止 checkpoint/评估路径直接调用 logger 的约束。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认日志后端短暂异常不会影响模型加载、保存和评估兜底动作返回。

### Step 59 - latest checkpoint 保存失败隔离

- 状态：完成
- Commit：`093f83a`
- 内容：
  - workflow 增加 `_save_latest_model()`，封装周期性 `agent.save_model(id="latest")`。
  - `latest` checkpoint 保存异常时记录 `save latest model failed` 并返回失败，不再让文件系统或平台模型目录异常直接终止训练循环。
  - 保存尝试后仍更新时间戳，避免持续失败时每个 epoch 高频刷日志或反复打断训练节奏。
  - 无平台依赖测试覆盖保存成功、保存失败且 logger 后端也失败的路径；静态测试增加 safe save helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 checkpoint 目录权限或磁盘短暂异常时训练仍继续，并在下一保存周期重试 `latest`。

### Step 60 - 平台训练指标读取失败隔离

- 状态：完成
- Commit：`16ce1cd`
- 内容：
  - workflow 增加 `_get_training_metrics()`，封装平台 `get_training_metrics()` 调用。
  - 指标服务抛错时记录 `get training metrics failed` 并返回空字典，避免非核心监控读取失败中断 episode。
  - 指标返回非字典对象时也按空指标处理，避免后续日志格式假设导致异常。
  - 无平台依赖测试覆盖指标读取成功、返回 `None` 和抛错且 logger 后端也失败的路径；静态测试增加 safe metrics helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 metrics 服务短暂异常时训练 episode 仍继续，并在后续轮次恢复指标日志。

### Step 61 - 样本发送失败隔离

- 状态：完成
- Commit：`fef197e`
- 内容：
  - workflow 增加 `_send_sample_data()`，封装 `agent.send_sample_data()` 调用。
  - 发送样本前继续传递 `list(g_data)` 浅拷贝，避免后续 `g_data.clear()` 清空 learner 可能异步消费的对象。
  - 样本池或 learner 通道抛错时记录 `send sample data failed` 并返回失败，不再让非环境交互路径直接终止 workflow。
  - 无平台依赖测试覆盖样本发送成功、发送失败且 logger 后端也失败、空样本跳过；静态测试增加 safe sample send helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本池或 learner 通道短暂异常时 actor 继续产生后续 episode，服务恢复后样本发送恢复。

### Step 62 - latest checkpoint 加载失败隔离

- 状态：完成
- Commit：`ee9f9bd`
- 内容：
  - workflow 增加 `_load_latest_model()`，封装每局开始时的 `agent.load_model(id="latest")`。
  - `latest` checkpoint 意外加载异常时记录 `load latest model failed` 并返回失败，不再让文件系统、并发写入或 target network 同步异常直接终止 episode。
  - 加载失败后 workflow 继续使用当前进程内模型参数运行，保持 actor 训练循环可恢复。
  - 无平台依赖测试覆盖加载成功、加载失败且 logger 后端也失败的路径；静态测试增加 safe latest load helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 checkpoint 文件权限、并发写入或坏文件导致的加载异常不会阻止后续 episode 启动。

### Step 63 - 训练配置读取失败隔离

- 状态：完成
- Commit：`5095119`
- 内容：
  - workflow 增加 `_read_usr_conf()`，封装 `read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)`。
  - 配置读取或平台校验工具抛错时记录 `read usr conf failed` 并返回 `None`，入口沿用已有 `usr_conf is None` 逻辑清晰退出。
  - 配置工具返回非字典结果时也按无效配置处理，避免未知配置对象传入 `env.reset()`。
  - 无平台依赖测试覆盖读取成功、返回非 dict 和抛错且 logger 后端也失败的路径；静态测试增加 safe config read helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认配置文件缺失、格式错误或校验工具异常时训练入口给出明确日志且不会产生二次崩溃。

### Step 64 - 平台容灾检测失败隔离

- 状态：完成
- Commit：`57897f2`
- 内容：
  - workflow 增加 `_handle_disaster_recovery()`，封装 reset 后和 step 后的 `handle_disaster_recovery(env_obs, logger)` 调用。
  - 容灾 helper 抛错时记录 `handle disaster recovery failed` 并返回 `False`，避免平台容灾 SDK 或异常 env_obs 格式中断训练循环。
  - 容灾 helper 返回非 bool 对象时显式转为 bool，保持原有触发容灾 break 的语义。
  - 无平台依赖测试覆盖容灾检测 True、非空对象、None 和抛错且 logger 后端也失败的路径；静态测试增加 safe disaster recovery helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认容灾 SDK 临时异常不会阻止 episode 继续执行；真实容灾信号仍能触发 break。

### Step 65 - reward shaping 失败隔离

- 状态：完成
- Commit：`0c1503e`
- 内容：
  - workflow 增加 `_shape_reward()`，封装中间 transition 和终局 transition 的 `reward_shaping()` 调用。
  - 奖励函数抛错时记录 `reward shaping failed` 并返回 `(0.0, 0.0)`，避免异常 observation、动作或 agent 状态直接中断 episode。
  - 奖励函数返回 NaN/Inf 或异常结构时继续复用 `_reward_components()` 清洗为有限二元 reward。
  - 无平台依赖测试覆盖 reward 返回非有限值、reward 函数抛错且 logger 后端也失败的路径；静态测试增加 safe reward shaping helper 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认单步 reward 计算异常只影响该 transition 奖励，不会导致 episode 或 workflow 崩溃。

### Step 66 - observation 处理失败隔离

- 状态：完成
- Commit：`0aaa71f`
- 内容：
  - workflow 增加 `_process_observation()`，封装决策帧 `agent.observation_process()` 调用。
  - observation 特征处理抛错时记录 `observation process failed`，预测路径回退到规则动作；如果仍需构造样本，则 `_obs_feature()` 使用全零固定维度特征占位。
  - workflow 增加 `_update_traffic_info()`，封装非决策帧 `agent.preprocess.update_traffic_info()`，失败时记录 `traffic info update failed` 并继续推进环境。
  - `_predict_action()` 对空 `obs_data` 直接走规则动作兜底，规则动作失败时仍返回 `[0, 0, MIN_GREEN_DURATION]`。
  - 无平台依赖测试覆盖 observation 处理成功/失败、零特征兜底、非决策帧 traffic update 成功/失败和空 obs_data 预测兜底；静态测试增加相关锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests`、`python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认单帧 observation/preprocess 异常只影响该帧特征质量，不会导致 actor episode 崩溃。

### Step 67 - sample_process 失败隔离

- 状态：完成
- Commit：`24eaa57`
- 内容：
  - workflow 增加 `_process_samples()`，封装终局路径和容灾路径的 `sample_process(collector)` 调用。
  - 样本转换抛错时记录 `sample process failed` 并返回空列表，避免单个异常 collector 直接触发 `run_episodes` 外层异常。
  - `sample_process()` 返回非 list 或空结果时 workflow 不再 yield，避免把无效样本批次继续送到样本池。
  - 无平台依赖测试覆盖样本转换成功、空 collector、返回非 list 和抛错且 logger 后端也失败的路径；静态测试增加 safe sample process helper 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认容灾或终局样本转换异常只丢弃当前 collector，不会导致 actor 训练循环崩溃。

### Step 68 - agent reset 失败隔离

- 状态：完成
- Commit：`cb0ec9d`
- 内容：
  - workflow 增加 `_reset_agent()`，封装每局 `agent.reset(env_obs)` 调用。
  - agent reset 抛错时记录 `agent reset failed` 并跳过当前 episode，避免半初始化的 `FeatureProcess` 状态继续参与特征、reward 和样本构造。
  - 无平台依赖测试覆盖 reset 成功、reset 失败且 logger 后端也失败的路径；静态测试增加 safe agent reset helper 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认偶发 agent reset 异常只影响当前 episode，不会导致 workflow 进程退出。

### Step 69 - env reset/step 失败隔离

- 状态：完成
- Commit：`22879d3`
- 内容：
  - workflow 增加 `_reset_env()`，封装 `env.reset(usr_conf=usr_conf)` 和 reset 返回值归一化。
  - reset 抛错时记录 `env reset failed` 并跳过当前 episode，下一 epoch 继续尝试启动新局。
  - workflow 增加 `_step_env()`，封装 `env.step(act)` 和 step 返回值归一化。
  - step 抛错时记录 `env step failed` 并中止当前 episode，避免在缺失下一状态时继续构造 transition 或发送未闭合样本。
  - 无平台依赖测试覆盖 env reset/step 成功、失败且 logger 后端也失败的路径；静态测试增加 safe env reset/step helper 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认偶发环境 RPC 或封装异常只影响当前 episode；若长期反复出现，再根据平台日志区分环境故障和非法动作问题。

### Step 70 - 样本批次 reward 聚合隔离

- 状态：完成
- Commit：`fc91ee0`
- 内容：
  - workflow 增加 `_sample_batch_stats()`，封装主循环里对 `g_data` 的长度、phase reward 和 duration reward 聚合。
  - 样本批次长度读取失败时记录 `sample batch length failed` 并按空批次统计，避免监控聚合异常中断训练。
  - 单个样本 `rew` 属性读取失败时记录 `sample reward read failed` 并按零 reward 统计，其它样本继续参与聚合。
  - 批次迭代异常时记录 `sample batch iteration failed` 并保留已聚合的可读 reward。
  - 无平台依赖测试覆盖正常 reward、NaN reward、异常 `rew` 属性、空批次和异常长度批次；静态测试增加 safe sample batch stats 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本批次偶发异常不会影响 reward/data_length 监控上报和后续样本发送。

### Step 71 - env.step 前动作合法化

- 状态：完成
- Commit：`bc5a7fc`
- 内容：
  - workflow 增加 `_safe_action()`，在每次 `env.step()` 前统一清洗最终动作。
  - 非决策帧动作保持 `[None, None, None]`，不改变平台跳帧语义。
  - 决策帧动作强制 `junction_id=0`，相位裁剪到 `0-3`，duration 裁剪到 `MIN_GREEN_DURATION` 到 `MIN_GREEN_DURATION + DIM_OF_ACTION_DURATION - 1`。
  - 动作结构异常、NaN/Inf 或不可转数值时记录 `invalid action, use default action` 并回退为 `[0, 0, MIN_GREEN_DURATION]`。
  - 无平台依赖测试覆盖合法动作、越界动作、非决策帧动作、字符串动作和 NaN 动作；静态测试增加 final action sanitation 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认异常预测或规则兜底动作不会触发环境非法动作错误，且非决策帧 `[None, None, None]` 语义保持不变。

### Step 72 - done 标记解析修正

- 状态：完成
- Commit：`4d2b6a0`
- 内容：
  - workflow 更新 `_safe_done_flag()`，显式解析 `terminated` / `truncated` 的 bool、数值和字符串形式。
  - `"true"`、`"1"`、`"yes"`、`"y"` 会视为 True；`"false"`、`"0"`、`"no"`、`"n"` 和空字符串视为 False。
  - NaN/Inf、未知字符串和异常对象按 False 处理，避免 `bool("False") == True` 导致 episode 过早结束。
  - 无平台依赖测试覆盖 int、字符串 true/false、未知字符串、Inf 和 object 输入；静态测试增加 done flag sanitation 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认不同封装返回的 `terminated` / `truncated` 类型不会误触发结束或漏掉真实结束。

### Step 73 - Agent.learn 失败隔离

- 状态：完成
- Commit：`29d0640`
- 内容：
  - `Agent.learn()` 增加 try/except，封装 `self.algorithm.learn(list_sample_data)`。
  - `Algorithm.learn()` 未预期抛错时记录 `learn failed` 并返回 `None`，避免 learner 因单个异常 batch 直接退出。
  - 静态测试增加 agent learn 失败日志和仍委托 algorithm 的锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本池偶发坏 batch 只影响该 batch 学习，不会导致 learner 进程退出；若连续出现，再保存 batch 定位字段问题。

### Step 74 - exploit 最终兜底动作

- 状态：完成
- Commit：`e43363a`
- 内容：
  - `Agent.exploit()` 在 observation 处理失败、模型预测为空或异常时不再直接调用 `rule_based_action()`，改为 `_safe_rule_based_action()`。
  - `_safe_rule_based_action()` 封装规则策略异常，规则策略失败时记录 `rule_based_action failed, use default action`。
  - 最终默认动作固定为 `[0, 0, Config.MIN_GREEN_DURATION]`，避免评估入口在最后兜底也失败时崩溃。
  - smoke 测试增加规则策略被 monkeypatch 为抛错时 `exploit()` 返回默认动作；静态测试增加 final exploit fallback 锚点。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认评估任务中异常 observation 或规则兜底异常不会导致 `exploit()` 抛错。

### Step 75 - 空合法动作 mask 审查

- 状态：完成
- Commit：`e266a74`
- 内容：
  - 复查 `Agent._phase_action_mask()` 和 `_joint_action_mask()`，确认全零相位 mask 与空 joint mask 行都会回退为可选全集。
  - 静态测试增加空 phase mask 和空 joint mask 行兜底锚点，避免后续重构重新引入随机探索空集合采样风险。
  - `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md` 补充 workflow 不决策语义与 Agent 直接推理兜底语义的区别。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认真实 `legal_action=0` 的帧不会进入 `predict()`，若平台评估入口直接调用 `exploit()`，空 mask 也不会触发空采样异常。

### Step 76 - 样本 done 标记解析修正

- 状态：完成
- Commit：`7bfee58`
- 内容：
  - `sample_process()` 的 `_not_done_flag()` 显式解析 bool、有限数值和 true/false 字符串。
  - `"true"`、`"1"`、`"yes"`、`"y"` 会写成终局 `not_done=0`；`"false"`、`"0"`、`"no"`、`"n"` 和空字符串写成非终局 `not_done=1`。
  - 未知字符串、NaN/Inf 和异常对象按非终局处理，避免平台字符串 done 标记被 `int()` 解析失败后误判。
  - 无平台依赖测试增加 `_not_done_flag()` 的 bool、字符串和 Inf 覆盖；静态测试增加字符串 done 解析锚点。
- 验证：
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认样本池终局 transition 的 `done` / `not_done` 语义与 learner target 一致。

### Step 77 - 样本 Frame 属性异常隔离

- 状态：完成
- Commit：`e70b479`
- 内容：
  - `sample_process()` 新增 `_safe_getattr()`，隔离 Frame 属性访问异常。
  - `obs` 或 `act` 属性读取失败时跳过当前帧；`rew`、`done` 或 `legal_action` 读取失败时使用默认值继续构造样本。
  - 无平台依赖测试增加 `rew`、`done`、`legal_action` 属性抛错的 Frame 覆盖；静态测试增加 `_safe_getattr()` 锚点。
- 验证：
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 sample collector 中个别异常 Frame 不再导致整段样本转换失败。

### Step 78 - 样本字段转换异常隔离

- 状态：完成
- Commit：`4798ecd`
- 内容：
  - `_fixed_float_list()` 捕获异常 array-like 转换失败，异常 observation 或 reward 会归一化为零向量。
  - `sample_process()` 检查 action 长度和前三个字段时隔离未预期异常，畸形 action 对象只跳过当前帧。
  - 无平台依赖测试增加 `__array__` 抛错的 obs/reward 和 `__len__` 抛错的 action 覆盖；静态测试增加字段转换异常锚点。
- 验证：
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认异常样本字段不会触发 `sample process failed` 并导致整段 collector 被丢弃。

### Step 79 - 合法动作归一化异常隔离

- 状态：完成
- Commit：`5379302`
- 内容：
  - `normalize_phase_legal_action()` 对异常 array-like 合法动作输入回退为四个相位都可选。
  - `_phase_array()` 同步隔离异常 array-like 转换失败，返回零相位数组。
  - `_finite_float()` 补充 `OverflowError` 处理，减少异常数值对象中断交通统计 helper 的风险。
  - 无平台依赖测试增加 `legal_action.__array__()` 抛错和 `_need_to_predict()` 复用该兜底的覆盖；静态测试增加 traffic helper 转换异常锚点。
- 验证：
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认真实 `legal_action` 的异常格式是否应按全相位可选处理，或需要更严格地区分不可决策帧。

### Step 80 - 模型输入非有限值清洗

- 状态：完成
- Commit：`03e171f`
- 内容：
  - `Model._prepare_input()` 在补零/截断后统一调用 `torch.nan_to_num()`，把输入 NaN/Inf 清零。
  - `Model._as_numpy_array()` 对异常 array-like observation 转换失败使用零向量或零行兜底。
  - 静态测试增加模型输入非有限值清洗和异常 numpy 转换隔离锚点。
- 验证：
  - 已运行 `python tests/test_target_dqn_static.py`，通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 有 `torch` 环境后补跑 `python tests/test_target_dqn_smoke.py`，确认直接模型 forward 对 NaN/Inf 输入输出有限 Q 值。

### Step 81 - learner 样本字段异常隔离

- 状态：完成
- Commit：`a793eaa`
- 内容：
  - `Algorithm.learn()` 新增 `_sample_field()`，读取样本字段时隔离属性访问异常。
  - `obs`、`_obs` 字段读取失败时使用零 observation；`act` 失败时使用 `[0, 0, MIN_GREEN_DURATION]`；`rew` 失败时使用 `[0.0, 0.0]`；`done` 失败时按非终局；`legal_action` 失败时按全相位可选。
  - `_normalize_tensor()` 捕获未预期 tensor 转换异常，坏字段转换为零张量后继续定宽补齐。
  - 静态测试增加 learner 安全字段读取和异常 tensor 转换锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/algorithm tests/test_target_dqn_static.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 有 `torch` 环境后补充/运行 smoke，确认异常样本字段不会让整批 `Algorithm.learn()` 失败。

### Step 82 - env.step 常见 tuple 兼容

- 状态：完成
- Commit：`d7176f4`
- 内容：
  - `_normalize_step_result()` 增加 Gym 四元返回 `(observation, reward, done, info)` 兼容。
  - `_normalize_step_result()` 增加 Gymnasium 五元返回 `(observation, reward, terminated, truncated, info)` 兼容。
  - 保留当前二元封装返回和作业文档六元返回路径，减少不同平台包装层切换时的训练入口差异。
  - 无平台依赖测试增加四元/五元返回归一化覆盖；静态测试增加 tuple 格式锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认真实 `env.step()` 是否还存在其他返回结构，并将原始返回样例补入测试。

### Step 83 - workflow 映射读取异常隔离

- 状态：完成
- Commit：`6123127`
- 内容：
  - `_safe_env_value()` 捕获 env_obs/obs 映射 `.get()` 抛出的异常，并回退默认值。
  - `_safe_legal_action()` 和 `_need_to_predict()` 改为复用 `_safe_env_value()`，避免异常 dict-like observation 中断合法动作读取和预测门控。
  - 无平台依赖测试增加 `.get()` 抛错的 env_obs 覆盖；静态测试增加集中读取和 legal_action 安全读取锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后确认 env_obs/obs 是否始终是普通 dict；如出现异常映射对象，需要保存原始类型和 repr。

### Step 84 - learner 样本批次归一化

- 状态：完成
- Commit：`88495da`
- 内容：
  - `Algorithm.learn()` 入口新增 `_sample_batch()`，先把样本批次安全归一化为 list。
  - 支持 generator 式 batch 输入，避免 `len()` 假设导致 learner 外层直接跳过可消费样本。
  - 无法迭代的异常 batch 容器会记录 `sample batch iteration failed` 并跳过当前更新，不再抛到 `Agent.learn()` 外层。
  - 静态测试增加 batch 归一化和异常 batch 日志锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/algorithm tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_static.py`，通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 有 `torch` 和平台样本池环境后确认 learner 实际收到的 batch 类型，必要时把原始类型写入日志。

### Step 85 - Agent 评估映射读取隔离

- 状态：完成
- Commit：`3f03ad0`
- 内容：
  - 新增 `_safe_mapping_get()`，隔离 Agent 侧 dict-like 对象 `.get()` 抛错。
  - `exploit()` 读取外层 observation、`obs` 包装和 `extra_info` 时改用安全读取，避免异常 observation 在兜底链路前直接抛出。
  - `observation_process()`、`rule_based_action()`、相位特征和相位年龄特征的关键字段读取改用安全 helper。
  - `observation_process()` 隔离 `preprocess.update_traffic_info()` 异常，特征构造失败前仍可继续使用当前 observation 做保守处理。
  - 静态测试增加 Agent 安全映射读取锚点，并防止评估入口回退到直接 `observation.get()` / `raw_obs.get()`。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_static.py`，通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 有 `torch` 环境后补跑 smoke，确认异常 observation 评估入口会返回默认合法动作而不是抛错。

### Step 86 - Agent 预测批次归一化

- 状态：完成
- Commit：`a007acd`
- 内容：
  - `Agent.__predict_detail()` 入口新增 `_obs_batch()`，先把 ObsData batch 安全归一化为 list。
  - 支持 generator 式 ObsData batch；无法迭代的异常 batch 会记录 `predict observation batch failed` 并返回空动作列表。
  - 新增 `_obs_data_field()`，读取 `feature` 和 `legal_action` 属性时隔离异常 property。
  - 静态测试增加预测 batch 归一化、异常 batch 日志和异常 ObsData 属性隔离锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_static.py`，通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 有 `torch` 环境后补跑 smoke，确认 generator 式 ObsData batch 和异常属性不会中断直接 `predict()` 调用。

### Step 87 - duration reward 动作空间对齐

- 状态：完成
- Commit：`12c2f51`
- 内容：
  - 新增 `_max_action_duration()`，统一样本动作裁剪和 reward duration 目标上限。
  - `reward_shaping()` 的目标 duration 从环境全局 `MAX_GREEN_DURATION=40` 改为模型当前可表达的 `MIN_GREEN_DURATION + DIM_OF_ACTION_DURATION - 1`。
  - 避免高压场景下 duration reward 鼓励模型追逐无法由 80 维联合动作头输出的 40 秒动作。
  - 无平台依赖测试增加饱和压力下最大可表达 duration 的零惩罚覆盖；静态测试增加 reward/action duration 上限共享锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台训练后观察 `duration_reward` 是否仍长期强负；若是，继续校准压力尺度或扩大 duration 动作空间。

### Step 88 - duration 分桶覆盖完整绿灯范围

- 状态：完成
- Commit：`ed4beac`
- 内容：
  - `Config` 新增 `DURATION_STEP`、`duration_index_to_seconds()` 和 `duration_seconds_to_index()`，集中管理 duration bin 与秒数互转。
  - 保持 80 维联合动作 Q head 不变，20 个 duration bin 从只覆盖 `8-27` 秒改为覆盖 `MIN_GREEN_DURATION=8` 到 `MAX_GREEN_DURATION=40` 秒。
  - `action_process()`、规则兜底、workflow 动作清洗、样本动作裁剪、reward target 和 learner 反向索引统一使用同一 duration 范围/分桶。
  - 无平台依赖测试增加 duration 映射端点覆盖；smoke 期望同步到 `MAX_GREEN_DURATION`。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests/test_target_dqn_features.py tests/test_target_dqn_static.py tests/test_target_dqn_smoke.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台训练后观察 duration 分布、duration_reward 和切换惩罚，确认 20 个分桶覆盖 `8-40` 秒是否优于原先线性 `8+idx`。

### Step 89 - 平台评分指标监控

- 状态：完成
- Commit：`08e045c`
- 内容：
  - workflow 新增 `ENV_SCORE_ALIASES`、`_env_score_metrics()` 和 `_update_env_metric_snapshot()`，从 `env.step()` 的 score、`env_obs.score`、`extra_info.score_info` 或对象属性中安全提取平台评分项。
  - 训练监控和 epoch 日志增加 `env_score`、`avg_delay`、`avg_queue_length`、`avg_waiting_time` 和 `switch_penalty`，方便平台短训后直接对齐作业评分指标。
  - 无平台依赖测试增加 dict、对象属性、NaN 和缺失 snapshot 的评分指标提取覆盖；静态测试增加 score metric 监控锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 score 字段实际名称是否命中当前 alias；若监控长期为 0，需要保存一次 `score` / `extra_info` 样例后补 alias。

### Step 90 - reward 尺度裁剪

- 状态：完成
- Commit：`efc4d7c`
- 内容：
  - `Config` 新增 `REWARD_DELAY_CAP=300.0` 和 `REWARD_CLIP=5.0`，集中管理 reward 稳定性参数。
  - `reward_shaping()` 的平均延误惩罚先按上限截断，避免少量极端车辆延误主导 phase reward。
  - 新增 `_clip_reward()`，将 `phase_reward` 和 `duration_reward` 裁剪到固定范围，并把异常 reward 值回退为 0。
  - 无平台依赖测试增加 NaN/上下界裁剪和极端延误车辆覆盖；静态测试增加 reward cap/clip 锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/conf agent_target_dqn/feature tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台训练后观察 `value_loss`、`model_grad_norm` 和平均 reward；如仍爆炸，优先降低 `LR` 或继续收紧 reward 权重。

### Step 91 - 兼容缺失 target_junction 的车辆记录

- 状态：完成
- Commit：`61a1ea7`
- 内容：
  - `traffic_utils` 新增 `_vehicle_value()`，隔离车辆字段读取，并支持 dict 或对象属性形式。
  - `on_enter_lane()` 不再要求车辆必须包含 `target_junction`；缺失时按单路口目标路口处理，避免作业文档式车辆记录被全部跳过。
  - `in_junction()` 和 `on_depart_lane()` 改用安全字段读取，减少车辆字段缺失导致的 KeyError。
  - 无平台依赖测试增加缺失 `target_junction` 的车辆样例，覆盖进口车道判断、车道统计和相位压力统计；静态测试增加字段兼容锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实车辆记录是否提供 `target_junction`；如没有，观察相位压力、排队统计和 reward 是否不再长期为 0。

### Step 92 - 等待时间统计兼容 target_junction 缺失

- 状态：完成
- Commit：`adfa19c`
- 内容：
  - 将车辆字段读取 helper 提升为 `vehicle_value()`，供 traffic utils 和 preprocessor 复用。
  - 新增 `_default_target_junction()` 和 `_waiting_target_junction()`，等待时间统计在 `target_junction` 缺失且车辆可识别为进口车道时按单路口目标路口归类。
  - `get_all_junction_waiting_time()` 和 `get_all_junction_waiting_time_by_origin()` 不再因文档式车辆记录缺少 `target_junction` 而漏计，同时跳过没有车辆 ID 或无法识别为进口车道的畸形 targetless 记录。
  - 无平台依赖测试增加缺失 `target_junction` 的等待时间统计覆盖；静态测试增加共享字段读取和等待统计默认目标锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实车辆记录缺少 `target_junction` 时，交叉口等待时间统计、相位压力和 reward 是否仍有非零信号。

### Step 93 - 兼容对象式平台协议记录

- 状态：完成
- Commit：`acab81e`
- 内容：
  - `traffic_utils` 新增 `record_value()`，统一支持 dict 和属性对象字段读取，`vehicle_value()` 复用该 helper。
  - `FeatureProcess` 的路网初始化、动态交通更新、等待时间、行驶距离、车道车辆数和交叉口等待统计改用安全字段读取，兼容作业文档中的 FrameState / Vehicle / InitState 对象形态。
  - `reward_shaping()`、Agent 观测编码、规则兜底、相位特征和 workflow 环境字段读取兼容 dict 与属性对象，避免真实平台返回协议对象时被当成空 observation。
  - 无平台依赖测试增加属性对象车辆、属性对象路网初始化、对象式 reward observation 和 workflow object env_obs 覆盖；静态测试增加对象协议字段读取锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实 Observation / Vehicle / Phase 是否以属性对象返回；如果仍有字段读取失败，保存原始类型和 repr 后补具体字段别名。

### Step 94 - 收紧协议记录列表解析

- 状态：完成
- Commit：`74ec647`
- 内容：
  - `_is_record()` 排除 `bool`、`int`、`float` 和 `complex`，避免标量 observation / extra_info / vehicle 字段被误当作有效协议对象。
  - Agent、reward 和 preprocessor 的列表字段解析支持单个非迭代协议对象，兼容平台把单个 Vehicle、Phase、Junction 或 lane container 直接作为对象返回的边界情况。
  - workflow 的 `_safe_observation()` 和 `_safe_extra_info()` 对标量 payload 回退为空对象，避免坏字段触发不必要预测路径。
  - 无平台依赖测试覆盖单个对象式 InitState/Vehicle、单个对象式 reward vehicles、标量 observation/extra_info 回退；静态测试增加标量伪记录与单对象容器锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/feature/definition.py agent_target_dqn/feature/preprocessor.py agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后如果 observation 容器还有 list/dict 之外的特殊 repeated 类型，保存原始类型、repr 和可迭代行为后继续扩展解析。

### Step 95 - workflow 保留对象式环境返回

- 状态：完成
- Commit：`e20ba7d`
- 内容：
  - `_normalize_reset_result()` 支持对象式 reset 返回，避免平台直接返回 env_obs 对象时被归一化为空 dict。
  - `_normalize_step_result()` 的二元 step 返回保留对象式 env_obs，裸对象式 step 返回也按 env_obs 处理。
  - Gym 四元、Gymnasium 五元 step 返回中的对象式 `extra_info` 不再被替换为空 dict，保留 score_info、frame_no 等平台字段给后续安全读取 helper。
  - 无平台依赖测试覆盖对象式 reset、二元对象式 env_obs、四元/五元对象式 extra_info 和裸对象式 step 返回；静态测试增加归一化阶段对象保留锚点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `env.reset()` / `env.step()` 真实返回形态；如果返回 custom tuple 或 protobuf repeated wrapper，再按原始 repr 扩展归一化。

## 2026-05-31

### Step 96 - 完整代码包复核并更新 AGENTS

- 状态：完成
- Commit：`1e294a0`
- 内容：
  - 复核平台完整 `codebase/`、四套 agent、配置入口、测试目录、打包脚本和文档台账。
  - 更新 `AGENTS.md` 的实际项目结构，补充 `tests/`、`scripts/check_offline.sh`、`scripts/package_submission.sh`、`RUNBOOK.md`、`EXPERIMENTS.md`、`REPORT_DRAFT.md` 和 `PROGRESS.md` 的职责。
  - 在 `AGENTS.md` 增加本次完整代码包复核结论，明确当前主线仍是 `agent_target_dqn`，其它三套 agent 仍视为平台模板/备选代码。
  - 在 `AGENTS.md` 补充剩余风险：真实平台 `frame_state.lanes` 还没有作为 vehicles 稀疏时的 fallback 信号，dict 形态 repeated 字段需要真实样例后再扩展。
  - 将实施计划和测试计划更新为当前实际流程：普通本地优先跑 `./scripts/check_offline.sh`，平台环境再跑 `cd codebase && python train_test.py`。
- 验证：
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台环境可用后采集真实 observation 样例，优先确认 `vehicles` / `lanes` / `legal_action` 的实际协议形态。

### Step 97 - 接入 lanes 聚合压力 fallback

- 状态：完成
- Commit：`e6a5606`
- 内容：
  - `traffic_utils` 新增 lane 字段读取、进口车道 ID 映射、lane 聚合统计、lane 聚合相位压力和车辆/lane 统计合并 helper。
  - `observation_process()` 在解析 `frame_state.lanes` 后，将 lanes 传入交通统计，并把 lanes 逐车道统计与车辆逐车道统计合并，保持 `Config.DIM_OF_OBSERVATION = 638` 不变。
  - `rule_based_action()` 在车辆相位压力全 0 时使用 lanes 聚合压力，避免车辆明细缺失时评估兜底固定选择 0 相位。
  - `reward_shaping()` 在 `vehicles` 无有效进口车辆但 `lanes` 有有效聚合信息时，使用 lanes 聚合压力计算 phase/duration reward。
  - 无平台依赖测试覆盖 dict/object lane 记录、非有限 lane 字段、lane-only 交通摘要和 lane-only reward；静态测试增加 lanes fallback 代码锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 lanes fallback 已处理及真实平台待确认点。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `frame_state.lanes` 的真实容器类型、字段名、数值尺度，以及 lanes fallback 是否能避免 reward/压力长期为 0。

### Step 98 - 兼容 dict 形态 repeated 字段

- 状态：完成
- Commit：`22239d3`
- 内容：
  - Agent 的 `_as_record_list()` 支持单条 dict 协议记录和 dict-of-records，同时用已知 Vehicle / Phase / Lane 字段限制展开范围，避免把任意 dict 当作有效车辆。
  - reward 和 preprocessor 的 `_safe_list()` 支持单条 dict 记录、dict-of-records、dict-of-scalar lane ID 容器和嵌套 list/tuple 容器。
  - `FeatureProcess.update_traffic_info()`、`reward_shaping()`、`observation_process()`、`rule_based_action()` 和 road init 路径都能复用新的 dict 容器解析。
  - 无平台依赖测试覆盖单条 dict 车辆、dict-of-records 车辆、dict 车道、dict-of-scalar lane ID、dict 容器预处理和 dict 容器 reward。
  - 静态测试增加 dict repeated 容器解析锚点；`AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md` 更新当前兼容范围。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/feature/definition.py agent_target_dqn/feature/preprocessor.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后如果仍遇到 protobuf repeated wrapper 或特殊容器，保存原始类型、repr 和迭代行为，再继续扩展解析。

### Step 99 - 兼容文档式路网初始化字段别名

- 状态：完成
- Commit：`6e6795f`
- 内容：
  - `FeatureProcess.init_road_info()` 新增 `_first_record_value()`，统一读取多个字段别名。
  - 路口、边、车道和车辆配置初始化同时兼容模板字段 `j_id/e_id/l_id/v_config_id` 与文档式字段 `junction_id/edge_id/lane_id/vehicle_config_id`。
  - 无平台依赖测试增加文档式 `init_state` 样例，覆盖 dict-of-records 的 junctions、edges、lane_configs、vehicle_configs 和 `enter_lanes_on_directions`。
  - 静态测试增加路网初始化字段别名锚点；`AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md` 记录当前兼容范围。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature/preprocessor.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `extra_info.init_state` 的真实字段名；如仍有未识别命名，保存样例后继续扩展别名。

### Step 100 - 清洗车辆路口 ID 字段

- 状态：完成
- Commit：`14e60bf`
- 内容：
  - `traffic_utils` 新增路口 ID 清洗逻辑，`on_enter_lane()`、`in_junction()` 和 `on_depart_lane()` 会先把 `junction` / `target_junction` 转成有限整数再判断。
  - `FeatureProcess` 的跨帧路口状态和交叉口等待时间统计也改用清洗后的路口 ID，兼容平台把 `"0"` / `"-1"` 作为字符串返回。
  - 等待时间目标路口匹配新增 `_junction_key()`，当 `junction_dict` 的 key 是字符串 `"0"` 而车辆 `target_junction` 是 `"0"` 或数值 `0` 时仍能正确归属。
  - 无平台依赖测试增加字符串 `junction` / `target_junction`、畸形 target、字符串路网 key 的等待时间统计覆盖；静态测试增加路口 ID 清洗锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录字符串路口 ID 已处理及真实平台哨兵值仍需确认。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature/traffic_utils.py agent_target_dqn/feature/preprocessor.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实车辆 `junction` / `target_junction` 的哨兵值是否只使用 `-1`；如存在其它目标字段或哨兵值，保存原始车辆样例后扩展清洗规则。

### Step 101 - 归一化数字协议 ID

- 状态：完成
- Commit：`3e6fe4e`
- 内容：
  - `FeatureProcess.init_road_info()` 会将 `junction_id`、`edge_id`、`lane_id`、`vehicle_config_id` 以及模板字段 `j_id/e_id/l_id/v_config_id` 清洗为整数后写入路网字典。
  - `FeatureProcess.cal_v_num_in_lane()` 会将车辆 `lane` 清洗为整数后更新 `lane_volume`，避免 `"11"` 和 `11` 形成两套车道 key。
  - `get_phase_pressure()` 改用清洗后的车辆 lane ID 查相位，避免字符串 lane 让 reward 和交通摘要压力变成 0。
  - `Agent.observation_process()` 新增字符串/整数 key 匹配 helper，车辆 `v_config_id` / `vehicle_config_id` 为字符串时仍能命中 `vehicle_configs_dict`，路网 key 异常时也会保留默认单路口网格。
  - 无平台依赖测试增加字符串 lane、字符串路网 ID、字符串车辆配置 ID 和字符串车道统计覆盖；静态测试增加 ID 清洗锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录数字协议 ID 字符串化已处理。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/feature/traffic_utils.py agent_target_dqn/feature/preprocessor.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实 observation 是否还有非数字 ID 或额外字段别名；如有，保留原始样例再扩展字段清洗和 alias 映射。

### Step 102 - 兼容裸 observation 环境返回

- 状态：完成
- Commit：`05befef`
- 内容：
  - workflow 新增 `_looks_like_observation()`，可识别直接包含 `frame_state` 或 `legal_action` 的原始 observation dict / 对象。
  - `_safe_observation()` 在缺少嵌套 `observation` 字段时，会将裸 observation 原样传给后续决策逻辑，避免 reset/step 直接返回原始 observation 时被误归一化为空 dict。
  - 保留现有嵌套 `observation`、对象式 env_obs、二元 step/reset、Gym 四元、Gymnasium 五元和文档六元 step 返回兼容逻辑。
  - 无平台依赖测试增加裸 dict observation、裸对象 observation、reset 原样返回和 step 原样返回覆盖；静态测试增加 workflow 裸 observation 识别锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 workflow 已兼容直接返回原始 observation 的平台封装。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `env.reset()` / `env.step()` 的真实返回是否为裸 observation、嵌套 env_obs 或其它封装；若仍有空观测日志，保存原始返回 repr 后继续扩展归一化。

### Step 103 - 兼容 observation 与 state 字段别名

- 状态：完成
- Commit：`c0b3297`
- 内容：
  - workflow 的 `_safe_observation()` 兼容 `observation`、`obs`、`_obs` 三种观测字段名。
  - workflow 的 `_safe_extra_info()` 兼容 `extra_info`、`_state`、`state` 三种额外信息字段名。
  - `Agent.exploit()` 兼容评估入口传入 `obs`、`observation`、`_obs` 观测包装和 `extra_info`、`_state`、`state` 信息包装，避免评估侧字段别名导致空 observation。
  - 无平台依赖测试增加 dict 与对象式 alias env_obs 覆盖；静态测试增加 workflow 和 exploit 字段别名锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录作业文档中的 `observation/_obs`、`extra_info/_state` 别名已覆盖。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 env_obs 和评估入口是否还有其它包装字段；若仍出现空 observation 或默认动作，保存原始类型和 repr 后继续补 alias。

### Step 104 - 兼容 dict/object step envelope

- 状态：完成
- Commit：`42e53e1`
- 内容：
  - workflow 新增 `_normalize_step_record_result()`，当 `env.step()` 直接返回 dict 或对象式 step envelope 时，会抽取 reward/score、observation、done、truncated 和 extra_info。
  - 新增 `_looks_like_step_envelope()` 和 `_first_env_value()`，集中识别 `observation` / `obs` / `_obs`、`reward` / `score` / `env_reward`、`terminated` / `done`、`extra_info` / `_state` / `state` 等常见字段别名。
  - 保留裸 observation 的优先级：直接带 `frame_state` / `legal_action` 的 dict 或对象仍原样作为 observation 处理，不会误当 step envelope。
  - 无平台依赖测试增加 dict step envelope、对象式 step envelope、`done` 字符串、`score` 数值、对象式 state 和裸 observation 优先级覆盖；静态测试增加 step envelope 识别锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 `env.step()` dict/object envelope 已兼容。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `env.step()` 是否还有其它字段名，例如独立 `info` 或自定义 score 容器；若有，保存原始返回样例后继续扩展。

### Step 105 - 兼容嵌套帧号别名

- 状态：完成
- Commit：`901f50d`
- 内容：
  - workflow 的 `_safe_frame_no()` 先读取顶层 `frame_no` / `frameNo`，缺失时再从 `extra_info` / `_state` / `state` 中回退读取同名帧号字段。
  - Gym 四元、Gymnasium 五元和 dict/object step envelope 的帧号归一化都改走 `_safe_frame_no()`，避免平台把帧号放入 extra info 或对象属性时日志和终局路径长期显示 0。
  - 无平台依赖测试增加对象式 `frameNo`、嵌套 `extra_info`、`_state` 和 `state` 帧号覆盖；静态测试增加帧号 alias 和 step fallback 锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 workflow 帧号读取已支持顶层与嵌套字段别名。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后保存一次真实 env_obs / step_result 样例，确认帧号字段是否还有 `frame`、`frameId` 或独立 `info` 容器等新别名。

### Step 106 - 兼容 info 额外状态别名

- 状态：完成
- Commit：`9275fa5`
- 内容：
  - workflow 将 `info` 纳入 `extra_info` / `_state` / `state` 之后的额外状态别名，dict/object step envelope、帧号 fallback 和 score/状态读取都能复用该容器。
  - `Agent.exploit()` 评估入口也会读取 `info` 作为额外信息，兼容 Gym 风格封装或评估侧把状态对象命名为 `info` 的情况。
  - 无平台依赖测试增加 `info` step envelope、`info.frameNo` 帧号 fallback 和 `_safe_extra_info()` 覆盖；静态测试增加 workflow 与 exploit 的 `info` alias 锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 `info` 已作为额外状态容器支持。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认 `info` 中是否还包含更细的 score 或环境状态子容器；若平台监控字段仍为空，保存原始 `info` 样例后扩展 score alias。

### Step 107 - 读取嵌套平台指标容器

- 状态：完成
- Commit：`cad2748`
- 内容：
  - workflow 新增 `METRIC_SOURCE_KEYS`，集中维护 `score`、`score_info`、`scoreInfo`、`metrics`、`env_info`、`info` 等平台指标容器别名。
  - `_append_metric_sources()` 改为带 `seen` 集合和最大深度的有界递归，能读取 `info.env_info.metrics` 这类嵌套指标，同时避免循环引用卡住监控提取。
  - 无平台依赖测试增加嵌套 `metrics` / `info.env_info.metrics` 中的 `total_score`、`avg_wait_time`、`switch_count` 提取覆盖；静态测试增加指标容器 alias 和有界递归锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录平台 score 监控已支持嵌套指标容器。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后对比 monitor 上报与评估页面指标；如果仍缺字段，保存 env_reward/env_obs/info 样例后补充对应容器或字段别名。

### Step 108 - 兼容终局与截断字段别名

- 状态：完成
- Commit：`aeefce7`
- 内容：
  - workflow 新增 `DONE_FIELD_ALIASES`，集中维护 `terminated` / `done` / `is_done` / `terminal` 和 `truncated` / `timeout` / `is_truncated` 等结束或截断字段别名。
  - dict/object step envelope 归一化和 `_safe_done_flag()` 都改走同一别名表，并会从 `extra_info` / `_state` / `state` / `info` 中回退读取截断标记。
  - 无平台依赖测试增加 `isDone`、`timeout`、`terminal`、`timedOut`、嵌套 `info.is_truncated` 覆盖；静态测试增加终局字段别名锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 workflow 已兼容更多终局/截断别名。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认终局和超时日志是否与真实 episode 结束原因一致；如平台使用其它字段名，保存 step_result 样例后扩展 `DONE_FIELD_ALIASES`。

### Step 109 - 兼容相位字段别名

- 状态：完成
- Commit：`4ea8b94`
- 内容：
  - `Agent` 新增 `_phase_record_value()`，相位特征和相位服务年龄统一通过安全 helper 读取字段别名。
  - 相位记录现在兼容 `s_id` / `signal_id` / `signal_idx`、`phase_id` / `phase_idx` / `current_phase` / `current_phase_id`、`remaining_duration` / `remaining_time` / `remain_duration` / `remain_time`。
  - `_RECORD_FIELD_KEYS` 同步加入相位别名，避免 dict-of-records 解析时把别名相位记录误当成普通嵌套 dict。
  - smoke 测试增加 `signal_id + current_phase + remaining_time` 观测覆盖；静态测试增加相位字段 alias 锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录相位字段别名兼容情况。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py tests/test_target_dqn_static.py tests/test_target_dqn_smoke.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实 `frame_state.phases` 是否还有 `signalId`、`phase`、`remainTime` 等驼峰字段；如有，保存样例后继续扩展相位 alias。

### Step 110 - 兼容相位驼峰字段别名

- 状态：完成
- Commit：`1ee80ec`
- 内容：
  - `Agent` 相位字段 alias 扩展到 `signalId` / `signalIdx`、`phase` / `phaseId` / `phaseIdx`、`currentPhase` / `currentPhaseId`、`remainingDuration` / `remainingTime` / `remainDuration` / `remainTime`。
  - `_RECORD_FIELD_KEYS` 同步加入驼峰相位字段，避免 dict 容器解析时漏掉仅使用驼峰命名的相位记录。
  - smoke 测试中的相位 alias 观测改用 `signalId + phase + remainingTime`，静态测试增加驼峰字段锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录相位驼峰字段别名已覆盖。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py tests/test_target_dqn_static.py tests/test_target_dqn_smoke.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认相位特征 one-hot、remaining duration 和相位年龄是否与真实 `frame_state.phases` 一致；如仍异常，保存原始 phases 样例继续补 alias。

### Step 111 - 兼容车道观测字段别名

- 状态：完成
- Commit：`536e61d`
- 内容：
  - `traffic_utils` 新增 lane 字段别名读取，`frame_state.lanes` 现在兼容 `laneId` / `laneIdx`、`vCount` / `vehicle_count` / `vehicleCount`、`queueLength` / `queue_count` / `queueCount` 和 `congestionLevel`。
  - `Agent` 与 reward 的 `_RECORD_FIELD_KEYS` 同步加入 lane 驼峰/别名字段，避免单条 dict lane 记录只使用驼峰字段时被 dict 容器解析丢弃。
  - 无平台依赖测试增加 lane alias 的统计、相位压力和 lane-only reward 覆盖；静态测试增加 lane alias 锚点。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录 lanes fallback 已覆盖常见驼峰字段。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/feature/traffic_utils.py agent_target_dqn/feature/definition.py agent_target_dqn/agent.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实 `frame_state.lanes` 是否还有其它字段名或单位差异；若 lanes fallback 仍为空，保存原始 lanes 样例继续扩展字段映射。

### Step 112 - 兼容合法动作字段别名

- 状态：完成
- Commit：`8b64227`
- 内容：
  - `Agent` 新增集中合法动作字段别名读取，`observation_process()` 和 `rule_based_action()` 现在兼容 `legalAction`、`phaseLegalAction`、`actionMask`、`phaseMask` 等字段。
  - workflow 新增 `LEGAL_ACTION_KEYS`，`_safe_legal_action()`、`_need_to_predict()` 和裸 observation 识别都通过同一别名表读取合法动作。
  - 无平台依赖测试增加 dict/object 裸 observation 的 `legalAction`、`phaseLegalAction`、`actionMask` 和 `phaseMask` 覆盖；smoke 观测改用 `legalAction` 验证 Agent 路径。
  - 更新 `AGENTS.md`、`RUNBOOK.md` 和 `REPORT_DRAFT.md`，记录合法动作字段别名兼容情况。
- 验证：
  - 已运行 `python -m compileall agent_target_dqn/agent.py agent_target_dqn/workflow/train_workflow.py tests/test_target_dqn_features.py tests/test_target_dqn_static.py tests/test_target_dqn_smoke.py`，通过。
  - 已运行 `python tests/test_target_dqn_features.py` 和 `python tests/test_target_dqn_static.py`，均通过。
  - 已运行 `./scripts/check_offline.sh`，所有离线检查通过；smoke 因当前本地缺少 `torch` 明确 skip。
- 下一步：
  - 平台运行后确认真实 observation 的合法动作字段是否命中当前别名；如果仍出现长期不决策或过度决策，保存原始 observation 样例继续校准门控语义。
