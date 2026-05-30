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
- 内容：
  - 新增 `REPORT_DRAFT.md`，整理问题定义、Target-DQN 方法、状态特征、动作映射、奖励设计、规则兜底、训练流程、本地验证、平台实验表和后续改进方向。
  - 报告草稿可迁移到 `icml2022.zip` 模板继续排版。
- 验证：
  - 已运行 `python tests/test_target_dqn_static.py`，静态约束检查通过。
  - 已运行 `git diff --check`，未发现空白错误。
  - 已检查 `REPORT_DRAFT.md`、`RUNBOOK.md`、`PROGRESS.md` 行数。
- 下一步：
  - 平台环境可用后补真实实验结果和曲线。
