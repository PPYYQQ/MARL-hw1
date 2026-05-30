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
