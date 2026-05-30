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
