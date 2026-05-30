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
