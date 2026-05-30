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
- `value_loss`：DQN TD loss，先看是否有限值且不爆炸。
- `q_value` / `target_q_value`：观察 Q 值是否数值稳定。
- `model_grad_norm`：梯度范数，频繁过大说明奖励或学习率可能不稳。
- 平台评分项：平均延误、平均排队长度、平均等待时间、信号切换惩罚。

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
- `load latest model` 找不到文件：当前代码会跳过首次 `latest` 加载，属于从零训练的正常情况。
- `latest` 模型结构不兼容：当前联合动作模型会跳过不兼容的旧 `latest` checkpoint，并从当前参数继续训练；若要强制加载指定模型 ID，结构不兼容仍会抛错。
- `legal_action` 是标量而不是列表：当前 workflow 会先归一化为 4 维相位 mask，再判断是否需要决策；若平台提供相位级 mask，也会沿用相位约束。
- `run_episodes error: ...`：优先看冒号后的原始异常信息和 Python chained traceback，当前 workflow 不再只抛通用错误。
- reward 长期为 0：检查 `reward_shaping()` 是否收到真实车辆字段、`vehicles` 是否为空、相位压力是否一直为 0。
- loss 爆炸或 NaN：优先降低 `Config.LR`，再缩小 reward 权重或检查 observation 是否含 NaN。
