# Mars-GP-IPP

火星磁场信息路径规划：GPy 共享核三输出在线估计和原生 PyTorch PPO。祝融训练、毅力测试；16×16 动作网格、8 个动作、70 步任务、31×31 全局评价网格。

## 专用环境

本项目是非 ROS 项目，使用专用 Conda 环境 `mars_gp_ipp`。共享的 `marl_env` 不作修改。

```bash
conda create -n mars_gp_ipp python=3.10 pip
conda activate mars_gp_ipp
# ROS 终端先隔离继承的 Python 搜索路径，禁止加载用户级包。
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pip install --dry-run -r requirements-lock.txt
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
```

`requirements.txt` 使用兼容范围，`requirements-lock.txt` 记录实际验收版本。新环境中的锁定安装包含 GPU 库；已有环境先检查安装预览，不覆盖共享环境。脚本 bootstrap 在导入计算库前限制线程并设置 CUDA 确定性参数，训练和测试继续禁用 TF32、启用确定性算法。

## 训练与正式评估

```bash
python scripts/train.py --config configs/smoke.yaml --output results/main/smoke_new/seed_0
python scripts/evaluate.py --checkpoint results/main/smoke_new/seed_0/checkpoint.pt --output results/main/smoke_new_evaluation
python scripts/plot_results.py results/main/smoke_new/seed_0
python scripts/plot_results.py results/main/smoke_new_evaluation --formats png pdf
python scripts/run_comparison.py --msr-checkpoint results/main/msr/seed_0/checkpoint.pt --output results/comparison/4baseline
```

默认只输出 PNG，`train.py`、`evaluate.py`、`run_main.py` 和 `plot_results.py` 可显式传入 `--formats png pdf`。现有结果目录默认拒绝覆盖；重新验证请指定新目录。

正式配置仍为 10000 Episode，每 200 Episode 终点评估并保存，使用固定测试种子 1000–1049；结束回合必定评估、保存。smoke 是 40 Episode、每 10 Episode、5 个测试种子。两项间隔若不同，任一到期均执行一次统一保存周期。保存周期与 PPO 的 2000 步 rollout 更新相互独立，完整训练结束才更新剩余 Buffer。

训练每回合只在终点计算全局指标，所有动作仍照常采样、更新 GP 和计算奖励。训练期评估只保存每个 seed 的终点；正式评估保留 prior、reset 和全部 70 步。

```text
results/main/<experiment>/seed_<seed>/
  config.yaml            metadata.json       checkpoint.pt
  train_metrics.csv      updates.csv         eval_endpoints.csv
  summary.json           run_status.json
  figures/training.png   figures/training_evaluation.png
results/main/<formal_evaluation>/
  config.yaml            metadata.json
  eval_metrics.csv       trajectories.csv    field_snapshot.npz
  figures/               # 逐步指标、场图和终点汇总
```

模型保存采用 format 2，仅含网络权重、完整配置、scaler、进度和模型版本，不支持断点续训。format 1 模型仍可用于正式评估，不读取续训状态。CSV 按周期增量追加；中途崩溃可能丢失当前未保存周期，不提供恢复。

## 正式测试图与论文样式

已有正式测试数据可直接重新绘图，不重新加载模型或计算 GP：

```bash
python scripts/plot_results.py results/main/full_rbf_eval_seed_0 \
  --formats png pdf
```

生成四张图（默认只输出 PNG；上面的命令同时导出 PDF）：

| 文件 | 内容 |
|---|---|
| `field_map.png` | 3行×2列，左侧离线参考场、右侧在线重构场；按 Br、Bφ、Bψ 排列，使用柔和的 hot 等值填色，同分量共享色阶 |
| `error_trajectory.png` | 1行×3列的绝对误差场，cool 蓝紫色色图，红色点线轨迹及圆形起点/方形终点 |
| `component_rmse.png` | 三分量 RMSE 均值同轴显示，Br error / Bφ error / Bψ error，青绿/红/紫虚线，无阴影 |
| `evaluation.png` | 1行×2列，总体 RMSE 与 GP Trace，均值±总体标准差 |

场图和轨迹使用保存快照中的第一个固定测试 seed（当前为1000）；曲线使用全部50个固定测试 seed 的统计。曲线0步是初次采样后的 reset，随后是1–70动作步；CSV仍保留采样前 prior。CSV中的 B2/B3 仅在显示时映射为 Bφ/Bψ，不改数据字段。

毅力场图使用绝对经纬度：横轴 Longitude 65–95°，纵轴 Latitude 5–35°；祝融场图使用对应场景原点。旧图曾将经纬度标签写反，现在采用数据实际含义。坐标偏移只在绘图中执行，保存的网格和轨迹仍是局部坐标。多算法比较时三分量图分别保存为 `component_rmse_<algorithm>.png`，总体图继续叠加各算法。

## 批量实验与汇总

```bash
python scripts/run_main.py --config configs/main/original.yaml --dry-run
python scripts/run_main.py --config configs/main/matern.yaml --dry-run
python scripts/run_main.py --config configs/main/lengthscale.yaml --dry-run
python scripts/run_ablation.py --config configs/ablation/reward_terms.yaml --dry-run
python scripts/run_ablation.py --config configs/ablation/reward_weights.yaml --dry-run
```

去掉 `--dry-run` 会启动配置中的完整实验，需要自行明确安排。单次 train 只运行选定训练 seed，批量入口才展开多种子。多目录汇总需先分别完成正式评估，再将这些评估目录传给 `plot_results.py` 并指定新的 `--output`，不能把终点评估冒充逐步曲线。

## 数据与方法

原始工作簿在 `data/raw/`，生产真值模型在 `data/models/`，敏感性实验自动生成并复用 `data/models/generated/` 的缓存。生成场不会覆盖两个原始模型，长度尺度倍率 1.0 仍独立生成和验证。

本仓库保留四份核心文档：

- [实验记录](docs/实验记录.md)：实验对比、指标和曲线。
- [GP Trace Greedy 与全局最优路径对比](docs/GP_Trace_Greedy与全局最优路径对比.md)。
- [GP Trace 与 RMSE 是否等价](docs/GP_Trace与RMSE是否等价问题.md)。
- [复现命令](docs/复现命令.txt)：本机与远端运行、同步命令记录。

实验记录直接引用的 25 张图片随仓库保留，继续使用原 `results/` 相对路径。其他 docs 文档和完整实验结果只保留在本地，不在默认上传范围内；训练与评估所需的完整产物仍需自行生成。

## B-Norm + No-MH + Add-Uncertain

新版13维主方法、统一在线GP、随机起点Cover及双场景五方法对照，见[实验记录](docs/实验记录.md)。完整实验入口为 `python scripts/run_bnorm_experiment.py`，复用现有Msr模型并保留全部旧实验产物。

## V2 Local–Global

22维局部/全局不确定度候选方法及GlobalOnly消融，对比结果见[实验记录](docs/实验记录.md)。入口 `scripts/run_local_global_experiment.py`，四Baseline直接复用已有产物。

## V3 / V3.1 测试地图

`scripts/complete_v3_1_recovery.py` 支持 test1～test5，默认冻结现有 V3、V3.1 seed-0 模型，与 GP-Trace 比较，不启动训练。V3.1 模型读取 `results/main/v3_1_gp_trace_global_raw_train1_recovery/seed_0/checkpoint.pt`。

```bash
python3 scripts/complete_v3_1_recovery.py --maps test1
# 单 seed 冒烟评估：使用新的输出目录
python3 scripts/complete_v3_1_recovery.py --maps test1 --seeds 1000 --output /tmp/mars_test1_smoke_new
```

默认使用 seeds 1000–1049、每回合 70 步，结果写到 `results/comparison/v3_1_gp_trace_global_raw_train1/<地图>/`，包含三方法原始结果、汇总和图表。`--maps` 可指定多张地图，省略时选择 test1～test5；选中目录已存在会停止，需使用新的 `--output`。其他地图及顶层历史报告不会改写。原来的完整历史恢复流程需显式传入 `--historical-recovery`，不能与地图评估参数混用。

## V3.1–V3.4 独立变体与五图训练

新入口 `scripts/run_v3_1_v3_4.py` 默认只预检；三个新策略独立训练，V3.1 冻结历史模型，GP-Trace 继续使用原有局部8邻域算法。`smoke` 仅运行每版30回合及五图双种子评估，不启动正式10000回合训练。

## V3.3 / V3.4 奖励权重实验

独立入口 `scripts/run_v3_weight_experiment.py` 默认只预检，支持七组辅助奖励权重的冒烟、正式训练、配对测试和CSV重绘。`smoke` 每版训练30回合，再执行五图双种子测试，不启动正式长训练；结果与旧实验隔离。

## V3.3.4 / V3.5：Action Mask 与三项奖励

独立入口 `scripts/run_v3_3_4_v3_5.py` 默认只读预检，不调用旧七组权重训练入口。两版显式开启 `ppo.action_mask`，旧配置缺省关闭；历史 checkpoint 保留原始无 mask 推理行为。checkpoint 仍仅用于推理恢复，不支持断点续训。

```bash
conda activate mars_gp_ipp
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python scripts/run_v3_3_4_v3_5.py preflight
python -m pytest -q tests/test_ppo.py tests/test_v3_variants.py tests/test_v3_weights.py tests/test_v3_masked.py
python scripts/run_v3_3_4_v3_5.py smoke
# 仅从已有 CSV 重绘到新目录，不载入场模型、不执行 GP 评估
python scripts/run_v3_3_4_v3_5.py plot --smoke-results --output /tmp/mars_masked_replot_new
```

smoke 每版30回合、2100步，每10回合保存并在五张训练图各使用5个固定 monitor seeds。随后在 test1～test5 使用 seeds=1000、1001 做70步配对评估；这些产物仅用于流程验收。V3.3.4 对比 GP-Trace、原始 V3.3.3；V3.5 对比 GP-Trace、新 V3.3.4、原始 V3.4.1。缺失历史模型时明确失败，不替代、不重训。

训练目录为 `results/main/v_3.3.4_smoke/`、`v_3.5_smoke/`，根目录仅包含 `checkpoint.pt`、`figure/`、`others/`；逐步奖励在 `others/train_steps.csv`，回合奖励在 `others/train_metrics.csv`。测试目录为 `results/comparison/v_3.3.4_smoke_test/`、`v_3.5_smoke_test/`，每张地图包含 `figure/`、`table/comparison.md`、`others/`。两版训练及五图测试均保存 `acceptance.json`；顶层 `others/execution.json` 记录历史文件哈希验证和耗时。非空输出目录拒绝覆盖，失败现场保留。

`train`、`evaluate` 阶段分别提供正式训练及正式评估能力，但不属于本次 smoke 执行范围。正式输出名为 `v_3.3.4_train` / `v_3.5_train` 和 `v_3.3.4_test` / `v_3.5_test`。V3.3.4 同时改变 mask 与重复惩罚；V3.5 同时增加磁场状态与 MH 奖励，不能从两版比较中单独归因。

### V3.3.3_m：仅加入出界动作 mask

`configs/main/V3_3_3_m.yaml` 继承 V3.3.3，保持 11 维观测和奖励权重 `[1.0, 0.005]`，仅启用出界动作 mask。训练 seed=0、10,000 回合×70步；每200回合在 train1–train5 使用固定5个种子监测。正式评估使用 test1–test5、每图50个配对种子1000–1049，只对比原始 V3.3.3 和 V3.3.3_m。最终检查点用于正式评估，原基准不重新训练。

在上述 `mars_gp_ipp` 环境内执行：

```bash
python scripts/run_v3_3_3_m.py preflight
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ppo.py tests/test_v3_variants.py tests/test_v3_weights.py tests/test_v3_masked.py tests/test_v3_3_3_masked.py
python scripts/run_v3_3_3_m.py smoke
python scripts/run_v3_3_3_m.py train
python scripts/run_v3_3_3_m.py evaluate
python scripts/run_v3_3_3_m.py plot --output /tmp/mars_v3_3_3_m_replot_new
```

正式目录为 `results/main/v_3.3.3_m_train/` 和 `results/comparison/v_3.3.3_m_test/`，冒烟目录在对应名称后追加 `_smoke`。`plot --smoke-results` 从冒烟 CSV 重绘。非空目录拒绝覆盖；重跑通过 `--train-root`、`--test-root` 指定新的带时间戳父目录，后续阶段使用相同参数。训练保存 `figure/`、`others/` 和检查点，评估每图保存两方法逐步指标、轨迹、30/50/70步汇总及图表，顶层 `others/summary_70.csv` 和 `table/report.md` 包含五地图平均及相对基准差值。基准缓存须通过配置、检查点、地图、种子及轨迹审计；不满足时只重评基准。执行记录保存历史文件哈希，冒烟结果不可作为正式性能结论。
