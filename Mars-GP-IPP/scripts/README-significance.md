# V3.5.7 显著性分析复现

仅分析现有数据，不启动训练、评估或 ROS。使用系统 Python 创建独立环境，不使用 conda：

```bash
# 如果当前已激活 conda，先运行 conda deactivate，直到退出所有 conda 环境。
which python3  # 应为 /usr/bin/python3
cd /home/bit/xhk_IPP_MARS/Mars-GP-IPP
/usr/bin/python3 -m venv ../.venv-significance
../.venv-significance/bin/python -m pip install -r scripts/requirements-significance.txt
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv-significance/bin/python -m pytest tests/test_v3_5_7_significance.py -q
../.venv-significance/bin/python scripts/analyze_v3_5_7_significance.py
```

默认输出 `results/comparison/v3_5_7_significance/`，含 50 对主分析和
`supplementary_250/` 中的 250 对补充分析。每个范围分别进行八次双侧配对 t 检验，
按指标进行四项 Holm 校正，并统一提供 Wilcoxon 补充检验及差值诊断。

输出目录必须为空；复跑时用 `--output-dir` 指定新的空目录，以防旧结果与失败审计混杂。
可用 `--project-root` 指定迁移后的项目目录；历史 metadata 中的两个项目绝对路径会按项目名称重新定位。
依赖版本和实际运行命令保存在 `environment.json`。地图 pickle 必须为可信的项目产物。

源数据、配置、metadata、地图模型、轨迹、检查点及算法标识依据的 SHA256
保存在 `source_manifest.json`；在计算前后核验不变。`raw_step70.csv` 和
`paired_samples.csv` 保留原始文件及 CSV 行号。审计失败只生成审计与说明，
不会发布显著性结果；失败输出目录也须保留或另选新目录。

50 对对应五图各 seed 1000–1009，250 对对应五图各 seed 1000–1049。
固定地图及固定 checkpoint 上的回合差值并不代表独立训练重复，跨地图复用相同
seed 也可能带来相关性；结果报告明确保留这些推断限制。
