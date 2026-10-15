# Mars-GP-IPP-Code

火星磁场高斯过程建模与信息路径规划。两个项目保持并列，部分实验脚本通过相对路径读取另一项目的模型。

| 项目 | 用途与运行入口 |
| --- | --- |
| [Mars_Env_GP](Mars_Env_GP/README.md) | 磁场 GP 建模、十区域数据集训练、模型与场图导出；环境和运行命令见项目说明。 |
| [Mars-GP-IPP](Mars-GP-IPP/README.md) | GP 在线估计、PPO 信息路径规划、基线对比与评估；环境和运行命令见项目说明。 |

两个项目分别使用其说明中的环境，不要求共用一个 Python 环境。核心文档：

- [实验记录](Mars-GP-IPP/docs/实验记录.md)
- [GP Trace Greedy 与全局最优路径对比](Mars-GP-IPP/docs/GP_Trace_Greedy与全局最优路径对比.md)
- [GP Trace 与 RMSE 是否等价](Mars-GP-IPP/docs/GP_Trace与RMSE是否等价问题.md)
- [复现命令](Mars-GP-IPP/docs/复现命令.txt)

## 默认上传内容

保留源码、配置、测试、依赖文件、当前数据工作簿、参考资料和小型 GP 模型。`Mars-GP-IPP/docs/` 仅保留上述四份核心文档，并附带文档直接引用的 25 张实验图片，沿用原 `results/` 路径。`Mars_Env_GP/output/` 包含部分实验直接使用的十区域模型，因此保留。

另外保留以下三个策略目录内的 `checkpoint.pt`、`config.yaml`、`metadata.json` 和 `summary.json`，共 12 个文件。路径均相对于 `Mars-GP-IPP/`：

| 策略 | 归档目录 |
| --- | --- |
| Msr | `results/main/msr/seed_0/` |
| V2.1 | `results/main/bnorm_no_mh_local_global_3x3_rbf_no_local/seed_0/` |
| V3.1 | `results/main/v3_1_gp_trace_global_raw_train1_recovery/seed_0/` |

这些检查点用于加载评估，不包含完整断点续训状态。运行还需要本仓库的源码、依赖和 `data/models/` 中对应的 GP 场模型。Cover、GP-Trace、GP-Greedy 为规则算法，无需策略检查点，代码与配置随项目保留。

默认排除虚拟环境、缓存、日志、工作簿备份、`Mars-GP-IPP/代码上传专用/` 重复副本、其他 docs 文档、未列入白名单的实验结果和检查点，以及 `data/models/generated/` 缓存。完整实验产物仍保留在本地；部分评估及绘图命令需要先自行生成对应产物。排除只影响 Git 文件收录，不会删除本地文件。

## 调整上传范围

统一编辑根目录 [.gitignore](.gitignore)。[项目内 .gitignore](Mars-GP-IPP/.gitignore) 仅保留说明，不再重复定义忽略规则。

根目录使用白名单；新增根目录文件或目录时，需要增加对应的 `!` 规则。新增核心文档或图片引用时，同步补充 `.gitignore` 中的精确文件路径白名单；results 的目录遍历规则本身不会放行文件。新增策略权重时，在 `*.pt`、`*.pth` 规则之后追加精确路径例外。放行被排除目录中的单个文件时，必须同时放行父目录。

调整后，在仓库根目录预览候选文件：

```bash
git status --short --untracked-files=all
git ls-files --others --exclude-standard
```

查询文件被哪一条规则排除（将路径换成实际文件）：

```bash
git check-ignore -v Mars-GP-IPP/results/comparison/test_maps_seed0/summary.csv
git check-ignore -v Mars_Env_GP/.venv/
```

首次提交前，核对候选文件是否包含本机路径、凭据或不希望发布的数据。Git 忽略规则不会自动移除已经跟踪的文件；首次提交之后调整范围时，还需检查暂存区。

## 手动提交与推送

本地准备阶段不暂存、不提交、不推送。确认忽略规则和候选清单后，可手动执行：

```bash
git add .gitignore README.md Mars_Env_GP Mars-GP-IPP
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add Mars environment and GP IPP projects"
git push -u origin main
```

后续在同一目录修改后，检查差异、暂存、提交，再执行 `git push`。多设备协作时，修改前先同步远程分支。
