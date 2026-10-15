# 火星磁场高斯过程复现

此目录按原 `GPy_true_field.py` 的方法训练模型，按原 `draw_true_field.py` 和论文图片绘图。祝融号与毅力号各使用一个 GPy 模型；每个模型一次接收 `Br`、`B2`、`B3` 三列目标数据。

## 运行

在本目录执行（已在 Python 3.10 下验证）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

也可以分别运行 `GPy_true_field.py`（训练、保存模型）和 `draw_true_field.py`（从模型绘图）。`run.py` 还会将新旧模型在相同网格上的预测差写入 `output/kernel_parameters.json`。脚本默认使用本目录的 `data/Mars_Rover_Magnetic_Field.xlsx`，并支持 `--workbook` 和 `--output-dir` 参数。

## 文件

- `data/`：两张工作表 `Zhurong_Magnetic_Field` 和 `Perseverance_Magnetic_Field`；沿用原数据列名。
- `reference/`：迁移来的两张论文原图、两个旧模型和旧网格数组。它们仅用于复现对照，不会被训练脚本覆盖。
- `output/`：`zhurong.png`、`perseverance.png`、两个同名 `.pkl` 模型和 `kernel_parameters.json`。

## 高斯过程流程

每张表的经纬度分别减去其区域左下角坐标：祝融号减去 `(95, 10)`，毅力号减去 `(65, 5)`。模型输入为局部二维坐标，输出为三列磁场。保持原脚本的 RBF 核初值 `variance=1.0, lengthscale=1.0`，使用 `GPy.models.GPRegression(X, Y, kernel)`，然后调用一次 `optimize()` 优化核方差、长度尺度和高斯噪声方差。在 0–30、步长 1 的 31×31 网格上预测三列均值并绘图。

三列目标共享一组核参数，但这个模型没有显式学习分量之间的交叉协方差。`.pkl` 保存了训练数据和完整模型，可以重新加载预测；JSON 便于直接查看核参数。`B2`、`B3` 在原绘图中分别标作 `Bφ`、`Bψ`，工作簿仍保留原列名。

祝融号图片参照 `reference/B_origin.png`，毅力号图片参照 `reference/test_set_1.png` 的左列。为保持数据含义准确，新图修正了原图写反的经纬度轴标题；祝融号图标明了局部坐标偏移量。

论文文字提到祝融号有 70 个观测点，但当前工作簿及旧模型只有 56 个；此复现严格使用现存数据。毅力号有 60 个点。

## 十区域数据集批量训练

独立 venv 中运行（不要使用 conda）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run.py --ten-datasets --replace-rovers
```

`--replace-rovers` 会先生成带时间戳的原工作簿备份，再将两张源工作表分别替换到
`train1` 和 `test1`。源数据五列及祝融号备注保留；复制格式时重映射样式，保留原
Excel 包的其他工作表 XML、共享字符串和元数据，并验证数值及格式一致。
完成首次替换后，后续训练使用 `run.py --ten-datasets` 即可；也支持 `--workbook`
和 `--output-dir`。不带 `--ten-datasets` 时仍执行原两火星车复现和旧模型对照。

默认输出为 `output/10datasets/`：十个同名模型、十张三分量 PNG、
`analysis.md`、`analysis.json` 和 `kernel_parameters.json`。
每个 pkl 仍是原生 GPy GPRegression 对象。预测输入是按对应区域左下角平移后的
经纬度（度），输出顺序为 Br、B2、B3；区域原点和依赖版本见 JSON。
三分量不标准化，共享核参数，核初值不变，每次拟合仅优化一次。
固定种子 42 的随机五折验证共额外拟合 50 次，不改变最终全数据训练的十个模型。

图片按 `output/zhurong.png` 的横排三分量样式绘制，使用局部 0–30° 坐标、绿色
星形观测点、灰色网格和独立色条，不叠加外推斜线或技术总标题。
`train1` 沿用参考图色阶，其余表按分量自动设置色阶，跨图比较须读取色条。
图片为模型预测均值，观测凸包覆盖率及外推网格比例见 JSON。
随机五折结果不能代替独立区域或空间分块验证。
项目沿用 nT 单位，但工作簿自身没有明确单位字段，也未证明数据采集来源。

验证：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```
