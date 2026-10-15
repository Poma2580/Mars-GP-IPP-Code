# 4×4地图中GP-Trace Greedy与全局最优路径的对比

## 一、实验设置

沿用此前的4×4地图实验：

| 参数 | 设置 |
|---|---|
| 地图规模 | 4×4，共16个预测位置 |
| 起点 | $(2,2)$ |
| 动作空间 | 8邻域 |
| 规划长度 | 4步（含起点共5个采样位置） |
| GP核函数 | RBF |
| 核长度尺度 | $\ell=0.9$ |
| 观测噪声标准差 | $\sigma_n=0.05$ |
| 真实磁场 | 此前构造的局部磁异常场 |
| 评价指标 | 最终全局GP Trace、Trace下降比例（Trr）和实际RMSE |

其中，**GP-Trace Greedy**指每一步从合法的8邻域候选位置中，选择使**全局GP Trace即时下降最多**的位置；若现有代码的`gp_trace`采用局部4邻域Trace，则须按实际实现重新计算。

## 二、实验结果

| 方法 | 最终GP Trace ↓ | Trr ↑ | RMSE ↓ |
|---|---:|---:|---:|
| GP-Trace Greedy | 7.5422 | 52.86% | 0.8973 |
| 全局最优Trace路径 | **6.8198** | **57.38%** | 0.7687 |
| 全局最优RMSE路径 | 9.9564 | 37.77% | **0.0011** |

> 注：以上数值摘自此前讨论中的计算结果，本文仅整理排版，未在本次重新运行代码验证。

### 1. GP-Trace Greedy路径

每一步选择使当前全局GP Trace下降最多的相邻位置。所得路径：

$$
(2,2)\rightarrow(3,3)\rightarrow(4,2)\rightarrow(3,1)\rightarrow(2,1)
$$

最终全局Trace：

$$
\boxed{\operatorname{Tr}(\Sigma_T^{\mathrm{Greedy}})=7.5422}
$$

### 2. 全局最优Trace路径

枚举从同一起点出发、满足地图边界及8邻域运动约束的全部1610条合法4步路径，逐一计算最终全局Trace。其中一条最优路径为：

$$
(2,2)\rightarrow(3,1)\rightarrow(4,2)\rightarrow(3,3)\rightarrow(2,4)
$$

最终全局Trace：

$$
\boxed{\operatorname{Tr}(\Sigma_T^{\mathrm{Trace\text{-}Optimal}})=6.8198}
$$

相较于GP-Trace Greedy，最终Trace降低比例为：

$$
\begin{aligned}
\eta_{\mathrm{Trace}}
&=\frac{\operatorname{Tr}(\Sigma_T^{\mathrm{Greedy}})-\operatorname{Tr}(\Sigma_T^{\mathrm{Trace\text{-}Optimal}})}{\operatorname{Tr}(\Sigma_T^{\mathrm{Greedy}})}\times100\%\\
&=\frac{7.5422-6.8198}{7.5422}\times100\%\\
&\approx\boxed{9.58\%}.
\end{aligned}
$$

**结论：GP-Trace Greedy在该4步规划问题中没有达到全局最小Trace。**

### 3. 全局最优RMSE路径

同样枚举全部合法路径，以最终全局RMSE为目标进行比较。所得一条最优路径为：

$$
(2,2)\rightarrow(1,2)\rightarrow(1,3)\rightarrow(1,4)\rightarrow(1,3)
$$

最终RMSE：

$$
\boxed{\operatorname{RMSE}_T^{\mathrm{RMSE\text{-}Optimal}}=0.001146}
$$

对应最终全局Trace：

$$
\boxed{\operatorname{Tr}(\Sigma_T^{\mathrm{RMSE\text{-}Optimal}})=9.9564}
$$

该路径虽然具有更低的RMSE，但其Trace高于GP-Trace Greedy及全局最优Trace路径，说明：

$$
\boxed{\arg\min_{\mathcal P}\operatorname{Tr}(\Sigma_T)\;\text{与}\;\arg\min_{\mathcal P}\operatorname{RMSE}_T\;\text{不必相同}}
$$

**注意：全局最优RMSE路径是在已知完整真实磁场的条件下枚举得到的离线最优解，仅用于展示理论上的最小可达误差，不能直接作为未知磁场环境中可执行的在线IPP策略。**

## 三、结论

1. GP-Trace Greedy每一步只优化即时Trace下降，因此不保证整条路径的最终Trace全局最优。
2. 在该4×4、4步规划案例中，枚举得到的全局最优Trace比GP-Trace Greedy低约**9.58%**，说明多步规划仍有潜在提升空间。
3. 全局最优Trace路径与全局最优RMSE路径不一致，进一步说明**最小化GP Trace不等价于最小化固定磁场上的实际RMSE**。
4. PPO理论上可能优于GP-Trace Greedy，但是否能够实现提升，仍取决于奖励设计、状态信息与策略训练效果；这个反例本身不能保证PPO会超过贪心基线。
