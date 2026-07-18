# ShapeNet-Car → DrivAerML 多精度迁移：论文证据审计

审计日期：2026-07-16

## 结论先行

当前证据支持一个“有条件值得做、但尚未被现有论文直接证明”的判断：ShapeNet-Car 预训练 AB-UPT 很可能能降低 DrivAerML 的高精度样本需求，尤其是共同存在的表面压力与体速度；但它同时跨越了几何分布、湍流/精度、输出字段和数据归一化四个边界，负迁移风险真实存在。现阶段不能把任一篇论文中的“约 2 倍”“3 倍”或某个低/高精度比例直接当作本项目的预期收益。

最重要的可执行结论如下。

1. 先做共同字段任务：`surface_pressure + volume_velocity`。ShapeNet-Car 对 DrivAerML 的表面摩擦、体压力和体涡量没有源标签；这些新字段必须使用新头，并单独报告。
2. 不应把“冻结主干、只训头”作为主方案。2026 年直接针对 AB-UPT 的车辆族迁移研究中，这一做法失败；全层 LoRA 成功。线性探测仍应保留为诊断基线。
3. 迁移前必须锁定坐标、单位、场变量定义和归一化契约。几何坐标映射不一致可以让一个本来有效的检查点变成远差于随机预测。
4. 压力/速度和近壁量要分开判断。给定论文的相同物理族多精度实验显示压力与速度能正迁移，而壁面剪切应力不能；这与 ShapeNet-Car 缺少摩擦标签共同构成强警告。
5. 需要固定测试集、严格嵌套的高精度训练子集、至少 3 个模型/子集种子，并把 scratch、全量微调、线性探测、渐进解冻和全层 LoRA 放在同一学习曲线上比较。只有在置信区间层面优于同样本 scratch，或以更少 DrivAerML 样本达到全量 scratch 的目标误差，才能声称“样本效率提升”。
6. 若原始 CFD 检查点不能稳定正迁移，GeoPT 是最接近任务的第二路线：它已经在 ShapeNet 几何上预训练并在 DrivAerML 上验证，预训练数据约 5 TB，低于 9 TB 的新增数据上限；但它用的是合成动力学自监督和 Transolver，而不是现有 AB-UPT CFD 检查点。

## 证据强度总表

| 证据 | 与本任务相同之处 | 关键差异 | 可支持的结论 | 证据等级 |
|---|---|---|---|---|
| Setinek et al., *Towards Multi-Fidelity Scaling Laws...* | 物理场、多精度、固定高精度测试、按生成成本配比 | 2D 翼型；同几何族；联合混合训练；两种均为 RANS；不是预训练→微调 | 先测场级低/高精度差距；压力/速度可能迁移，近壁量可能负迁移 | 中等，机制证据强，任务外推有限 |
| 邬晓敬等，2024 | 汽车气动、低/高精度 CFD、少量高精度、明确成本比较 | 只预测标量阻力系数；同一 10 维 MIRA 参数空间；配对/对齐输入；没有点云全场 | 低精度不是越多越好；多精度相关性必须先验证；成本应按 CFD 小时核算 | 中低，汽车域相关但任务形式差异大 |
| Bleeker et al., arXiv:2502.09692v1 | RANS DrivAerNet → HRLES DrivAerML；表面压力；从 scratch 对照 | GP-UPT 旧版本；不是 ShapeNet；结果在后续 TMLR v4 中被移除 | 汽车低→高精度迁移可行，是直接假设来源 | 中低：最接近精度迁移，但仅旧预印本结果 |
| Keum & Warey, arXiv:2605.27968 | AB-UPT；跨车辆族；20 训练 + 10 验证；FFT/LFT/LoRA | 源/目标使用相同求解器、边界条件、字段与归一化；只做表面场；私有数据 | 几何编码器需要适配；优先全层 LoRA；归一化一致是硬门槛 | 高（对适配策略），中等（对本任务收益） |
| Wu et al., GeoPT, arXiv:2602.20399 | ShapeNet 几何 → DrivAerML；小样本曲线；工业三维点云 | 无 ShapeNet CFD 标签；Transolver；合成动力学预训练；不同检查点 | ShapeNet→DrivAerML 并非不可行；纯几何预训练会负迁移，动力学提升很关键 | 高（对替代路线），间接（对现有检查点） |
| Zhang et al., 2024 风场多步迁移 | 不同物理近似造成空间局部失配；少量 HF | 风场、二维/规则区域、工作坊论文 | vanilla 微调会让缺失物理区域污染原有特征；可按区域/字段分阶段迁移 | 中低，方法启发 |

## 给定论文一：多精度缩放规律

来源：M. Setinek, T. Galletti, J. Brandstetter, *Towards Multi-Fidelity Scaling Laws of Neural Surrogates in CFD*, NeurIPS 2025 AI4Science Workshop，arXiv:2511.01830。稳定链接：[arXiv](https://arxiv.org/abs/2511.01830)，[OpenReview PDF](https://openreview.net/pdf?id=GTnY1rowTj)。以下页码为 PDF 页码。

### 方法和数据

- 研究问题是：在固定 CFD 数据生成预算 `D_b`（core-hours）下，最优的高精度比例 `D_c` 是多少。模型大小和训练计算量固定，只改变数据生成预算与配比（pp. 3–5）。
- 数据是基于 AirfRANS 设定的二维外流翼型。OpenFOAM v2506、`simpleFOAM`、不可压缩（Mach < 0.3）、Reynolds 数 2×10^6–6×10^6、攻角 −5°–15°、k–ω SST（p. 4）。
- 两个精度都求解 RANS，差异不只是分辨率，而是近壁物理处理（Table 1, p. 4）：

| 项目 | 高精度 | 低精度 |
|---|---:|---:|
| 黏性底层 | 直接解析 | 壁函数建模 |
| 第一层高度 | 2 μm | 1,200 μm |
| 第一层中心 `y+` | < 1 | 30–300 |
| 平均模拟成本 | 13.4 core-hours | 4.8 core-hours |
| 平均节点数 | 180k | 96k |
| 总数据量 | 18 GB | 7.8 GB |

- 共 611 组同工况低/高精度配对模拟；491 组用于训练/验证，120 组仅作高精度测试（p. 4）。
- 模型是约 4M 参数的 Transolver。输入为初始条件和网格节点坐标；测试只在 120 个未见高精度样本上进行。预算为 1,000、1,500、2,000、2,500、3,500、4,500、5,500 core-hours，以及约 6,600 core-hours 的全高精度基准；每组 4 个随机种子（pp. 5–6）。
- p. 5 写“预测六个量”，随后只列出二维速度 2 分量、压力 1 分量、壁面剪切 2 分量，共 5 个标量分量。这是文内计数不一致，不影响图表的字段解释，但复述时不应写成确定的 6 分量。

### 主要结果

- 在紧预算下，体/表面压力和体速度的低高混合训练优于把全部预算用于少数高精度样本；预算越小，最优方案通常越偏向更多低精度。预算越充足，最优配比越偏向高精度（Figures 2–3, pp. 5–6）。
- 壁面剪切应力两个分量没有观察到正迁移：所有预算下，提高高精度比例都持续改善结果（Figure 3, p. 6）。
- 作者把这种差异归因于低精度壁函数没有解析黏性底层。配对场差异的 nMAE（Table 2, p. 6）为：

| 字段 | 表面 | 体域 |
|---|---:|---:|
| x 速度 | — | 0.118 |
| y 速度 | — | 0.303 |
| 压力 | 0.043 | 0.040 |
| x-WSS | 0.405 | — |
| y-WSS | 0.796 | — |

- 附录 p. 12 定义：

```text
nMAE = Σ_i |ŷ_i^LF − y_i^HF| / Σ_i |y_i^HF|
```

  其中低精度场先以最近邻插值到对应高精度网格，再对测试样本取平均。
- 训练配置（p. 12）：AdamW，weight decay 1e-4，β=(0.9,0.999)，500 epochs；连续 250 epochs 无验证改善则提前停止；10-epoch warmup 到 5e-4 后余弦衰减；梯度裁剪；float32。Transolver 维度 256、4 heads、8 layers、slice base 128、MLP expansion 2、dropout 0.1。

### 局限和迁移判断

- 只考察单一二维翼型族和一种 RANS 近壁处理差异；没有 LES/HRLES/DNS（p. 7）。
- 只有离散两级精度；固定模型大小和训练计算；没有证明跨几何数据集、跨架构或预训练→微调（p. 7）。
- 低/高精度样本来自同一配对池，训练是从混合数据联合训练，不是先训源域再微调目标域。因此它不能直接证明 ShapeNet 检查点可迁移到 DrivAerML。
- 可迁移的是诊断原则：先构造少量同几何或近邻几何桥接集，按字段计算 LF/HF 差距；先做压力和速度；把摩擦/WSS、涡量和近壁区域列为高风险字段。论文中的最优比例和 13.4/4.8 成本比不能搬用到本项目。
- 本项目已有 ShapeNet 数据和检查点，其成本是 sunk cost。主坐标轴应是“需要多少个 DrivAerML 高精度样本”，而不是重新优化已发生的 ShapeNet 生成成本；只有计划新增低精度 DrivAerML 时才需要真正的 `D_b` 配比实验。

## 给定论文二：汽车 MFDNN 优化

来源：邬晓敬、高然、马龙，*基于多精度深度神经网络的汽车气动外形优化设计方法*，《空气动力学学报》2024, 42(7):103–111，DOI：[10.7638/kqdlxxb-2023.0164](https://doi.org/10.7638/kqdlxxb-2023.0164)。PDF p. 1 是引文页；PDF pp. 2–10 对应期刊 pp. 103–111。

### 方法和数据

- 任务是快背式 MIRA 标准车模的单目标、无约束减阻，约束只有 10 个形状变量各自的上下界（Eq. 1, PDF p. 3 / journal p. 104）：

```text
min C_D(X_i),  X_i,min ≤ X_i ≤ X_i,max
```

- 十个变量是接近角、发动机罩前缘、前/后风窗、后备箱盖和离去角的 y/z 方向形变量（Figure 5、Table 4, PDF p. 5 / journal p. 106）。
- CFD 使用 Star-CCM+；不可压缩空气 ρ=1.225 kg/m³，入口 30 m/s，出口相对压力 0 Pa，计算域壁面滑移、车表面无滑移（PDF p. 4 / journal p. 105）。
- 两级精度（Table 2, PDF p. 4 / journal p. 105）：

| 项目 | 高精度 | 低精度 |
|---|---:|---:|
| 体网格 | 4,788,261 | 795,757 |
| 离散 | 二阶迎风 | 一阶迎风 |
| 方程 | Navier–Stokes | Euler |
| 湍流模型 | k–ω | 无 |
| 迭代数 | 1,300 | 550 |
| 单次耗时 | 1.74 h | 0.18 h |

  论文硬件为 32-core Intel Xeon Gold 6226R 2.9 GHz；高/低精度耗时比约 9.67。高精度网格 4.8M 与 5.12M 都给出 `C_D=0.274`，风洞为 0.278（Table 3, PDF p. 4 / journal p. 105）。
- MFDNN 使用 Meng–Karniadakis 复合网络（Eq. 2, PDF p. 5 / journal p. 106）：

```text
y_H(x) = α F_l(x_H, y_L) + (1 − α) F_nl(x_H, y_L)
```

  三个 DNN 分别学习低精度映射、低高精度线性相关和非线性相关；线性分支不使用激活函数（Figure 7, PDF pp. 5–6 / journal pp. 106–107）。
- 一维间断函数演示使用 41 个 LF + 5 个 HF，而单精度 DNN 只用同样 5 个 HF；MFDNN 的拟合更接近真实函数（Figure 8, PDF p. 6 / journal p. 107）。这只是教学算例，不能当作汽车全场证据。
- 汽车代理只预测标量阻力系数 `C_D`。作者随机抽取 10 个工况画出 LF/HF 趋势，但没有报告相关系数（Figure 9, PDF p. 7 / journal p. 108）。模型精度用 50 个高精度测试样本评估，并给出 `R²` 和 RRAAE 定义（Eqs. 5–6, 同页）；Figure 10 扫描 10–50 个 HF 与 30/40/50 个 LF，但没有把精确曲线数值制成表格。

### 优化结果

在线 MFDNN 起始用 30/40/50 个 LF + 5 个 HF，每次把当前代理最优点做一次 HF CFD 并加入重训。对照包括在线 DNN，以及用 20/50/100 个 HF 的离线 DNN。最终结果（Table 5, PDF p. 8 / journal p. 109）：

| 方法 | 最终 `C_D` | 降阻 | 收敛所需 HF | 时间 |
|---|---:|---:|---:|---:|
| MFDNN online, 30 LF | 0.2496 | 8.91% | 37 | 69.80 h |
| MFDNN online, 40 LF | 0.2486 | 9.27% | 14 | 29.74 h |
| MFDNN online, 50 LF | 0.2523 | 7.92% | 15 | 35.00 h |
| DNN online | 0.2491 | 9.08% | 49 | 83.65 h |
| DNN offline, 20 HF | 0.2671 | 2.52% | 20 | 34.85 h |
| DNN offline, 50 HF | 0.2664 | 2.77% | 50 | 87.13 h |
| DNN offline, 100 HF | 0.2536 | 7.45% | 100 | 174.27 h |

作者据此称 MFDNN 框架相对离线 DNN 的收敛速度为 5.85×、相对在线 DNN 为 2.81×。关键的反例也在同页：50 LF 比 40 LF 更差；作者明确归因于过多 LF 掩盖 HF 特征，以及部分 LF 在设计空间中提供错误趋势。

### 局限和迁移判断

- 这是标量 `C_D` 代理和优化研究，不是表面/体域点云场预测。它不能证明 AB-UPT 的全场 trunk 能迁移。
- LF/HF 共用同一个 10 维 MIRA 参数空间，几何语义和输出完全对齐；ShapeNet 与 DrivAerML 是不配对、非同一参数化的跨数据集迁移。
- 仅一个优化流程，未报告多种子、标准差、显著性或置信区间；10 点趋势图证据较弱；Figure 10 无精确数表。
- 时间比较混合了在线/离线策略与不同 HF 调用次数，神经网络训练时间也未拆分；不能把 5.85× 或 2.81× 当作纯模型收益。
- Euler LF 完全删除黏性和湍流，尤其不能支持壁面摩擦/WSS 的迁移。
- 商业求解器、网络层宽/深/优化超参数和数据均未充分开放，没有官方复现代码。
- 其复合网络只有在同一个 `x` 上能得到或插值到 `y_L(x)` 时才自然适用。当前 ShapeNet→DrivAerML 没有同几何 LF 场，直接把 Eq. 2 搬到 AB-UPT 不成立。若未来为少量 DrivAerML 几何补做 LF CFD，才可尝试 fidelity-conditioned head 或 residual/correction head。

## 更直接的原始研究与官方代码

### 1. AB-UPT 跨车辆族 LoRA：最强适配策略证据

S. Keum, A. Warey, *Adapting Automotive Aerodynamics Surrogates to New Vehicle Families via Transfer Learning*, 2026，[arXiv:2605.27968](https://arxiv.org/abs/2605.27968)。

- 使用 61.47M、surface-only AB-UPT：几何编码器 6 个 Perceiver blocks、18.9M；表面分支 12 blocks、38.8M；解码器 3.8M（Sec. 3.1, Table 3）。
- 私有 PowerFLOW 数据共 511 个算例、5 个拓扑不同的 SUV/crossover 族；每族约 100 个几何，所有族使用相同求解器、边界条件、网格方案和字段。每次留一族，411 个源样本预训练；目标族只用 20 train + 10 val，约 70 test（Secs. 3.2–3.4）。
- FFT 更新 61.47M；LFT 冻结整个几何编码器和前 10/12 表面 blocks，仅训 6.83M；LoRA 在几何编码器、共享层和表面分支的 Q/K/V、attention output 与两个 MLP 线性层中都注入适配器，共 158 层、rank 64、约 10.13M 可训练参数（Sec. 3.3）。
- LoRA 形式为：

```text
h = W0 x + (α/r) B A x
```

  其中 `A∈R^(r×d)`、`B∈R^(d×r)`；论文设置 `r=64, α=128`。
- 五族平均，LoRA 的压力力 `R²≈0.851±0.020`、剪切力 `R²≈0.946±0.027`；表面压力相对 L2 约 `3.21e−4`，摩擦约 `0.198`。冻结编码器的 LFT 压力 `R²` 为负，FFT 明显更不稳定（Secs. 4.2–4.5）。
- Large SUV #3 上，30 个目标样本的 LoRA 总阻力 `R²=0.87`、MAE 9.2 N；从 scratch 用 30 个为 0.77/12.3 N，用 103 个为 0.81/10.0 N（Table 14）。但 scratch 直接对照只做了一个目标族。
- 论文发现坐标与场归一化必须继承源域契约；误重算坐标 bounds 曾让同一 LoRA 检查点的力预测从约 `R²=0.86` 降到约 `−30`（Sec. 5）。
- 局限：所有数据在同一不可压缩高速公路外流物理区间；只有表面场；rank sweep 只在一族；数据私有；adapter 是族专属；没有体域或跨精度验证。

对本项目的直接含义：LoRA 应作为优先 PEFT 基线，并且必须覆盖几何编码器；但由于本项目还跨 solver/fidelity 和字段，这篇论文的 20 个训练样本不能视作本项目样本数承诺。

### 2. GeoPT：最接近 ShapeNet → DrivAerML 的公开路线

H. Wu et al., *GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training*, ICML 2026，[arXiv:2602.20399](https://arxiv.org/abs/2602.20399)，[官方代码](https://github.com/Physics-Scaling/GeoPT)，[项目页](https://physics-scaling.github.io/GeoPT/)。

- 用 ShapeNet v1 的 car/airplane/watercraft，共 13,463 个几何；每个几何生成 100 个随机动力学监督，共 1,346,300 样本。每样本 32,768 体点 + 4,096 表面点；统一尺度/朝向（Sec. 4.2–4.3）。
- 不是静态几何重建：对随机速度场下、受几何边界约束的输运轨迹预测 vector-distance 序列。纯几何 vector-distance 预训练在 DrivAerML 上反而显著负迁移；“动力学提升”是核心（Secs. 4.1–4.2）。
- 默认 backbone 是 Transolver；base/large/huge 为 8/16/32 layers、3/7/15M 参数（Sec. 4.3）。
- 离线自监督数据约 5 TB，作者报告 80 CPU cores 约 3 天生成；这在 9 TB 新增数据约束内，但工程成本仍不可忽视（Sec. 4.3）。
- DrivAerML 小样本 Table 8：Transolver scratch 用 100 个样本、200 epochs 的相对 L2 为 0.093；GeoPT 用 60 个、200 epochs 为 0.091，即约 40% 标签节省；GeoPT 用全部 100 个为 0.075。5-run 标准差与 95% 等价置信区间在 Appendix Table 7 报告。
- 论文也给出 450-sample surface-only 对齐结果：压力系数 MSE 从 Transolver 的 0.004223 降为 0.003370，Mean AE 从 0.04125 降为 0.03617（Appendix Table 5）。
- 官方代码只公开预训练数据生成；下游数据通过 Hugging Face/原数据源获得。官方 README 明确要求下游几何与预训练归一化域对齐，并正确配置动力学 prompt。
- 局限：主实验证据是 Transolver，不是 AB-UPT；它利用无标签几何而不是现有 ShapeNet CFD 权重；动力学 prompt 不能精确表达所有材料/物理设置；论文主要聚焦复杂几何边界（Appendix G）。

对本项目的直接含义：这是最强的“ShapeNet→DrivAerML 可以产生样本效率收益”证据，也是一个独立 baseline。不能据此宣称当前 CFD 检查点会同样有效；更合理的研究问题是比较 `scratch`、`ShapeNet-CFD checkpoint`、`GeoPT-style pretraining`，以及二者是否能顺序或多任务组合。

### 3. 一个必须降级使用的直接先例：NeuralCFD v1

arXiv:2502.09692 的 [v1](https://arxiv.org/abs/2502.09692v1) 名为 *NeuralCFD*，Sec. 4.4 用 4,000 个低精度 RANS DrivAerNet 表面压力预训练 GP-UPT，再在 HRLES DrivAerML 子集上微调；报告用约一半 DrivAerML 数据即可超过全数据 scratch，并在全部 DrivAerML 上微调进一步降误差。

但该 arXiv 的当前 [v4/TMLR 版本](https://arxiv.org/abs/2502.09692) 已改为 AB-UPT 论文，并在 p. 20 明确把低→高精度迁移列为未来工作；旧 Sec. 4.4 和对应结果不在经评审最终版本中。因此：

- 它是非常接近本目标的假设生成证据；
- 不能按 TMLR 已验证结果引用；
- 其源域是 DrivAerNet，不是 ShapeNet-Car，而且只验证表面压力。

### 4. 不同物理近似下的多步迁移

D. Zhang et al., *Transfer Learning in Multi-fidelity Surrogate Modeling: A Wind Farm Case*, ICML 2024 AI4Science Workshop，[OpenReview](https://openreview.net/forum?id=yBTDCqNcan)。

论文强调低精度不总是“同算法粗网格”，还可能删除或近似某些物理；这些缺失物理只在空间局部占主导，直接微调可能污染原本有效的低精度特征。方法使用共享 backbone、LF/HF 头和逐步放松的 LF/HF 损失权重，并对区域 LF/HF 相似性做评估，再加入高置信伪 HF 区域。它支持本项目采用“字段/区域分开评估、逐步解冻”的设计，但风场工作坊实验和伪标签风险使它不适合作为首个实现。

### 5. 其它算子多精度论文：可作方法对照，不是最近任务证据

- L. Lu et al., *Multifidelity Deep Neural Operators...*, [arXiv:2204.06684](https://arxiv.org/abs/2204.06684)：两个 DeepONet 通过 residual learning 和 input augmentation 耦合；同 HF 数量下可达约一个数量级更低误差。应用是声子 Boltzmann 热输运，输入/网格语义与汽车点云不同。
- Y. Lyu et al., *Multi-fidelity prediction ... using FNO*, [arXiv:2304.06972](https://arxiv.org/abs/2304.06972)：FNO 预训练→微调三种流/热场；主要优势依赖规则网格和分辨率不变性。
- H. Tang et al., *Multi-fidelity FNO for ... Geological Carbon Storage*, [arXiv:2308.09113](https://arxiv.org/abs/2308.09113)：报告约 81% 数据生成成本降低，并测试不同地质模型/模拟器；仍是规则储层网格，不可直接外推到任意汽车几何。
- X. Meng, G. Karniadakis, *A composite neural network that learns from multi-fidelity data*, JCP 401 (2020) 109020，[DOI](https://doi.org/10.1016/j.jcp.2019.109020)：给定中文论文 Eq. 2 的原始方法。适用于同一输入空间和可对齐 LF/HF 输出；不是无配对跨数据集迁移。

## 与当前 Noether/AB-UPT 实现的逐点对应

本节基于仓库当前实现与已生成的兼容性证据，不把论文架构参数误当成本地检查点参数。

1. 字段交集。
   - ShapeNet preset 只有 `surface_pressure:1` 和 `volume_velocity:3`：`recipes/aero_cfd/src/aero_cfd/presets/shapenet_car.py:43-78`。
   - DrivAerML 有 `surface_pressure:1, surface_friction:3` 与 `volume_pressure:1, volume_velocity:3, volume_vorticity:3`：`recipes/aero_cfd/src/aero_cfd/presets/drivaerml.py:41-81`。
   - 当前 DrivAerML loader 中名为 `volume_pressure` 的训练文件实际是 `volume_cell_totalpcoeff.pt`，即总压系数而非 ShapeNet 未提供的同名静压；不能仅凭字段名假设语义一致。
   - 因此 common-field 任务是最干净的因果检验；full-field 任务混合了迁移字段和完全新字段，必须后做并逐字段报告。
2. 采样尺度。
   - Python preset 的轻量默认值不是源检查点实际设置。源 checkpoint resolved config 使用 3,586 个 geometry/surface anchors 和 4,096 个 volume anchors；当前原生 DrivAerML YAML 使用 65,536 geometry points、每域 16,384 anchors。Python preset 另有 1,024 geometry supernodes、每域 512 anchors，正式实验不能混用这两套入口。
   - AB-UPT 本身支持可变点数，但输入分布、局部密度和 anchor 覆盖发生变化。需要把“相同模型权重、不同采样密度”作为单独消融，不能把采样改变误判成 fidelity 效应。
3. 归一化。
   - 源 checkpoint 的位置 scale 是 1,000、RoPE max wavelength 是 10,000；当前 DrivAerML YAML 改为 scale 100,000、max wavelength 40,000，而 Python preset 仍写 scale 1,000。这不是小差异：radius graph 直接在归一化坐标上建立，必须先统计每个 supernode 的邻居数以排除退化图。
   - 坐标轴也不一致：ShapeNet 平均速度主分量沿 `+z`，DrivAerML 沿 `+x`。源模型含绝对位置编码和 RoPE，不具旋转等变性。至少要比较原坐标与将 DrivAer `(x,y,z)` 映射到源轴 `(y,z,x)` 的 canonical-axis 方案；position、velocity、friction、vorticity及其统计量必须同步置换。
   - 场变量各自使用 mean/std，DrivAerML 涡量另用 logscale。需要记录并比较实际 bounds、单位、来流方向和 pressure/velocity 定义。若重置输出头，目标场统计可以用于新头；但进入共享 trunk 的坐标映射必须与源权重的空间语义一致，或先做显式对齐消融。
4. 检查点可装载性。
   - `research/multi_fidelity/evidence/transfer_compatibility_best_total.json` 显示，主动重置旧 readout 后，7,006,538 个非 readout 目标参数全部形状兼容；兼容参数约占 common 目标模型的 99.978%。
   - 这证明“技术上可加载”，不证明“语义上可迁移”。由于 readout 被重置，本实验实际检验的是几何/物理 trunk 表示迁移，而不是直接保留 LF 输出头。
5. 当前实验脚本。
   - `recipes/aero_cfd/scripts/train_drivaerml_multifidelity.py` 已有 scratch、finetune、linear_probe、gradual_unfreeze，并使用固定 permutation 的嵌套前缀子集和固定 val/test。
   - 文献表明还缺一个关键方法：对所有线性层、尤其几何编码器注入 LoRA。建议将 LoRA 加入正式矩阵，但保留现有 4 项以诊断“头可迁移性、全模型可迁移性和解冻时机”。
   - 源 checkpoint 的 `best` 是按 ShapeNet test loss 选出的，而该数据没有 validation split。主结论应优先用固定 500-epoch `latest`，test-selected `best` 只作敏感性分析，避免把源测试选择偏差带入迁移结论。

## 推荐实验矩阵和判定标准

### 阶段 0：不训练的迁移门槛

- 核对 source/target 的长度单位、坐标轴、来流方向、几何裁剪、压力是静压/总压/系数还是原始 Pa、速度是否归一到自由流、时间平均定义和近壁量定义。
- 保存 source 与 target 的实际位置 min/max、字段 mean/std；对同一 DrivAerML 样本分别走 source-style 与 target-style 坐标处理，比较几何 encoder 激活范数/分布。
- 计算 zero-shot common-field 指标。预期可以很差；其用途是量化域差，不是最终方案。
- 若能找到近几何或能为 5–20 个 DrivAerML 几何补算低精度 CFD，计算与 p. 12 同形式的逐字段 nMAE，并按表面位置/尾流/近壁区域分层。

### 阶段 1：最小可证伪学习曲线

- 固定官方 400-train / 34-val / 50-test；训练子集使用同一 permutation 的严格嵌套前缀，例如 `n={5,10,20,40,80,160,400}`。
- 至少 3 个独立 subset seeds × 3 个 model seeds；资源不足时先做 `n={10,40,160}` 筛选，再补全最有希望的策略。
- 两个任务：
  1. common：表面压力 + 体速度；
  2. full：全部 5 个字段，且新字段头明确随机初始化。
- 至少五种策略：scratch、全量微调、只训 readout/decoder、渐进解冻、全层 LoRA。LoRA 先试 `r={8,32,64}`，rank sweep 只在一个中等 `n` 完成，再冻结设置用于全部子集，避免测试集调参。
- 所有策略使用相同训练预算、验证选择规则和数据增强；单独记录 trainable parameters、峰值显存、GPU-hours 和 wall-clock。

### 指标和“样本效率”定义

- 主指标：每字段、每样本的物理单位 relative L2 / MAE，并报告均值、median、90/95 分位和最坏案例。
- 次指标：由表面压力/摩擦积分得到的阻力/升力或相应系数，不能只报告归一化训练损失。
- 统计：对相同测试几何做 paired bootstrap；学习曲线跨种子给 95% CI。测试集只做最终读数，所有模型选择基于验证集。
- 正迁移：在相同 `n` 与训练预算下，迁移策略的测试误差 CI 明显低于 scratch。
- 样本节省：定义达到 `scratch(n=400)` 某目标误差所需的最小 `n`，并报告 `1 − n_transfer/400`；不能用单点最好结果代替曲线。
- 负迁移停止规则：若 common-field 在两个连续 `n`、多数种子上劣于 scratch，停止扩大 full-field；先查归一化/坐标对齐，再查是否需要更强 adapter 或源数据 replay。

## 最终研究路线排序

1. **首选、最省改动**：现有 ShapeNet-CFD trunk + 重置 readout，跑 common-field 的 scratch/FFT/probe/gradual-unfreeze；补全全层 LoRA。
2. **若压力正迁移而速度/近壁字段不稳定**：字段专属 loss/head；共享几何 trunk，压力/速度 adapter 分开；新字段只用目标监督。必要时对尾流/近壁区域加权，但不能用伪标签替代真实测试。
3. **若现有检查点无稳定收益**：复现 GeoPT-Transolver 作为公开强基线，或把 dynamics-lifted 目标移植到 AB-UPT 几何分支。约 5 TB 新预训练数据符合 9 TB 限制。
4. **若允许新增 LF CFD**：优先在 DrivAerML 几何上生成小规模配对 LF/HF 桥接集，再做 fidelity-conditioned AB-UPT、residual correction 或混合训练。这时 Setinek 的预算/字段诊断和 MFDNN 的相关性思想才最直接适用。

## 参考入口

- AB-UPT 当前论文与最终 TMLR 版本：[arXiv:2502.09692](https://arxiv.org/abs/2502.09692)
- Noether 官方实现：[Emmi-AI/noether](https://github.com/Emmi-AI/noether)
- UPT 官方实现：[ml-jku/UPT](https://github.com/ml-jku/UPT)
- Transolver 官方实现：[thuml/Transolver](https://github.com/thuml/Transolver)
- GINO / NeuralOperator 官方实现：[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
- DrivAerML 数据论文：[arXiv:2408.11969](https://arxiv.org/abs/2408.11969)
- GeoPT 官方代码：[Physics-Scaling/GeoPT](https://github.com/Physics-Scaling/GeoPT)
- 多物理预训练 MPP：[论文](https://arxiv.org/abs/2310.02994)，[官方代码](https://github.com/PolymathicAI/multiple_physics_pretraining)
- Poseidon PDE foundation model：[论文](https://arxiv.org/abs/2405.19101)，[官方代码](https://github.com/camlab-ethz/poseidon)
- 无监督算子预训练：[arXiv:2402.15734](https://arxiv.org/abs/2402.15734)，[官方代码](https://github.com/delta-lab-ai/data_efficient_nopt)

## 审计边界

- 两份给定 PDF 已逐页渲染并对关键图表、公式与提取文本交叉核对。
- 网络检索只采用论文页面、出版 DOI、OpenReview、作者项目页和官方代码仓库作为事实依据。
- 未把搜索摘要中的宣传数值当作独立证据；GeoPT 的 DrivAerML 数值来自论文 Table 8，LoRA 数值来自论文表格/正文。
- 由于 arXiv:2502.09692v1 的直接低→高精度实验未进入当前 v4/TMLR，本文明确把它降级为预印本版本证据。
