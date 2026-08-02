# ShapeNet-Car 预训练 AB-UPT 能否减少 DrivAerML 高保真样本？

证据审计、风险判断与可执行验证方案
日期：2026-07-18
状态：首个目标任务前已冻结（`frozen_before_first_target_job`）

> **历史文档。** 本文是 2026-07-18 冻结时的证据审计，其中"P0/P1 尚未运行"等表述已被后续
> 真实运行取代（实际状态以 NCSA provenance、metrics 与 Slurm 记录为准）。文中引用的
> `tools/materialize_study_manifests.py`、`tools/assess_validation_gate.py`、
> `tools/analyze_transfer_results.py`、`evidence/manifests/`、`evidence/commands/`
> 已被替换或删除。当前实现要报告的那一版实验定义见 [`PAPER_RUN.md`](PAPER_RUN.md)。

## 执行结论

**当前结论不是“已经证明能迁移”，也不是“技术上已完成端到端迁移”，而是：checkpoint 与目标侧源兼容架构具有很高的张量形状兼容性，严格初始化和 train+val 配置构造链已经验证；真实 DrivAerML P0/P1 尚未运行，实际优化稳定性、预测质量和样本效率都仍待证。**

现有 ShapeNet-Car checkpoint 在重置旧输出 readout 后，可以严格装入当前代码的 DrivAerML 源架构匹配模型：common 任务的 7,008,004 个可训练参数中，7,006,464 个可迁移，比例 99.9780%；full 任务比例 99.9588%。epoch-500 `latest` checkpoint 已实际执行 `strict=True` 加载，missing/unexpected keys 都为空。这证明形状兼容和初始化路径不是纯概念提议，但没有证明一次真实 DrivAerML 训练、checkpoint 产出、冻结评估和统计分析已经端到端完成。

但 99.98% 的**形状兼容**不等于物理语义兼容。ShapeNet 与 DrivAerML 同时跨越：车辆几何族、CFD/湍流精度、表面/体网格密度、来流坐标轴、场统计、输出字段和 readout API。尤其 ShapeNet 主流向是 +z，DrivAerML 是 +x；AB-UPT 使用绝对位置编码和 RoPE，并非旋转等变。如果不先对齐坐标，负迁移可能只是预处理错误。

综合两篇指定论文和更接近的工作，本研究值得先投入 P0/P1：

- 指定的多精度缩放论文表明，压力和速度可从低精度数据获益，而壁面剪切应力没有正迁移；这支持先做共同字段、后做近壁字段。
- 指定的汽车 MFDNN 论文说明少量高精度数据可以纠正大量低精度数据，但“更多低精度”并不单调更好；50 个 LF 反而弱于 40 个 LF。
- GeoPT 已公开验证 ShapeNet 几何预训练到 DrivAerML，论文的一个小样本对照中，60 个预训练样本达到或略优于 100 个 scratch 样本，约等于 40% 标签节省；但它是合成动力学自监督 + Transolver，不是当前 CFD checkpoint。
- 2026 年 AB-UPT 车辆族迁移研究显示：只冻结几何编码器会失败，全层 LoRA 比 full fine-tuning 稳定；但其源/目标求解器、边界条件、字段和归一化一致，域差小于本项目。
- NeuralCFD arXiv v1 曾直接报告 DrivAerNet RANS→DrivAerML HRLES 用不到一半 HF 数据，但该实验被后续 TMLR/v4 删除，最终论文把低→高精度迁移写成未来工作。因此它只能生成假设，不能当作经同行评审的证明。

本冻结预注册协议规定了一个比“看 best loss”更强的成功定义：只有迁移在 common 任务的整条学习曲线上优于同构 scratch、估计样本效率至少 1.5×，并且 N=400 不出现超过 5% 的负迁移，才允许称为“减少 DrivAerML 高保真样本”。统计显著性和置信区间的精确定义见后文；当前没有任何真实目标训练结果满足或否定这些条件。

## 研究问题的准确表述

这里不是传统的配对多精度回归。ShapeNet-Car 与 DrivAerML 没有同一几何上的 LF/HF 成对场，ShapeNet 数据和 checkpoint 又已经存在，其生成成本是 sunk cost。因此主问题是：

> 在目标侧 optimizer updates 相同、网络结构相同、训练 ID 相同、评估点相同的条件下，ShapeNet-Car 初始化能否让模型以更少的 DrivAerML 高保真 run 达到固定测试误差？

这是一项“预训练→目标微调”的高保真样本效率研究。只有未来为 DrivAerML 几何额外生成低精度 CFD 时，才应把 LF/HF 生成 core-hours 和混合比例作为主坐标轴。

## 源 checkpoint 事实

主源 run：

```text
/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train
```

三个候选权重：

| 权重 | 选择方式 | SHA256 | 本研究角色 |
|---|---|---|---|
| `ab_upt_cp=latest_model.th` | 固定 epoch 500 | `261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38` | confirmatory primary |
| `ab_upt_cp=best_model.loss.test.total_model.th` | ShapeNet test total，epoch 144 | `221ce5a1a6803876b5d0fc6157a8f06aa5299abe104c4ee9fcbef3f0aaffc5f7` | planned source sensitivity |
| `ab_upt_ema=0.9999_cp=latest_model.th` | epoch-500 EMA | `f1200056bee2a2be2cac0f390bf9aaa049d1b1f29b680bab0721969ad8ba4aeb` | planned source sensitivity |

ShapeNet split 是 789 train + 100 test，没有 validation。原 run 用 test total 选择 best，因此主实验使用固定 epoch-500 `latest`，避免源端 test 选模成为主结论的隐含自由度。

严格 confirmatory runner 读取协议中的 `source.primary_checkpoint_tag` 和完整 SHA，并同时与代码内审计常量比较；CLI 不能把它替换为另一份源权重。当前 runner 因而只执行 `latest` 主源。表中的 source best-test 和 source EMA 只是预注册的计划敏感性，尚无独立执行路径，不能被描述成当前 strict runner 已支持或已运行。

resolved source 架构是 hidden 192、geometry depth 1、10 个 physics blocks、每域 2 个 decoder blocks、3 heads、radius 9、位置尺度 1,000、RoPE 默认 10,000。网络只吃坐标，不使用 SDF/normals 等 physics features。源目标为 surface pressure 和 volume velocity。

当前 API 把旧版直接 Linear readout 改成了 LayerNorm+Linear。旧 checkpoint 有 265 个 state tensors，当前同架构模型有 269 个。删除旧 `backbone.domain_decoder_projections` 并从目标模型实例化同名当前 readout 后，261 个其余 tensors 全部名称/shape 对齐。

当前 native DrivAerML 6-physics/6-decoder 模型只可迁移约 59.47% 参数，所以不能用它与源架构 transfer 做主 scratch 对照。主实验中 scratch 和 transfer 都使用源架构；native 结构只能作为第二组架构迁移实验。

## DrivAerML 数据事实

NCSA 权威路径：

```text
/scratch/andyye2/data/drivaerml_subsampled_10x
```

只读审计得到：357,054,090,936 bytes，`du -sh=333G`，共 484 个 `run_*`。缺失的 16 个 ID 与官方 hidden test 完全相同，因此可见数据恰好组成固定 train 400 / val 34 / test 50。

每个完整 run 有 15 个 `.pt` 张量和一个 `geo_ref_<id>.csv`。表面 VTP 字段按点对齐，体字段按 cell 对齐。local cache 只适合 smoke：run 102 缺 total-pressure coefficient，run 104 缺 velocity，且 run 102 的 vorticity 有约 3.57e9 极值。正式训练前必须逐 selected-run 做存在性、shape、finite 检查，失败要记录，不能静默丢 run。

一个重要语义陷阱：Noether 的 `volume_pressure` property 实际映射到 `volume_cell_totalpcoeff.pt`，而不是同时保存的 `volume_cell_pressure.pt`。full 任务和论文表述应称它为“volume total-pressure coefficient”，除非另行修改并验证 file map。

`geo_ref` CSV 只看到参考长度、面积和 force center，没有 drag/lift coefficient或 16 维 morph 参数。因此本研究不承诺形态分层 acquisition，也不把阻力系数作为已有 ground truth 主指标。

## 两篇指定论文能与不能支持什么

### Setinek et al., *Towards Multi-Fidelity Scaling Laws...*

论文在 2D AirfRANS 风格翼型上比较两级 RANS：HF 直接解析黏性底层、首层 2 µm、y+<1、平均 13.4 core-hours；LF 用壁函数、首层 1,200 µm、y+=30-300、平均 4.8 core-hours。611 组配对条件中 491 用于 train/val，120 只作 HF test；每个预算跑 4 seeds。

压力和体速度在紧预算下常从 LF/HF 混合数据获益，但 WSS 不获益。论文给出的 LF/HF 场 nMAE：表面/体压力约 0.043/0.040，x/y 速度 0.118/0.303，x/y WSS 0.405/0.796。这个字段差异与本项目 ShapeNet 缺少 friction 标签的事实一致。

它不能直接给出本项目最优 LF/HF 比例，因为它是同一二维几何族、配对模拟、混合训练，两级又都是 RANS；本项目是不配对的跨数据集预训练→微调。

### 邬晓敬等，*基于多精度深度神经网络的汽车气动外形优化设计方法*

论文在同一 10 维 MIRA 参数空间预测标量 `C_D`。HF 是约 4.79M 网格、二阶 NS+k-omega、1.74 h；LF 是约 0.80M 网格、一阶 Euler、0.18 h。MFDNN 形式是：

```text
y_H(x) = alpha F_linear(x_H, y_L) + (1-alpha) F_nonlinear(x_H, y_L)
```

在线 MFDNN 的 40 LF 初始集最终使用 14 次 HF，时间 29.74 h；30 LF 需要 37 次 HF；50 LF 需要 15 次 HF而最终阻力更差。作者明确警告，过多错误趋势的 LF 会掩盖 HF 特征。

它支持“先验证相关性、负迁移真实存在、成本要按 HF 调用核算”，但不能证明点云全场 trunk 迁移：任务只是同一参数空间的标量 `C_D`，没有体场，且没有多种子/置信区间。

逐页证据、公式、图表和限制见 `literature_evidence_audit.md`。

## 更接近的论文与代码

### GeoPT：最接近的公开替代路线

GeoPT 在 ShapeNet car/airplane/watercraft 的 13,463 个几何上生成约 134.63 万个合成动力学样本，再微调 Transolver。作者强调纯静态几何预训练会负迁移，加入 dynamics lifting 才是关键。其离线预训练数据约 5 TB，在用户允许的 9 TB 新增数据上限内。

DrivAerML 小样本表中，scratch 100 样本、200 epochs 的 relative L2 为 0.093；GeoPT 60 样本为 0.091，100 样本为 0.075。这是 ShapeNet→DrivAerML 可提升样本效率的最强公开证据，但不等价于当前 AB-UPT CFD checkpoint。官方代码是 `Physics-Scaling/GeoPT`；本轮没有克隆或下载它。

### AB-UPT 跨车辆族 LoRA

Keum & Warey 使用 61.47M surface AB-UPT，在 411 个源车辆上预训练，对每个新车辆族只用 20 train + 10 val。冻结 geometry encoder 的 limited fine-tuning 失败；全层 LoRA（包括 geometry encoder 内线性层）平均优于 full fine-tuning，并在一个族上超过使用约 3 倍目标数据的 scratch。

这说明 LoRA 是必要的 gated extension，且不能只在 readout 加 adapter。但论文所有族使用相同 solver、边界条件、网格、字段和 normalization；其 20 个样本不能当成本项目承诺。

### NeuralCFD v1 的版本降级

arXiv:2502.09692v1 的 Sec. 4.4 用 4,000 个 DrivAerNet RANS surface-pressure 样本预训练 GP-UPT，再微调 DrivAerML HRLES，并声称少于一半 HF 数据可匹配 scratch。当前 v4/TMLR 已改为 AB-UPT 论文，删除该结果，并把 LF→HF 迁移列作未来工作。报告必须明确引用 v1，而不能说“TMLR 已证明”。

## 关键不兼容与控制方式

| 因素 | ShapeNet-Car | DrivAerML | 主实验控制 |
|---|---|---|---|
| 共同字段 | surface pressure, volume velocity | 两者都有 | common confirmatory |
| 新字段 | 无 friction/volume pressure/vorticity | 有 | fresh readout，full gated secondary |
| 来流轴 | +z | +x | 先置换 `[1,2,0]`，raw axis 作消融 |
| 位置尺度/RoPE | 1k / 10k | 新 YAML 100k / 40k，preset 仍 1k | 主实验固定 1k/10k |
| 架构 | 10 physics / 2 decoder | native 6/6 | scratch/transfer 都用 10/2 |
| readout | 旧 Linear | 当前 LN+Linear | 整个 readout 重置 |
| 网格密度 | 约 3.6k surface / 28.5k volume | 可达 0.88M surface / 14.7M volume | 同方法固定 sampling；密度另作消融 |
| 归一化 | source stats | target stats | 每个 N 只用该 train subset 标签拟合 |
| source 选模 | test-selected best | 固定 val/test | epoch-500 latest 主，best 仅敏感性 |

坐标对齐必须在 normalizer 之前同时应用于 position、velocity、friction、vorticity 和未来可能使用的 normals：

```text
(x_shape, y_shape, z_shape) = (y_drivaer, z_drivaer, x_drivaer)
```

这个循环置换 determinant=+1，是 proper rotation，因此 vorticity 也按同样分量置换。已在真实 run 1 的 14,744,958 个 velocity 点上验证输出严格等于 native `[..., (1,2,0)]`。

这里实现的是 DrivAerML→ShapeNet frame 的**前向**坐标/向量分量置换，并且发生在 normalizer 之前。冻结评估在计算误差前反归一化到物理量纲，但当前指标回调不会再把向量旋回 DrivAer native frame；因此只能声称“forward transform + inverse normalization”，不能声称实现了 inverse coordinate transform。

## 协议与完整性边界

`experiment_protocol.yaml` 的角色是冻结预注册协议的原始文件 SHA256 完整性锚，不是 strict runner 全部运行参数的唯一真值源。runner 只解析其中的 study ID、冻结状态、8 组种子和主源 tag/SHA 等强绑定字段；完整运行事实还来自代码常量、CLI、resolved config、manifest、subset-stat artifact、训练 sidecar 和评估 audit。任何协议改动都会改变 raw-file SHA，因此正式运行前必须按最终协议重新物化 manifests 并重算依赖它们的 stats。

每个 manifest 内保存 canonical payload SHA；raw-file SHA 由 stats/training 消费者对完整文件字节计算，并写入下游 artifact/provenance。训练入口重新计算 canonical payload SHA，并用 manifest raw SHA 绑定 subset stats；它还比较 manifest 中的 implementation commit 和 dirty 布尔值与运行时 Git 状态。这个边界不能证明两个 dirty worktree 的具体 diff 相同。正式 production 必须从 clean commit 物化 `implementation_git_dirty=false` 的 manifests，并通过 `--manifest-root` 把它们写在 Git worktree 之外；仓库内现有 dirty manifests 只作开发证据，不能直接投产。

subset-stat artifact 记录 protocol/study、manifest raw SHA、`train_subset_size`、coordinate frame、selected IDs 和 train-only access attestation；preset 还要求当前任务所需 normalizer keys。运行缓存键保持 `[manifest_sha256, train_subset_size, task, coordinate_frame]`。训练 sidecar 进一步绑定 manifest raw/payload SHA、stats 文件 SHA、source SHA、budget/frame/seeds、resolved-config SHA、Git 状态及所有目标 checkpoint SHA；冻结评估再核对 sidecar、实际 checkpoint 和这些上游身份。

## 严格实验设计

### 数据 ladder

固定 N={25,50,100,200,400}。使用 8 个预注册 PCG64 acquisition ladders，每个 ladder 只对官方 400 train IDs 排列，然后取嵌套前缀。每个 manifest 同时记录 `SubsetWrapper` 所需 base-dataset index 和真实 design ID，并保存官方 val/test IDs。正式 clean manifests 必须在最终协议和实现 commit 上重新物化到仓库外路径。

同一 `(replicate,N)` 下，scratch/transfer 共用训练 ID、顺序、model/DataLoader seed 和 val 设计。训练 pipeline seed 为 `None`，每次 dataset visit 由 worker RNG 重采样 geometry/anchors；相同 DataLoader seed 使 S/P-FT 得到相同且可复现的采样流。val/test 使用固定 seed 4242，在相同代码、PyTorch 版本和设计顺序下运行时再生相同的 anchor/output-point indices；当前没有物化索引文件。

### 两种训练预算

主分析固定 40,000 optimizer updates，batch size 1：

| N | 等价 epochs | updates |
|---:|---:|---:|
| 25 | 1,600 | 40,000 |
| 50 | 800 | 40,000 |
| 100 | 400 | 40,000 |
| 200 | 200 | 40,000 |
| 400 | 100 | 40,000 |

实现直接使用 `trainer.max_updates=40000`。实用成本控制固定 100 epochs，对应 2,500-40,000 updates。前者回答 matched-compute sample efficiency，后者回答每个已购买样本看相同次数时的 GPU 成本。

训练没有 early stopping，也不生成 target EMA。`latest` 是训练结束的 final raw 主 checkpoint；`best_model.loss.val.total` 是 best-validation 目标敏感性。二者不要与 source checkpoint 表中的 source EMA 混淆。训练配置只实例化 train+val，不含 test dataset 或 callback。

### 方法

- `S`：同构 scratch，全部随机初始化。
- `P-FT`：严格加载协议锁定的 source trunk，重置 readout，第 0 step 全量训练。
- `P-LP`：只训 readout，诊断线性可读性。
- `P-GU`：0-10% updates 训 heads，10-30% 加 domain decoders，30-100% 全解冻；默认 head/decoder/body LR 比 1/0.3/0.1。
- `P-LoRA`：core 结果后的 gated extension，必须覆盖 geometry encoder 和所有线性层；rank 只在一个中等 N 的 val pilot 选择。当前尚未实现到 strict runner。

### 分阶段矩阵

| 阶段 | 内容 | jobs |
|---|---|---:|
| P0 | common 四方法 + full S/P-FT smoke | 6 |
| P1 | common S/P-FT，前 3 replicates，全 N | 30 |
| P2 | common S/P-FT，新增 5 replicates，全 N | 50 |
| P2a | fixed-epoch control，前 3 reps | 30 |
| P2b | P-LP/P-GU，N=50/100，前 3 reps | 12 |
| P2c | raw-axis S/P-FT 消融，N=50/100 | 12 |
| P3 | gated full S/P-FT，N=50/100/200/400，前 5 reps | 40 |

命令生成器按 phase 固化 budget 和 frame；训练与 statistics generator 都支持显式外部 `--manifest-root`。评估 generator 只从 `training_provenance.json` 发现 cell，并可按 method/replicate/N/task/**budget**/**frame** 过滤及选择 `latest` 或 `best_model.loss.val.total` checkpoint tag，避免手填科学标签。生产主分析应显式筛选 `compute_matched`、`shapenet`、`latest`。

真实 P0/P1 尚未运行，风险控制目前只是已设计/部分实现，不能称为已验证。P1 gate 通过后才释放 P2；P3 必须在 common gate 通过后才可训练，而且 full field×N 推断还要等待对应多重性实现。

## 指标、统计与成功门槛

### 冻结评估边界

评估 runner 每次只实例化一个 official split（`val` 或 `test`），runner 自身默认 `val`；evaluation generator 要求显式选择 split。test 有两道代码门：generator 只有收到 `--confirm-test-release` 才生成 test 命令，生成的每条命令还携带同一 flag，runner 再独立拒绝未确认的 test 运行。

主评估每个 design、每域使用 16,384 个运行时 anchor/output points，不是 dense full-grid。固定 seed 4242 只在相同代码、PyTorch 版本和 design order 下再生相同 indices。test audit 会写出官方 design IDs 和数量；`analyze_transfer_results.py` 还要求显式 `--confirm-frozen-test`，并逐项核对 8 个 replicate、5 个 N 和官方 50 个 test IDs。chunked dense full-grid sensitivity 目前是 `planned_not_implemented`。

evaluation generator 的 cell 唯一性包含 task、method、replicate、N、frame、budget 和 checkpoint identity；它从训练 sidecar 取得 manifest/stats/run/checkpoint 来源，并同时核对 sidecar checkpoint SHA 与实际文件 SHA。`latest` final raw 是主分析，`best_model.loss.val.total` 是目标 checkpoint 敏感性；当前训练没有 target EMA。

对每个 design 和 field，在反归一化后的物理量纲中计算：

```text
relative_L2 = ||prediction-truth||_2 / (||truth||_2 + epsilon)
MAE = mean(abs(prediction-truth))
```

组件仍位于本次运行选择的 coordinate frame；评估没有 inverse coordinate rotation。不能把数百万 mesh points 当独立统计样本。common 主分数是 pressure 与 velocity 的等权 log macro。每个 N 报告 paired geometric error ratio：

```text
R_N = exp(mean(log_error_PFT - log_error_S))
```

在 `log2(N)` 上积分 log-error 得到 normalized AULC。以 scratch N=200 的误差为预注册阈值，用单调 isotonic interpolation 求 transfer 所需 N；禁止在观测区间外 extrapolate。

### 已实现的 common 推断

置信区间用至少 10,000 次 **paired two-way crossed bootstrap**：分别重采样训练 replicates 和 test designs，并跨 method/N/field 保持同一重采样索引。它不是把一层嵌在另一层的 hierarchical bootstrap。

AULC 和每个 N 都使用 replicate-level、单侧 exact paired sign-flip enumeration，备择为 transfer error 更低。Holm 只调整全部 common N 的 per-N **p 值**；AULC 是单一主假设，不进入该 per-N Holm family。所有 bootstrap CI 都是 pointwise、未作 simultaneous/multiplicity correction，不能把 Holm p-value 写成 CI 也被校正。

样本效率 CI 只有在每一个 bootstrap draw 都能在观测 N 区间内识别 crossing 时才报告；只要存在 null/不可识别 crossing，当前分析器就 withholding CI，因此成功 gate 不能通过，禁止拿可识别子集的条件 CI 代替。

“减少 HF 样本”必须同时满足：

1. common compute-matched AULC ratio 点估计 <=0.90，且 pointwise 95% CI 上界 <1；
2. scratch-N=200 阈值下，样本效率点估计 >=1.5，且完整可识别的 pointwise 95% CI 下界 >1；
3. N=400 的 `R_400` pointwise 95% CI 上界 <1.05。

若只有早期收敛更快但第二条不成立，只称“优化/计算效率收益”。若某 N 的 ratio 95% CI 下界 >1.05，定义为 material negative transfer。

full 任务的 field×N Holm family 仍是 `not_implemented`。因此即使 common gate 已通过并运行了 P3，full confirmatory inference 仍被阻塞，直到实现、测试并冻结该 family；现阶段只能输出描述性 field-wise 指标，不能声称通过 full confirmatory gate。

### Val-only futility gate 的工具边界

N=50/100、前三个 paired replicates 中，若 P-FT 两个 N 的 median ratio 都 >=1.10 且 6/6 pairs 全部更差，则停止 raw P-FT 并先审计坐标和 normalization。`assess_validation_gate.py` 能校验 CSV 的矩形配对结构、N/replicate 数量和数值规则，但长表 schema 不含 split、checkpoint stream 或选择来源；`--confirm-final-raw-val` 只是 caller assertion，不是密码学证明。正式 gate 必须同时核对训练 sidecar、每个 val evaluation audit 和 merge provenance，以证明输入确为 final-raw/latest、raw non-EMA、validation-only；test 不参与任何 gate。

## GPU 与 I/O 预算

定义 `C0` 为实际目标 GPU 上一次 N=400、40k updates、源兼容模型的实测时间。仓库约 7 H100-hours 的 showcase 使用另一架构，只能作乐观参考。core 计划约 144 C0，另留 15% smoke、profiling、失败重跑和最终评估余量；必须按 P1→P2→P3 分批释放。

subset stats 是 staged artifacts，不应同时启动：P0/P1 只先计算这两阶段所需的 15 个 common artifacts 与 1 个 P0 full-smoke artifact。统计 CLI 在任何 target-label I/O 前强制核对 frozen protocol、protocol SHA/study 与 manifest；同一 seed/task/frame 的多个嵌套 N 只扫描最大前缀一次，并在 N 边界快照 Welford 状态，因此 P0/P1 用 4 个 CPU jobs 产出 16 个 artifacts，且不会读取最大已购前缀之外的标签。

## 已实现的可执行链

- `tools/materialize_study_manifests.py`：嵌套 ladders、indices/IDs 和 canonical payload SHA；消费者另算 raw-file SHA；正式协议冻结后仍需在 clean commit、仓库外重物化。
- `tools/compute_subset_statistics.py`：selected-train-only Welford float64、vorticity signed-log1p、固定位置 envelope 和 provenance。
- `aero_cfd/datasets/transfer_drivaerml.py`：normalizer 前的 forward native→ShapeNet frame 变换。
- `aero_cfd/presets/drivaerml_transfer.py`：拒绝 manifest raw hash、N、frame 或 required stats 不匹配，禁止退回 full-train stats。
- `run_drivaerml_transfer_strict.py`：协议主源 tag/SHA 强绑定、train+val only、strict trunk/fresh heads、预算和成功后原子训练 provenance sidecar。
- `tools/audit_radius_graph.py`：从真实 strict dry-run resolved config 重建 train dataset/pipeline，审计 capped radius graph 并执行 zero-neighbor gate。
- `tools/generate_statistics_commands.py` / `generate_training_commands.py`：分阶段命令和外部 `--manifest-root`。
- `tools/generate_evaluation_commands.py`：只从训练 sidecar 生成单-split评估命令，绑定 budget/frame/checkpoint identity，并实施第一道 test-release gate。
- `eval_drivaerml_transfer_frozen.py`：val 默认、单 split、第二道 test-release gate、上游/checkpoint hash 核验、官方 split audit 和长表导出。
- `tools/merge_metric_csvs.py`：确定性合并、唯一键检查、输入/输出 SHA provenance。
- `tools/assess_validation_gate.py`：实现预注册 futility 数值规则，同时明确 caller provenance assertion 的边界。
- `tools/analyze_transfer_results.py`：完整 common 配对检查、AULC/PAVA、样本效率、exact sign-flip、per-N Holm p 值及 paired two-way crossed bootstrap。
- NCSA CPU/GPU SLURM templates。

目前的离线证据包括实际 source strict-load、真实大张量 forward 坐标变换、manifest/stat/config 构造和合成统计校验；这些仍不等于真实目标端到端训练。冻结后离线验证结果为：Ruff 全通过；上述 10 个测试模块共 59 passed；两个 SLURM 模板通过 Bash 语法检查；协议/源 SHA、8 份 manifests、180 条训练命令及 4/16 条单遍统计命令的一致性检查通过。

## 当前尚未执行与明确边界

本冻结预注册协议没有伪造训练结论：尚未运行 DrivAerML P0/P1，所以不能回答“实际节省了多少样本”。当前交付只证明：

1. source checkpoint 与源兼容目标架构的张量形状兼容，并完成 strict 初始化/config 构造验证；
2. 文献支持值得做受控验证，但不能外推本项目效应量；
3. 负迁移风险及控制规则已经识别和编码，真实有效性仍待 P0/P1；
4. 允许声称 HF 样本减少的 gate 已写入冻结预注册协议；
5. 真实 strict dry-run 尚需 clean code、外部 clean manifests、对应 subset stats、数据路径和 source checkpoint 全部 staging 到位。

NCSA 已有数据集和 `/scratch/andyye2/ABUPT/noether`，但当前仓库内 manifests 仍绑定旧 commit 且 `dirty=true`，远端也尚未具备最终 clean 实现、外部 manifests、重算 stats 和主源 checkpoint 的完整一致链。依照用户要求，本轮没有从 GitHub/PyPI 下载、没有向 NCSA 上传，也没有提交训练作业。具体 staging 和命令见 `EXECUTION.md`。

## 证据文件

- `source_evidence_audit.md`
- `data_evidence_audit.md`
- `literature_evidence_audit.md`
- `experiment_protocol.yaml`
- `experiment_protocol.md`
- `evidence/strict_transfer_load_latest.json`
- `evidence/transfer_compatibility_latest.json`
- `evidence/manifests/`
- `evidence/commands/`

## 最终建议

本次已批准 P0 + P1 的真实训练准备，不批准一次性释放完整 144 C0。协议已在首个 target job 前冻结；正式 P0 前仍须把实现与冻结协议纳入最终 clean commit，向 Git worktree 外重物化 manifests，重算所需 stats，并完成代码、数据路径和主源 checkpoint staging；随后才能执行真实 strict dry-run 和 graph audit。P0 必须验证文件完整、finite loss、source/protocol checksum、strict load、邻居数分布、forward coordinate transform、inverse normalization 和 provenance。

P1 用 common fields、aligned axes、source-compatible scratch/P-FT 和前三个 paired ladders回答最小可证伪问题。只有 common gate 通过后才释放 P3；在 full field×N Holm 实现与验证完成前，P3 即使运行也不能产生 confirmatory full inference。若 P-FT 不稳定但 common 表征仍有信号，再实现覆盖 geometry encoder 的全层 LoRA。若当前 CFD checkpoint 完全无稳定收益，再把 GeoPT 作为约 5 TB 的独立第二路线；届时因需要从 GitHub/论文数据源获取内容，必须按用户指示先暂停并给出下载清单。
