# ShapeNet-Car → DrivAerML 迁移负效应审计报告（P1 复盘）

日期：2026-07-27
角色：模型架构 / 代码审计（exploratory 命名空间；未修改任何冻结协议、未覆盖任何既有结果、未提交任何作业）
审计对象 commit：本地 `multi-fidelity` @ `88a9d09e`（评估实现）；P1 训练实现 `cf974078`
NCSA 访问：`NCSA_CONNECTION_OK`（cc-login5），全部为有界只读查询

---

## 0. 一句话结论

**负迁移的原因不是"domain gap"，而是一处可量化的输入几何渲染错配：DrivAerML 的位置归一化把车渲染成源模型从未见过的尺度，导致 (a) 位置编码在车体上几乎不分辨结构，(b) 几何编码器的 radius 图 100% 被截断、丢弃 99% 的邻域信息。** 两侧都已实测，非推断。

---

## 1. 现象复盘（实测，816 行 per-design/per-field 长表）

数据源：`.../cf974078…/reports/P1_final_raw_val.csv`（56,153 bytes，已下载到
`research/multi_fidelity_exploratory/evidence/`）。408 个配对单元 = 2 methods × 2 N × 3 reps × 34 designs × 2 fields。

### 1.1 分 field × replicate 的配对几何比（P-FT / S）

| N | rep | field | geo ratio | P-FT gm | S gm | 变差的 design 数 |
|---:|---:|---|---:|---:|---:|---:|
| 50 | 0 | surface_pressure | 1.0663 | 0.1511 | 0.1417 | 33/34 |
| 50 | 0 | volume_velocity | 1.0321 | 0.1511 | 0.1464 | 26/34 |
| 50 | 1 | surface_pressure | 1.0317 | 0.1544 | 0.1496 | 25/34 |
| 50 | 1 | volume_velocity | **1.4893** | 0.2223 | 0.1492 | **34/34** |
| 50 | 2 | surface_pressure | 1.0300 | 0.1467 | 0.1425 | 27/34 |
| 50 | 2 | volume_velocity | 1.0539 | 0.1513 | 0.1435 | 34/34 |
| 100 | 0 | surface_pressure | 1.0127 | 0.1360 | 0.1343 | 20/34 |
| 100 | 0 | volume_velocity | 1.0336 | 0.1404 | 0.1359 | 32/34 |
| 100 | 1 | surface_pressure | 1.0850 | 0.1497 | 0.1380 | 33/34 |
| 100 | 1 | volume_velocity | 1.0518 | 0.1449 | 0.1378 | 33/34 |
| 100 | 2 | surface_pressure | **0.9979** | 0.1356 | 0.1359 | 16/34 |
| 100 | 2 | volume_velocity | 1.0563 | 0.1429 | 0.1353 | 33/34 |

按 N×field 的 replicate 中位数：

| N | surface_pressure | volume_velocity |
|---:|---:|---:|
| 50 | 1.0317 | 1.0539 |
| 100 | 1.0127 | 1.0518 |

**结论 1**：两个 field 同向变差 → 机制作用于**共享 trunk / 输入表征**，不是某个 readout 的问题。
**结论 2**：`volume_velocity` 6/6 单元全部 >1 且更稳定（1.032–1.489）；`surface_pressure` 波动更大，N=100/r2 甚至轻微更好（0.9979）。体速度对几何编码质量更敏感——与根因 R2（几何编码器邻域被摧毁）一致。

### 1.2 N=50/r1 离群单元：优化失败，非设计族问题

该单元 volume_velocity 比值 1.4893。逐 design 分解：

- **34/34 个 design 全部变差**；比值 min=1.286, p25=1.424, median=1.479, p75=1.546, max=1.720。
- 分布是**整体平移**，没有少数 design 主导。

→ 明确是该次训练进入了明显更差的盆地（优化/初始化异常），**不是**数据子集分布或特定设计族问题。协议规定的"基础设施故障可重跑一次"不适用（这不是崩溃）。

### 1.3 效应量（决定单 replicate 对照的分辨力）

per-design 配对 log-ratio 的标准误（n=34）：

| 单元 | sd(log) | se(log) |
|---|---:|---:|
| rep0 N=50 pressure | 0.0356 | 0.0061 |
| rep0 N=100 velocity | 0.0327 | 0.0056 |
| rep2 N=100 velocity | 0.0236 | 0.0040 |

→ 单个 replicate 的 34 个配对 design 足以把 **~1–2% 的效应**解析出来（se ≈ 0.4–0.9%）。§4 的单次 A/B 对照在统计上是可行的（但仍只是单 replicate 的 exploratory 证据）。

### 1.4 绝对精度水平

| N | field | S | P-FT |
|---:|---|---:|---:|
| 50 | surface_pressure | 0.1445 | 0.1507 |
| 50 | volume_velocity | 0.1464 | 0.1719 |
| 100 | surface_pressure | 0.1361 | 0.1403 |
| 100 | volume_velocity | 0.1363 | 0.1427 |

对照：本仓库 README 记录的 DrivAerML AB-UPT（全量数据、native 6/6 架构、生产配置）relative L2 为 surface pressure 0.0386、volume velocity 0.0600。当前两条臂都在 0.136–0.17，是参考值的 **3.5–4 倍**。样本量小是部分原因，但这个绝对水平本身提示：**两条臂共同受制于输入几何渲染**（R1/R2 对 S 同样有害）。

---

## 2. 根因清单（按 证据强度 × 预期精度影响 排序）

### R1【实测确证】位置归一化使车体在模型坐标系中只有 39 个单位，源模型见到的是 ~600

**代码位置**
- 归一化：`normalized = (x − raw_pos_min)/(raw_pos_max − raw_pos_min) × scale`
  — [normalizers.py:111-156](src/noether/data/preprocessors/normalizers.py)
- 迁移 preset 固定 `scale=1000` — [drivaerml_common.py:38-48](recipes/aero_cfd/src/aero_cfd/presets/drivaerml_common.py)、[drivaerml.py:63-68](recipes/aero_cfd/src/aero_cfd/presets/drivaerml.py)
- 目标包络固定 `[-40, 80]` — [compute_subset_statistics.py:457-459](research/multi_fidelity/tools/compute_subset_statistics.py)
- 源包络 `[-4.5, 6.0]` — [stats.yaml](src/noether/data/datasets/cfd/shapenet_car/stats.yaml)
- 位置编码与 RoPE 的 `max_wavelength=10000`，**均非尺度等变** — [continuous_sincos_embed.py:53-54](src/noether/modeling/modules/layers/continuous_sincos_embed.py)、[rope_frequency.py:45-49](src/noether/modeling/modules/layers/rope_frequency.py)

**实测数据**

| | 源 ShapeNet（预训练时） | 目标 DrivAerML（当前） |
|---|---:|---:|
| 车体 bbox 最大轴（raw） | 5.30–7.20（25 designs，p0.5–p99.5 稳健值 5.30） | 4.7036 m |
| 归一化包络长度 | 10.5 | 120 |
| **归一化后车长** | **505–686 单位** | **39.2 单位** |
| 车长 / 最粗波长(10000) | **5.0–6.9%** | **0.39%** |

→ 车体在位置编码里被压缩 **13–18 倍**。

**为什么降低精度**
1. RoPE 只依赖 token 间相位差。源 trunk 的全部 QK 滤波器是在"车体跨度约 600 单位"的统计下学出来的；目标把同一物理关系压到 39 单位，预训练注意力模式整体失谐 → 初始特征系统性错误 → 优化落入更差盆地（与 6/6 稳定轻度负迁移、随 N 增大而减弱一致）。
2. sincos/RoPE 最细波长 2π≈6.3 单位。在源上 = 1% 车长；在目标上 = **16% 车长**。**S 与 P-FT 都**无法在位置编码层面分辨小于 ~0.75 m 的结构（后视镜、轮拱、边界层）——解释了 §1.4 的绝对精度偏低。
3. 更浪费的是：实测 **99.5% 的体网格 cell 位于 ±10 m 内**（中位半径 2.71 m，p95=4.65 m），而归一化按 120 m 包络进行 → **96% 的坐标量程分配给了只含 0.3% 数据的远场**。

**置信度**：高（两侧均为真实点云实测）。

### R2【实测确证】radius-9 supernode 池化 100% 饱和，丢弃 99% 邻域

**代码位置**
- `radius` 注入 SupernodePoolingConfig — [model_defaults.py:22-36](src/noether/core/presets/model_defaults.py)（默认 9）
- `max_degree=32` — [encoders.py:19-20](src/noether/core/schemas/modules/encoders.py)
- 截断按**点索引顺序**而非最近邻 — [geometric.py](src/noether/modeling/functional/geometric.py) 的 `radius_pytorch` / `radius_triton` / PyG 路径

**实测数据（16,384 几何点 / 1,024 supernodes，数值安全的分块距离计算）**

| 配置 | 物理半径 | 平均度 | 中位度 | zero_frac | **cap32_frac** |
|---|---:|---:|---:|---:|---:|
| **源 ShapeNet**（scale 1000, r=9, 3586 点全为 supernode） | 0.0945 raw | **5.8** | 5.0 | 0.000 | **0.000** |
| **目标当前**（scale 1000, r=9） | **1.08 m** | **3,185** | **3,421** | 0.000 | **1.0000** |
| 目标 scale 8000, r=9 | 0.135 m | 233.8 | 130 | 0.000 | 0.7842 |
| 目标 scale 10000, r=9 | 0.108 m | 166.7 | 73 | 0.000 | 0.7285 |
| **目标 scale 10000, r=1** | **0.012 m** | **5.3** | 3 | 0.000 | **0.0010** |

`cap32_frac = 1.0000` 与 NCSA 上 P0 官方审计报告完全一致（`P0_common_scratch_radius_graph.json` /
`P0_common_finetune_radius_graph.json`：mean=32.0，**全部分位数 0%/1%/5%/50%/95%/99%/100% 均为 32.0**）——
即 1,024 个 supernode **无一例外**全部触顶。

**为什么降低精度**：几何编码器实际收到的是"以 1.08 m（23% 车长）为半径的球内、按索引序任取 32 点"的聚合，**丢弃 99% 的局部几何**。对 P-FT 额外有害：源编码器学的是 ~5.8 个邻居、跨度 1.3% 车长的局部算子，目标把它变成 23% 车长的模糊算子，权重语义完全失配。

**注意 P0 gate 的盲区**：预注册 gate 只检查 `zero_fraction ≤ 0.001`（实测 0.000，通过），**没有对 `cap_fraction` 设门**。协议正文虽写明"capped degree 32 is not an uncapped physical-neighbor count"，但该风险未被任何自动检查捕获。

**置信度**：高（NCSA 官方 P0 审计 + 独立复算，两者吻合）。

### R3【佐证】仓库自己的 DrivAerML 生产配置用的是完全不同的几何参数化

| 参数 | ShapeNet 源（预训练） | **DrivAerML 生产 YAML** | **多保真迁移 preset** |
|---|---|---|---|
| position scale | 1,000 | **100,000** | **1,000** |
| supernode radius | 9 | **0.25** | **9** |
| max_wavelength | 10,000 | **40,000** | **10,000** |
| geometry points / supernodes | 3,586 / 3,586 | **65,536 / 16,384** | **16,384 / 1,024** |
| decoder blocks | 2/2 | 6/6 | 2/2 |

配置链已核实：`train_drivaerml.yaml → train_caeml.yaml → dataset_normalizers: caeml_dataset_normalizers`
（[caeml_dataset_normalizers.yaml:7,20](recipes/aero_cfd/configs/dataset_normalizers/caeml_dataset_normalizers.yaml) = scale 100000）
+ [experiment/drivaerml/ab_upt.yaml:25-38](recipes/aero_cfd/configs/experiment/drivaerml/ab_upt.yaml)（radius 0.25、max_wavelength 40000）。

**含义**：框架作者为 DrivAerML 自行选定的几何参数化与 ShapeNet 的完全不同；迁移研究把 **ShapeNet 的一套（1000/9/10000）直接套在 DrivAerML 上**，两边都不匹配。README 中 0.0386/0.0600 的参考精度出自 scale=100,000 那一套。

**重要陷阱**：`max_wavelength` 决定 `rope.omega` / `pos_embed.omega` **buffer**，而这些 buffer 在
`state_dict` 中，会被 `PreviousRunInitializer` 的 strict load **用源值静默覆盖**
（[previous_run.py:77-90](src/noether/core/initializers/previous_run.py)）。因此迁移臂**不能**靠改
`max_wavelength` 生效，除非把 omega 加入 `patterns_to_instantiate`。这一点 `source_evidence_audit.md`
已警告，本报告实测确认其机制。→ 迁移方案必须在 `max_wavelength = 10,000` 的约束下解决问题。

### R4【中高置信】优化预算把初始化冲淡为"选盆地"

Lion 每步每权重恰好移动 ±lr（[lion.py:190-205](src/noether/core/optimizer/lion.py)）。40k updates、cosine 5e-5→1e-6、warmup 5% → 累计 |Δw| 预算 ≈ **0.9**，而权重量级仅 0.02–0.2（truncnormal002）。优化器有能力把每个权重重写 5–45 次。

→ 6/6 单元 1.02–1.07 的小幅劣化，符合"源初始化选了略差的盆地"，而非"源特征被保留但不适配"。

**关键推论**：在 R1/R2 未修复前，**降低 trunk LR、冻结、或 LoRA 只会更强地保留失配特征，预期使负迁移更糟**（与 Keum & Warey"只冻结几何编码器失败"同向）。这是本方案不选 LR 分层/冻结作为改进的直接理由。

### R5【低影响】绝对相位偏置

车心绝对归一化位置随 scale 变化。RoPE 平移不变不受影响；sincos 低频带的常量偏置可被 MLP/LayerNorm 早期吸收。不作为根因。

### 审计排除项（回答任务书 A 部分，均非根因）

1. **checkpoint 加载语义正确**：`patterns_to_remove/instantiate = backbone.domain_decoder_projections`
   （[run_drivaerml_transfer_strict.py:48,466-477](recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py)）
   + strict `load_state_dict`；261/269 张量按名加载、8 个 readout 张量重置，与 evidence JSON 一致。
2. **输出通道语义一致**：两侧 common task 都是 surface_pressure(1) + volume_velocity(3)；目标场统计仅用当前 train subset 拟合；readout（LN+Linear）整体重置。不存在"shape 相同但含义不同仍被加载"的输出层张量。
3. **坐标置换正确**：`(x,y,z)→(y,z,x)`（det=+1 真旋转）在 normalizer 之前应用于全部位置与向量场，文件名边界枚举完整
   （[transfer_drivaerml.py:28-94](recipes/aero_cfd/src/aero_cfd/datasets/transfer_drivaerml.py)）。已用 run_1 实测 bbox 验证流向轴映射正确（DrivAer x → ShapeNet z）。
4. **decoder reset 边界合理**：`domain_decoder_blocks` 被加载而非重置并非错误——其输入输出都是 hidden 表征，语义由 trunk 决定；失配发生在更上游的位置表征。重置更多层只会减少可迁移参数而不解决根因。
5. **评估链无泄漏、无硬错误**：单 split 实例化、双重 test gate、sidecar/checkpoint SHA 双向核验、指标在 inverse-normalization 后按物理量纲计算、relative L2 对 3 分量整体取范数（置换不改范数）。
6. **S 与 P-FT 配对严格**：同 manifest、同 model/DataLoader seed、readout 同 draw；唯一差异是 trunk 初始化。

---

## 3. 模型结构审计图（common task，7,008,004 可训练参数）

| 模块 | P-FT 初始化 | 可训练 | LR |
|---|---|---|---|
| `backbone.rope.omega` / `pos_embed.omega`（buffer） | **源加载**（锁定 max_wavelength=10000） | 非参数 | — |
| `backbone.encoder`（SupernodePooling: pos_embed + message MLP + proj） | 加载 | ✓ | 5e-5 |
| `backbone.geometry_blocks[0]`（1 × TransformerBlock） | 加载 | ✓ | 5e-5 |
| `backbone.domain_biases.{surface,volume}` | 加载 | ✓ | 5e-5 |
| `backbone.physics_blocks[0]`（Perceiver → geometry） | 加载 | ✓ | 5e-5 |
| `backbone.physics_blocks[1..9]`（self/cross ×9） | 加载 | ✓ | 5e-5 |
| `backbone.domain_decoder_blocks.{surface,volume}[0..1]` | 加载 | ✓ | 5e-5 |
| `backbone.domain_decoder_projections.surface`（LN+Linear→1） | **重置**（774 参数） | ✓ | 5e-5 |
| `backbone.domain_decoder_projections.volume`（LN+Linear→3） | **重置**（766 参数） | ✓ | 5e-5 |

- 加载 7,006,464 可训练参数（99.978%）+ 74 buffer 元素；重置 1,540（0.022%）。
- P-FT 无任何 LR 分层（body/decoder multiplier 均 1.0 → 不生成 modifiers）。
- **语义不兼容边界不在任何参数张量内部，而在输入端：物理坐标 → 模型坐标的仿射映射，以及由它派生的 radius 图。**

---

## 4. 唯一改进方案（B）：源匹配的几何渲染

### 4.1 定义

**B = 现有 P-FT + `--position-scale 10000` + `--supernode-radius 1.0`**，其余全部不变。

这是**一个原则**（"让 DrivAerML 的输入几何在统计上等同于源预训练时所见"）的两个耦合常数，两者都由 §2 的实测**导出**而非调参：

| 判据 | 源实测 | 目标 @ B | 目标当前 |
|---|---:|---:|---:|
| 图平均度 | 5.8 | **5.3** ✓ | 3,185 ✗ |
| 图 32-截断比例 | 0.000 | **0.001** ✓ | 1.000 ✗ |
| 图 zero 比例 | 0.000 | **0.000** ✓ | 0.000 |
| 车长 / 最粗波长 | 5.0–6.9% | **3.9%** ✓ | 0.39% ✗ |
| 域内最大归一化坐标 | 1,000 | **9,971** ✓（< 10,000，无混叠） | 1,000 |

- `scale = 10000` 是**上界**：整个 [-40, 80] 域恰好映射到 [0, 9971]，正好一个最粗周期内，**零混叠**；同时把车放大 10 倍。不能再大，否则远场绕回（补丁已加此校验）。
- `radius = 1.0`（物理 1.2 cm）由"匹配源平均度 5.8"直接解出。

### 4.2 为什么是它（排除其它候选）

| 候选 | 排除理由 |
|---|---|
| trunk 低 LR / 冻结 / 渐进解冻 | R4 推论：R1/R2 未修复时保留的是失配特征，预期更糟 |
| 扩大 reset（重置 decoder blocks） | 减少可迁移参数，不触及根因（§2 排除项 4） |
| 提高 max_wavelength 到 40,000（生产值） | **对迁移臂无效**：omega buffer 会被源 checkpoint 静默覆盖（R3） |
| 采用生产 scale=100,000 | 需配套 wavelength 40,000（同上无效）；且远场绕过 10 个周期 |
| 增大 anchors / geometry points | 改变每 update 计算量，破坏 compute_matched 可比性 |
| LoRA | 协议标记 not_implemented；同样不触及 R1/R2 |

B 是唯一同时命中 R1 与 R2、且在 `max_wavelength=10000` 被 checkpoint 锁死的约束下可行的改动。

### 4.3 预注册（在看到任何 B 结果前冻结）

- 单元：task=`common`、N=`100`、replicate=`0`（subset seed 1103 / model seed 7103）、frame=`shapenet`、budget=`compute_matched`（40k updates）、DataLoader seed=model seed、val seed=4242。
- **A** = 现有 P-FT（`cf974078`，复用 checkpoint，不重训）；**B** = 上述定义，一次训练。
- **主 endpoint**：`best_model.loss.val.total`（A、B 同规则）；`latest` 同时报告。
- 只用 `val`。禁止 test、禁止 grid search、禁止追加 replicate、禁止并行第二配置。

### 4.4 成功判据

1. **主要**：B 的 val equal-field log-macro 明显优于 A，且 34 个 design 的配对差分布支持（不只看 aggregate）。参照 §1.3，se(log)≈0.4–0.9%，可分辨 1–2% 效应。
2. **次要**：B 不再落后于同单元 scratch（S r0/n100，只读引用）。
3. 任一 field 显著退化即判失败，不以 aggregate 掩盖。
4. 单 replicate 仅作 exploratory 证据，不宣称统计学证明或样本节省。

### 4.5 必须声明的归因边界（重要）

R1/R2 同样损害 scratch。因此 **B 优于 A 证明的是"修正几何渲染后迁移更好"，不能单独证明"迁移本身有效"**；完整归因需要在新渲染下再训一条 scratch 臂（本次不做，因为只允许一个新训练作业）。判据 2 的"B vs 现有 S"因此是**跨渲染的对照**，须如实标注。

### 4.6 风险与守护

- zero-neighbor 风险：实测 `zero_fraction = 0.0000`（scale 10000 / r=1）；仍按协议对 B 的真实 dry-run resolved config 跑 `audit_radius_graph.py` 门（≤0.001）。
- fp16：位置只进入 sincos/RoPE，内部强制 fp32（[continuous_sincos_embed.py:78-80](src/noether/modeling/modules/layers/continuous_sincos_embed.py)），无溢出风险。
- 数值：本报告的度数测量使用分块 float64 差值（非 mm-trick），已与 NCSA 官方 P0 审计交叉验证一致。

---

## 5. 已实现的最小补丁（默认行为逐字节不变）

已完成并验证：

| 文件 | 改动 |
|---|---|
| [drivaerml_transfer.py](recipes/aero_cfd/src/aero_cfd/presets/drivaerml_transfer.py) | mixin 新增 `position_scale`（默认 1000.0；校验 finite/positive/≤ max_wavelength）；position 归一化器改用该值 |
| [run_drivaerml_transfer_strict.py](recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py) | 新增 `--position-scale`、`--supernode-radius`；写入 resolved config、audit、provenance sidecar；非默认值时 run_id 加 `-ps…-sr…` 后缀 |
| [eval_drivaerml_transfer_frozen.py](recipes/aero_cfd/scripts/eval_drivaerml_transfer_frozen.py) | 同名两个 flag；与 sidecar 严格一致性校验（旧 sidecar 无该键 → 视为 1000.0 / 9.0，A 完全兼容）；记入 audit |
| [test_position_scale.py](tests/unit/recipes/aero_cfd/test_position_scale.py) | **21 个新测试**：默认不变、两个 normalizer 同步、非法值拒绝、边界 10000 允许、runner 绑定、sidecar 记录、legacy sidecar 兼容、scale/radius 不匹配拒绝 |
| test_strict_transfer_runner.py / test_frozen_eval_config.py | 补齐两个新 Namespace 属性 |
| [README.MD](recipes/aero_cfd/README.MD) | 新增 position scale 说明段 |

**未修改**：冻结协议 YAML/MD、manifest 生成器、统计工具、gate/merge/analyze 工具、`src/noether` 核心代码。

### 验证结果

- `ruff check`：All checks passed
- `ruff format --check`：全部已格式化
- `pytest tests/unit/recipes/aero_cfd/ research/multi_fidelity/tools/`：**81 passed, 0 failed**
  （在一个临时干净 worktree 中验证——`build_eval_config` 设计上要求 worktree 干净；临时 worktree 已删除，主仓库未受影响）

---

## 6. 证据索引

| 证据 | 位置 | 用途 |
|---|---|---|
| P1 长表 816 行 | `research/multi_fidelity_exploratory/evidence/P1_final_raw_val.csv`（已下载，56,153 B） | §1 全部复盘 |
| P0 radius audit | NCSA `.../reports/P0_common_{scratch,finetune}_radius_graph.json` | R2 的 cap_fraction=1.0 |
| 目标几何实测 | NCSA run_1 表面 882,809 点 / 体 14,744,958 cell（有界只读计算，未传输） | R1、R2、B 参数标定 |
| 源几何实测 | 本地 `data/shapenet_car_raw/.../quadpress_smpl.vtk`（25 designs） | 源车长与源图度数 |
| 生产配置对照 | `recipes/aero_cfd/configs/{dataset_normalizers,experiment/drivaerml}` | R3 |

## 7. 与冻结文档的不一致（按要求指出，未修改冻结协议）

1. `experiment_protocol.md` / `.yaml` 与 `RESEARCH_REPORT.zh-CN.md` 多处仍称 P0/P1 `real_status: pending`、
   "尚未运行任何真实目标训练"；实际 P0/P1 已完成（NCSA array 9656853/9656854 COMPLETED，12 个 val cells，816 行指标）。
2. `experiment_protocol.yaml` 的 `evidence_scope.not_demonstrated` 仍列
   `completed_real_train_checkpoint_frozen_evaluation_chain`，该链实际已完成一次。
3. 预注册 graph preflight 只对 `zero_neighbor_fraction` 设门，未对 `cap_fraction` 设门；实测 cap_fraction=1.0 通过了 gate（§2 R2）。建议后续协议修订时补充该门，但本轮不改冻结文件。
