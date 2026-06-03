# 暂缓事项(latter)

记录已识别、但本阶段**暂不实施**的问题。等核心机制验证有效后再回头处理。

## L1. 评估的循环论证(top-score-bin MSE 自我印证)

### 问题
计划 Step 5 的诊断指标 "top-score-bin MSE" 用 baseline 残差定义 bin,再去验证 adaptive
在这些 bin 上更好。score 和 metric **同源**(都来自同一份 baseline 误差),所以
"adaptive 在高分点上变好" 几乎是定义决定的,接近自我印证,**不能**作为模型整体变强的证据,
甚至可能掩盖全场退化。

### 处理顺序(等核心机制生效后再做)
1. **先确认机制有效**:用诊断指标(top-score-bin、selected-score vs full-score、ESS)
   确认 adaptive 采样确实把高残差区域更多地纳入 anchor —— 这一步**允许**用 score 派生指标,
   因为此处目的就是"验证机制按设计运转",不是验证泛化。
2. **再看全局是否更好**:主判据必须换成**不由 baseline 误差派生**的留出指标:
   - global volume velocity MSE(全场)
   - wake region MSE / non-wake MSE(物理分区)
   - near-wall SDF bands: 0.005L / 0.01L / 0.02L
   top-score-bin 永远只作诊断,不作结论。

### 一句话
"机制是否生效" 可以用 score 派生指标自证;"模型是否更好" 必须用与 score 无关的物理/全场指标。
两步分开,顺序不能反。
