# 旧方案接口清理记录

日期：2026-09-16。范围依据用户确认：**保留通用 LoRA 基础能力，删除 Q8/D8 等专用模式和队列**。
当前状态为 V1 搬移策略 + 显式选层的普通 LoRA；MoE-LoRA 只更新了[实施计划](moe_lora_implementation_plan.md)，尚未实现。

## 1. 实际移除内容

| 范围 | 已移除 |
|---|---|
| LoRA 模式分发 | ADAPTATION_MODES、旧固定目标别名、query/key/decoder/full 等适配训练模式 |
| 旧适配结构 | post_attention_lora.py、post_tanh_residual.py，以及对应安装/冻结/保存/加载分支 |
| 策略变体 | V2 栈位置扩展、V3 preference hyper-decoder、crane/preference 前向参数、score 诊断返回分支 |
| 训练入口 | preference/scalarization/双目标配置、专用 priority_distribution 开关、RPN-LOO 损失选项、旧 score 诊断与固定 epoch 筛选输出接口 |
| 实验队列 | Q8/D8/D8F/FULL、早期筛选、score/mid 的运行脚本、续跑管道和分析器 |
| 专用测试与文档 | 与上述已删除功能直接绑定的 7 个测试文件，以及 2 份过时方案文档 |

本次共从活动目录移除 **26 个文件**：2 个旧适配模块、15 个实验脚本、7 个专用测试、2 份文档。
逐项列表和基于清理前工作区的变更清单在 [verification.json](../archive/20260916_interface_cleanup/verification.json)。
这些数字不包含本轮开始前已经存在的删除或归档变动。

## 2. 保留的主线

- 原始 V1 编码、LSTM、attention、pointer decoder 与显式禁用 autocast 的 FP32 最终评分。
- 通用 LoRALinear、零增量初始化、显式目标列表、A/B 训练、独立 adapter 保存/加载、普通 LoRA merge。
- H20 基座严格加载、基座 hash 绑定、公开观测与非法动作 mask。
- 标量/向量化环境、数据清单、split/重叠校验、原始 V1 训练与 resume。
- 共享 rollout、replay、validation、通用训练控制；原 H20 续训需要的 loss_weighting 兼容仍保留。
- 适配训练的种子控制、精度选择、梯度裁剪、更新/时间预算、best/latest 与训练日志。
- 普通 LoRA / BASE / Leveling 评估、清单生成、吞吐工具和原始训练曲线工具。

底层环境与物理状态/数据工具保留原实现；本次删除的是策略、训练和适配入口中的旧研究分支，没有重写环境。
原始训练器中的 full resume 指恢复训练状态，和已删除的 FULL 专用适配模式不是同一接口。

## 3. 接口变化和兼容性

| 清理前 | 清理后 |
|---|---|
| install_lora 可依赖默认目标 | install_lora(model, target_modules=(...), rank=..., alpha=...)，目标必填 |
| configure_trainable_parameters(model, mode) | configure_trainable_parameters(model)，只开放已安装的普通 LoRA A/B |
| adapt CLI --mode、--lora-target-modules | --target-modules 必填；没有适配模式别名 |
| AdaptationConfig 的 mode/loss_mode 等 | domain 与 target_modules 必填，使用原 normalized-cost RL 及动作熵 |
| 多类型旧 checkpoint 分发 | 严格基座加载 + 普通 load_lora_adapter |
| 模型 preference/crane/诊断参数 | action_log_probabilities(priorities, forbidden) 返回 Tensor |
| replay 可返回 score 诊断 | _backward_replay 返回 float 损失 |
| 吞吐脚本 --modes 多模式 | 指定 --target-modules、rank/alpha，只测普通 LoRA |

普通 adapter 的格式字符串仍为 mijnrl_lora_adapter_v1，目标顺序保留，仍核对基座 hash。
**旧命令行不再兼容**；旧 post-attention、post-tanh 和 full/decoder 专用适配 checkpoint 不再由当前评估入口加载。
历史文件没有被转换或覆盖。需要复现旧方案时，在独立副本中按快照恢复其代码依赖。

同时修正了清理中涉及的通用校验：冻结参数检查不能跳过消失或被解冻的参数；verify 使用实际目标结构、执行真正保存重载，检查有限值和 mask，并在失败时返回非零。

## 4. 代码规模变化

以下按本轮开始时的工作区快照统计，不使用 Git HEAD 替代用户已有修改：

| 核心文件 | 清理前行数 | 清理后行数 |
|---|---:|---:|
| scrp/scale_diverse_policy.py | 396 | 214 |
| scrp/lora.py | 394 | 336 |
| scrp/adaptation.py | 193 | 52 |
| experiments/adapt_lora.py | 908 | 635 |
| experiments/train_online_crp_scale_diverse.py | 2,170 | 1,454 |

剩余活动模块职责见[项目 README](../README.md)。未增加通用插件注册器、第二套训练器或新的实验队列。

## 5. 验证结果

- 完整回归：**39 passed**。命令为 `python -B -m pytest -q -p no:cacheprovider tests`；完整输出在 [pytest.txt](../archive/20260916_interface_cleanup/pytest.txt)。
- 保留的测试覆盖环境、公开信息隔离、数据 split、原始训练/恢复、标量/向量化 rollout 与 replay、普通 LoRA 零增量和更新。
- 新增 7 项用例覆盖一次真实适配更新、冻结基座、adapter 保存/重载/合并评估、CLI 拒绝旧模式、拒绝 NaN/Inf 或变化的 mask、拒绝旧格式，以及 CUDA autocast 下最终评分保持 FP32。
- 原 H20 基座在 **4 种布局、8 个固定 validation 状态**上的 log-probability 与清理前逐位一致，参数量仍为 **1,105,842**。这项检查是固定状态回归，不代表所有可能输入或训练轨迹的形式化等价证明。
- 剩余 16 个源码模块可导入，CLI parser 可生成帮助；活动源码没有旧模式名或已删除模块的导入；Git diff 空白检查通过。
- 清理前 53 份文件快照的 SHA256 全部校验通过。

清理前的 67 项测试包含已经废弃的专用功能；相关测试随实现归档。当前为保留主线的 32 项测试加 7 项新回归，不能把两个总数直接当成覆盖率比较。

核对未变化的 SHA256：

| 资产 | SHA256 |
|---|---|
| checkpoints/h20.best.pt | aedd5e744607359318352f699cb4320f85dbbfc2c12b510047e04bd0013ef2d0 |
| 大规模域 adapt_train.json | 42a05cef63a9ebb123b62b79f43d29c5d214c2d36e81e314fdcf2d6b3b1d5cfe |
| 大规模域 validation.json | 45d3c0dc67dff431284de9b7e20eb62fd5e59ad77a777e3e7783a020f11d928d |

本次未改写 checkpoint、现有数据或历史结果，没有读取项目的封存测试实例，也没有运行长期训练。
回归中的数据测试使用临时生成的独立测试清单，不是项目封存数据。

## 6. 快照与后续交接

清理前文件保存在 [archive/20260916_interface_cleanup](../archive/20260916_interface_cleanup/README.md)。
包含原路径、SHA256、清理前工作区状态，以及 .snapshot 形式的源码/测试/文档；这些文件不参与当前导入和测试收集。
本次保留了既有未提交工作，未执行 reset/clean、提交或推送。

后续模型直接执行[更新后的 MoE-LoRA 计划](moe_lora_implementation_plan.md)：固定 stack_projection.2、共享冻结基座、两个低秩专家与输入 router，复用当前训练循环。
不要按快照里的上一版计划恢复旧接口。
