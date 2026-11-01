# MoE-LoRA 实施计划：基于已清理的 V1 + 通用 LoRA

更新时间：2026-09-17。**本计划的工程实现已完成，尚无正式性能结论。**
验收结果：54 项测试通过；CPU/FP32 与 CUDA/FP16 smoke 均完成 3 次有效更新、0 次跳过，冻结基座及 adapter 保存重载检查通过。真实 H20 基座零增量检查按 `2e-5` 容差通过（最大差 `1.9073486328125e-6`，动作一致，并非逐位一致）；adapter 保存重载逐位一致。产物分别位于 `output/moe_lora_smoke_cpu` 与 `output/moe_lora_smoke_cuda`。
用户最新要求是先移除旧方案接口、保留通用 LoRA。该清理已完成，见[清理记录](interface_cleanup.md)。
上一版计划中“保留旧模式、扩充 ADAPTATION_MODES、兼容 FULL/post-tanh”等要求作废；原文保存在清理快照中。

## 1. 已确定的目标与边界

将现有 stack_projection[2] 从普通线性层替换为共享基座 + 多个 LoRA 增量专家 + 输入路由。
固定 H=20 基座；只训练专家 A/B 和 router。默认研究配置：2 个 dense 专家、rank=8、alpha=16、dropout=0。
选层是优先级与布局融合后的 512→128 投影，位于 self-attention 之前。

保留公开观测、动作 mask、LSTM、attention、decoder 和搬移次数目标。此次不引入第二个训练器，不恢复 Q8/D8、FULL/decoder、post-attention/post-tanh、V2/V3、偏好或多目标模式。
先完成工程与小规模 smoke，不自行启动长期训练、超参搜索、封存测试或修改论文结论。

## 2. 当前代码事实

### 2.1 模型位置

~~~text
priorities [B,S,T]
  ├─ 相对优先级 + tier_position
  │  → container_embedding: 2→64→128
  │  → stack_lstm: 128→128
  │  → concat(mean(outputs), hidden[-1]): 256
  │  → priority_projection: 256→128
  └─ 高度/剩余容量/源栈标记/全局占用率
     → layout_projection: 4→64→128

concat(priority,layout): [B,S,256]
  → stack_projection[0]: 256→512
  → ReLU
  → stack_projection[2]: 512→128    ← 仅替换这里
  → 3 × self-attention + FFN
  → context_projection → cross_attention
  → query_projection / key_projection
  → FP32 dot / tanh / mask / log_softmax
~~~

真实 checkpoint 为 checkpoints/h20.best.pt：V1、embed_dim=128、ffn_dim=512、heads=8、3 个 encoder block，1,105,842 个参数。
SHA256 为 aedd5e744607359318352f699cb4320f85dbbfc2c12b510047e04bd0013ef2d0。
安装时维度仍必须从原层读取，以支持小模型测试。

### 2.2 当前调用链和已存在能力

~~~text
experiments/adapt_lora.py::adapt
  → scrp/adaptation.py::load_h20_base_policy（严格 V1/H20）
  → scrp/lora.py::install_lora（target_modules 必填）
  → configure_trainable_parameters(model)（只开放普通 LoRA A/B）
  → train_online_crp_scale_diverse.py
      _collect_scale_rollouts（无梯度采样）
      _backward_replay（逐子批前向与 backward，返回 float）
      evaluate_fixed_validation（greedy 验证）
  → save_lora_adapter（best/latest）

evaluate_lora_adaptation.py::H20AdaptationSelector
  → load_h20_base_policy → load_lora_adapter → 可选普通 LoRA merge
~~~

现在没有 ADAPTATION_MODES、load_adaptation_checkpoint、_save_run_checkpoint、loss_mode、return_score_diagnostics 或旧专家头。
模型 action_log_probabilities(priorities, forbidden) 只返回 Tensor；forward 额外接受 greedy。
通用训练 CLI 使用 --target-modules；没有 --mode 或 --lora-target-modules。

已有的冻结快照按照 requires_grad=False 收集，并检查名字存在、仍冻结和值未改变。
verify 工具已经按实际目标检查零增量，执行真正保存重载，检查 mask、有限值和 greedy 动作；失败返回 1。
上述修复直接复用，不应按旧计划再改一次。

原始训练器仍被适配依赖，不要拆成新框架或归档它。
现有 39 项测试通过；其中包括普通 LoRA 完整训练保存加载、原训练 resume、标量/向量化和 replay 梯度验证。
旧专用模式测试已归档，不要以“恢复之前 67 项”为目标把废弃接口加回来。

## 3. 两篇论文提供的思路与本项目取舍

前次论文审读依据：

- [Mixture of LoRA Experts，ICLR 2024](../../参考论文/ICLR-2024-mixture-of-lora-experts-Paper-Conference.pdf)，§3.2–3.3：将多个已训练 LoRA 的输出门控混合，加回基座；组合阶段主要学习门控，含均衡项。原方法使用专家输出构造门控输入。
- [Mixture-of-experts enhanced dynamic algorithm configuration](../../参考论文/1-s2.0-S0957417426027090-main.pdf)，§3.2：融合表示经过两个异构普通专家和 dense 路由进行残差增强；路由使用搜索进度/改进信号，残差系数 0.1。**这篇没有 LoRA。**

本项目采用共享冻结线性基座、多低秩增量、每栈输入路由，并联合训练专家与 router。
这是结合论文思路的适配设计；没有复现预训练专家组合流程，也不移植 GCN、AFM、HV 奖励、进度路由或 0.1 残差系数。
不要在代码注释或论文中写成完整复现任一原方法。

## 4. 新层的精确定义

令 x 为 ReLU 后的 [B,S,d_in]：

~~~text
g(x) = softmax(router(x), dim=-1)             # [B,S,E]
delta_i(x) = (alpha/rank) * B_i(A_i(x))       # [B,S,d_out]
y = linear(x, W0, b0) + Σ_i g_i(x)*delta_i(x)
~~~

- W0/b0 只保存、计算一次。专家仅有 A/B，无额外 bias、激活或全秩主分支。
- A_i 各自独立 Kaiming uniform，B_i 全零，router weight/bias 全零。初始 y 与原层一致。
- router 是 Linear(d_in,E,bias=True)。默认 E=2、r=8、alpha=16；全部专家参与，不实现 top-k、容量限制、噪声、温度搜索或额外 LayerNorm。
- router Linear 和 softmax 必须在禁用 autocast 的 FP32 区域中运行；专家乘法遵守外层精度，混合归约 FP32，增量转回 base output dtype 再相加。保留模型现有 FP32 最终评分。
- dropout 本版必须为 0，避免无梯度 rollout 与 replay 使用不同随机路径。
- 真实空栈仍是模型输入；不能把动作非法 mask 当路由 padding mask。
- 复制原层的 weight/bias、device、dtype、train/eval 状态。模型 forward 签名和返回值不变。
- 默认新增可训练参数为 2*8*(512+128)+(512*2+2)=11,266，总参数为 1,117,108。
- 零初始化首步 A/router 的任务梯度可以为零；先让 B 更新，再检验 A/router 的有效梯度。

## 5. 文件级实施任务

### 5.1 新增 scrp/moe_lora.py

集中实现以下接口，不引入 AdapterRegistry 或自动发现：

~~~python
MOE_LORA_TARGET_MODULE = "stack_projection.2"
MOE_LORA_ADAPTER_FORMAT = "mijnrl_moe_lora_adapter_v1"

class LoRAExpert(nn.Module): ...
class MoELoRALinear(nn.Linear): ...

def install_moe_lora(model, *, num_experts=2, rank=8, alpha=16.0,
                     dropout=0.0) -> MoELoRALinear: ...
def configure_moe_lora_training(model) -> tuple[str, ...]: ...
def moe_lora_config(model) -> dict[str, object]: ...
def moe_lora_state_dict(model) -> dict[str, torch.Tensor]: ...
def save_moe_lora_adapter(path, model, *, base_checkpoint_sha256,
                         metadata) -> Path: ...
def load_moe_lora_adapter(model, path, *,
                         expected_base_checkpoint_sha256=None,
                         map_location="cpu") -> dict[str, object]: ...
def capture_moe_lora_routing(model): ...  # contextmanager
~~~

MoELoRALinear 不继承 LoRALinear，防止被普通 LoRA 的枚举、保存、合并误识别。
安装器只替换固定目标，拒绝重复安装或与普通 LoRA 混装；E/rank 为正整数，alpha 有限且正，dropout=0。
E=1 留作退化测试。冻结 helper 先冻结全模型，再仅开放当前 MoE 的专家 A/B 和 router。

参数命名固定为 stack_projection.2.experts.0.lora_A / lora_B、experts.1.*、router.weight / router.bias。
原 weight/bias 键名和形状保持。临时路由数据不作为 buffer/Parameter，不进入 checkpoint。
scrp/lora.py 的普通接口与格式不扩展旧“模式表”，也不把 MoE 参数塞进普通 lora_state_dict。

### 5.2 新 adapter 保存加载

~~~python
{
    "format": "mijnrl_moe_lora_adapter_v1",
    "base_checkpoint_sha256": "...",
    "moe_lora_config": {
        "target_module": "stack_projection.2",
        "in_features": 512, "out_features": 128, "bias": True,
        "num_experts": 2, "rank": 8, "alpha": 16.0, "dropout": 0.0,
        "routing": "dense_per_stack_input_softmax", "temperature": 1.0,
    },
    "adapter_state_dict": {
        # 仅专家 A/B 和 router 参数，包含完整模块前缀
    },
    "metadata": {
        # 训练配置、种子、清单哈希、选择指标、验证记录
    },
}
~~~

加载顺序：严格加载原基座 → 核对格式/hash/目标/维度 → 在临时层上校验全部键和形状 → 加载成功后安装。
缺少/额外 adapter 键、夹带 base weight/bias、错误 hash 或结构均报错；失败不得留下半安装模型。
固定 W0 不写入 adapter，训练不会改写原 checkpoint。

在 scrp/adaptation.py 新增一个只分发普通 LoRA 与 MoE-LoRA 两种格式的 load_adapter helper，供评估/verify 共用。
保留 load_h20_base_policy，不恢复旧 full/residual/post-attention 加载器。
普通 mijnrl_lora_adapter_v1 格式与目标顺序保持。

### 5.3 在共享 _backward_replay 接入路由辅助项

当前返回 float 的默认行为保留，新增 router_balance_coefficient=0.0 和 return_diagnostics=False 关键字参数。
均衡系数必须有限且非负。普通调用不传时损失和返回语义不变；非 MoE 模型传非零均衡系数须报错。

capture_moe_lora_routing 的上下文覆盖每个 replay 子批的前向、损失和 backward。
只在此上下文中暂存带梯度的路由概率；每次开始清空，正常/异常退出都关闭并清空。
rollout、validation、普通 inference 不保存计算图；不根据 model.training 隐式决定是否缓存。

对子批 c，n_c 为决策数，S 为栈数，N 为本次完整 replay 的决策总数：

~~~text
q_i = mean_{decision,stack} g_i
L_bal(c) = -mean_i log(clamp(q_i,min=1e-8)) - log(E)
L(c) = 现有 normalized-cost RL + 动作熵损失
       + coefficient * (n_c/N) * L_bal(c)
~~~

N 不是 loss_denominator 的实例数；N=0 返回零损失和零计数，不除零。
均衡是子批目标：n_c/N 避免随子批个数直接放大，但非线性 log(mean(g)) 意味着非零辅助项不保证不同 chunk 划分梯度等价。
固定正式 replay batch size=2048，并记录配置；smoke 可用 32。

return_diagnostics=True 时返回 (float_loss, diagnostics)。diagnostics 仅包含新路由所需信息，不恢复被删除的 score 诊断：
decision_count、token_count、router_weight_sums、router_entropy_sum、balance_loss_weighted_sum，以及每专家增量范数和。
统计由同一次前向取得，全部 detach 为 Python 标量/列表，不保存历史概率图，不额外跑整次前向。

### 5.4 扩展 experiments/adapt_lora.py，复用同一训练循环

1. AdaptationConfig 增加 adapter_kind="lora"、num_experts=2、router_balance_coefficient=0.0。CLI 为 --adapter-kind lora|moe_lora、--num-experts、--router-balance-coefficient。不恢复 --mode。
2. target_modules 保持必填。MoE 必须恰好是 ("stack_projection.2",)，拒绝多目标；普通 LoRA 仍可显式选任意兼容线性层。MoE 拒绝非零 dropout。
3. rank/alpha 通用默认仍为 4/8；正式 MoE 命令显式传 8/16、均衡系数 0.01。普通 LoRA 要求均衡系数为 0；不要只加 CLI 而不接入实际损失。
4. 加载基座后，按 adapter_kind 选择安装/冻结 helper；优化器仍只接受 requires_grad=True 参数。复用现有冻结快照与结束校验。
5. adaptation_structure 按类型记录实际结构；新增 MoE run_format="mijnrl_h20_moe_lora_run_v1"；普通格式保留。
6. 保存根据类型分到普通/MoE save helper，best/latest 仍用 validation_mean_relocations_per_container 选择，epoch 0 仍可为 best。
7. replay 接入辅助项并取得统计。按 token/decision 加权汇总专家权重、路由熵、辅助损失和专家增量范数；epoch 0 没有训练统计时填 null。
8. epoch_mean_normalized_loss 在 MoE 下包含辅助项，metadata 明确损失定义；辅助损失另列。验证选模只看搬移 R/N。
9. 保留输出拒绝覆盖、train/validation split 与重叠检查。此轮不增加适配训练精确 resume，不声称 adapter 含 optimizer/RNG 恢复状态。

### 5.5 评估、不变量与吞吐入口

- evaluate_lora_adaptation.py：用新的两格式加载 helper，普通 action 调用不变。MoE 遇到 merge_lora_for_inference=True 必须明确拒绝；动态输入路由不能静态平均合并。metadata 记录 dynamic_moe_lora。
- verify_lora_invariants.py：有 adapter 时推断真实类型、目标/rank/E；无 adapter 时用新增 --adapter-kind 和显式目标。与文件结构冲突的显式参数报错。保留现有零增量、真正 round-trip、mask/有限值/动作比较和失败退出码。
- MoE 输出 merge_supported=false 及原因，不把未执行合并检查写成“通过”。普通 LoRA 合并检查继续保留。
- benchmark_adaptation_throughput.py：只扩展支持 lora/moe_lora，安装、冻结和辅助项与 adapt 一致。无 FULL/decoder 或历史模式列表。这里的短测不能代替正式吞吐结论。
- 当前 final scoring FP32 修复保留；不要用 Git HEAD 的旧模型覆盖当前工作区。

### 5.6 新增 experiments/smoke_moe_lora.py

该文件只负责准备小数据并调用 adapt/verify，不复制训练循环或加入实验队列。

- 参数：--base-checkpoint（默认 checkpoints/h20.best.pt）、--output-root（必填）、--device、--precision-mode、--seed。默认 CPU/FP32。
- 从现有大规模域 adapt_train/validation 按 shape、canonical_id 排序，每个 shape 各取 2 个，共各 8 个；输出新子集，保留 metadata。记录来源路径/hash 和选中 ID，不读或生成封存测试。
- MoE E2/r8/a16/dropout0，LR=3e-4，均衡系数 .01，batch2，K2，replay32，epochs1，updates_per_epoch3、max_updates3，vectorized、runtime_validation=True。
- 输出 latest/best adapter、metrics、summary、invariants；拒绝覆盖已有产物。
- 要求 3 applied、0 skipped、冻结基座不变、合法有限输出、保存重载通过。10 分钟适配预算；超时或少于 3 次更新必须标记未完成并返回非零，不能自动延长。
- CPU FP32 必跑；CUDA 可用再跑 FP16。小模型测试可用生成的独立训练/验证数据，不能触碰项目封存测试。

## 6. 验收与执行顺序

| 阶段 | 交付内容 | 必须通过 |
|---|---|---|
| A | 核对当前清理状态与基座/清单 hash；运行现有测试 | 当前 39 项通过，保留已有未提交工作 |
| B | 新层、安装、冻结 helper | 零增量、公式、参数量、多形状、梯度、动态路由 |
| C | 新格式和两格式加载；评估/verify | 往返、错误输入无半安装、拒绝 MoE merge、普通 LoRA 不退化 |
| D | replay、adapt、吞吐最小接入 | 均衡权重、统计聚合、异常缓存清理、普通调用返回行为不变 |
| E | CPU/CUDA smoke、完整回归、更新 README | 三次有效更新、哈希不变、清晰记录结果；不进入长训练 |

新增 tests/test_moe_lora.py 和 tests/test_moe_lora_integration.py，至少覆盖：

1. 单层与完整模型 B=0 时 CPU FP32 输出逐位一致；多 B/S/T、空栈处理、mask 与 greedy 一致。
2. E=1 同权重等价普通 LoRA；E=2 等价显式专家加权公式，W0 只计算一次；输入不同能产生不同路由。
3. 默认 trainable=11,266，冻结集合正确；多步非退化损失下专家和 router 有有限有效梯度。
4. CUDA autocast 下 router 与最终评分 FP32；零增量同精度比较容差 2e-5 且 greedy 一致；不得拿 FP16 与 FP32 位级一致作假要求。
5. 均匀路由均衡项约 0，偏斜更大且有 router 梯度；尾部子批 n_c/N 权重正确。系数为 0 时保留原 replay 梯度等价。
6. 正常与异常退出均清空缓存，rollout/inference 无路由图残留；聚合统计不因不同子批大小而偏置。
7. round-trip、错误 hash/维度/缺失或多余键/基座参数混入拒绝，失败后模型仍未改装。
8. MoE merge 请求报错；普通 LoRA 原格式/merge 仍通过；不变量失败退出非零。
9. 公开 H20 观测相同而未揭示未来排列不同，输出保持一致。
10. smoke 拒绝覆盖、只用训练/验证、真正完成三步并保存/重载/评估。

## 7. 后续交付命令与交接要求

下列 MoE 命令仅在实现完成后可运行：

~~~powershell
$Py = 'D:\Anaconda\envs\fine_tuning\python.exe'
& $Py -B -m pytest -q -p no:cacheprovider tests
& $Py -B -m experiments.smoke_moe_lora --device cpu --precision-mode fp32 --output-root output/moe_lora_smoke_cpu
& $Py -B -m experiments.smoke_moe_lora --device cuda --precision-mode fp16 --output-root output/moe_lora_smoke_cuda
~~~

正式配置所需关键参数（不在本轮执行长期训练）：

~~~text
--adapter-kind moe_lora --target-modules stack_projection.2
--num-experts 2 --rank 8 --alpha 16 --dropout 0
--router-balance-coefficient 0.01 --replay-decision-batch-size 2048
~~~

交付时说明修改文件、测试/smoke 结果、可训练参数与 hash 校验，更新 README 为“已实现、尚无正式性能结论”。
不执行 reset/clean、不自动提交、不覆盖已有输出，不修改研究稿件与历史结论。
接手模型应以本文和当前源代码为准；archive 中的旧接口和旧计划只供追溯。
