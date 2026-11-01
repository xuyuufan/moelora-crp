# Baseline Solution Audit

审计日期：2026-09-21

结论：GP、Leveling 和当前正式 MoE checkpoints 产生的解均通过本次校对。正式结果中没有发现实际执行的 illegal action、relocation count mismatch、容器状态破坏或 future-information leakage；没有发现会影响后续公平 baseline comparison 的 solution correctness 问题。

## 1. Scope

本次只检查三种方法已有正式输出的 solution/action correctness、计数与 provenance。没有统一接口，没有把三种方法接入同一个 validator，没有修改算法、配置、模型或训练/评估结果，也没有重新训练 GP/MoE。审计后的跟进修复只修改了 `GP/summary.py` 的绘图与报告 provenance，并新增对应回归测试。

新增的 [baseline_solution_audit.py](D:/DSAI/containter_new/baseline_solution_audit.py) 是独立只读审计器。它读取正式 rule/checkpoint/result，调用现有环境做 replay，并在每一步独立检查 source、blocker、destination、容量、restricted-CRP 条件、自动提箱、计数、容器集合/位置映射和最终空场。聚合证据位于 `baseline_solution_audit_work/`。

状态判定仅针对“解是否正确”。审计发现的 GP 绘图汇总问题已在本次跟进中解决，不改变 GP solution 的 PASS。

## 2. GP

### Implementation and result checked

- 正式配置：`GP/configs/h20_four_terminal_sets.json`。
- 正式结果：`GP/results/h20_four_terminal_sets`，包含 H=20、四种 terminal set、每组 30 seeds，共 120 个完整 run；没有把 smoke run 混入正式结果。
- 最终 rule 从每个 `run_complete.json` 的 `best_prefix` 重建，并逐项确认其 expression、tree size/depth 与对应 `train_complete.json` 一致。
- seed 按配置公式重新计算；120/120 的 seed、H、terminal set、训练/测试 manifest hash、config hash、implementation hash 均吻合。
- 当前正式配置只有 H=20，因此本次 GP 正式 replay 的范围是 120 个 H=20 rules，不把 smoke 中的 H=0/5/10 当成正式结果。

配置 canonical SHA-256 为 `e4ae0339361e3b9321410e8d7fc0415d2e0c2e87eba6190cf425b091c2e96a3b`；产生正式解的 run implementation SHA-256 为 `19b2f3421d688e8d05a2d4b376b31dcf2f072d2bef8ec20d145a9c5f634d52d2`，修复后的 reporting implementation SHA-256 为 `f8f5571de176aaf466137c43c3ede3fed4014c5ad3d68641fb9ff5e88a67305d`。两者分开记录，避免把报告层修复误写成产生正式解的代码。

### Full solution replay

对 120 个 rules 在 840 个原始 Caserta instances 上执行了 100,800 个完整 solution replay，共校对 2,992,283 次 relocation。每个加速 kernel trace 都与 `OnlineCRPEnvironment` 的标量状态逐步对照：

- 当前 target 所在 source 正确；被移动对象始终是 source 顶部、且位于 target 上方的 blocker。
- destination 始终属于当前 legal set，不是 source，且移动前未满载。
- 每次 relocation counter 只增加 1；自动 retrieval 不增加 relocation counter。
- 每步前后容器数量、identity、location map 一致；没有丢失、重复或凭空产生容器。
- retrieval order 与给定 sequence 一致；100,800/100,800 个最终 yard 均完全清空。
- 每个 instance 的 replayed count 与对应 `run_complete.json` 的 reported count 相等；mismatch 为 0。

### Horizon/information check

H=20 的含义确认为“当前 target + 最多 20 个 future targets”，即最多 21 个可见 priority。未揭露容器在 GP public state 中为 0；ID 相关 terminals 只读取已揭露相对 priority。完整 retrieval sequence 保留在 environment 私有状态，rule 只接收 public state 与 legal destinations。

另外，对 120 个正式 rules 各构造一对“可见前缀相同、H 之外隐藏后缀反转”的状态；120/120 对的 public state、legal set 与所选 action 均保持不变。未发现 future-information leakage。

### Tests executed

GP 与 Leveling 的联合相关测试共 54 项通过。GP 测试实际覆盖表达式/保护除法、五种 crossover、六种 mutation、深度限制、固定 seed/resume、一致性、unknown terminal、隐藏后缀置换、Numba-vs-scalar 状态差分、非法 destination 防护与 36,024 regression；新增测试还覆盖单 horizon、四 horizon、旧 run provenance 兼容和混合 implementation hash 拒绝。本次结论不只依赖测试，还包含上述正式结果全量 replay。

**GP solution status: PASS.**

## 3. Leveling

### Implementation checked

`Leveling/heuristic.py` 的 destination rule 是：在排除 source 且未满载的 stacks 中选最小高度，平局取最左 stack。执行逻辑只移动当前 target 上方的顶部 blocker；target 可达后直接 retrieve，不计 relocation。该方法不读取 retrieval horizon 或 future target priority。

### Full solution replay and regression

对原始 Caserta 840/840 instances 做了完整逐步 replay，共检查 36,024 次 relocation：

- actual illegal actions：0；容量、restricted-CRP、source/destination 与 target unblock/retrieval 检查全部通过。
- 840/840 个 yard 最终清空；容器身份和计数始终一致。
- 对结果文件中 H=0/5/10/20 的 3,360 行均逐项核对。四列只是比较标签，Leveling 行为不随 H 改变；count mismatch 为 0。

**Expected Caserta relocations = 36,024**  
**Actual replayed relocations = 36,024**  
**Strict equality = YES**

**Leveling solution status: PASS.**

## 4. MoE

### Formal checkpoints and configuration

根据 `Mijn_RL/README.md`、正式输出和 checkpoint metadata，当前正式模型是 H=20 base `Mijn_RL/checkpoints/h20.best.pt` 加以下三份 validation-selected adapters：

| Method | Seed | Selected epoch | Adapter SHA-256 |
|---|---:|---:|---|
| MoE_seed20260913 | 20260913 | 2 | `a4c83c37cd20b8867ce679aace0087a2833f7d998d297a90d2d2b894128ce47b` |
| MoE_seed20260923 | 20260923 | 2 | `63d0b77a0a4e5c79ba65c508d985e81cc41be80928eb4bd3b429c5f705496c0a` |
| MoE_seed20260933 | 20260933 | 4 | `abf410c8ac6fc155afb821830f99adf2ff96b8ff21c52a02e048ddcc9bc60058` |

Base SHA-256 为 `aedd5e744607359318352f699cb4320f85dbbfc2c12b510047e04bd0013ef2d0`。三个 adapter 的 base hash、seed、target module `stack_projection.2`、rank 8、alpha 16、2 experts、dropout 0、dense softmax routing 和 temperature 1.0 均与正式 provenance 一致。推理为 frozen、greedy、FP32；没有误载旧模型或 debug checkpoint。

### Model output to action mapping

模型 action dimension 与实际 `num_stacks` 完全相同，argmax 的零基 index 直接作为 destination stack index；没有 index offset、source/destination 反转或 padding stack 映射。正式 Caserta evaluator 按相同 `(num_stacks, max_tiers)` 分组，sealed evaluator逐实例运行，两者都没有把 padding stack 加入动作维度。

`forbidden` mask 把 source 和 full stacks 在最终 scores 中设为 `-inf`，然后才执行 argmax。没有 fallback、retry 或事后强制改 action；若 masked policy 仍选中 forbidden destination，现有代码会立即抛出异常。

作为 mask 行为的附加检查，在 Caserta 全量 replay 中计算了“不使用 mask 的反事实 argmax”：三个 adapters 分别有 339、355、354 次会指向非法栈。这不属于实际 illegal action，因为发布 evaluator 在 argmax 前实施 mask；实际执行的 masked illegal actions 为 0。这也说明 mask 是正式 policy 行为的必要组成，而非审计器替它修正动作。

### Caserta full replay

对三个 adapters、四个 H、840 instances 执行了 10,080 个完整 adapter solution replay，共检查 349,767 次实际 relocation；另对 base reference 做了 3,360 个完整 solution replay、122,816 次 relocation。每步同时比对正式 vectorized environment 与独立 scalar environment 的 observation、legal mask、action、retrieval/relocation count、terminal flag 和物理状态。

| Method | H=0 | H=5 | H=10 | H=20 |
|---|---:|---:|---:|---:|
| MoE_seed20260913 | 36,672 | 29,335 | 26,381 | 24,028 |
| MoE_seed20260923 | 36,872 | 29,379 | 26,409 | 24,027 |
| MoE_seed20260933 | 36,875 | 29,365 | 26,397 | 24,027 |

所有逐实例 counts 与 `moe_lora_caserta_20260918/episode_results.jsonl` 一致；illegal actions 0，count mismatch 0，全部 yard 清空。

### Official sealed full replay

正式 large-scale sealed set 的 500 instances 也使用三个 adapter 全量逐实例 replay，共 1,500 个 solutions、174,305 次 relocation：

| Method | Reported/replayed total |
|---|---:|
| MoE_seed20260913 | 58,163 |
| MoE_seed20260923 | 58,074 |
| MoE_seed20260933 | 58,068 |

三者逐实例 count 全部一致，actual illegal actions 0，1,500/1,500 个最终 yard 有效。sealed manifest 未用于训练或 checkpoint selection。

### Horizon and information check

模型训练 horizon 为 H=20。Caserta 正式评估在 frozen 模型上改变 environment observation window，分别测试 H=0/5/10/20；sealed 正式评估为 H=20。每个 H 都只把当前 target 映射为 1、未来 H 个 target 映射为 2..H+1，其他 occupied positions 映射为 unknown `-1`，empty tiers 为 0。模型只收到 public priority tensor 与 legal mask，不收到完整 retrieval sequence、物理 container ID 或 training-only data。

对 base 与三个正式 adapters 的四个 horizon 各执行隐藏后缀置换检查，共 16 对；相同 public observation 均得到相同 action。结合全量 vector/scalar state 对照，未发现 horizon information leakage。

### Data provenance

- Adaptation train：1,024 instances；validation：200；official sealed test：500；external Caserta：840。
- train/validation/sealed/Caserta 的所有两两 canonical-ID overlap 均为 0。
- Caserta 是外部 public benchmark，未用于 adapter selection；sealed test 也未用于 selection。
- GP、Leveling 和 MoE 的 Caserta replay 使用同一 manifest SHA-256：`c28c975c8026a59a1b7995f8afe8a41e9e48b469835b9d2227a9bbe7a5371a12`。不存在三者意外使用不同 Caserta test set 的情况。

MoE 相关 unit/release tests 为 11 passed；两条 warning 仅是 Windows ACL 阻止 pytest 写 `.pytest_cache`，不影响测试或模型输出。

**MoE solution status: PASS.**

## 5. Detected issues

### GP single-horizon convergence plot failure — RESOLVED

- Algorithm/component：GP reporting pipeline。
- File/function：`GP/summary.py::_plot_convergence`。
- Instance/step：不适用；120 个正式 run 的训练与评估均已完成，错误发生在最后画 convergence figure 时。
- Expected：单 horizon H=20 配置写完数值表后生成 `convergence.png` 并正常退出。
- Actual before fix：固定的四个 subplot axes 与仅一个 configured horizon 使用 `zip(..., strict=True)`，抛出 `ValueError: zip() argument 2 is shorter than argument 1`。
- Impact：`summary.json`、`main_table.*`、run/group metrics 和 100,800 行 episode results 已在异常前写出且本次全部 replay 通过；缺少的是 H=20 convergence plot。它不改变任何 action、rule 或 relocation count，也不影响本次 solution correctness 与公平比较数值。
- Resolution：subplot 行列数现在随 configured horizons 动态计算，多余 axes 会隐藏；绘图改用非交互 Matplotlib `Figure`/Agg API，避免 Windows Tk/headless backend 错误。汇总器保留正式 run implementation hash，并另记当前 reporting implementation hash；不同 run hash 混用仍会被拒绝。
- Verification：`py -3.12 -u -m GP.reproduce --config GP\configs\h20_four_terminal_sets.json --stage summarize --workers 1` 已退出 0，并生成 [convergence.png](D:/DSAI/containter_new/GP/results/h20_four_terminal_sets/convergence.png)（54,992 bytes）。正式 `episode_results.jsonl` SHA-256 仍为 `a778f43f3ff15229de11df71b551f9c5d9c36eb9212406796b3285d30ae1524a`，与修复前完全相同。

除此之外，**No correctness issue detected in the checks performed.**

## 6. Validation coverage

| Method | Instances checked | Full/Sampled | Illegal actions | Count mismatch | Final yard valid |
|---|---:|---|---:|---:|---|
| GP | 100,800 solutions（840 × 120 rules） | FULL | 0 | 0 | 100,800/100,800 |
| Leveling | 840 solutions；另核对 3,360 reported rows | FULL | 0 | 0 | 840/840 |
| MoE | 11,580 adapter solutions（10,080 Caserta + 1,500 sealed） | FULL | 0 | 0 | 11,580/11,580 |

没有把 sampled validation 写成 full。本表的 FULL 指每个纳入审计的正式 solution 都从 initial state replay 到 terminal state；MoE 还额外全量核对了 3,360 个 Caserta base-reference solutions。

## 7. Reproducibility

实际执行的测试命令：

```powershell
py -3.12 -m pytest GP\tests Leveling\tests -q

Set-Location D:\DSAI\containter_new\Mijn_RL
py -3.12 -m pytest tests\test_moe_lora.py tests\test_release_artifacts.py -q
Set-Location D:\DSAI\containter_new
```

实际执行的审计命令：

```powershell
py -3.12 -u baseline_solution_audit.py --stage gp
py -3.12 -u baseline_solution_audit.py --stage leveling
py -3.12 -u baseline_solution_audit.py --stage provenance
py -3.12 -u baseline_solution_audit.py --stage moe-caserta --device cuda
py -3.12 -u baseline_solution_audit.py --stage moe-sealed --device cuda
py -3.12 -u -m GP.reproduce --config GP\configs\h20_four_terminal_sets.json --stage summarize --workers 1
```

第一次把所有 pytest 从 workspace root 合并调用时，MoE tests 因其 `scrp` 顶层 import 要求工作目录为 `Mijn_RL` 而在 collection 阶段失败；随后按项目约定在 `Mijn_RL` 目录重跑，11 项全部通过。这个调用路径问题没有执行测试主体，也不是 baseline solution bug。

机器可读汇总为 [baseline_solution_audit.json](D:/DSAI/containter_new/baseline_solution_audit.json)。各阶段原始聚合证据为：

- `baseline_solution_audit_work/gp.json`
- `baseline_solution_audit_work/leveling.json`
- `baseline_solution_audit_work/moe_provenance.json`
- `baseline_solution_audit_work/moe_caserta.json`
- `baseline_solution_audit_work/moe_sealed.json`

## 8. Conclusion

- GP 的正式 H=20 解通过校对。
- Leveling 的解通过校对，并严格复现 36,024。
- 三个正式 MoE checkpoint 在 Caserta 与 official sealed test 上均通过校对。
- 实际 illegal action = 0；relocation count mismatch = 0；future-information leakage 未发现。
- 没有发现会影响公平 baseline comparison 的 solution correctness 问题。GP 单-horizon convergence plot 的汇总阶段异常已解决；修复不影响解或计数。
