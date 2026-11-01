# MijnRL：H=20 动态 MoE-LoRA 在线集装箱搬移

[English README](README_EN.md)

本仓库是论文实验的最小可复现版本，只保留最终采用的动态 MoE-LoRA、固定数据、发布 checkpoint、密封测试结果和论文通用材料。普通 LoRA、smoke test、旧数据生成器、Caserta/高密度域接口及旧训练入口均不属于当前发布协议。

## 1. 实验定义

- 问题：在线集装箱搬移问题（Online CRP）。
- 观测：当前目标以及最多 20 个已公开的未来目标，即 `H=20`；完整提取顺序不会暴露给策略。
- 基座：冻结的 `scale_diverse_online_2d_v1` 策略，文件为 `checkpoints/h20.best.pt`。
- 适配域：`large_scale_n100_150`，容器数 100–150，填充率 0.60–0.80。
- MoE-LoRA：只替换 `stack_projection.2`，两个 dense experts，每个 rank 8，`alpha=16`，FP32 动态路由器，共 11,266 个可训练参数。
- 主指标：逐实例计算 `R/N` 后取平均；这不同于 `sum(R)/sum(N)`。
- 数据隔离：`adapt_train` 用于更新，`validation` 用于选 best epoch，`test_sealed` 只在训练全部完成后评估。训练代码不接收测试集路径。

固定规模为 1,024 个训练实例、200 个验证实例和 500 个密封测试实例，三个集合按 canonical ID 两两不相交。

## 2. 目录

```text
checkpoints/h20.best.pt                      冻结的 H=20 基座
data/.../large_scale_n100_150/               固定 train/validation/test_sealed
scrp/                                        环境、策略、MoE-LoRA 与训练核心
experiments/train_moe_lora.py                单个 seed 的正式训练
experiments/evaluate_moe_lora.py             FP32 greedy 密封测试
experiments/evaluate_moe_caserta.py          Caserta 四 horizon 外部评估
experiments/reproduce.py                     三 seed 端到端复现入口
experiments/verify_release.py                发布文件 SHA-256 校验
output/moe_lora_long_20260917_seed*/          三个发布 adapter 与训练指标
output/moe_lora_test_sealed_20260917/         发布的逐实例结果与摘要
output/moe_lora_caserta_20260918/             Caserta 840 实例评估结果
output/pdf/                                   通用问题定义的编译 PDF
figures/gen_fig_moe_sealed_test.py            从摘要重画论文图
docs/                                        通用论文定义与参考文献
tests/                                       MoE 与发布协议回归测试
```

## 3. 环境安装

已验证环境：Windows 11、Python 3.12.13、PyTorch 2.12.0+cu126、CUDA 12.6、cuDNN 91002、NumPy 2.4.4、Matplotlib 3.11.2、pytest 9.1.0。正式结果在 NVIDIA GeForce RTX 4060 Laptop GPU 上生成。

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS 的激活命令为 `source .venv/bin/activate`。正式复现需要 CUDA GPU；CPU 可验证代码链路，但由于底层数值运算不同，个别 greedy 决策可能与发布的 CUDA 结果分叉。

先确认下载内容未损坏：

```powershell
python -B -m experiments.verify_release
```

预期输出：`verified 30 release artifacts`。

## 4. 先做快速闭环检查

下面命令只训练一个 seed、执行一次参数更新，并评估前 4 个密封实例。它用于检查“训练 → 保存 → 重新加载 → 评估”，不是论文结果：

```powershell
python -B -m experiments.reproduce `
  --quick-check `
  --device cuda `
  --precision-mode fp16 `
  --output-root output/reproduction_quick
```

输出目录必须不存在；脚本会拒绝覆盖已有实验。成功后应看到：

```text
output/reproduction_quick/
  seed20260913/best.adapter.pt
  seed20260913/latest.adapter.pt
  seed20260913/metrics.jsonl
  sealed_test/summary.json
  sealed_test/episode_results.jsonl
  reproduction.json
```

## 5. 完整三 seed 复现

```powershell
python -B -m experiments.reproduce `
  --device cuda `
  --precision-mode fp16 `
  --output-root output/reproduction_full
```

此命令按顺序训练 `20260913`、`20260923`、`20260933`，分别根据验证集平均 `R/N` 保存 best adapter；随后统一评估 500 个密封实例并生成 PNG/PDF 图。RTX 4060 Laptop GPU 上每个 seed 约 48–49 分钟，三 seed 训练约 2.4 小时，评估时间另计。

正式配置固定在代码中，CLI 故意不开放论文超参数，避免误把改动后的实验标为复现：

| 配置 | 值 |
|---|---:|
| epochs | 5 |
| batch size | 16 |
| rollouts / instance | 8 |
| replay decision batch | 2048 |
| learning rate | 3e-4 |
| entropy coefficient | 0.002 |
| router balance coefficient | 0.01 |
| gradient clip norm | 1.0 |
| training precision | FP16 |
| validation / test precision | FP32 greedy |

如需单独运行一个 seed：

```powershell
python -B -m experiments.train_moe_lora `
  --seed 20260913 `
  --device cuda `
  --precision-mode fp16 `
  --output-dir output/reproduce_seed20260913
```

如只想重新评估仓库内三份发布 adapter：

```powershell
python -B -m experiments.evaluate_moe_lora `
  --device cuda `
  --output-dir output/recheck_released_sealed
```

评估始终动态执行 MoE 路由；该 adapter 不能静态 merge 到基座线性层。

## 6. 发布结果参考

验证集选出的 best epoch：

| seed | best epoch | validation mean R/N |
|---:|---:|---:|
| 20260913 | 2 | 0.928947 |
| 20260923 | 2 | 0.928233 |
| 20260933 | 4 | 0.928554 |

500 个密封实例上的发布结果：

| 方法 | mean R/N | mean relocations |
|---|---:|---:|
| MoE seed 20260913 | 0.920944 | 116.326 |
| MoE seed 20260923 | 0.919516 | 116.148 |
| MoE seed 20260933 | 0.919412 | 116.136 |
| 冻结基座 | 0.946317 | 119.584 |
| Leveling | 1.189675 | 150.268 |

不同 GPU、驱动或数学库可能导致极小浮点差异，并在分数非常接近时改变后续 greedy 轨迹。严格逐实例对照应使用上述 CUDA 软件栈；跨平台复现应重点核对整体指标与结论。

## 7. Caserta 外部公开基准

三个最终 MoE adapter 已冻结后在原始 Caserta Table 2 的 840 个实例上测试，未根据这些结果重新训练或选择 checkpoint。运行命令为：

```powershell
python -B -m experiments.evaluate_moe_caserta `
  --device cuda `
  --output-dir output/moe_lora_caserta_20260918
```

表中数值为逐实例 `R/N` 的平均值；括号内为三个 MoE seed 的平均值相对基座变化，负数更好。

| 方法 | H=0 | H=5 | H=10 | H=20 |
|---|---:|---:|---:|---:|
| 冻结基座 | 1.247424 | 0.890973 | 0.788302 | 0.733953 |
| MoE seed 20260913 | 1.076285 | 0.853679 | 0.783186 | 0.734343 |
| MoE seed 20260923 | 1.083585 | 0.855367 | 0.784189 | 0.734237 |
| MoE seed 20260933 | 1.078470 | 0.854207 | 0.783943 | 0.734512 |
| MoE 三 seed 平均 | 1.079447 (-13.47%) | 0.854418 (-4.10%) | 0.783773 (-0.57%) | 0.734364 (+0.06%) |

基座在四个 horizon 的总搬移数分别为 41,277、30,907、26,639 和 23,993，与历史 Caserta 评估完全一致。三个 MoE 在 H=0、5、10 均改善；H=20 基本持平且略差，三个 seed 的配对 bootstrap 区间都跨过 0，因此不能声称 H=20 Caserta 有提升。该结果是外部公开基准，不是密封测试，也不参与 MoE 选型。

仅比较 Leveling 与 MoE 的论文图见 `figures/fig_moe_caserta_horizons.pdf`，可复现脚本为 `figures/gen_fig_moe_caserta_horizons.py`。

若要查看 21 个 Caserta 规模组（每组 40 个实例）的逐组对比，运行：

```powershell
python -B -m experiments.summarize_moe_caserta_groups
```

主表为 `output/moe_lora_caserta_20260918/caserta_21_group_comparison.xlsx`；同目录还提供 `group_comparison.md`、`group_comparison.csv`、`group_seed_detail.csv` 和机器可读的 `group_comparison.json`。比较对象为 Leveling 与三个最终 MoE seed 在 H=0、5、10、20 下的表现。Leveling 只使用公开堆栈高度，因此其完整动作序列在四个 horizon 上完全一致；脚本会对全部 840 个实例重新验证这一点。

可直接放入论文的三线表源码为 `docs/tables/tab_moe_caserta_groups.tex`，PDF 预览为 `output/pdf/tab_moe_caserta_groups.pdf`。运行 `python -B -m figures.gen_table_moe_caserta_groups` 可从逐组结果重新生成表格源码。

## 8. 测试与输出解释

```powershell
python -B -m pytest -q -p no:cacheprovider
```

- `best.adapter.pt`：按验证集平均 `R/N` 选择，用于最终测试。
- `latest.adapter.pt`：最后完成 epoch 的状态，不用于论文主结果。
- `metrics.jsonl`：epoch 0、每个训练 epoch 和 `run_complete` 记录，可用于审计 seed、路由统计、更新时间和验证得分。
- `episode_results.jsonl`：每个算法、每个密封实例的搬移次数和运行时间。
- `summary.json`：平均指标、配对比较和 bootstrap 区间。

问题定义的 LaTeX 与文献入口见 [docs/README.md](docs/README.md)。历史代码和过程记录已移至仓库外 `D:\DSAI\containter_new\archive`，不会参与当前导入或复现。
