# MoE-LoRA 研究文件归档

本目录汇总了 `D:\DSAI\containter_new` 中与 MoE-LoRA 直接相关的可复现发布包、历史实现与结果、早期计划、独立结果审计和参考论文。文件按原工作区相对路径保存在 `sources/` 下；源文件没有移动或删除。

## 本地完整包与公开仓库副本

本地资料包保留 158 个源文件（49,824,265 字节）。推送到 GitHub 的公开子集为 156 个源文件：只排除本地参考目录中的第三方 ICLR 论文 PDF 及已有全文提取件；原件保留在本地。提交不包含本地全量 `MANIFEST.csv` 和 `SHA256SUMS`，因为它们覆盖被排除的两个文件；公开子集使用 `PUBLIC_MANIFEST.csv` 和 `PUBLIC_SHA256SUMS`。两个本地全量清单保持原样。
## 核心入口

- 中文说明：[`sources/archive/20261004_mainline_cleanup/legacy_moe_release/README.md`](sources/archive/20261004_mainline_cleanup/legacy_moe_release/README.md)
- English README：[`sources/archive/20261004_mainline_cleanup/legacy_moe_release/README_EN.md`](sources/archive/20261004_mainline_cleanup/legacy_moe_release/README_EN.md)
- 实现：`legacy_moe_release/scrp/moe_lora.py`、`scrp/moe_training.py`
- 训练与评估：`experiments/train_moe_lora.py`、`evaluate_moe_lora.py`、`evaluate_moe_caserta.py`、`reproduce.py`
- 发布完整性工具：`experiments/verify_release.py`

核心子目录同时包含 H=20 基座 checkpoint、训练/验证/密封测试清单、Caserta 数据、三个 seed 的 adapters 与指标、sealed/Caserta 逐实例结果、图表、表格、测试和 `requirements.txt`。原归档说明和历史发布哈希差异记录分别见 `sources/archive/20261004_mainline_cleanup/README.md` 与包内 `checksum_audit.json`。

## 历史与溯源

- `archive/20260917_moe_final_cleanup/`：MoE 消融、seed-13 复核、旧数据与训练代码。
- `archive/20260917_moe_code_cleanup/`：MoE-LoRA 实施计划、5-epoch 复核、旧代码和测试。
- `archive/20260916_interface_cleanup/`：仅保留早期 MoE-LoRA 计划快照和快照说明；普通 LoRA 通用代码快照未复制。
- `archive/20261004_mainline_cleanup/previous_baseline_audit/`：跨方法 solution audit 包，其中包含 MoE provenance、sealed 和 Caserta 机器可读证据。
- `archive/20260917_moe_focus/README.md`：归档范围说明；该目录内的非 MoE 训练结果没有复制。
- 参考论文（不纳入公开仓库）：本地文件名为 `ICLR-2024-mixture-of-lora-experts-Paper-Conference.pdf`，已有文本提取件为 `ICLR-2024-mixture-of-lora-expe.txt`。公开引用见 [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ce806d8b4bf38cd92d483e5a0490d983-Abstract-Conference.html)；两份本地文件由 `.gitignore` 排除。

PLAN-001 归档中有两个引用已发布 MoE checkpoint 的跨方法比较脚本，因此仅复制了这两个脚本和归档 README。该原归档没有相应比较结果目录；A09/FULL 对照输入和其他跨方法材料不在本资料包内，这两份脚本不构成独立复现入口。

## 校验

- `MANIFEST.csv` 列出每个收录源文件的原路径、目标相对路径、字节数、源 SHA-256 和目标 SHA-256。
- `SHA256SUMS` 对 `sources/` 中收录文件提供源 SHA-256 列表。
- 已执行独立复核：158/158 个收录源文件的源与目标 SHA-256 均一致，总计 49,824,265 字节。
- 本次复制校验与历史发布包校验是两件事：本次逐文件复制哈希已全部一致；原发布包 `checksum_audit.json` 记录旧 `SHA256SUMS` 中 29/30 项一致，`output/pdf/tab_moe_caserta_groups.pdf` 有一项历史哈希差异。原始校验文件与 PDF 均未改写。

## 未纳入范围

不包含普通 LoRA 独立筛选和训练档案、MoE-LoRA 无直接依赖关系的 UW-SAA TW-MoE、PLAN-001 跨方法比较所需的其他方法输入/输出，也不包含 `.git`、虚拟环境、缓存、Python 字节码和生成的 TeX 辅助文件。具体扫描到的缓存排除项见 [`EXCLUSIONS.md`](EXCLUSIONS.md)。