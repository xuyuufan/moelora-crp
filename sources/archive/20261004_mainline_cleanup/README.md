# 2026-10-04 主线整理归档

当前主线已转为纯 Attention 多尺度联合训练；本目录保存旧方案与历史材料。

| 子目录 / 文件 | 内容 |
|---|---|
| `legacy_moe_release/` | 旧 H=20 MoE-LoRA 发布版本：旧 README、训练/评估入口、专属代码、测试、三 seed adapter、密封测试与 Caserta 结果、图表、PDF、SHA256SUMS |
| `historical_projects/` | CRP_RL、hier_pg 和 Mijn_RL_archive_20260913_pre_lora；保留已有文件与 Git 元数据 |
| `historical_outputs/` | 原工作区 outputs 中的历史产物 |
| `previous_baseline_audit/` | 以前的基线审计脚本、摘要和中间证据 |
| `delivery_packages/` | 原 Leveling 与在线 GP 的交付 ZIP |
| `temporary_materials/` | 旧临时打包目录、PDF 编译材料及工作区缓存 |
| `move_manifest.json` | 实际移动的原路径、新路径、文件清单、大小与验证结果 |
| `plan_move.json` | 当前执行计划的新旧路径与 SHA256 校验 |
| `protected_files_before.json` | 33 个当前主线源码、协议、配置、数据和检查点的整理前 SHA256 |
| `git_status_before.txt` | 整理前主项目已有的 Git 改动 |
| `organize.ps1` | 本次整理脚本，拒绝重复执行和覆盖目标 |

历史资产采用移动归档，没有永久删除。当前主线需要的共享源码、旧冻结基座与大规模数据保留在原路径；旧发布包获得独立副本，保留原入口与数据。

## 旧发布包校验

在 `legacy_moe_release/` 目录运行 `python -B -m experiments.verify_release SHA256SUMS_ARCHIVED`，验证本次归档保存的 30 个资产。原 SHA256SUMS 保持未修改，逐项对比记录在 checksum_audit.json：29 个一致，output/pdf/tab_moe_caserta_groups.pdf 与原发布哈希不同。该 PDF 原样保留；本次没有替换它或将新哈希冒充原发布哈希。原 README 是历史版本，不能用于判断当前主线。

历史文件内的绝对路径、运行日志与来源记录保持原样；部分路径已经迁移，查找资产以移动清单为准。旧审计与旧论文图表是历史证据，不是当前 Attention 的结果。

## 恢复

按移动清单的 Source、Destination 反向移动即可恢复某项。先确认原路径没有新文件，不要覆盖新产物；共享副本不必反向移动。
