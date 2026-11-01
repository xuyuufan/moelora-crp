# 2026-10-04：按 PLAN-001 整理 Mijn_RL

当前依据：`Mijn_RL/docs/planning/PLAN_INDEX.md` 和 `PLAN_001_H20_A1_PILOT_AND_REVIEW.md`。当前 H20-A1 任务书明确复用共享 Attention 代码、数据和两份对照证据；这些资产保留原路径。

本归档转移 445 个旧文件，共 1,041,485,830 bytes（约 993 MiB），没有永久删除实验资产。所有转移文件 SHA256 校验通过，78 个保留文件（共享代码、tests、全部固定数据、基座、计划、任务书、对照报告、共享协议）内容未变。

## 内容与证据

- `original_paths/`：旧 Attention 配置、train_attention 及两个旧比较脚本；旧阶段文档、回执、状态；A07–A09／smoke runs、评估与暂停曲线。目录结构对应原项目相对路径。
- `shared_snapshot/`：整理时共享代码／测试副本、原 README、共享协议与旧工作区入口，仅作版本证据。
- `move_manifest.json`：逐文件原路径、归档路径、大小、SHA256；可定位旧报告中的路径。
- `retained_sha256.json`：本次保留资产的 SHA256 清单。
- `verification.json`：转移与保留资产核验结果。
- `git_status_before.txt`：整理前已有修改、删除和未跟踪文件；没有 reset、clean、commit 或 push。

## 恢复边界

本归档是按原路径保存的可恢复文件集合，不是独立可运行项目。旧脚本仍使用原工作区、归档 MoE 来源、共享数据及旧运行合同。不要直接在本目录运行训练或数据生成器。

用户之后明确要求恢复旧 Attention 时，先检查 move_manifest.json 的 SHA256，再将需要的配置、入口、文档及对应 runs 按原相对路径移回 Mijn_RL。目标已有文件时停止核对，不覆盖新工作。共享代码如已变更，先与 shared_snapshot 比较并核对运行合同、源码哈希；不要整份覆盖当前代码或恢复旧 README 来替代当前计划。

暂停 seed 的原始恢复说明位于 `original_paths/docs/attention_multiscale/PAUSED_20261004.md`；其“保留原路径”等文字是旧整理时的历史快照，本次以 move_manifest.json 为准。第三个 seed 保持暂停，本次没有启动训练或模型评估。

`.pytest_cache` 访问被拒绝，保留未处理。当前计划、任务书和共享协议未改写。旧报告保留原样，其历史 runs 引用使用清单定位。

## 整理后验证

现有回归测试 20 passed（11.54 秒），本地模块依赖与导航链接检查通过；重新核验 78 个保留文件 SHA256 均未变化。git diff --check 通过。测试没有启动正式训练或最终 split 模型评分。
