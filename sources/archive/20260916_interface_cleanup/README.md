# 2026-09-16 旧接口清理快照

本目录用于查阅清理前状态，不属于活动训练路径。

- `snapshot/`：53 个工作区文件的原始内容。沿用原相对路径，在文件名末尾附 `.snapshot`，避免被 Python/pytest 当作活动模块或测试。
- `manifest.json`：快照路径、原文件路径与 SHA256；记录清理开始时 Git HEAD。
- `working_tree_before.txt`：清理前工作区状态。快照包括原有未提交修改，不能用 Git HEAD 代替它来恢复。
- `baseline_logits.pt`：原 H20 基座在 4 种布局、共 8 个 validation 状态上的观测、动作 mask 和 log-probability，供本次行为回归。
- `cleanup_*.py`：本次使用的一次性源码转换记录。**不要在当前代码上再次运行**，也不要导入这些脚本作为生产代码。
- `verification.json`：本次收尾的基座/数据哈希、输出一致性、快照完整性和文件变更清单。

恢复旧方案时，先复制项目到独立工作目录，再按 `manifest.json` 的 `path` 恢复对应 `.snapshot` 内容；恢复前校验 SHA256，按依赖成组恢复。不要直接覆盖当前主线。
快照不是整个仓库备份：环境、数据、基座 checkpoint 和历史训练输出继续保留在原位置。

当前移除范围与接口迁移详见 [清理记录](../../docs/interface_cleanup.md)。上一版 MoE 计划保存在 `snapshot/docs/moe_lora_implementation_plan.md.snapshot`，其中要求兼容旧接口的部分已被新计划取代。
