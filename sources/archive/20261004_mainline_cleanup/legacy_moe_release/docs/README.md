# 论文材料索引

[English](README_EN.md) · [项目复现说明](../README.md)

本目录只保留与具体适配方法无关的通用论文材料：

- `h20_problem_definition.tex`：H=20 在线集装箱搬移问题定义、公开观测、动作约束与目标函数。
- `h20_problem_definition_references.bib`：问题定义所引用的文献。

问题定义应保持方法无关。MoE-LoRA 的结构、训练协议和复现命令以项目根目录的中英文 README 及活动代码为准，不应写进该问题定义章节。

中文 TeX 使用 LuaLaTeX 和嵌入字体；交叉引用稳定时应编译两次：

```powershell
lualatex h20_problem_definition.tex
lualatex h20_problem_definition.tex
```

旧实验计划和训练复核记录已移至 `D:\DSAI\containter_new\archive\20260917_moe_code_cleanup\docs`，仅用于历史追溯。
