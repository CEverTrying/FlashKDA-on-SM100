# FlashKDA 报告索引

本目录统一保存面向阅读和展示的成品报告。原始数据、复现脚本、NCU 报告与 SASS 仍保存在 `../experiments/`，避免报告与实验产物混在一起。

## 建议阅读顺序

1. [项目导读](PROJECT_GUIDE.md) 面向第一次接触 KDA 和 CUDA kernel 的读者，解释项目结构、执行流程、实验与当前进展。
2. [交互流程图](PROJECT_WORKFLOW.html) 展示输入、K1、workspace、K2、验证证据和 SM100 判断之间的关系。
3. [完整技术报告](FINAL_REPORT.md) 汇总 B300 复现、六项分析、SM100 实验和最终结论。
4. [PRE 准备稿](PREPARATION.md) 提供 12 页讲述结构与答辩问题。

## 基准报告

| 设备 | 报告 | 说明 |
|---|---|---|
| B300 | [BENCHMARK_B300.md](BENCHMARK_B300.md) | 2026-08-28 的 Blackwell B300 基准 |
| GB200 | [BENCHMARK_GB200.md](BENCHMARK_GB200.md) | 官方 Blackwell GB200 对照 |
| H20 | [BENCHMARK_H20.md](BENCHMARK_H20.md) | Hopper H20 对照 |

## 流程图附件

- [PROJECT_WORKFLOW.json](PROJECT_WORKFLOW.json) 保存 Archify 源规格。
- [PROJECT_WORKFLOW.visual-check.json](PROJECT_WORKFLOW.visual-check.json) 保存自动视觉检查回执。当前环境缺少 Chrome，截图检查状态为 skipped，showcase 结构验证为 9/9 通过。

实验结果与复现入口见[实验索引](../experiments/README.md)。
