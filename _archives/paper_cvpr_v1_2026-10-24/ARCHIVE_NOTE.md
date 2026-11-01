# 存档说明:paper_cvpr v1(2026-10-24)

**存档时间**:2026-10-24
**存档原因**:论文定位重构——从"theory + Push-T 扩散策略验证"(CVPR 向)转为"纯理论 + 数值验证 + toy 训练桥"(选项 C,NeurIPS 向)。Push-T 平台与 batch-1 实验移作论文二素材。本目录是重构前最后一版完整快照。

## 快照状态

- **主文** `main.pdf`:9 页(8 页正文 + 参考文献),review 格式,编译零警告。
- **补充材料** `suppl.pdf`:L1–L4 证明、C1/H 判据推导、T2(汇率 + 单峰性 + 剂量律)、T1-b(固定 p 完整 ridge)、Zhao 型地板、数值协议 v5–v16。
- **参考文献**:4 条占位条目已于存档当日修复为真实文献(EgoBridge=NeurIPS 2025 / Zeva-Ego=arXiv 2609.24411 / Chen-Wu-Mondelli=arXiv 2609.38011 / Lei et al.=arXiv 2604.13645)。
- **审稿自查** `review/cvpr_review.md`:内部 CVPR 式评审(Weak Reject 倾向,S1–S5 / W1–W8)。
- **已知遗留**:Zeva-Ego 完整作者列表待核;"Fig. in suppl." 缺图;paperID 未填。

## 关联资产(不在本目录)

- Push-T 平台与数据:`wjm/ego/l1_pusht/`(双域数据 122+128 条、batch-1 共 19 runs、`outputs/batch_summary.jsonl`)
- M2.1 标定分析:`l1_pusht/analysis/theory_calib_fit_extrapolate.py` + `outputs/theory_calib_*`(λ* 0.12→0.42 修复证据,δ_scale=0.325)
- 数值验证代码:`doc/.../M1_T1定理/verify_*.py` 等
- 台账:`doc/.../执行台账.md`

## 恢复指引

直接整目录复制回 `../paper_cvpr` 或在原位编译(latexmk -pdf main.tex / suppl.tex)。v1 的正文文本(main/sec/*.tex)在选项 C 重构时大部分可复用:Sec 3–4(模型与理论)、Sec 7 的边界与局限;Sec 5–6(平台与实验)在新定位下整节移出。
