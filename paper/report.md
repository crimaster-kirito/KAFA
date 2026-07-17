当前状态：已经形成可编译的完整英文工作稿，但还不是最终投稿版。重构后的匿名主文为 6 页，补充材料 3 页；29 篇参考文献、匿名性、字体嵌入、旧实验数字和矩阵边界均通过验证。最终 IMDb 结果段、核心实验图表和独立 Conclusion 必须等待冻结官方矩阵完整结束后写入。
论文文件：[main.pdf](/root/projects/KAFA/paper/main.pdf)
英文原文：[main.tex](/root/projects/KAFA/paper/main.tex)
部分	完成度	当前状态
标题与核心论点	90%	已改为中性的 selection-purity/transfer-utility audit，避免预设 IMDb 结论
Abstract	80%	服务器主线英文已重写；仍缺完整 IMDb primary outcome 句
Introduction	95%	三点贡献、A/B/C/D estimands、matched-K 与推断角色完整
Related Work	95%	已补 FedFD、Selective-FD、Fed-DFA、LESS 等直接近邻工作
Problem/Method	95%	概率迁移与 class-conditional routing 有正式定义
Experimental Design	98%	TREC discovery、IMDb prospective、AG task-specific context 及统计 family 已说明
Results	65%	TREC 与既有 AG 已落笔；IMDb 40+30 和 AG secondary 尚未完成
Discussion	85%	服务器审计逻辑已重写；最终结论取决于 IMDb 冻结 gate
Limitations/Ethics	98%	复用测试、合成污染、score-ECE、隐私与外推限制均已披露
Reproducibility	95%	协议、哈希、矩阵验证与 fail-closed 生成链已写入
Conclusion	0%	有条件英文逻辑已设计，但不会在完整结果前伪造结论

图表情况：
主框架图已经完成第二版：[framework.pdf](/root/projects/KAFA/paper/figures/framework.pdf)。它展示 A/B/C/D 四臂、B-A/C-B/D-C 三个估计量、matched-K 对照，以及相互独立的 TREC discovery、IMDb prospective test 和 AG task-specific evidence 三种推断角色；不再暗示错误的实验时间顺序。
旧实验图 [evidence.pdf](/root/projects/KAFA/paper/figures/evidence.pdf) 已移到补充材料。主文当前有一张 evidence-ledger 表；补充材料有四张详细表。IMDb 最终实验图和结果表由 fail-closed 生成器在 70 条官方记录全部验证后自动产生，当前不会生成半成品。
新增的关键 TREC 四臂实验也已经全部完成：
10 个新种子 × 4 个方法，共 40 个正式记录。
Severe uniform 相对 clean uniform：+1.58 pp，Holm p=.015。
Severe routed 相对 severe uniform：−1.42 pp，Holm p=.0074，但 NLL/ECE 更好。
Routed 相对 anchor 的净差异：−0.04 pp，区间跨零。
这支持“选择更纯、校准更好，不保证服务器 accuracy 更高”，但属于复用 TREC 测试集的事后机制分析，不能包装成独立确认实验。
工程状态不是“整个仓库全绿”，只能作如下精确表述：
最新主文 PDF：6 页 US Letter；补充材料 3 页。
当前主文 1 张矢量框架图、1 张证据表；补充材料 1 张实验图、4 张详细表；联合解析 29 references。
字体嵌入、匿名信息、引用和数字复算通过。
开发集 12 条、正式四臂 40 条记录通过协议验证。
聚焦研究回归测试为 134 passed；完整仓库仍有可选依赖相关失败，不能宣称全绿。
IMDb 前瞻官方矩阵正在资源感知调度，实时记录数以
`kafa_script/analysis_outputs/imdb_server_role_reversal/official_evaluation`
为准；本文件不记录易过期的进程快照。
真正还没达到 100% 的既包括 CCF-A 说服力，也包括最终写作。剩余关键路径是：完整 IMDb 40/40 → matched-K 30/30 → AG secondary 40/40 → 生成图表与统计 → 重写摘要结果讨论并新增 Conclusion → 最终七页边界和全量构建审计。
最后一轮母语级英文压缩与审稿人视角攻击性审读。
因此可以确认：英文原文、摘要、绪论、方法、框架图、已有证据图和表格已经落到可编译文件；当前主文是 6 页工作稿。最终 IMDb/AG 结果图表、结果段、摘要结论句和独立 Conclusion 尚未完成，不能把当前稿称为完整 7 页投稿版。
