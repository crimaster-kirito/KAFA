当前状态：已经形成包含正式 IMDb 结果、AG secondary 收尾结果和 IMDb selector-component 诊断的可编译英文投稿工作稿。最新完整构建验证为匿名主文 7 页、补充材料 4 页、reproducibility checklist 2 页。31 篇参考文献、匿名性、字体嵌入、报告数字和矩阵边界均通过当前验证。
论文文件：[main.pdf](/root/projects/KAFA/paper/main.pdf)
英文原文：[main.tex](/root/projects/KAFA/paper/main.tex)
部分	完成度	当前状态
标题与核心论点	90%	已改为中性的 selection-purity/transfer-utility audit，避免预设 IMDb 结论
Abstract	99%	已写入 TREC、IMDb primary、matched-K 和 AG secondary 结果；最终仅需按页数压缩做微调
Introduction	97%	三点贡献、A/B/C/D estimands、matched-K 与推断角色完整；已做一轮审稿风险措辞收紧
Related Work	98%	已补 FedFD、Selective-FD、FedID、FedHDS、Fed-DFA、LESS 等直接近邻工作，并在补充材料加入 closest-work boundary 表
Problem/Method	95%	概率迁移与 class-conditional routing 有正式定义
Experimental Design	98%	TREC discovery、IMDb protocol-frozen test、AG task-specific context 及统计 family 已说明
Results	99%	TREC、IMDb 40+30、既有 AG、AG secondary 40/40 和 IMDb selector-component 30/30 均已落笔或进入补充材料
Discussion	98%	已按 IMDb 正向反转、matched-K 和 AG secondary 重写
Limitations/Ethics	98%	复用测试、合成污染、score-ECE、隐私与外推限制均已披露
Reproducibility	95%	协议、哈希、矩阵验证与 fail-closed 生成链已写入
Conclusion	98%	独立英文结论已落笔；已通过最终构建一致性检查

图表情况：
主框架图已经完成第二版：[framework.pdf](/root/projects/KAFA/paper/figures/framework.pdf)。它展示 A/B/C/D 四臂、B-A/C-B/D-C 三个估计量、matched-K 对照，以及相互独立的 TREC discovery、IMDb protocol-frozen test 和 AG task-specific evidence 三种推断角色；不再暗示错误的实验时间顺序。
旧实验图 [evidence.pdf](/root/projects/KAFA/paper/figures/evidence.pdf) 已移到补充材料。主文当前有 2 张矢量图和 2 张表：框架图、IMDb 正式实验图、evidence ledger 和 IMDb 正式结果表。补充材料有 1 张实验图、5 张表和新增 selector-component 文字诊断段，包括 closest-work boundary。IMDb 六项正式资产由 70 条验证记录自动生成。
新增的关键 TREC 四臂实验也已经全部完成：
10 个新种子 × 4 个方法，共 40 个正式记录。
Severe uniform 相对 clean uniform：+1.58 pp，Holm p=.015。
Severe routed 相对 severe uniform：−1.42 pp，Holm p=.0074，但 NLL/ECE 更好。
Routed 相对 anchor 的净差异：−0.04 pp，区间跨零。
这支持“选择更纯、校准更好，不保证服务器 accuracy 更高”，但属于复用 TREC 测试集的事后机制分析，不能包装成独立确认实验。
IMDb protocol-frozen 四臂矩阵与 matched-K 次级矩阵已经完整验证：
Primary D-C 为 +0.291 pp，95% CI [+0.072,+0.510]，paired p=.014940；选择纯度从 40.23% 升至 66.53%，但方向显著为正，因此冻结的负方向确认门禁失败。
Matched routed-K 相对 target-eligible random-K 为 +0.323 pp，Holm p=.002044；maximum-purity-K 相对 routed-K 为 +0.047 pp，Holm p=.487230。该结果支持“排序含有有效信号，但更高纯度不构成单调效用保证”。
IMDb selector-component post-hoc 诊断已经完整验证：
10 个种子 × 3 个 component arms，共 30 条新记录；复用既有 10 条 severe routed-K matched reference。Full routed-K 均值 91.46%，gradient-only 91.30%，loss-only 91.50%，margin-only 91.36%。三组 full-minus-component 对比为：+0.16 pp（Holm p=.135）、−0.04 pp（Holm p=.505）、+0.10 pp（Holm p=.135）。该诊断不能改变 primary gate 或 matched-K family；它只支持“当前数据没有分离出 Holm 显著的单组件替换缺口”。
工程状态不是“整个仓库全绿”，只能作如下精确表述：
最新已验证主文 PDF：7 页 US Letter，其中 References 从第 6 页开始；补充材料 4 页；reproducibility checklist 2 页。
当前主文 2 张矢量图、2 张表；补充材料 1 张实验图、5 张表和新增 selector-component 文字诊断；联合解析 31 references。
字体嵌入、匿名信息、引用和数字复算通过。
开发集 12 条、正式四臂 40 条记录通过协议验证。
`paper/build_submission.sh` 已在纳入 selector-component verifier/summarizer/tests 后通过，包含 147 个研究相关测试。完整仓库仍有可选依赖相关风险，不能未经复测宣称全仓库全绿。
IMDb original 40/40、matched-K 30/30 和条件实现合同均已验证；AG secondary 40/40 已验证并生成 `kafa_script/analysis_outputs/AGNEWS_SERVER_ABLATION_SECONDARY.md`。跨任务 server-audit 资产已由 150 条记录生成并通过 verify-only：TREC 40、IMDb original 40、IMDb matched-K 30、AG News 40。
真正还没达到 100% 的关键路径是：作者/利益冲突/OpenReview 人工字段确认，以及人工决定是否再做投稿前英文微调。已完成一轮审稿人视角攻击性审读，并新增内部准备文件 `paper/REVIEWER_ATTACK_NOTES.md`；该文件不进入匿名 artifact。
OpenReview 注册包已补齐标题、183-word registration abstract、关键词、上传文件清单、验证命令、证据边界和 human-only 字段提醒；`paper/SUBMISSION_HANDOFF_INDEX.md` 已整理成人工提交总入口且不进入匿名 artifact；`paper/OPENREVIEW_COPYPASTE_FIELDS.md` 已整理成可直接复制到 OpenReview 的人工填表入口且不进入匿名 artifact；`paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md` 已整理作者、冲突、funding、data/code、simultaneous-submission 和 AI-policy 等人工字段采集模板且不进入匿名 artifact；`paper/OPENREVIEW_SUBMISSION_RECEIPT.md` 已整理提交后记录 OpenReview ID、上传 hash 和确认材料的模板且不进入匿名 artifact；`paper/WORKTREE_HANDOFF.md` 已分类说明当前 dirty worktree 中的上传文件、helper 文件、实验输出和代码测试；`paper/SUBMISSION_FILE_MANIFEST.md` 已记录四个上传文件的页数、大小和 SHA-256，方便提交前核对；`.venv/bin/python paper/final_submission_preflight.py` 已提供一键本地提交前复查；`paper/COMPLETION_AUDIT.md` 已逐项标注目标完成证据与未完成人工字段；`SUBMISSION_RISK_CHECKLIST.md` 已列出 supported claims、must-not-claim 边界和最高审稿风险。作者、单位、冲突和政策回答仍需人工填写。
匿名 artifact 打包脚本 `paper/package_anonymous_artifact.py` 已通过 dry-run/build；当前 `paper/artifacts/KAFA_anonymous_artifact.zip` 包含 765 个 manifest 文件、766 个 zip entries，SHA-256 为 `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`。zip 文本复扫未检出旧提交片段、local absolute path、原始 Git 地址、repository-owner/user 标识或明显凭据字符串。OpenReview 注册摘要已压缩到本地计数 183 words。
因此可以确认：英文原文、摘要、绪论、方法、IMDb 正式结果、AG secondary 结果、selector-component 诊断、框架图、实验图、表格、独立 Conclusion、完整构建和匿名 artifact 均已落到主稿/补充材料/生成资产。当前不能称最终提交版的原因已经从“实验/构建未完成”收束为“最后人工投稿字段与审稿压缩尚未完成”。
