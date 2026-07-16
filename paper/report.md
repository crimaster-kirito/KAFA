当前状态：已经形成一篇完整英文论文，不是提纲或半成品。最新版匿名稿为 7 页，包含 4 张表、2 张矢量图、23 篇参考文献，提交检查和 75 个测试均已通过。
论文文件：[main.pdf](/root/projects/KAFA/paper/main.pdf)
英文原文：[main.tex](/root/projects/KAFA/paper/main.tex)
部分	完成度	当前状态
标题与核心论点	95%	已收敛到“选择纯度不足以代表迁移效用”
Abstract	95%	已写入 AG、TREC 四臂、GSM8K、自然噪声及证据边界
Introduction	90%	动机、研究问题、贡献定位和框架入口完整
Related Work	80%	已成文，但相对 CCF-A 标准仍略薄
Problem/Method	93%	路由、概率迁移、anchor control、teacher ranking 均有正式定义
Experimental Protocol	96%	数据、模型、种子、划分、Holm 检验和事后属性均已说明
Results	98%	主结果、负结果、消融、成本和置信区间均已落笔
Discussion	94%	accuracy–calibration reversal 和 purity–utility 分离已经形成逻辑闭环
Limitations/Ethics	98%	测试集复用、合成污染、隐私与外推限制均已披露
Reproducibility	96%	协议、哈希、失败运行历史和矩阵验证均已写入

图表情况：
主框架图已经完成：[framework.pdf](/root/projects/KAFA/paper/figures/framework.pdf)。它展示三种异构客户端、clean/conflict/off-task 数据、路由与概率迁移、服务器端和本地端两条实验支路。
主要实验图已经完成：[evidence.pdf](/root/projects/KAFA/paper/figures/evidence.pdf)。包含 GSM8K 多模型严重度曲线，以及 Hausa/Yoruba 自然弱标签结果。
四张表已经进入 PDF：AG/TREC server 结果与 TREC 四臂消融、AG Co-Teaching、跨模型 GSM8K、1,024 样本严重度曲线。
新增的关键 TREC 四臂实验也已经全部完成：
10 个新种子 × 4 个方法，共 40 个正式记录。
Severe uniform 相对 clean uniform：+1.58 pp，Holm p=.015。
Severe routed 相对 severe uniform：−1.42 pp，Holm p=.0074，但 NLL/ECE 更好。
Routed 相对 anchor 的净差异：−0.04 pp，区间跨零。
这支持“选择更纯、校准更好，不保证服务器 accuracy 更高”，但属于复用 TREC 测试集的事后机制分析，不能包装成独立确认实验。
工程状态已经全绿：
最新 PDF：7 页 US Letter。
4 tables、2 vector figures、23 references。
字体嵌入、匿名信息、引用和数字复算通过。
开发集 12 条、正式四臂 40 条记录通过协议验证。
75 passed。
当前没有占用显存的实验进程。
真正还没达到 100% 的是“CCF-A 说服力”，而不是写作。当前文字完成度约 95%，但我对实验竞争力的诚实评估约为 65%–70%。下一步最有价值的是补：
AG News 同构四臂消融，避免无法归因其负迁移。
FedMD/FedDF 一类成熟方法的同预算基线。
一个未复用测试集的确认性实验或额外 server task。
最后一轮母语级英文压缩与审稿人视角攻击性审读。
因此可以确认：英文原文、摘要、绪论、方法、框架图、实验图、表格和完整 7 页 PDF 都已经完成；现在进入的是“补强 CCF-A 证据强度”，而不是“继续把论文基本章节写完”。