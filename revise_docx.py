# -*- coding: utf-8 -*-
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

SRC = r"F:\KAFA\ywp附件6_修改稿.docx"
OUT = r"F:\KAFA\ywp附件6_修改稿.docx"

def text(p): return p.text.strip().replace('\r','').replace('\a','')
def find(doc, value):
    for p in doc.paragraphs:
        if text(p) == value: return p
    raise ValueError(value)
def remove(p): p._element.getparent().remove(p._element)
def move_after(el, anchor): anchor._element.addnext(el)
def style(p, size=10.5, bold=False, align=None):
    if align is not None: p.alignment = align
    for r in p.runs:
        r.font.name = '宋体'; r._element.rPr.rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia','宋体')
        r.font.size = Pt(size); r.bold = bold
def para_after(doc, anchor, content, size=10.5, bold=False, align=None):
    p = doc.add_paragraph(content); style(p,size,bold,align); move_after(p._element,anchor); return p
def table_after(doc, anchor, lines):
    tbl=doc.add_table(rows=len(lines), cols=1); tbl.style='Table Grid'; tbl.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,line in enumerate(lines):
        cell=tbl.cell(i,0); cell.text=line; cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            style(p,9)
            if line.startswith('  '): p.paragraph_format.left_indent=Pt(14)
    move_after(tbl._element,anchor); return tbl

doc=Document(SRC)

# Replace the misplaced implementation section and remove the three English algorithm images with it.
h2=find(doc,'（二）完成算法与实验系统实现')
h3=find(doc,'（三）系统模型与问题定义')
start=doc.paragraphs.index(h2); end=doc.paragraphs.index(h3)
h2.text='（二）完成方法设计与实验系统实现'; style(h2,10.5,True)
for p in list(doc.paragraphs[start+1:end]): remove(p)
p=para_after(doc,h2,'已完成 KAFA 的模块化实验系统实现。系统以公共知识锚点和规范标签空间为统一接口，支持异构客户端在不交换原始训练数据和完整模型参数的条件下进行概率知识传输；客户端侧实现模型相对的数据路由与低秩微调，服务器侧实现规范概率校验、聚合以及软硬联合更新。')
p=para_after(doc,p,'实验系统基于 PyTorch、Hugging Face 和 FederatedScope 搭建，并记录锚点顺序、标签顺序、协议摘要、载荷校验结果和样本选择摘要。相关实现使每轮训练过程可以追踪、复核和复现；具体算法流程统一置于后文“KAFA方法”之后。')

# Correct the fifth/sixth level sequence and merge experiment matrix with results.
exp=find(doc,'（三）完成核心实验矩阵和机制分析'); exp.text='（五）核心实验矩阵与阶段性结果'; style(exp,10.5,True)
remove(find(doc,'（四）获得的主要阶段性结果'))
prog=find(doc,'（五）完成比例'); prog.text='（六）完成比例'; style(prog,10.5,True)

# Replace the first poor figure with an editable Chinese process diagram.
goal=find(doc,'第一，建立跨模型统一的知识表达接口。利用公共锚点和有限规范标签集合，将不同模型对候选答案的评价转换为固定维度的标签概率，从而消除模型结构、词表和分词器差异对知识交换的直接影响。')
figp=doc.paragraphs[doc.paragraphs.index(goal)-1]
remove(figp)
flow=doc.add_table(rows=1, cols=5); flow.style='Table Grid'; flow.alignment=WD_TABLE_ALIGNMENT.CENTER
for c,label in zip(flow.rows[0].cells,['公共锚点与规范标签','客户端相对数据路由','低秩本地微调','规范概率上传与聚合','服务器软硬联合更新']):
    c.text=label; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for pp in c.paragraphs: style(pp,9,True,WD_ALIGN_PARAGRAPH.CENTER)
move_after(flow._element,goal._element.getprevious())
cap=para_after(doc,Paragraph(flow._element,doc._body),'图1 KAFA知识锚点对齐与联邦微调流程',10,False,WD_ALIGN_PARAGRAPH.CENTER)
note=para_after(doc,cap,'注：公共锚点为异构客户端提供统一的任务语义参照，流程自左向右表示一次通信轮次中的知识传输顺序。',8.5,False,WD_ALIGN_PARAGRAPH.CENTER)

# Put editable Chinese algorithms beside the KAFA method.
anchor=exp
intro=para_after(doc,anchor,'4.8 KAFA算法流程',10.5,True)
last=para_after(doc,intro,'以下算法给出 KAFA 一轮通信中客户端数据路由、规范概率聚合与服务器联合更新的实现流程。')
algorithms=[
('算法1 客户端本地数据路由',['输入：客户端模型、公共锚点、本地混合数据、路由阈值与保留比例','输出：筛选后的本地训练子集与路由记录','1：按规范标签对公共锚点分组，计算类别条件锚点梯度；','2：计算本地数据和公共锚点的一致性分布，并得到模型相对门控量；','3：若门控量不低于阈值，则保留全部本地数据并结束；','4：否则，针对每个本地样本计算梯度一致性、训练损失和标签不一致边际；','5：将三类信号转换为客户端内部百分位秩，并计算综合风险分数；','6：对于每个规范标签类别：','  6.1：按风险分数从低到高排序；','  6.2：选择预设比例的低风险样本；','7：将所有已选样本及其路由记录输出。']),
('算法2 异构客户端规范概率生成与聚合',['输入：公共锚点、规范标签、候选答案映射、客户端集合','输出：聚合教师概率矩阵','1：服务器固定锚点顺序、标签顺序和协议摘要，并广播给各客户端；','2：对于每个客户端：','  2.1：计算各候选答案的长度归一化损失；','  2.2：将损失转换为规范标签概率，并按固定顺序组成概率矩阵；','  2.3：校验矩阵形状、数值范围、行和与顺序摘要；','3：服务器仅聚合通过校验的客户端概率矩阵；','4：对每个公共锚点和规范标签位置计算有效客户端概率均值。']),
('算法3 服务器软硬联合更新',['输入：公共锚点及真实标签、聚合教师概率、服务器模型','输出：更新后的服务器低秩适配参数','1：按照当前轮次随机种子确定公共锚点顺序；','2：对于每个训练步骤：','  2.1：计算服务器模型在候选答案上的规范标签概率；','  2.2：使用锚点真实标签计算硬监督损失；','  2.3：若存在有效教师概率，则计算软监督损失；否则将其置零；','  2.4：加权组合两类损失，并更新服务器低秩适配参数；','3：记录有效客户端数、目标函数值和教师聚合摘要。'])]
for title,lines in algorithms:
    last=para_after(doc,last,title,10,True,WD_ALIGN_PARAGRAPH.CENTER)
    last=Paragraph(table_after(doc,last,lines)._element,doc._body)

# Caption the four remaining experimental figures and give each a Chinese note.
caps=['图2 TREC任务上的路由蒸馏结果','图3 IMDb任务上的路由蒸馏结果','图4 IMDb固定数量选择的机制分析结果','图5 GSM8K严重污染条件下教师路由对照结果']
large=[]
for sh in doc.inline_shapes:
    if sh.width.pt>250 and sh.height.pt>100:
        el=sh._inline
        while el.tag != '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p': el=el.getparent()
        p=Paragraph(el,doc._body)
        if p._element.sourceline is None or True: large.append(p)
# The recreated flow table is not an inline shape; after deletion, these are exactly the result figures.
for p,captext in zip(large[-4:],caps):
    cap=para_after(doc,p,captext,10,False,WD_ALIGN_PARAGRAPH.CENTER)
    para_after(doc,cap,'注：图中数据来自对应任务的正式实验记录，具体指标与统计口径见正文说明。',8.5,False,WD_ALIGN_PARAGRAPH.CENTER)

doc.save(OUT)
