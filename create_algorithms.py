# -*- coding: utf-8 -*-
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=r'F:\KAFA\KAFA算法中文图表.docx'
def font(run,size=10.5,bold=False):
    run.font.name='Times New Roman'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); run.font.size=Pt(size); run.bold=bold
def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell(cell,txt,bold=False,indent=0):
    p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.LEFT; p.paragraph_format.space_after=Pt(0); p.paragraph_format.space_before=Pt(0); p.paragraph_format.left_indent=Pt(indent)
    r=p.add_run(txt); font(r,10,bold); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
def alg(doc,title,inputs,outputs,steps):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(9); p.paragraph_format.space_after=Pt(3); font(p.add_run(title),11,True)
    t=doc.add_table(rows=2+len(steps),cols=1); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'; t.autofit=True
    set_cell(t.cell(0,0),'输入：'+inputs,False); shade(t.cell(0,0),'F2F2F2')
    set_cell(t.cell(1,0),'输出：'+outputs,False); shade(t.cell(1,0),'F2F2F2')
    for i,(s,ind) in enumerate(steps): set_cell(t.cell(i+2,0),s,False,ind)
doc=Document(); sec=doc.sections[0]; sec.top_margin=Cm(2.1); sec.bottom_margin=Cm(2.1); sec.left_margin=Cm(2.4); sec.right_margin=Cm(2.4)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; font(p.add_run('KAFA 方法中文算法表'),16,True)
p=doc.add_paragraph('以下内容可直接复制到论文中，或在 Word 中另存为图片后插入。'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; font(p.runs[0],10)
alg(doc,'算法1 客户端本地数据路由','客户端模型、公共锚点、本地混合数据、路由阈值和保留比例','筛选后的本地训练子集与路由记录',[
('1：按规范标签对公共锚点分组，计算类别条件锚点梯度。',0),('2：计算本地数据和公共锚点的一致性分布，得到模型相对门控量。',0),('3：若门控量不低于阈值，则保留全部本地数据并结束。',0),('4：否则，针对每个本地样本计算梯度一致性、训练损失和标签不一致边际。',0),('5：将三类信号转换为客户端内部百分位秩，计算综合风险分数。',0),('6：对于每个规范标签类别：',0),('按风险分数从低到高排序。',18),('选择预设比例的低风险样本。',18),('7：输出所有已选样本及其路由记录。',0)])
alg(doc,'算法2 异构客户端规范概率生成与聚合','公共锚点、规范标签、候选答案映射、客户端集合','聚合教师概率矩阵',[
('1：服务器固定锚点顺序、标签顺序和协议摘要，并广播给各客户端。',0),('2：对于每个客户端：',0),('使用本地分词器计算各候选答案的长度归一化损失。',18),('将损失转换为规范标签概率，并按固定顺序组成概率矩阵。',18),('校验矩阵形状、数值范围、行和与顺序摘要。',18),('3：服务器仅聚合通过校验的客户端概率矩阵。',0),('4：对每个公共锚点和规范标签位置计算有效客户端概率均值。',0)])
alg(doc,'算法3 服务器软硬联合更新','公共锚点及真实标签、聚合教师概率、服务器模型','更新后的服务器低秩适配参数',[
('1：按照当前轮次随机种子确定公共锚点顺序。',0),('2：对于每个训练步骤：',0),('计算服务器模型在候选答案上的规范标签概率。',18),('使用锚点真实标签计算硬监督损失。',18),('若存在有效教师概率，则计算软监督损失；否则将其置零。',18),('加权组合两类损失，并更新服务器低秩适配参数。',18),('3：记录有效客户端数、目标函数值和教师聚合摘要。',0)])
doc.save(OUT)
