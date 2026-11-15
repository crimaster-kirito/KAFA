# -*- coding: utf-8 -*-
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=r'F:\KAFA\KAFA算法中文图表_三线表.docx'
def set_font(run,size=10.5,bold=False):
    run.font.name='Times New Roman'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); run.font.size=Pt(size); run.bold=bold
def border(el,edge,val='nil',sz='0'):
    e=OxmlElement(f'w:{edge}'); e.set(qn('w:val'),val); e.set(qn('w:sz'),sz); e.set(qn('w:space'),'0'); e.set(qn('w:color'),'000000'); el.append(e)
def set_borders(table):
    tblPr=table._tbl.tblPr
    old=tblPr.first_child_found_in('w:tblBorders')
    if old is not None: tblPr.remove(old)
    b=OxmlElement('w:tblBorders')
    border(b,'top','single','12'); border(b,'bottom','single','12')
    border(b,'left'); border(b,'right'); border(b,'insideH'); border(b,'insideV')
    tblPr.append(b)
    # single separator line below the first row, the middle line of the three-line table
    tcPr=table.cell(0,0)._tc.get_or_add_tcPr(); tcB=OxmlElement('w:tcBorders'); border(tcB,'bottom','single','8'); tcPr.append(tcB)
def ptext(p,txt,size=10.5,bold=False,indent=0):
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT; p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(1); p.paragraph_format.left_indent=Pt(indent)
    r=p.add_run(txt); set_font(r,size,bold)
def add_algorithm(doc,title,io,steps):
    cap=doc.add_paragraph(); cap.alignment=WD_ALIGN_PARAGRAPH.CENTER; cap.paragraph_format.space_before=Pt(9); cap.paragraph_format.space_after=Pt(3); set_font(cap.add_run(title),11,True)
    t=doc.add_table(rows=2,cols=1); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=True; set_borders(t)
    hp=t.cell(0,0).paragraphs[0]; ptext(hp,io,10,False)
    cell=t.cell(1,0); cell.text=''; body=cell.paragraphs[0]
    for i,(s,ind,bold) in enumerate(steps):
        p=body if i==0 else cell.add_paragraph(); ptext(p,s,10,bold,ind)
doc=Document(); sec=doc.sections[0]; sec.top_margin=Cm(2.1); sec.bottom_margin=Cm(2.1); sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)
title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; set_font(title.add_run('KAFA 方法中文算法表'),16,True)
sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER; set_font(sub.add_run('三线表格式，可直接复制或截图粘贴到论文中。'),10)
add_algorithm(doc,'算法1 客户端本地数据路由','输入：客户端模型 f_c、公共锚点集合 A、本地数据 D_c、门控阈值 τ、保留比例 ρ    输出：筛选子集 S_c、路由记录 R_c',[
('1：按规范标签 l ∈ Y 对 A 分组，计算类别条件锚点梯度 ḡ_c,l。',0,False),('2：计算本地一致性分布与锚点一致性分布，得到模型相对门控量 γ_c。',0,False),('3：若 γ_c ≥ τ，则令 S_c ← D_c，并返回 (S_c, R_c)。',0,False),('4：否则，对每个样本 x_i ∈ D_c 计算一致性 a_i、损失 ℓ_i 和标签边际 m_i。',0,False),('5：计算 r_i ← Rank(a_i) + Rank(ℓ_i) + Rank(m_i)。',0,False),('6：对于每个规范标签 l ∈ Y 执行：',0,True),('令 I_l ← {i : y_i = l}，按 r_i 从小到大排序。',20,False),('将前 ⌈ρ|I_l|⌉ 个样本加入 S_c。',20,False),('7：输出 S_c 及其对应的 R_c。',0,False)])
add_algorithm(doc,'算法2 异构客户端规范概率生成与聚合','输入：公共锚点 A、规范标签集 Y、候选答案映射 h、客户端集合 C    输出：聚合教师概率矩阵 Q̄ ∈ R^(N_A×L)',[
('1：服务器固定 A、Y、h 的顺序及协议摘要 σ，并广播给所有 c ∈ C。',0,False),('2：对于每个客户端 c ∈ C 执行：',0,True),('对每个锚点 a_j ∈ A、标签 l ∈ Y，计算长度归一化损失 ℓ_c,j,l。',20,False),('令 q_c,j ← Softmax(−ℓ_c,j,·)，并按固定顺序堆叠为 Q_c。',20,False),('若 Q_c 不满足形状、范围、行和或摘要 σ 的校验，则拒绝 Q_c。',20,False),('3：令 C_v ← {c ∈ C : Q_c 通过校验}。',0,False),('4：对每个 (j,l)，计算 Q̄_j,l ← |C_v|^(−1) Σ_(c∈C_v) Q_c,j,l。',0,False)])
add_algorithm(doc,'算法3 服务器软硬联合更新','输入：锚点 A 及真实标签 Y_A、聚合教师概率 Q̄、服务器模型 f_s、权重 λ_A 和 λ_D    输出：更新后的服务器适配参数 θ_s',[
('1：按当前轮次随机种子确定锚点顺序 π(A)。',0,False),('2：对于每个锚点批次 B ⊂ π(A) 执行：',0,True),('计算服务器规范标签概率 p_s(·|x; θ_s)。',20,False),('计算硬监督损失 L_hard ← CE(Y_A, p_s)。',20,False),('若 C_v ≠ ∅，计算软监督损失 L_soft ← CE(Q̄, p_s)；否则令 L_soft ← 0。',20,False),('令 L_s ← λ_A L_hard + λ_D L_soft，并更新 θ_s ← Optimizer(θ_s, ∇L_s)。',20,False),('3：记录 |C_v|、L_s 与教师概率摘要。',0,False)])
doc.save(OUT)
