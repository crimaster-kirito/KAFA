$ErrorActionPreference = 'Stop'

$src = 'F:\BaiduNetdiskDownload\ywp附件6.doc'
$dst = 'F:\KAFA\ywp附件6_修改稿.doc'
$wdFormatXMLDocument = 0
$wdCollapseEnd = 0
$wdAlignParagraphCenter = 1
$wdAutoFitFixed = 0

function Find-Paragraph($doc, [string]$needle) {
  for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
    $p = $doc.Paragraphs.Item($i)
    $t = $p.Range.Text -replace '[\r\a]', ''
    if ($t.Trim() -eq $needle) { return $p }
  }
  throw "Paragraph not found: $needle"
}

function Set-RunFont($range, [int]$size, [bool]$bold = $false, [string]$font = '宋体') {
  $range.Font.NameFarEast = $font
  $range.Font.Name = $font
  $range.Font.Size = $size
  $range.Font.Bold = [int]$bold
}

function Add-Note($doc, $pos, [string]$note) {
  $r = $doc.Range($pos, $pos)
  $r.InsertAfter("`r$note`r")
  $noteRange = $doc.Range($pos + 1, $pos + 1 + $note.Length)
  $noteRange.ParagraphFormat.Alignment = $wdAlignParagraphCenter
  Set-RunFont $noteRange 8 $false '宋体'
  return $pos + $note.Length + 2
}

function Add-Algorithm($doc, $pos, [string]$title, [string[]]$lines) {
  $r = $doc.Range($pos, $pos)
  $r.InsertAfter("`r$title`r")
  $titleRange = $doc.Range($pos + 1, $pos + 1 + $title.Length)
  $titleRange.ParagraphFormat.Alignment = $wdAlignParagraphCenter
  Set-RunFont $titleRange 10 $true '宋体'
  $tablePos = $pos + $title.Length + 2
  $tableRange = $doc.Range($tablePos, $tablePos)
  $tbl = $doc.Tables.Add($tableRange, $lines.Count, 1)
  $tbl.AllowAutoFit = $true
  $tbl.Borders.Enable = 1
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $cell = $tbl.Cell($i + 1, 1)
    $cell.Range.Text = $lines[$i]
    $cell.Range.ParagraphFormat.SpaceAfter = 0
    $cell.Range.ParagraphFormat.SpaceBefore = 0
    Set-RunFont $cell.Range 9 $false '宋体'
    if ($lines[$i] -match '^    ') { $cell.Range.ParagraphFormat.LeftIndent = 18 }
    elseif ($lines[$i] -match '^  ') { $cell.Range.ParagraphFormat.LeftIndent = 9 }
    else { $cell.Range.ParagraphFormat.LeftIndent = 0 }
  }
  $tbl.Rows.Item(1).Cells.Item(1).Range.Font.Bold = 1
  $end = $tbl.Range.End
  return $end
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($src, $false, $false)
  $doc.SaveAs2($dst, $wdFormatXMLDocument)

  # Rebuild the first figure as a concise, Chinese, editable flow diagram.
  $goal = Find-Paragraph $doc '第一，建立跨模型统一的知识表达接口。利用公共锚点和有限规范标签集合，将不同模型对候选答案的评价转换为固定维度的标签概率，从而消除模型结构、词表和分词器差异对知识交换的直接影响。'
  $figPara = $goal.Previous()
  $figPara.Range.Text = ''
  $figRange = $figPara.Range
  $flow = $doc.Tables.Add($figRange, 1, 5)
  $flow.AllowAutoFit = $false
  $labels = @('公共锚点与规范标签','客户端相对数据路由','低秩本地微调','规范概率上传与聚合','服务器软硬联合更新')
  for ($i=1; $i -le 5; $i++) {
    $cell = $flow.Cell(1,$i)
    $cell.Range.Text = $labels[$i-1]
    $cell.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    Set-RunFont $cell.Range 9 $true '宋体'
    $cell.Shading.BackgroundPatternColor = 15132390
    $cell.VerticalAlignment = 1
  }
  $captionPos = $flow.Range.End
  $caption = '图1 KAFA知识锚点对齐与联邦微调流程'
  $captionRange = $doc.Range($captionPos, $captionPos)
  $captionRange.InsertAfter("`r$caption`r")
  $cr = $doc.Range($captionPos + 1, $captionPos + 1 + $caption.Length)
  $cr.ParagraphFormat.Alignment = $wdAlignParagraphCenter
  Set-RunFont $cr 10 $false '宋体'
  Add-Note $doc ($captionPos + $caption.Length + 2) '注：公共锚点为异构客户端提供统一的任务语义参照，箭头表示一次通信轮次中的知识传输顺序。' | Out-Null

  # Replace the misplaced algorithm section with an implementation summary.
  $h2 = Find-Paragraph $doc '（二）完成算法与实验系统实现'
  $h3 = Find-Paragraph $doc '（三）系统模型与问题定义'
  $replace = $doc.Range($h2.Range.Start, $h3.Range.Start)
  $summary = @"
（二）完成方法设计与实验系统实现
已完成 KAFA 的模块化实验系统实现。系统以公共知识锚点和规范标签空间为统一接口，支持异构客户端在不交换原始训练数据和完整模型参数的条件下进行概率知识传输；客户端侧实现模型相对的数据路由与低秩微调，服务器侧实现规范概率校验、聚合以及软硬联合更新。
实验系统基于 PyTorch、Hugging Face 和 FederatedScope 搭建，并记录锚点顺序、标签顺序、协议摘要、载荷校验结果和样本选择摘要。相关实现使每轮训练过程可以追踪、复核和复现；具体算法流程统一置于后文“KAFA方法”之后。

"@
  $replace.Text = $summary

  # Correct section sequence and merge the thin experiment-matrix item into results.
  $oldExp = Find-Paragraph $doc '（三）完成核心实验矩阵和机制分析'
  $oldResults = Find-Paragraph $doc '（四）获得的主要阶段性结果'
  $oldExp.Range.Text = '（五）核心实验矩阵与阶段性结果'
  $oldResults.Range.Text = ''
  $oldProgress = Find-Paragraph $doc '（五）完成比例'
  $oldProgress.Range.Text = '（六）完成比例'

  # Add the detailed Chinese algorithms after Section 4.7, immediately before experimental results.
  # Keep the existing paragraph object as the insertion anchor; Word may normalize
  # paragraph marks while replacing its text, so a fresh exact-text lookup is brittle.
  $anchor = $oldExp
  $insertPos = $anchor.Range.Start
  $intro = '4.8 KAFA算法流程'
  $introRange = $doc.Range($insertPos, $insertPos)
  $introRange.InsertAfter("$intro`r以下算法给出 KAFA 一轮通信中客户端数据路由、规范概率聚合与服务器联合更新的实现流程。`r")
  $introText = $doc.Range($insertPos, $insertPos + $intro.Length)
  Set-RunFont $introText 10 $true '宋体'
  $pos = $insertPos + $intro.Length + 1 + '以下算法给出 KAFA 一轮通信中客户端数据路由、规范概率聚合与服务器联合更新的实现流程。'.Length + 1

  $alg1 = @(
    '算法1 客户端本地数据路由',
    '输入：客户端模型、公共锚点、本地混合数据、路由阈值与保留比例',
    '输出：筛选后的本地训练子集与路由记录',
    '1：按规范标签对公共锚点分组，计算类别条件锚点梯度；',
    '2：计算本地数据和公共锚点的一致性分布，并得到模型相对门控量；',
    '3：若门控量不低于阈值，则保留全部本地数据并结束；',
    '4：否则，针对每个本地样本计算梯度一致性、训练损失和标签不一致边际；',
    '5：将三类信号转换为客户端内部百分位秩，并计算综合风险分数；',
    '6：对于每个规范标签类别：',
    '  6.1：按风险分数从低到高排序；',
    '  6.2：选择预设比例的低风险样本；',
    '7：将所有已选样本及其路由记录输出。'
  )
  $pos = Add-Algorithm $doc $pos '算法1 客户端本地数据路由' $alg1
  $alg2 = @(
    '算法2 异构客户端规范概率生成与聚合',
    '输入：公共锚点、规范标签、候选答案映射、客户端集合',
    '输出：聚合教师概率矩阵',
    '1：服务器固定锚点顺序、标签顺序和协议摘要，并广播给各客户端；',
    '2：对于每个客户端：',
    '  2.1：使用本地分词器计算各候选答案的长度归一化损失；',
    '  2.2：将损失转换为规范标签概率，并按固定顺序组成概率矩阵；',
    '  2.3：校验矩阵形状、数值范围、行和与顺序摘要；',
    '3：服务器仅聚合通过校验的客户端概率矩阵；',
    '4：对每个公共锚点和规范标签位置计算有效客户端概率均值。'
  )
  $pos = Add-Algorithm $doc $pos '算法2 异构客户端规范概率生成与聚合' $alg2
  $alg3 = @(
    '算法3 服务器软硬联合更新',
    '输入：公共锚点及真实标签、聚合教师概率、服务器模型',
    '输出：更新后的服务器低秩适配参数',
    '1：按照当前轮次随机种子确定公共锚点顺序；',
    '2：对于每个训练步骤：',
    '  2.1：计算服务器模型在候选答案上的规范标签概率；',
    '  2.2：使用锚点真实标签计算硬监督损失；',
    '  2.3：若存在有效教师概率，则计算软监督损失；否则将其置零；',
    '  2.4：加权组合两类损失，并更新服务器低秩适配参数；',
    '3：记录有效客户端数、目标函数值和教师聚合摘要。'
  )
  $pos = Add-Algorithm $doc $pos '算法3 服务器软硬联合更新' $alg3

  # Caption the remaining experimental figures from back to front to preserve ranges.
  $captions = @(
    '图5 GSM8K严重污染条件下教师路由对照结果',
    '图4 IMDb固定数量选择的机制分析结果',
    '图3 IMDb任务上的路由蒸馏结果',
    '图2 TREC任务上的路由蒸馏结果'
  )
  $figs = @()
  for ($i=1; $i -le $doc.InlineShapes.Count; $i++) {
    $s = $doc.InlineShapes.Item($i)
    if ($s.Width -gt 250 -and $s.Height -gt 100 -and $s.Range.Start -gt $anchor.Range.Start) { $figs += $s }
  }
  $figs = $figs | Sort-Object { $_.Range.Start } -Descending
  for ($i=0; $i -lt [Math]::Min($figs.Count,4); $i++) {
    $s = $figs[$i]
    $p = $s.Range.Paragraphs.Item(1)
    $at = $p.Range.End
    $cap = $captions[$i]
    $rr = $doc.Range($at, $at)
    $rr.InsertAfter("$cap`r")
    $capRange = $doc.Range($at, $at + $cap.Length)
    $capRange.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    Set-RunFont $capRange 10 $false '宋体'
    Add-Note $doc ($at + $cap.Length + 1) '注：图中数据来自对应任务的正式实验记录，具体指标与统计口径见正文说明。' | Out-Null
  }

  $doc.Save()
  $doc.Close()
} finally {
  $word.Quit()
}
