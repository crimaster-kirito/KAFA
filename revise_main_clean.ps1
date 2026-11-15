$ErrorActionPreference='Stop'
$src='F:\BaiduNetdiskDownload\ywp附件6.doc'
$dst='F:\KAFA\ywp附件6_图题引用修改稿.doc'
$word=New-Object -ComObject Word.Application
$word.Visible=$false; $word.DisplayAlerts=0
function Find-P($doc,[string]$needle){
  for($i=1;$i -le $doc.Paragraphs.Count;$i++){
    $p=$doc.Paragraphs.Item($i); $t=$p.Range.Text -replace '[\r\a]',''
    if($t.Trim() -eq $needle){return $p}
  }
  throw "Not found: $needle"
}
function Style-Caption($range){
  $range.ParagraphFormat.Alignment=1
  $range.Font.NameFarEast='宋体'; $range.Font.Name='宋体'; $range.Font.Size=10
}
function Caption-After($doc,$shape,[string]$caption){
  $at=$shape.Range.Paragraphs.Item(1).Range.End
  $r=$doc.Range($at,$at); $r.InsertAfter("$caption`r")
  $cr=$doc.Range($at,$at+$caption.Length); Style-Caption $cr
}
try{
  $doc=$word.Documents.Open($src,$false,$false)
  $doc.SaveAs2($dst,0)
  # Citations in the original prose; keep the original first framework figure untouched.
  $p=Find-P $doc '本文的总体目标是在不交换客户端原始训练数据、且不要求参与模型结构一致的条件下，构建面向异构语言模型的联邦知识传输方法。具体目标包括以下三个方面。'
  $p.Range.InsertAfter("`r如图1所示，KAFA以公共锚点为统一语义参照，依次完成客户端数据路由、本地微调、规范概率聚合和服务器联合更新。")
  $stageTexts=@(
   '阶段结果1：TREC 四臂实验使用 10 个种子、共 40 条正式记录。严重污染条件下，路由蒸馏相对未过滤蒸馏的服务器准确率下降 1.42 个百分点，但 NLL 和 ECE 更好。这说明选择纯度与校准性的提升不必然带来准确率提升。',
   '阶段结果2：IMDb 预先冻结的 40 条正式记录中，路由蒸馏相对严重污染的未过滤蒸馏提高 0.291 个百分点，95% 置信区间为 [0.072, 0.510]；所选样本纯度由 40.23% 提高到 66.53%。该方向与 TREC 不一致，证明迁移效用具有任务依赖性。',
   '阶段结果3：IMDb 固定 K=128 的次级机制实验中，路由选择相对目标任务合格池内的随机选择提高 0.323 个百分点（Holm p=0.002044）；最大纯度上界相对路由选择仅高 0.047 个百分点（Holm p=0.487230）。这表明路由排序含有有效信号，但更高的纯度不构成单调效用保证。',
   '阶段结果4：GSM8K 官方留出集的 3 个种子中，严重污染条件下教师路由相对直接传输的平均精确匹配率提高 3.34 个百分点。由于种子数较少，该结果仅作为描述性证据，不作强显著性结论。'
  )
  for($i=0;$i -lt 4;$i++){ $sp=Find-P $doc $stageTexts[$i]; $sp.Range.Text=("如图{0}所示，" -f ($i+2))+$stageTexts[$i]+"`r" }
  # Captions: figure 1 is the original framework, figures 2–5 are the four result plots.
  Caption-After $doc $doc.InlineShapes.Item(2) '图1 KAFA知识锚点对齐与联邦微调流程'
  Caption-After $doc $doc.InlineShapes.Item(131) '图2 TREC任务上的路由蒸馏结果'
  Caption-After $doc $doc.InlineShapes.Item(132) '图3 IMDb任务上的路由蒸馏结果'
  Caption-After $doc $doc.InlineShapes.Item(133) '图4 IMDb固定数量选择的机制分析结果'
  Caption-After $doc $doc.InlineShapes.Item(134) '图5 GSM8K严重污染条件下教师路由对照结果'
  # Correct section ordering without adding new content.
  $e=Find-P $doc '（三）完成核心实验矩阵和机制分析'; $e.Range.Text='（五）核心实验矩阵与阶段性结果'+"`r"
  $r=Find-P $doc '（四）获得的主要阶段性结果'; $r.Range.Text=''
  $c=Find-P $doc '（五）完成比例'; $c.Range.Text='（六）完成比例'+"`r"
  $doc.Save(); $doc.Close()
}finally{$word.Quit()}
