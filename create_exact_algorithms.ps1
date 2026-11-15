$ErrorActionPreference='Stop'
$out='F:\KAFA\KAFA算法中文翻译_保持原版式_精确版.docx'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0

function Set-Font($range,[double]$size=10,[bool]$bold=$false){
  $range.Font.Name='Times New Roman'
  $range.Font.NameFarEast='宋体'
  $range.Font.Size=$size
  $range.Font.Bold=[int]$bold
}
function Clear-Borders($table){
  foreach($idx in @(-1,-2,-3,-4,-5,-6)){ $table.Borders.Item($idx).LineStyle=0 }
}
function Set-Edge($cell,[int]$edge,[int]$width=4){
  $b=$cell.Borders.Item($edge); $b.LineStyle=1; $b.LineWidth=$width; $b.Color=0
}
function Add-Mixed($doc,$cell,[string]$markup,[double]$size=10,[bool]$bold=$false,[double]$indent=0){
  $p=$cell.Range.Paragraphs.Item(1)
  $p.Range.Text=''
  $p.Range.ParagraphFormat.SpaceBefore=0
  $p.Range.ParagraphFormat.SpaceAfter=0
  $p.Range.ParagraphFormat.LineSpacingRule=0
  $p.Range.ParagraphFormat.LeftIndent=$indent
  $matches=[regex]::Matches($markup,'<<(.*?)>>')
  $last=0
  foreach($m in $matches){
    $plain=$markup.Substring($last,$m.Index-$last)
    if($plain.Length -gt 0){
      $pos=$cell.Range.End-1; $r=$doc.Range($pos,$pos); $r.Text=$plain; Set-Font $r $size $bold
    }
    $expr=$m.Groups[1].Value
    $pos=$cell.Range.End-1; $er=$doc.Range($pos,$pos); $er.Text=$expr
    $null=$doc.OMaths.Add($er)
    if($er.OMaths.Count -gt 0){ $er.OMaths.Item(1).BuildUp() }
    Set-Font $er $size $false
    $last=$m.Index+$m.Length
  }
  $tail=$markup.Substring($last)
  if($tail.Length -gt 0){ $pos=$cell.Range.End-1; $r=$doc.Range($pos,$pos); $r.Text=$tail; Set-Font $r $size $bold }
}
function Add-Algorithm($doc,[string]$title,[string]$inputText,[string]$output,[object[]]$steps,[bool]$pageBreak=$true){
  $inputParts=$inputText -split '\|\|'
  $rows=2+$inputParts.Count+$steps.Count
  $r=$doc.Range($doc.Content.End-1,$doc.Content.End-1)
  $table=$doc.Tables.Add($r,$rows,2)
  $table.AllowAutoFit=$false
  $table.Rows.AllowBreakAcrossPages=0
  $table.Columns.Item(1).Width=$word.CentimetersToPoints(0.75)
  $table.Columns.Item(2).Width=$word.CentimetersToPoints(16.1)
  Clear-Borders $table

  $titleCell=$table.Cell(1,1); $titleCell.Merge($table.Cell(1,2))
  Add-Mixed $doc $titleCell $title 10.5 $true 0
  for($j=0;$j -lt $inputParts.Count;$j++){
    $row=2+$j
    $inputCell=$table.Cell($row,1); $inputCell.Merge($table.Cell($row,2))
    $prefix=if($j -eq 0){'输入：'}else{'　　'}
    Add-Mixed $doc $inputCell ($prefix+$inputParts[$j]) 10 $false 0
    if($j -eq 0){
      $inputCell.Range.Characters.Item(1).Font.Bold=1
      $inputCell.Range.Characters.Item(2).Font.Bold=1
      $inputCell.Range.Characters.Item(3).Font.Bold=1
    }
  }
  $outputRow=2+$inputParts.Count
  $outputCell=$table.Cell($outputRow,1); $outputCell.Merge($table.Cell($outputRow,2))
  Add-Mixed $doc $outputCell ('输出：'+$output) 10 $false 0
  $outputCell.Range.Characters.Item(1).Font.Bold=1
  $outputCell.Range.Characters.Item(2).Font.Bold=1
  $outputCell.Range.Characters.Item(3).Font.Bold=1

  Set-Edge $titleCell -1 8
  Set-Edge $titleCell -3 4
  Set-Edge $outputCell -3 4

  for($i=0;$i -lt $steps.Count;$i++){
    $row=$outputRow+1+$i
    $num=$table.Cell($row,1)
    $num.Range.Text=[string]($i+1)
    Set-Font $num.Range 9.5 $false
    $num.Range.ParagraphFormat.SpaceAfter=0
    $num.Range.ParagraphFormat.SpaceBefore=0
    $num.Range.ParagraphFormat.Alignment=0
    $entry=$steps[$i]
    Add-Mixed $doc $table.Cell($row,2) $entry.Text 10 $entry.Bold $entry.Indent
  }
  Set-Edge $table.Cell($rows,1) -3 8
  Set-Edge $table.Cell($rows,2) -3 8
  $table.Range.ParagraphFormat.SpaceAfter=0
  $table.Range.Cells.VerticalAlignment=1
  if($pageBreak){
    $after=$doc.Range($doc.Content.End-1,$doc.Content.End-1)
    $after.InsertAfter("`r")
    $after.Collapse(0)
    $after.InsertBreak(7)
  }
}

try{
  while($word.Documents.Count -gt 0){ $word.Documents.Item(1).Close(0) }
  $doc=$word.Documents.Add()
  $sec=$doc.Sections.Item(1)
  $sec.PageSetup.TopMargin=$word.CentimetersToPoints(1.4)
  $sec.PageSetup.BottomMargin=$word.CentimetersToPoints(1.4)
  $sec.PageSetup.LeftMargin=$word.CentimetersToPoints(1.7)
  $sec.PageSetup.RightMargin=$word.CentimetersToPoints(1.7)

  $a1=@(
    [pscustomobject]@{Text='按规范标签划分公共锚点。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='对每个 <<y∈𝒴>> 执行';Indent=0;Bold=$true},
    [pscustomobject]@{Text='计算关于可训练 LoRA 参数的类别条件锚点梯度 <<a_(cy)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='使用每个锚点及其对应的 <<a_(cy)>> 计算锚点参考余弦分布。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='对每个样本 <<z∈D_c>> 执行';Indent=0;Bold=$true},
    [pscustomobject]@{Text='计算样本梯度 <<g_c(z)>> 和训练损失 <<ℓ(z)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='若 <<z>> 是目标任务样本，则使用其声明标签计算 <<s_g(z)>> 和 <<m(z)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='否则，使用最大锚点梯度余弦值，并赋予超出范围的语义边际。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='计算 <<r_c=(Q_(0.25)(s_g(D_c)))/(max(Q_(0.25)(s_g(A)),ε))>>。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='若 <<r_c≥γ>>，则返回 <<D_c^(sel)←D_c>>，并记录门控未触发。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='将 <<−s_g(z)>>、<<ℓ(z)>> 和 <<m(z)>> 转换为客户端内部百分位秩，并计算 <<R(z)>>。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='令 <<K_y←max(1,round(ρ|D_c[y]|/L))>>。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='对每个声明标签 <<y∈𝒴>> 执行';Indent=0;Bold=$true},
    [pscustomobject]@{Text='按 <<(R(z),source_id(z))>> 的升序对合格样本排序。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='选择前 <<K_y>> 个样本；离题样本不计入类别配额。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='合并各类别选择结果，构造 <<Q_c>>，并返回 <<D_c^{sel}>> 和 <<Q_c>>。';Indent=0;Bold=$false}
  )
  Add-Algorithm $doc '算法 1  模型相对的公共锚点数据路由' '客户端模型 <<f_c>>、分词器 <<τ_c>>、本地混合数据 <<D_c>>、公共锚点 <<A>>||规范标签集 <<𝒴>>、门控阈值 <<γ>>、保留比例 <<ρ>>' '选定的训练子集 <<D_c^(sel)>> 和路由记录 <<Q_c>>' $a1 $true

  $a2=@(
    [pscustomobject]@{Text='广播固定的锚点顺序、标签顺序、候选答案映射和协议摘要。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='对每个客户端 <<c∈𝒞>> 执行';Indent=0;Bold=$true},
    [pscustomobject]@{Text='对每个公共锚点 <<x_i>> 执行';Indent=18;Bold=$true},
    [pscustomobject]@{Text='对每个规范答案 <<v_k>> 执行';Indent=36;Bold=$true},
    [pscustomobject]@{Text='使用 <<τ_c>> 对提示和 <<v_k>> 编码，并计算按答案词元归一化的负对数似然 <<l_(cik)>>。';Indent=54;Bold=$false},
    [pscustomobject]@{Text='计算 <<q_(ci)←softmax(−l_(ci))>>。';Indent=36;Bold=$false},
    [pscustomobject]@{Text='按照固定锚点顺序堆叠所有 <<q_(ci)>>，形成 <<Q_c∈ℝ^(N_A×L)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='校验形状、数值有限性、概率范围、行和、锚点摘要和标签顺序摘要。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='若校验通过，则序列化并上传 <<Q_c>>；否则拒绝载荷并记录原因。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='收集全部有效载荷，并定义有效客户端集合 <<𝒞_v>>。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='对每个锚点 <<x_i>>，计算 <<q̅_i←|𝒞_v|^(−1)∑_(c∈𝒞_v)q_(ci)>>。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='堆叠所有 <<q̅_i>> 形成 <<Q̅>>，并返回 <<Q̅>>。';Indent=0;Bold=$false}
  )
  Add-Algorithm $doc '算法 2  规范标签概率生成与客户端聚合' '更新后的客户端模型 <<{f_c}_(c=1)^C>>、分词器 <<{τ_c}_(c=1)^C>>、公共锚点 <<A>>||规范答案文本 <<{v_k}_(k=1)^L>>' '聚合教师概率矩阵 <<Q̅∈ℝ^(N_A×L)>>' $a2 $true

  $a3=@(
    [pscustomobject]@{Text='根据当前轮次的随机种子生成确定性的锚点顺序。';Indent=0;Bold=$false},
    [pscustomobject]@{Text='对 <<s=1,2,…,S>> 执行';Indent=0;Bold=$true},
    [pscustomobject]@{Text='抽取锚点小批次 <<B>>，以及对应的硬标签和教师概率行。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='计算服务器候选答案损失矩阵 <<l_(sB)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='对每个 <<x_i∈B>>，计算 <<p_s(·|x_i)←softmax(−l_s(i))>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='计算 <<ℒ_(hard)←CE(y_i,p_s(·|x_i))>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='若 <<Q̅>> 可用，则计算 <<ℒ_(soft)←−∑_k q̅_i(k)log p_s(k|x_i)>>；否则令 <<ℒ_(soft)←0>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='计算 <<ℒ_s←λ_A ℒ_(hard)+λ_D ℒ_(soft)>>。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='反向传播 <<ℒ_s>>，并更新服务器 LoRA 参数。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='记录服务器平均目标、有效客户端数量和教师聚合摘要。';Indent=18;Bold=$false},
    [pscustomobject]@{Text='返回更新后的服务器模型 <<f_s>>。';Indent=0;Bold=$false}
  )
  Add-Algorithm $doc '算法 3  软硬联合监督下的服务器更新' '服务器模型 <<f_s>>、分词器 <<τ_s>>、公共锚点 <<A>>、聚合教师矩阵 <<Q̅>>||权重 <<λ_A,λ_D>>、服务器更新步数 <<S>>' '更新后的服务器模型 <<f_s>>' $a3 $false

  $doc.SaveAs2($out,12)
  Write-Output ("tables={0} equations={1} pages={2}" -f $doc.Tables.Count,$doc.OMaths.Count,$doc.ComputeStatistics(2))
  $doc.Close(0)
}finally{
  $word.Quit()
}
