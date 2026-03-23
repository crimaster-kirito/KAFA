# 数据集下载指南

## 概述

代码已修改为**仅使用本地数据集**，不再自动下载。请按照本指南提前下载所需数据集。

---

## 数据集分类

### 1. LLM数据集（用于知识锚点算法）

这些数据集存储在 `data/` 目录下。

#### 必需数据集（知识锚点当前使用）

| 数据集名称 | 配置中的名称 | 文件名 | 存储位置 | 下载地址 |
|-----------|------------|--------|---------|---------|
| GSM8K | `gsm8k@llm` | `gsm8k_train.jsonl` | `data/gsm8k_train.jsonl` | https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl |

**注意**: 下载后请重命名为 `gsm8k_train.jsonl`

#### 可选数据集（用于异构数据实验）

| 数据集名称 | 配置中的名称 | 文件名 | 存储位置 | 下载地址 |
|-----------|------------|--------|---------|---------|
| Alpaca | `alpaca@llm` | `alpaca_data.json` | `data/alpaca_data.json` | https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json |
| Alpaca Cleaned | `alpaca_cleaned@llm` | `alpaca_data_cleaned.json` | `data/alpaca_data_cleaned.json` | https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json |
| Dolly-15k | `dolly-15k@llm` | `databricks-dolly-15k.jsonl` | `data/databricks-dolly-15k.jsonl` | https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl |
| Rosetta Alpaca | `rosetta_alpaca@llm` | `rosetta_alpaca.json` | `data/rosetta_alpaca.json` | https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json |
| Code Search Net | `code_search_net@llm` | 多个文件 | `data/{language}/final/jsonl/train/` | 需要运行 `python federatedscope/llm/dataset/code_search_net.py` |

---

### 2. Hetero NLP Tasks数据集（用于异构任务实验）

这些数据集存储在 `datasets/` 目录（或配置的 `data.root`）下，每个数据集一个子目录。

| 数据集名称 | 存储位置 | 下载地址 | 目录结构 |
|-----------|---------|---------|---------|
| IMDB | `datasets/imdb/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/imdb.zip | `imdb/train/pos/`, `imdb/train/neg/`, `imdb/test/` |
| AGNews | `datasets/agnews/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/agnews.zip | `agnews/train.csv`, `agnews/test.csv` |
| SQuAD | `datasets/squad/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/squad.zip | `squad/train.json`, `squad/test.json` |
| NewsQA | `datasets/newsqa/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/newsqa.zip | `newsqa/train.jsonl.gz`, `newsqa/test.jsonl.gz` |
| CNN/DM | `datasets/cnndm/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/cnndm.zip | `cnndm/train.src`, `cnndm/train.tgt`, `cnndm/test.src`, `cnndm/test.tgt` |
| MSQG | `datasets/msqg/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/msqg.zip | `msqg/train.src`, `msqg/train.tgt`, `msqg/test.src`, `msqg/test.tgt` |

---

## 快速下载脚本

### 下载LLM数据集

创建 `download_llm_datasets.sh`:

```bash
#!/bin/bash
# 下载LLM数据集到 data/ 目录

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "下载GSM8K数据集..."
wget -O "$DATA_DIR/train.jsonl" \
  "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
mv "$DATA_DIR/train.jsonl" "$DATA_DIR/gsm8k_train.jsonl"

echo "下载Alpaca数据集..."
wget -O "$DATA_DIR/alpaca_data.json" \
  "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"

echo "下载Alpaca Cleaned数据集..."
wget -O "$DATA_DIR/alpaca_data_cleaned.json" \
  "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json"

echo "下载Dolly-15k数据集..."
wget -O "$DATA_DIR/databricks-dolly-15k.jsonl" \
  "https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl"

echo "下载Rosetta Alpaca数据集..."
wget -O "$DATA_DIR/rosetta_alpaca.json" \
  "https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json"

echo "✓ 所有LLM数据集下载完成！"
```

### 下载Hetero NLP Tasks数据集

创建 `download_hetero_datasets.sh`:

```bash
#!/bin/bash
# 下载Hetero NLP Tasks数据集到 datasets/ 目录

DATASETS_DIR="datasets"
BASE_URL="https://federatedscope.oss-cn-beijing.aliyuncs.com"

mkdir -p "$DATASETS_DIR"

datasets=("imdb" "agnews" "squad" "newsqa" "cnndm" "msqg")

for dataset in "${datasets[@]}"; do
    echo "下载 $dataset 数据集..."
    wget -O "$DATASETS_DIR/${dataset}.zip" "${BASE_URL}/${dataset}.zip"
    
    echo "解压 $dataset..."
    unzip -q "$DATASETS_DIR/${dataset}.zip" -d "$DATASETS_DIR"
    
    # 清理zip文件
    rm "$DATASETS_DIR/${dataset}.zip"
    
    echo "✓ $dataset 下载完成"
done

echo "✓ 所有Hetero NLP Tasks数据集下载完成！"
```

---

## Python下载脚本

创建 `download_datasets.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集下载脚本
"""
import os
import urllib.request
import zipfile
import shutil

def download_file(url, dest_path, filename=None):
    """下载文件"""
    if filename is None:
        filename = os.path.basename(url)
    
    file_path = os.path.join(dest_path, filename)
    
    if os.path.exists(file_path):
        print(f"文件已存在: {file_path}")
        return file_path
    
    print(f"下载: {url}")
    print(f"保存到: {file_path}")
    
    os.makedirs(dest_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)
    print(f"✓ 下载完成: {file_path}")
    return file_path

def download_llm_datasets():
    """下载LLM数据集"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {
        'gsm8k_train.jsonl': 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl',
        'alpaca_data.json': 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json',
        'alpaca_data_cleaned.json': 'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json',
        'databricks-dolly-15k.jsonl': 'https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl',
        'rosetta_alpaca.json': 'https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json',
    }
    
    for filename, url in datasets.items():
        if filename == 'gsm8k_train.jsonl':
            # GSM8K需要重命名
            temp_file = download_file(url, data_dir, 'train.jsonl')
            if os.path.exists(temp_file):
                os.rename(temp_file, os.path.join(data_dir, filename))
        else:
            download_file(url, data_dir, filename)

def download_hetero_datasets():
    """下载Hetero NLP Tasks数据集"""
    datasets_dir = 'datasets'
    base_url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
    
    datasets = ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']
    
    for dataset in datasets:
        zip_url = f'{base_url}/{dataset}.zip'
        zip_path = os.path.join(datasets_dir, f'{dataset}.zip')
        
        print(f"\n下载 {dataset} 数据集...")
        download_file(zip_url, datasets_dir, f'{dataset}.zip')
        
        # 解压
        print(f"解压 {dataset}...")
        extract_dir = os.path.join(datasets_dir, dataset + '_raw')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 移动文件
        if os.path.exists(os.path.join(extract_dir, dataset)):
            shutil.move(os.path.join(extract_dir, dataset), datasets_dir)
        
        # 清理
        os.remove(zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        print(f"✓ {dataset} 完成")

if __name__ == '__main__':
    print("=" * 60)
    print("下载LLM数据集")
    print("=" * 60)
    download_llm_datasets()
    
    print("\n" + "=" * 60)
    print("下载Hetero NLP Tasks数据集")
    print("=" * 60)
    download_hetero_datasets()
    
    print("\n✓ 所有数据集下载完成！")
```

---

## 目录结构

下载完成后，目录结构应该是：

```
FedBiOT/
├── data/                          # LLM数据集
│   ├── gsm8k_train.jsonl          # ✓ 必需（知识锚点当前使用）
│   ├── alpaca_data.json           # 可选
│   ├── alpaca_data_cleaned.json   # 可选
│   ├── databricks-dolly-15k.jsonl # 可选
│   └── rosetta_alpaca.json        # 可选
│
└── datasets/                      # Hetero NLP Tasks数据集（可选）
    ├── imdb/
    │   ├── train/
    │   │   ├── pos/
    │   │   └── neg/
    │   └── test/
    ├── agnews/
    │   ├── train.csv
    │   └── test.csv
    ├── squad/
    │   ├── train.json
    │   └── test.json
    ├── newsqa/
    │   ├── train.jsonl.gz
    │   └── test.jsonl.gz
    ├── cnndm/
    │   ├── train.src
    │   ├── train.tgt
    │   ├── test.src
    │   └── test.tgt
    └── msqg/
        ├── train.src
        ├── train.tgt
        ├── test.src
        └── test.tgt
```

---

## 必需数据集（知识锚点算法）

**当前知识锚点算法只需要**：
- ✅ `data/gsm8k_train.jsonl`

其他数据集是可选的，用于：
- 异构数据实验
- 多数据集对比实验
- 扩展功能测试

---

## 验证数据集

运行以下命令验证数据集是否存在：

```bash
# 检查LLM数据集
ls -lh data/gsm8k_train.jsonl

# 检查Hetero NLP Tasks数据集
ls -d datasets/*/
```

---

## 注意事项

1. **文件大小**: 某些数据集可能较大，请确保有足够的磁盘空间
2. **网络**: 如果下载失败，可以手动下载并放置到正确位置
3. **权限**: 确保有写入权限
4. **路径**: 确保路径与配置文件中的 `data.root` 一致

---

## 修改说明

代码已修改为：
- ❌ **不再自动下载**数据集
- ✅ **仅使用本地数据集**
- ✅ **如果数据集不存在，会给出清晰的错误提示和下载地址**

如果数据集不存在，运行时会看到类似错误：
```
FileNotFoundError: 数据集文件不存在: data/gsm8k_train.jsonl
请手动下载 gsm8k_train.jsonl 并放置到: data/
下载地址: https://...
```

