# 数据集需求清单

## 知识锚点算法必需数据集

### 当前必需（单数据集模式）

| 数据集 | 文件名 | 存储位置 | 下载地址 |
|--------|--------|---------|---------|
| **GSM8K** | `gsm8k_train.jsonl` | `data/gsm8k_train.jsonl` | https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl |

**注意**: 
- 下载后需要重命名为 `gsm8k_train.jsonl`
- 这是知识锚点算法当前使用的唯一必需数据集

---

## 可选数据集（用于扩展功能）

### LLM数据集（用于异构数据实验）

如果要在知识锚点中测试多个同任务数据集，可以下载：

| 数据集 | 文件名 | 存储位置 | 下载地址 | 用途 |
|--------|--------|---------|---------|------|
| Alpaca | `alpaca_data.json` | `data/alpaca_data.json` | https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json | 指令跟随任务 |
| Alpaca Cleaned | `alpaca_data_cleaned.json` | `data/alpaca_data_cleaned.json` | https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json | 指令跟随任务（清理版） |
| Dolly-15k | `databricks-dolly-15k.jsonl` | `data/databricks-dolly-15k.jsonl` | https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl | 指令跟随任务 |
| Rosetta Alpaca | `rosetta_alpaca.json` | `data/rosetta_alpaca.json` | https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json | 代码生成任务 |

### Hetero NLP Tasks数据集（用于异构任务实验）

这些数据集用于测试完全不同的任务类型，存储在 `datasets/` 目录：

| 数据集 | 存储位置 | 下载地址 | 任务类型 |
|--------|---------|---------|---------|
| IMDB | `datasets/imdb/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/imdb.zip | 情感分类 |
| AGNews | `datasets/agnews/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/agnews.zip | 新闻分类 |
| SQuAD | `datasets/squad/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/squad.zip | 问答 |
| NewsQA | `datasets/newsqa/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/newsqa.zip | 问答 |
| CNN/DM | `datasets/cnndm/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/cnndm.zip | 文本摘要 |
| MSQG | `datasets/msqg/` | https://federatedscope.oss-cn-beijing.aliyuncs.com/msqg.zip | 问题生成 |

---

## 快速下载

### 方法1: 使用Python脚本（推荐）

```bash
cd /home/tyx/ywp/FedBiOT
python3 download_datasets.py
```

然后选择：
- `4` - 仅下载必需数据集（gsm8k）
- `1` - 下载所有LLM数据集
- `3` - 下载全部数据集

### 方法2: 手动下载

#### 下载GSM8K（必需）

```bash
cd /home/tyx/ywp/FedBiOT/data
wget https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl
mv train.jsonl gsm8k_train.jsonl
```

#### 下载其他LLM数据集（可选）

```bash
cd /home/tyx/ywp/FedBiOT/data

# Alpaca
wget -O alpaca_data.json \
  https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json

# Dolly-15k
wget -O databricks-dolly-15k.jsonl \
  https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl

# Rosetta Alpaca
wget -O rosetta_alpaca.json \
  https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json
```

#### 下载Hetero NLP Tasks数据集（可选）

```bash
cd /home/tyx/ywp/FedBiOT
mkdir -p datasets

# 下载并解压每个数据集
for dataset in imdb agnews squad newsqa cnndm msqg; do
    wget -O datasets/${dataset}.zip \
      https://federatedscope.oss-cn-beijing.aliyuncs.com/${dataset}.zip
    unzip -d datasets datasets/${dataset}.zip
    rm datasets/${dataset}.zip
done
```

---

## 验证数据集

下载完成后，验证数据集是否存在：

```bash
# 检查必需数据集
ls -lh data/gsm8k_train.jsonl

# 检查所有LLM数据集
ls -lh data/*.json* data/*.jsonl

# 检查Hetero NLP Tasks数据集
ls -d datasets/*/
```

---

## 总结

### 知识锚点算法最小需求
- ✅ **仅需**: `data/gsm8k_train.jsonl`

### 扩展功能需求
- 异构数据实验: 多个LLM数据集（如gsm8k, alpaca, dolly-15k）
- 异构任务实验: Hetero NLP Tasks数据集

---

## 代码修改说明

✅ **已完成修改**:
1. `federatedscope/nlp/hetero_tasks/dataset/get_data.py` - 移除自动下载
2. `federatedscope/llm/dataloader/dataloader.py` - 移除自动下载，改为检查本地文件

如果数据集不存在，代码会：
- 抛出 `FileNotFoundError`
- 显示清晰的错误信息和下载地址
- 不会自动下载

