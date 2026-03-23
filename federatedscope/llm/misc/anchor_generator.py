import torch
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from torch.nn.functional import softmax


def generate_task_template(task_description: str, model, tokenizer, num_templates=5):
    """
    用 LLM 从任务描述生成任务模板（prompt templates），确保任务相关性
    """
    prompt = f"""
    根据以下任务描述，生成 {num_templates} 个不同的指令微调模板（Alpaca风格）。
    每个模板应包含 {{instruction}} 和 {{input}}/{{output}} 占位符，确保通用性。
    任务描述：{task_description}

    输出格式：仅列出模板，用编号1-{num_templates}。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    templates = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n')
    # 简单清洗，只取有效模板
    templates = [t.strip() for t in templates if '{{instruction}}' in t or '{{input}}' in t][:num_templates]
    return templates


def extract_intermediate_features(model, dataloader):
    """
    提取模型中间层 hidden states（取最后一层前平均）
    """
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model(**inputs, output_hidden_states=True)
            # 取倒数第二层 hidden states 并平均序列维度
            hidden = outputs.hidden_states[-2].mean(dim=1)  # [batch, hidden_dim]
            all_hidden.append(hidden.cpu())
    return torch.cat(all_hidden, dim=0)  # [num_samples, hidden_dim]


def build_knowledge_graph(features: torch.Tensor, texts: list, num_clusters=50):
    """
    简单知识图谱构建：
    - 用 KMeans 聚类得到“概念节点”
    - 节点间边基于句子共现或相似度
    """
    # 转 numpy 聚类
    features_np = features.numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features_np)

    G = nx.Graph()
    # 添加节点（簇中心代表概念）
    for i in range(num_clusters):
        G.add_node(f"Concept_{i}", embedding=kmeans.cluster_centers_[i])

    # 添加边：如果两个句子在同一簇或相似度高，加边
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if labels[i] == labels[j]:
                G.add_edge(f"Concept_{labels[i]}", f"Concept_{labels[j]}", weight=1.0)

    # 计算图 embedding：所有节点 embedding 平均
    embeddings = [data['embedding'] for _, data in G.nodes(data=True)]
    kg_embedding = np.mean(embeddings, axis=0) if embeddings else None

    return G, torch.tensor(kg_embedding)


def distill_soft_labels(model, tokenizer, public_dataloader, temperature=2.0):
    """
    生成软标签：运行模型得到 softened logits
    """
    model.eval()
    soft_labels_list = []
    with torch.no_grad():
        for batch in public_dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch.get('labels').to(model.device)

            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab]

            # 软化
            soft_logits = logits / temperature
            soft_probs = softmax(soft_logits, dim=-1)

            # 只取 labels 位置的软标签（或全序列）
            if labels is not None:
                soft_labels_list.append(soft_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1))
            else:
                soft_labels_list.append(soft_probs)

    return torch.cat(soft_labels_list, dim=0) if soft_labels_list else None


def generate_knowledge_anchor(public_dataloader, texts, task_description, model, tokenizer):
    """
    主函数：生成知识锚点
    """
    # 1. 生成任务模板
    task_templates = generate_task_template(task_description, model, tokenizer)

    # 2. 提取中层特征
    intermediate_features = extract_intermediate_features(model, public_dataloader)

    # 3. 构建知识图谱 + KG embedding
    kg_graph, kg_embedding = build_knowledge_graph(intermediate_features, texts)

    # 4. 蒸馏软标签
    soft_labels = distill_soft_labels(model, tokenizer, public_dataloader, temperature=4.0)

    # 5. 组合成锚点
    anchor = {
        'task_templates': task_templates,
        'kg_graph': kg_graph,  # 可序列化保存
        'kg_embedding': kg_embedding,  # tensor，用于客户端对比或损失
        'soft_labels': soft_labels,  # 用于 distillation loss
        'intermediate_features_mean': intermediate_features.mean(dim=0)  # 通用知识中心
    }

    return anchor

# def generate_knowledge_anchor(public_dataset, task_description, model):
#     # 示例：用公共数据集 + 任务描述 prompt LLM 生成模板
#     # 提取中层特征 -> 构建知识图谱（用 networkx 或简单 embedding 平均）
#     # 蒸馏软标签
#     with torch.no_grad():
#         features = model.get_intermediate_layers(public_dataset)  # 伪代码
#         kg = build_knowledge_graph(features, task_description)
#         soft_labels = distill_soft_labels(kg, model)
#     return {'soft_labels': soft_labels, 'kg_embedding': kg}
#