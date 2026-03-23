"""
Knowledge Anchor Data Splitter
支持40%服务器，40%客户端（3个，支持non-IID/异构/任务不相关），20%测试的数据分配
扩展支持多个同任务数据集（异构数据源，但任务类型一致）
"""
import numpy as np
import logging
import os
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import dirichlet_distribution_noniid_slice

logger = logging.getLogger(__name__)


class KnowledgeAnchorSplitter(BaseSplitter):
    """
    知识锚点数据分配器：
    - 40% 服务器训练数据
    - 40% 客户端训练数据（3个客户端，支持non-IID）
    - 20% 测试数据
    - 支持异构分布、任务不相关数据
    - 扩展：支持多个同任务数据集（hetero_datasets）
    """
    def __init__(self, client_num, server_ratio=0.4, client_ratio=0.4, 
                 test_ratio=0.2, non_iid_alpha=0.5, task_irrelevant_ratio=0.1,
                 hetero_datasets=None, hetero_distribution='uniform',
                 irrelevant_dataset=None, **kwargs):
        """
        Args:
            client_num: 客户端数量（固定为3）
            server_ratio: 服务器数据比例（默认0.4）
            client_ratio: 客户端总数据比例（默认0.4）
            test_ratio: 测试数据比例（默认0.2）
            non_iid_alpha: Dirichlet分布参数，控制non-IID程度（越小越non-IID）
            task_irrelevant_ratio: 任务不相关数据比例（用于模拟异构场景）
            hetero_datasets: 多个同任务数据集列表（如 ['gsm8k@llm', 'alpaca@llm']），None表示单数据集模式
            hetero_distribution: 异构数据集分配方式 ('uniform', 'proportional', 'dirichlet')
            irrelevant_dataset: 用于注入无关数据的数据集名称（如 'alpaca@llm'），None表示从当前数据集抽取
        """
        super(KnowledgeAnchorSplitter, self).__init__(client_num)
        assert abs(server_ratio + client_ratio + test_ratio - 1.0) < 1e-6, \
            f"数据比例总和必须为1.0，当前为{server_ratio + client_ratio + test_ratio}"
        assert client_num == 3, "知识锚点算法固定使用3个客户端"
        
        self.server_ratio = server_ratio
        self.client_ratio = client_ratio
        self.test_ratio = test_ratio
        self.non_iid_alpha = non_iid_alpha
        self.task_irrelevant_ratio = task_irrelevant_ratio
        self.irrelevant_dataset = irrelevant_dataset
        
        # 异构数据集支持
        self.hetero_datasets = hetero_datasets if hetero_datasets else None
        self.hetero_distribution = hetero_distribution
        self.hetero_mode = self.hetero_datasets is not None and len(self.hetero_datasets) > 1
        
        if self.hetero_mode:
            logger.info(f"启用异构数据集模式: {self.hetero_datasets}, 分配方式: {self.hetero_distribution}")
        
        if self.irrelevant_dataset:
            logger.info(f"将使用数据集 {self.irrelevant_dataset} 作为任务不相关数据源")

    def __call__(self, dataset, prior=None, **kwargs):
        """
        分配数据到客户端（FL split）
        
        注意：
        - BaseDataTranslator已经做了ML split（train/val/test），所以这里只需要做FL split
        - 只在train split时注入无关数据（通过prior参数判断：train时prior为None）
        - 返回格式：只返回客户端数据列表 [client1_data, client2_data, client3_data]
        - 服务器数据由BaseDataTranslator处理（key=0使用原始train数据）
        """
        from torch.utils.data import Dataset, Subset
        
        # 判断是否是train split（train split时prior为None，val/test split时prior不为None）
        is_train_split = (prior is None)
        
        # 转换为列表格式便于处理
        if isinstance(dataset, Dataset):
            dataset_list = [dataset[i] for i in range(len(dataset))]
        else:
            dataset_list = dataset
        
        total_size = len(dataset_list)
        indices = np.random.permutation(np.arange(total_size))
        
        # 对于所有split（train/val/test），都只分配给客户端
        # 注意：服务器数据由BaseDataTranslator处理，这里不处理
        client_data_indices = indices
        
        # 3. 客户端数据分配（支持non-IID）
        # 尝试提取标签信息（用于non-IID分配）
        try:
            # 尝试从数据中提取标签
            if isinstance(dataset_list[0], dict):
                # LLM数据集通常是dict格式
                labels = []
                for item in dataset_list:
                    # 尝试多种可能的标签字段
                    if 'label' in item:
                        labels.append(item['label'])
                    elif 'labels' in item:
                        labels.append(item['labels'])
                    elif 'category' in item:
                        labels.append(item['category'])
                    else:
                        # 如果没有标签，使用索引作为伪标签（用于随机分配）
                        labels.append(hash(str(item)) % 10)
            elif isinstance(dataset_list[0], tuple):
                labels = [item[1] if len(item) > 1 else 0 for item in dataset_list]
            else:
                labels = None
                
            if labels is not None and len(set(labels)) > 1:
                # 有标签信息，使用Dirichlet分布进行non-IID分配
                client_data_labels = [labels[i] for i in client_data_indices]
                client_data_labels = np.array(client_data_labels)
                
                # 使用Dirichlet分布分配
                client_idx_slices = dirichlet_distribution_noniid_slice(
                    client_data_labels, 
                    self.client_num, 
                    self.non_iid_alpha
                )
                
                # 将索引映射回原始索引
                client_indices = [
                    [client_data_indices[idx] for idx in slice_idx] 
                    for slice_idx in client_idx_slices
                ]
            else:
                # 无标签信息，使用IID分配
                logger.warning("无法提取标签信息，使用IID分配")
                client_indices = np.array_split(client_data_indices, self.client_num)
                client_indices = [idx.tolist() for idx in client_indices]
        except Exception as e:
            logger.warning(f"标签提取失败，使用IID分配: {e}")
            client_indices = np.array_split(client_data_indices, self.client_num)
            client_indices = [idx.tolist() for idx in client_indices]
        
        # 4. 可选：为某些客户端注入任务不相关数据（模拟异构场景）
        # 注意：只在train split时注入无关数据，val/test split不注入
        irrelevant_data_count = 0
        gsm8k_data_count = 0
        if is_train_split and self.task_irrelevant_ratio > 0:
            # 计算需要注入的无关数据量（基于客户端数据总量）
            total_client_data_size = sum(len(idx) for idx in client_indices)
            irrelevant_size = int(self.task_irrelevant_ratio * total_client_data_size)
            
            # 统计gsm8k数据量（主数据集）
            gsm8k_data_count = total_client_data_size
            
            if irrelevant_size > 0:
                import random
                irrelevant_data = []
                
                # 如果指定了外部无关数据集（如IMDB），从该数据集加载
                if self.irrelevant_dataset:
                    logger.info(f"从外部数据集 {self.irrelevant_dataset} 加载无关样本...")
                    irrelevant_data = self._load_irrelevant_dataset(
                        self.irrelevant_dataset, 
                        irrelevant_size,
                        kwargs.get('data_root', 'data/')
                    )
                    
                    if irrelevant_data:
                        irrelevant_data_count = len(irrelevant_data)
                        logger.info(f"✓ 成功加载 {irrelevant_data_count} 个无关样本（目标: {irrelevant_size}）")
                    else:
                        logger.warning(f"✗ 无法从 {self.irrelevant_dataset} 加载无关样本，将从当前数据集抽取")
                        # 回退到从当前数据集抽取
                        irrelevant_data = None
                
                # 如果没有外部数据集或加载失败，从客户端训练数据中随机抽取
                if not irrelevant_data:
                    all_client_indices_flat = []
                    for idx_list in client_indices:
                        all_client_indices_flat.extend(idx_list)
                    
                    # 随机选择一部分作为无关数据
                    irrelevant_indices = random.sample(
                        all_client_indices_flat, 
                        min(irrelevant_size, len(all_client_indices_flat))
                    )
                    irrelevant_data_count = len(irrelevant_indices)
                    
                    # 从原始客户端数据中移除这些索引
                    for i in range(len(client_indices)):
                        client_indices[i] = [idx for idx in client_indices[i] if idx not in irrelevant_indices]
                    
                    # 随机分配给某个客户端（模拟数据异构）
                    target_client = random.randint(0, self.client_num - 1)
                    client_indices[target_client].extend(irrelevant_indices)
                    
                    logger.info(f"✓ 为客户端{target_client+1}注入{irrelevant_data_count}个任务不相关样本（从当前数据集抽取）")
                else:
                    # 使用外部数据集的无关样本
                    # 随机分配给某个客户端
                    target_client = random.randint(0, self.client_num - 1)
                    
                    # 将无关数据添加到目标客户端
                    # 注意：这些数据需要转换为与主数据集相同的格式
                    # 重要：如果dataset是Dataset对象，不能直接append，需要创建新的Subset
                    if isinstance(dataset, Dataset):
                        # 对于Dataset对象，需要扩展索引范围
                        # 但Subset不支持动态扩展，所以我们需要在返回时处理
                        irrelevant_start_idx = len(dataset_list)
                        # 将无关数据添加到dataset_list（临时存储）
                        for i, irrelevant_item in enumerate(irrelevant_data):
                            dataset_list.append(irrelevant_item)
                            # 使用负数索引标记这些是无关数据（在构建Subset时处理）
                            client_indices[target_client].append(irrelevant_start_idx + i)
                    else:
                        # 对于列表格式，直接添加
                        irrelevant_start_idx = len(dataset_list)
                        for i, irrelevant_item in enumerate(irrelevant_data):
                            dataset_list.append(irrelevant_item)
                            client_indices[target_client].append(irrelevant_start_idx + i)
                    
                    logger.info(f"✓ 为客户端{target_client+1}注入{irrelevant_data_count}个任务不相关样本（来自 {self.irrelevant_dataset}）")
                
                logger.info(f"  这些样本将在训练时通过原型库相似度判断被识别为无关并排除")
        
        # 记录数据分配统计（只在train split时）
        if is_train_split:
            total_samples = gsm8k_data_count + irrelevant_data_count
            logger.info("=" * 80)
            logger.info("数据分配统计（Train Split）")
            logger.info("=" * 80)
            logger.info(f"  📊 GSM8K任务相关数据: {gsm8k_data_count} 个样本")
            if irrelevant_data_count > 0:
                logger.info(f"  📊 IMDB任务无关数据: {irrelevant_data_count} 个样本")
            logger.info(f"  📊 总数据量: {total_samples} 个样本")
            logger.info(f"  📊 客户端数据分配: {[len(idx) for idx in client_indices]} (客户端1, 客户端2, 客户端3)")
            logger.info("=" * 80)
        
        # 构建返回结果
        # 返回格式：只返回客户端数据列表 [client1_data, client2_data, client3_data]
        # 注意：BaseSplitter期望返回客户端数据列表
        # 服务器数据由BaseDataTranslator处理（key=0使用原始train数据）
        result = []
        
        # 只返回客户端数据
        # 如果dataset是Dataset对象，需要处理无关数据的索引
        original_dataset_size = len(dataset) if isinstance(dataset, Dataset) else len(dataset_list)
        
        for client_idx in client_indices:
            if isinstance(dataset, Dataset):
                # 过滤掉超出原始数据集大小的索引（这些是无关数据）
                valid_indices = [idx for idx in client_idx if 0 <= idx < original_dataset_size]
                invalid_indices = [idx for idx in client_idx if idx >= original_dataset_size]
                
                # 对于有效的索引，使用Subset
                if valid_indices:
                    client_subset = Subset(dataset, valid_indices)
                    # 对于无效索引（无关数据），需要从dataset_list中获取
                    if invalid_indices:
                        # 将无关数据转换为列表并合并
                        subset_list = [client_subset[i] for i in range(len(client_subset))]
                        for idx in invalid_indices:
                            if idx < len(dataset_list):
                                subset_list.append(dataset_list[idx])
                        result.append(subset_list)
                    else:
                        result.append(client_subset)
                else:
                    # 所有索引都是无关数据
                    result.append([dataset_list[idx] for idx in invalid_indices if idx < len(dataset_list)])
            else:
                # 列表格式，直接使用索引
                result.append([dataset_list[i] for i in client_idx if i < len(dataset_list)])
        
        # 记录分配信息（简化输出）
        if is_train_split:
            logger.debug(f"Train数据分配完成: 客户端={[len(idx) for idx in client_indices]}")
        else:
            logger.debug(f"{'Val' if prior is not None and 'val' in str(prior).lower() else 'Test'}数据分配完成: "
                        f"客户端={[len(idx) for idx in client_indices]}")
        
        return result
    
    def _load_irrelevant_dataset(self, dataset_name, num_samples, data_root='data/'):
        """
        从外部数据集加载无关样本（如IMDB）
        
        Args:
            dataset_name: 数据集名称（如 'imdb' 或 'imdb@hetero'）
            num_samples: 需要加载的样本数
            data_root: 数据根目录
            
        Returns:
            List: 无关样本列表，格式与主数据集一致
        """
        try:
            irrelevant_data = []
            
            # 处理数据集名称（可能包含@符号）
            if '@' in dataset_name:
                dataset_type = dataset_name.split('@')[0]
            else:
                dataset_type = dataset_name
            
            # 加载IMDB数据集
            if dataset_type.lower() == 'imdb':
                imdb_path = os.path.join(data_root, 'imdb', 'train.jsonl')
                if os.path.exists(imdb_path):
                    import json
                    with open(imdb_path, 'r', encoding='utf-8') as f:
                        all_imdb = [json.loads(line) for line in f]
                    
                    # 随机采样
                    import random
                    sampled_imdb = random.sample(all_imdb, min(num_samples, len(all_imdb)))
                    
                    # 转换为与主数据集（gsm8k）一致的格式
                    # LLM数据加载器期望格式: {'instruction': ..., 'input': ..., 'output': ...}
                    # IMDB格式是: {'text': ..., 'label': ...}
                    # 我们需要转换为LLM格式以便tokenization
                    for item in sampled_imdb:
                        text = item.get('text', '')
                        label = item.get('label', 0)
                        
                        # 转换为指令格式（情感分类任务）
                        # 注意：这会被识别为无关任务（情感分类 vs 数学问题）
                        converted_item = {
                            'instruction': f"Classify the sentiment of the following text as positive or negative:",
                            'input': text,
                            'output': 'positive' if label == 1 else 'negative',
                            # 保留原始字段以便识别
                            'text': text,
                            'label': label,
                            '_source': 'imdb',  # 标记来源，便于识别
                            '_is_irrelevant': True  # 标记为无关样本
                        }
                        irrelevant_data.append(converted_item)
                    
                    logger.info(f"✓ 从IMDB加载了 {len(irrelevant_data)} 个样本（情感分类任务）")
                else:
                    logger.warning(f"✗ IMDB数据集文件不存在: {imdb_path}")
                    logger.warning(f"  请确保IMDB数据集已下载到: {data_root}/imdb/train.jsonl")
                    return []
            else:
                logger.warning(f"✗ 不支持的数据集类型: {dataset_type}")
                return []
            
            return irrelevant_data
            
        except Exception as e:
            logger.error(f"✗ 加载无关数据集失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []


def call_knowledge_anchor_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'knowledge_anchor':
        return KnowledgeAnchorSplitter(client_num, **kwargs)
    return None

