"""
Knowledge Anchor Data Translator
自定义translator，正确处理服务器数据和客户端数据分配
"""
import logging
import numpy as np
from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.core.auxiliaries.splitter_builder import get_splitter

logger = logging.getLogger(__name__)


class KnowledgeAnchorDataTranslator(BaseDataTranslator):
    """
    知识锚点数据转换器：
    - 重写split方法，正确处理40/40/20的数据分配
    - 服务器数据（40%）从train中提取
    - 客户端数据（40%）分配给3个客户端
    - 测试数据（20%）保持不变
    """
    def __init__(self, global_cfg, client_cfgs=None):
        super(KnowledgeAnchorDataTranslator, self).__init__(global_cfg, client_cfgs)
        self.splitter = get_splitter(global_cfg)
        
        # 从splitter获取服务器和客户端比例
        if hasattr(self.splitter, 'server_ratio'):
            self.server_ratio = self.splitter.server_ratio
            self.client_ratio = self.splitter.client_ratio
            self.test_ratio = self.splitter.test_ratio
        else:
            # 默认值
            self.server_ratio = 0.4
            self.client_ratio = 0.4
            self.test_ratio = 0.2

    def split(self, dataset):
        """
        执行ML split和FL split，正确处理服务器数据
        
        Returns:
            dict of ``ClientData`` with client_idx as key
        """
        from torch.utils.data import Dataset, Subset
        
        # 1. 先做ML split（train/val/test）
        # 注意：这里train包含服务器和客户端数据，需要进一步分割
        train, val, test = self.split_train_val_test(dataset)
        
        # 2. 从train中分离服务器数据和客户端数据
        # 转换为列表格式便于处理
        if isinstance(train, Dataset):
            train_list = [train[i] for i in range(len(train))]
        else:
            train_list = train
        
        total_train_size = len(train_list)
        indices = np.random.permutation(np.arange(total_train_size))
        
        # 计算服务器和客户端数据大小
        # train中：server_ratio / (server_ratio + client_ratio) 是服务器数据
        # client_ratio / (server_ratio + client_ratio) 是客户端数据
        server_size = int(self.server_ratio / (self.server_ratio + self.client_ratio) * total_train_size)
        client_train_size = total_train_size - server_size
        
        server_train_indices = indices[:server_size]
        client_train_indices = indices[server_size:]
        
        # 构建服务器训练数据
        if isinstance(train, Dataset):
            server_train = Subset(train, server_train_indices)
        else:
            server_train = [train_list[i] for i in server_train_indices]
        
        # 构建客户端训练数据
        if isinstance(train, Dataset):
            client_train = Subset(train, client_train_indices)
        else:
            client_train = [train_list[i] for i in client_train_indices]
        
        logger.info(f"Train数据分离: 服务器={len(server_train_indices)}, "
                   f"客户端={len(client_train_indices)}")
        
        # 3. 使用splitter分配客户端数据（train/val/test）
        client_num = self.global_cfg.federate.client_num
        split_train, split_val, split_test = [[None] * client_num] * 3
        train_label_distribution = None
        
        # Split train/val/test to client
        if len(client_train) > 0:
            split_train = self.splitter(client_train)  # 只传入客户端训练数据
            if self.global_cfg.data.consistent_label_distribution:
                try:
                    train_label_distribution = [[j[1] for j in x]
                                                for x in split_train]
                except:
                    logger.warning(
                        'Cannot access train label distribution for '
                        'splitter, split dataset without considering train '
                        'label.')
        if len(val) > 0:
            split_val = self.splitter(val, prior=train_label_distribution)
        if len(test) > 0:
            split_test = self.splitter(test, prior=train_label_distribution)
        
        # 4. 构建数据字典
        # key 0: 服务器数据（只有train，没有val/test）
        data_dict = {
            0: ClientData(self.global_cfg, train=server_train, val=[], test=[])
        }
        
        # key 1-N: 客户端数据
        logger.info(f'客户端数量: {client_num}')
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            
            # 统计客户端数据量
            train_size = len(split_train[client_id - 1]) if split_train[client_id - 1] is not None else 0
            val_size = len(split_val[client_id - 1]) if split_val[client_id - 1] is not None else 0
            test_size = len(split_test[client_id - 1]) if split_test[client_id - 1] is not None else 0
            
            logger.info(f'  客户端{client_id}: train={train_size}, val={val_size}, test={test_size}')
            
            data_dict[client_id] = ClientData(client_cfg,
                                              train=split_train[client_id - 1],
                                              val=split_val[client_id - 1],
                                              test=split_test[client_id - 1])
        
        return data_dict

