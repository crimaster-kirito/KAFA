"""Data translator for knowledge-anchor experiments."""

import logging

import numpy as np
from torch.utils.data import Dataset, Subset

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import ClientData
from federatedscope.core.data.base_translator import BaseDataTranslator

logger = logging.getLogger(__name__)


class KnowledgeAnchorDataTranslator(BaseDataTranslator):
    """Keep server-anchor data separate from client training data."""

    def __init__(self, global_cfg, client_cfgs=None):
        super(KnowledgeAnchorDataTranslator, self).__init__(global_cfg,
                                                            client_cfgs)
        self.splitter = get_splitter(global_cfg)

        if hasattr(self.splitter, 'server_ratio'):
            self.server_ratio = self.splitter.server_ratio
            self.client_ratio = self.splitter.client_ratio
            self.test_ratio = self.splitter.test_ratio
        else:
            self.server_ratio = 0.4
            self.client_ratio = 0.4
            self.test_ratio = 0.2

    def split(self, dataset):
        train, val, test = self.split_train_val_test(dataset)

        if isinstance(train, Dataset):
            train_list = [train[i] for i in range(len(train))]
        else:
            train_list = train

        server_indices, client_indices = self._server_client_indices(
            train_list)
        server_train = self._take_split(train, train_list, server_indices)
        client_train = self._take_split(train, train_list, client_indices)

        logger.info("Knowledge-anchor split: server=%s, client=%s",
                    len(server_indices), len(client_indices))

        client_num = self.global_cfg.federate.client_num
        split_train = [None] * client_num
        split_val = [None] * client_num
        split_test = [None] * client_num
        train_label_distribution = None

        if len(client_train) > 0:
            split_train = self.splitter(client_train)
            if self.global_cfg.data.consistent_label_distribution:
                try:
                    train_label_distribution = [[j[1] for j in x]
                                                for x in split_train]
                except Exception:
                    logger.warning(
                        "Cannot access train label distribution; splitting "
                        "without train-label prior.")
        if len(val) > 0:
            split_val = self.splitter(val, prior=train_label_distribution)
        if len(test) > 0:
            split_test = self.splitter(test, prior=train_label_distribution)

        data_dict = {
            0: ClientData(self.global_cfg, train=server_train, val=[], test=test)
        }

        logger.info("Number of clients: %s", client_num)
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg

            train_size = len(split_train[client_id - 1]) \
                if split_train[client_id - 1] is not None else 0
            val_size = len(split_val[client_id - 1]) \
                if split_val[client_id - 1] is not None else 0
            test_size = len(split_test[client_id - 1]) \
                if split_test[client_id - 1] is not None else 0
            logger.info("Client %s: train=%s, val=%s, test=%s", client_id,
                        train_size, val_size, test_size)

            data_dict[client_id] = ClientData(
                client_cfg,
                train=split_train[client_id - 1],
                val=split_val[client_id - 1],
                test=split_test[client_id - 1])

        return data_dict

    def _server_client_indices(self, train_list):
        roles = [
            item.get('split_role') if isinstance(item, dict) else None
            for item in train_list
        ]
        if any(role == 'server' for role in roles):
            server_indices = [
                idx for idx, role in enumerate(roles) if role == 'server'
            ]
            client_indices = [
                idx for idx, role in enumerate(roles) if role == 'client'
            ]
            if not client_indices:
                client_indices = [
                    idx for idx, role in enumerate(roles) if role != 'server'
                ]
            return server_indices, client_indices

        total_train_size = len(train_list)
        indices = np.random.permutation(np.arange(total_train_size))
        server_size = int(self.server_ratio /
                          (self.server_ratio + self.client_ratio) *
                          total_train_size)
        server_indices = indices[:server_size].tolist()
        client_indices = indices[server_size:].tolist()
        return server_indices, client_indices

    @staticmethod
    def _take_split(dataset, dataset_list, indices):
        if isinstance(dataset, Dataset):
            return Subset(dataset, indices)
        return [dataset_list[i] for i in indices]
