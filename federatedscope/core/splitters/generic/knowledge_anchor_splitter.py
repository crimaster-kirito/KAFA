"""
Knowledge Anchor splitter with in-task dirty supervision injection.

The server split stays clean. Client train splits may receive GSM8K-style
corrupted supervision, but the input distribution is unchanged. Dirty metadata
is kept only for measurement; trainers must not use it to decide filtering.
"""

import copy
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice

logger = logging.getLogger(__name__)

FINAL_ANSWER_RE = re.compile(r"(?:####|The answer is)\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^<>]+)=(-?\d+(?:\.\d+)?)>>")


class KnowledgeAnchorSplitter(BaseSplitter):
    """Split client data and inject same-task dirty supervision if requested."""

    def __init__(self,
                 client_num,
                 server_ratio=0.4,
                 client_ratio=0.4,
                 test_ratio=0.2,
                 non_iid_alpha=0.5,
                 dirty_ratio=None,
                 dirty_type='answer_corruption',
                 dirty_client_ratios=None,
                 dirty_seed=42,
                 task_irrelevant_ratio=0.0,
                 irrelevant_dataset=None,
                 hetero_datasets=None,
                 hetero_distribution='uniform',
                 **kwargs):
        super(KnowledgeAnchorSplitter, self).__init__(client_num)
        assert abs(server_ratio + client_ratio + test_ratio - 1.0) < 1e-6, \
            f"Data ratios must sum to 1.0, got {server_ratio + client_ratio + test_ratio}"

        self.server_ratio = server_ratio
        self.client_ratio = client_ratio
        self.test_ratio = test_ratio
        self.non_iid_alpha = non_iid_alpha
        self.dirty_ratio = (
            float(dirty_ratio)
            if dirty_ratio is not None else float(task_irrelevant_ratio or 0.0)
        )
        self.dirty_type = dirty_type
        self.dirty_client_ratios = dirty_client_ratios
        self.dirty_seed = int(dirty_seed)
        self.hetero_datasets = hetero_datasets
        self.hetero_distribution = hetero_distribution

        if irrelevant_dataset:
            logger.warning(
                "`irrelevant_dataset` is ignored for KnowledgeAnchorSplitter. "
                "Client corruption now uses in-task dirty supervision, not IMDB/OOD data.")
        if task_irrelevant_ratio and dirty_ratio is None:
            logger.warning(
                "`task_irrelevant_ratio` is deprecated here; treating it as "
                "`dirty_ratio` for same-task dirty supervision.")

    def __call__(self, dataset, prior=None, **kwargs):
        is_train_split = prior is None
        dataset_list = [dataset[i] for i in range(len(dataset))] \
            if isinstance(dataset, Dataset) else list(dataset)
        total_size = len(dataset_list)
        indices = np.random.permutation(np.arange(total_size))

        client_indices = self._split_indices(dataset_list, indices)
        if is_train_split and self.dirty_ratio > 0:
            result = self._build_dirty_client_datasets(dataset, dataset_list,
                                                       client_indices)
        else:
            result = self._build_client_datasets(dataset, dataset_list,
                                                 client_indices)

        if is_train_split:
            dirty_counts = [
                sum(bool(self._as_dict(item).get('_is_dirty', False))
                    for item in self._iter_dataset(client_data))
                for client_data in result
            ]
            logger.info(
                "KnowledgeAnchor train split: client sizes=%s, dirty=%s, "
                "dirty_type=%s",
                [len(client_data) for client_data in result], dirty_counts,
                self.dirty_type)
        else:
            logger.debug("KnowledgeAnchor eval split: client sizes=%s",
                         [len(client_data) for client_data in result])

        return result

    def _split_indices(self, dataset_list, indices):
        labels = self._extract_labels(dataset_list)
        try:
            if labels is not None and len(set(labels)) > 1:
                client_labels = np.array([labels[i] for i in indices])
                idx_slices = dirichlet_distribution_noniid_slice(
                    client_labels, self.client_num, self.non_iid_alpha)
                return [[int(indices[idx]) for idx in idx_slice]
                        for idx_slice in idx_slices]
        except Exception as error:
            logger.warning("Dirichlet split failed, falling back to IID: %s",
                           error)

        return [idx.tolist() for idx in np.array_split(indices,
                                                       self.client_num)]

    @staticmethod
    def _extract_labels(dataset_list):
        if not dataset_list:
            return None
        labels = []
        for idx, item in enumerate(dataset_list):
            item = KnowledgeAnchorSplitter._as_dict(item)
            if 'label' in item:
                labels.append(item['label'])
            elif 'labels' in item and not hasattr(item['labels'], 'shape'):
                labels.append(item['labels'])
            elif 'category' in item:
                labels.append(item['category'])
            elif 'categories' in item:
                labels.append(item['categories'])
            else:
                labels.append(idx % 10)
        return labels

    def _build_client_datasets(self, dataset, dataset_list, client_indices):
        if isinstance(dataset, Dataset):
            return [Subset(dataset, idxs) for idxs in client_indices]
        return [[dataset_list[i] for i in idxs] for idxs in client_indices]

    def _build_dirty_client_datasets(self, dataset, dataset_list,
                                     client_indices):
        tokenizer = self._find_tokenizer(dataset)
        if tokenizer is None:
            logger.warning(
                "Cannot rebuild tokenized dirty client datasets because no "
                "tokenizer was found; falling back to metadata-only records.")
            return [
                self._inject_dirty_records([dataset_list[i] for i in idxs],
                                           client_id)
                for client_id, idxs in enumerate(client_indices, start=1)
            ]

        from federatedscope.llm.dataset.llm_dataset import LLMDataset

        all_records = [
            self._raw_record_at(dataset, idx, dataset_list[idx])
            for idx in range(len(dataset_list))
        ]
        result = []
        for client_id, idxs in enumerate(client_indices, start=1):
            records = [copy.deepcopy(all_records[i]) for i in idxs]
            records = self._inject_dirty_records(records, client_id,
                                                 all_records)
            result.append(LLMDataset(records, tokenizer))
        return result

    def _inject_dirty_records(self,
                              records: Sequence[Dict[str, Any]],
                              client_id: int,
                              donor_pool: Optional[Sequence[Dict[str,
                                                                 Any]]] = None):
        records = [copy.deepcopy(self._as_dict(record)) for record in records]
        ratio = self._dirty_ratio_for_client(client_id)
        dirty_count = int(round(len(records) * ratio))
        if dirty_count <= 0:
            return records

        rng = random.Random(self.dirty_seed + client_id)
        dirty_indices = set(rng.sample(range(len(records)),
                                   min(dirty_count, len(records))))
        donor_pool = list(donor_pool or records)

        for idx, record in enumerate(records):
            record['_dirty_client_id'] = client_id
            if idx not in dirty_indices:
                record.setdefault('_is_dirty', False)
                record.setdefault('_dirty_type', 'clean')
                continue
            records[idx] = self._corrupt_record(record, rng, donor_pool,
                                                client_id)
        return records

    def _dirty_ratio_for_client(self, client_id):
        ratios = self.dirty_client_ratios
        if ratios is None:
            return self.dirty_ratio
        if isinstance(ratios, dict):
            return float(ratios.get(str(client_id), ratios.get(client_id, 0.0)))
        if isinstance(ratios, (list, tuple)) and client_id - 1 < len(ratios):
            return float(ratios[client_id - 1])
        return self.dirty_ratio

    def _corrupt_record(self, record, rng, donor_pool, client_id):
        clean_output = str(record.get('output', ''))
        dirty_type = self.dirty_type
        if dirty_type == 'mixed':
            dirty_type = rng.choice([
                'answer_corruption', 'reasoning_corruption',
                'answer_mismatch'
            ])

        corrupted = copy.deepcopy(record)
        corrupted['_is_dirty'] = True
        corrupted['_dirty_type'] = dirty_type
        corrupted['_clean_output'] = clean_output
        corrupted['_dirty_client_id'] = client_id

        if dirty_type == 'answer_corruption':
            corrupted['output'] = self._corrupt_final_answer(clean_output, rng)
        elif dirty_type == 'reasoning_corruption':
            corrupted['output'] = self._corrupt_reasoning(clean_output, rng)
        elif dirty_type == 'answer_mismatch':
            donors = [
                item for item in donor_pool
                if item.get('output') and item.get('instruction') !=
                record.get('instruction')
            ]
            donor = rng.choice(donors or donor_pool)
            corrupted['output'] = str(donor.get('output', clean_output))
        else:
            raise ValueError(f"Unknown dirty_type: {self.dirty_type}")
        return corrupted

    @staticmethod
    def _final_answer(output):
        match = FINAL_ANSWER_RE.search(str(output))
        return match.group(1) if match else None

    @staticmethod
    def _replace_final_answer(output, value):
        output = str(output)
        if FINAL_ANSWER_RE.search(output):
            return FINAL_ANSWER_RE.sub(lambda m: m.group(0).replace(
                m.group(1), str(value)), output, count=1)
        return output.rstrip() + f"\nThe answer is {value}"

    def _corrupt_final_answer(self, output, rng):
        final = self._final_answer(output)
        old_value = int(float(final)) if final is not None else rng.randint(
            1, 100)
        offset = rng.choice([-41, -23, -13, -7, 11, 19, 37])
        bad_value = old_value + offset
        if bad_value == old_value:
            bad_value += 1
        return self._replace_final_answer(output, bad_value)

    def _corrupt_reasoning(self, output, rng):
        output = str(output)
        matches = list(CALC_RE.finditer(output))
        if not matches:
            return self._corrupt_final_answer(output, rng)
        target = rng.choice(matches)
        old_value = int(float(target.group(2)))
        bad_value = old_value + rng.choice([-10, -6, -3, 2, 5, 9])
        corrupted = output[:target.start(2)] + str(bad_value) + \
            output[target.end(2):]
        return self._replace_final_answer(corrupted, bad_value)

    @staticmethod
    def _as_dict(item):
        if isinstance(item, dict):
            return item
        return dict(item)

    @staticmethod
    def _iter_dataset(dataset):
        return [dataset[i] for i in range(len(dataset))]

    def _raw_record_at(self, dataset, idx, fallback):
        base, raw_idx = self._resolve_base_dataset(dataset, idx)
        if base is not None and hasattr(base, 'raw_data') and \
                raw_idx < len(base.raw_data):
            return copy.deepcopy(base.raw_data[raw_idx])
        item = self._as_dict(fallback)
        return {
            'instruction': item.get('instruction', item.get('input', '')),
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'category': item.get('category', item.get('categories', None)),
        }

    def _resolve_base_dataset(self, dataset, idx):
        current = dataset
        current_idx = int(idx)
        while isinstance(current, Subset):
            current_idx = int(current.indices[current_idx])
            current = current.dataset
        return current, current_idx

    def _find_tokenizer(self, dataset):
        current = dataset
        while isinstance(current, Subset):
            current = current.dataset
        return getattr(current, 'tokenizer', None)


def call_knowledge_anchor_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'knowledge_anchor':
        return KnowledgeAnchorSplitter(client_num, **kwargs)
    return None


class FixedClientSplitter(BaseSplitter):
    """Use materialized `_client_id` fields instead of re-sampling clients."""

    def __init__(self, client_num):
        super(FixedClientSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        grouped = [[] for _ in range(self.client_num)]
        for idx in range(len(dataset)):
            item = dataset[idx]
            if not isinstance(item, dict):
                item = dict(item)
            if item.get('split_role') == 'server':
                continue
            client_id = item.get('_client_id', item.get('client_id', None))
            if client_id is None:
                grouped[idx % self.client_num].append(item)
                continue
            client_idx = int(client_id) - 1
            if client_idx < 0 or client_idx >= self.client_num:
                raise ValueError(f"Invalid _client_id={client_id}")
            grouped[client_idx].append(item)
        return grouped
