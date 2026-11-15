import copy
import logging
import random
import re

import numpy as np

from federatedscope.core.splitters import BaseSplitter

logger = logging.getLogger(__name__)


class DirtyLabelIIDSplitter(BaseSplitter):
    """
    IID splitter with same-task dirty-label injection for LLM datasets.

    The original dataset object is left untouched, so the server-side split
    remains clean. Only client training splits are rebuilt from selected raw
    records with a small fraction of corrupted outputs.
    """
    def __init__(self,
                 client_num,
                 ratio=0.0,
                 client_ratios=None,
                 strategy='answer_corruption',
                 seed=42):
        super(DirtyLabelIIDSplitter, self).__init__(client_num)
        self.ratio = float(ratio or 0.0)
        self.client_ratios = list(client_ratios or [])
        self.strategy = strategy
        self.seed = int(seed)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset
        from federatedscope.llm.dataset.llm_dataset import LLMDataset

        length = len(dataset)
        index = [x for x in range(length)]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(index)
        idx_slice = np.array_split(np.array(index), self.client_num)

        is_train_split = prior is None
        can_rebuild_llm = isinstance(dataset, LLMDataset) and \
            hasattr(dataset, 'raw_data') and hasattr(dataset, 'tokenizer')
        if not is_train_split or not can_rebuild_llm:
            if isinstance(dataset, Dataset):
                return [Subset(dataset, idxs) for idxs in idx_slice]
            return [[dataset[idx] for idx in idxs] for idxs in idx_slice]

        data_list = []
        dirty_counts = []
        for client_id, idxs in enumerate(idx_slice, start=1):
            raw_records = [
                copy.deepcopy(dataset.raw_data[int(idx)])
                for idx in idxs
            ]
            dirty_count = self._poison_records(raw_records, client_id)
            dirty_counts.append(dirty_count)
            data_list.append(
                LLMDataset(raw_records,
                           dataset.tokenizer,
                           output_tag='output'))

        logger.info(
            'Dirty-label IID split enabled: strategy=%s, ratio=%s, '
            'client dirty counts=%s',
            self.strategy, self.ratio, dirty_counts)
        return data_list

    def _client_ratio(self, client_id):
        if self.client_ratios:
            idx = client_id - 1
            if idx < len(self.client_ratios):
                return float(self.client_ratios[idx])
        return self.ratio

    def _poison_records(self, records, client_id):
        ratio = max(0.0, min(1.0, self._client_ratio(client_id)))
        if ratio <= 0 or not records:
            return 0

        rng = random.Random(self.seed + client_id * 1009)
        dirty_count = int(round(len(records) * ratio))
        dirty_count = min(dirty_count, len(records))
        dirty_indices = rng.sample(range(len(records)), dirty_count)

        clean_outputs = [record.get('output', '') for record in records]
        for idx in dirty_indices:
            record = records[idx]
            clean_output = record.get('output', '')
            strategy = self._select_strategy(rng)
            if strategy == 'answer_mismatch' and len(records) > 1:
                new_output = self._make_mismatched_output(
                    clean_outputs, idx, rng)
            else:
                new_output = self._make_corrupted_answer(clean_output, rng)
                strategy = 'answer_corruption'

            if new_output == clean_output:
                new_output = f'{clean_output}\nThe answer is {rng.randint(2, 999)}'

            record['_is_dirty'] = True
            record['_dirty_type'] = strategy
            record['_clean_output'] = clean_output
            record['_dirty_client_id'] = client_id
            record['_source'] = record.get('_source', 'gsm8k')
            record['output'] = new_output

        for idx, record in enumerate(records):
            if idx not in dirty_indices:
                record['_is_dirty'] = False
                record['_source'] = record.get('_source', 'gsm8k')
        return dirty_count

    def _select_strategy(self, rng):
        if self.strategy == 'mixed':
            return rng.choice(['answer_corruption', 'answer_mismatch'])
        return self.strategy

    @staticmethod
    def _make_mismatched_output(clean_outputs, idx, rng):
        candidates = [i for i in range(len(clean_outputs)) if i != idx]
        if not candidates:
            return clean_outputs[idx]
        return clean_outputs[rng.choice(candidates)]

    @staticmethod
    def _make_corrupted_answer(output, rng):
        matches = list(re.finditer(r'[-+]?\d+(?:\.\d+)?', str(output)))
        if not matches:
            return f'{output}\nThe answer is {rng.randint(2, 999)}'

        match = matches[-1]
        original = match.group(0)
        try:
            if '.' in original:
                value = float(original)
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                corrupted = str(value + delta)
            else:
                value = int(original)
                delta = rng.choice([-7, -5, -3, -2, -1, 1, 2, 3, 5, 7])
                corrupted_value = value + delta
                if corrupted_value == value:
                    corrupted_value += 1
                corrupted = str(corrupted_value)
        except ValueError:
            corrupted = str(rng.randint(2, 999))

        return f'{output[:match.start()]}{corrupted}{output[match.end():]}'
