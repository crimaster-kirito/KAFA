import torch
import logging
import re

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None

import accelerate
from accelerate import Accelerator

try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataloader.dataloader import get_tokenizer

logger = logging.getLogger(__name__)

import sys

sys.setrecursionlimit(100000)

GSM8K_FINAL_ANSWER_RE = re.compile(
    r"(?:####|The answer is)\s*(-?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE)
GSM8K_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]")
SQUAD_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        self.grad_accum_step = max(config.llm.grad_accum_step,
                                   config.grad.grad_accum_count)

        if config.llm.accelerator.use:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_step,
                mixed_precision='bf16')
            device = self.accelerator.device

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        self.eval_metrics = config.eval.metrics
        self.dataset_name = str(config.data.type).split('@')[0].lower()
        self.classification_datasets = {'imdb', 'agnews'}
        self.qa_datasets = {'gsm8k', 'squad'}
        self.generation_datasets = {'alpaca', 'dolly-15k'}
        self.enable_rich_task_metrics = self.dataset_name in \
            (self.classification_datasets | self.qa_datasets |
             self.generation_datasets)

    @staticmethod
    def _normalize_numeric_answer(answer):
        if answer is None:
            return None
        text = str(answer).strip()
        if not text:
            return None
        text = text.replace(',', '')
        try:
            number = float(text)
        except Exception:
            return text.lower()
        if number.is_integer():
            return str(int(number))
        return f'{number:.10g}'

    @classmethod
    def _extract_gsm8k_answer(cls, text):
        if text is None:
            return None
        text = str(text)
        match = GSM8K_FINAL_ANSWER_RE.search(text)
        if match:
            return cls._normalize_numeric_answer(match.group(1))
        numbers = GSM8K_NUMBER_RE.findall(text)
        if not numbers:
            return None
        return cls._normalize_numeric_answer(numbers[-1])

    @staticmethod
    def _normalize_text(text):
        if text is None:
            return ""
        text = str(text).strip().lower()
        text = SPACE_RE.sub(" ", text)
        return text

    @classmethod
    def _normalize_qa_text(cls, text):
        text = cls._normalize_text(text)
        text = NON_ALNUM_RE.sub(" ", text)
        text = SQUAD_ARTICLE_RE.sub(" ", text)
        text = SPACE_RE.sub(" ", text).strip()
        return text

    @classmethod
    def _token_overlap_f1(cls, pred_text, ref_text, qa_style=False):
        if qa_style:
            pred_norm = cls._normalize_qa_text(pred_text)
            ref_norm = cls._normalize_qa_text(ref_text)
        else:
            pred_norm = cls._normalize_text(pred_text)
            ref_norm = cls._normalize_text(ref_text)

        pred_tokens = [t for t in pred_norm.split(' ') if t]
        ref_tokens = [t for t in ref_norm.split(' ') if t]
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1

        overlap = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1

        if overlap <= 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        denom = precision + recall
        return 2.0 * precision * recall / denom if denom > 0 else 0.0

    def _map_class_label(self, dataset_name, text):
        norm = self._normalize_text(text)
        if not norm:
            return None

        if dataset_name == 'imdb':
            if 'positive' in norm:
                return 'positive'
            if 'negative' in norm:
                return 'negative'
            return None

        if dataset_name == 'agnews':
            if 'world' in norm:
                return 'world'
            if 'sport' in norm:
                return 'sports'
            if 'business' in norm:
                return 'business'
            if 'sci/tech' in norm or 'science' in norm or \
                    'technology' in norm or 'tech' in norm:
                return 'sci/tech'
            return None

        return None

    @staticmethod
    def _compute_classification_f1(true_labels, pred_labels, class_names):
        if not true_labels or len(true_labels) != len(pred_labels):
            return 0.0, 0.0

        class_f1 = []
        class_support = []
        for cls_name in class_names:
            tp = 0
            fp = 0
            fn = 0
            support = 0
            for y_true, y_pred in zip(true_labels, pred_labels):
                if y_true == cls_name:
                    support += 1
                    if y_pred == cls_name:
                        tp += 1
                    else:
                        fn += 1
                elif y_pred == cls_name:
                    fp += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + recall
            f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
            class_f1.append(f1)
            class_support.append(support)

        macro_f1 = sum(class_f1) / len(class_f1) if class_f1 else 0.0
        total_support = sum(class_support)
        weighted_f1 = 0.0
        if total_support > 0:
            weighted_f1 = sum(
                f * s for f, s in zip(class_f1, class_support)) / total_support
        return macro_f1, weighted_f1

    def _append_auto_task_metrics(self, ctx, eval_results, avg_loss):
        if not self.enable_rich_task_metrics:
            return

        split = ctx.cur_split

        # Always provide these core stability/quality metrics for target tasks.
        if f'{split}_perplexity' not in eval_results:
            eval_results[f'{split}_perplexity'] = float(
                torch.exp(torch.tensor(min(float(avg_loss), 20.0))).item())
        if f'{split}_token_acc' not in eval_results and \
                int(ctx.total_tokens) > 0:
            eval_results[f'{split}_token_acc'] = (
                float(ctx.correct_tokens) / float(ctx.total_tokens)
                if int(ctx.total_tokens) > 0 else 0.0)
        if f'{split}_nan_rate' not in eval_results:
            eval_results[f'{split}_nan_rate'] = (
                float(ctx.nan_batches) / float(ctx.forward_batches)
                if int(ctx.forward_batches) > 0 else 0.0)
        if f'{split}_stability_nan_rate' not in eval_results:
            eval_results[f'{split}_stability_nan_rate'] = \
                eval_results[f'{split}_nan_rate']
        if f'{split}_skipped_batch_rate' not in eval_results:
            eval_results[f'{split}_skipped_batch_rate'] = (
                float(ctx.skipped_batches) / float(ctx.forward_batches)
                if int(ctx.forward_batches) > 0 else 0.0)

        if int(ctx.text_metric_total) > 0:
            eval_results[f'{split}_text_exact_match'] = \
                float(ctx.text_em_correct) / float(ctx.text_metric_total)
            eval_results[f'{split}_text_token_overlap_f1'] = \
                float(ctx.text_overlap_f1_sum) / float(ctx.text_metric_total)

        if self.dataset_name == 'gsm8k' and int(ctx.gsm8k_em_total) > 0:
            eval_results[f'{split}_gsm8k_exact_match_accuracy'] = \
                float(ctx.gsm8k_em_correct) / float(ctx.gsm8k_em_total)

        if self.dataset_name == 'squad' and int(ctx.squad_total) > 0:
            eval_results[f'{split}_squad_exact_match'] = \
                float(ctx.squad_em_correct) / float(ctx.squad_total)
            eval_results[f'{split}_squad_f1'] = \
                float(ctx.squad_f1_sum) / float(ctx.squad_total)

        if self.dataset_name in self.classification_datasets and \
                int(ctx.cls_total) > 0:
            class_names = \
                ['positive', 'negative'] if self.dataset_name == 'imdb' else \
                ['world', 'sports', 'business', 'sci/tech']
            macro_f1, weighted_f1 = self._compute_classification_f1(
                ctx.cls_true_labels, ctx.cls_pred_labels, class_names)
            known_pred = sum(
                1 for pred in ctx.cls_pred_labels if pred != '__unk__')
            eval_results[f'{split}_classification_accuracy'] = \
                float(ctx.cls_correct) / float(ctx.cls_total)
            eval_results[f'{split}_classification_macro_f1'] = float(macro_f1)
            eval_results[f'{split}_classification_weighted_f1'] = \
                float(weighted_f1)
            eval_results[f'{split}_classification_known_pred_rate'] = \
                float(known_pred) / float(ctx.cls_total)

        if self.dataset_name in self.generation_datasets and \
                int(ctx.gen_total) > 0:
            eval_results[f'{split}_generation_nonempty_rate'] = \
                float(ctx.gen_nonempty_pred) / float(ctx.gen_total)
            eval_results[f'{split}_generation_token_overlap_f1'] = \
                float(ctx.gen_overlap_f1_sum) / float(ctx.gen_total)
            eval_results[f'{split}_generation_length_ratio'] = \
                float(ctx.gen_length_ratio_sum) / float(ctx.gen_total)

    def _update_gsm8k_and_dirty_proxy_stats(self, ctx, labels, logits):
        """
        Compute task-adaptive metrics from teacher-forced logits.
        Supports GSM8K/dirty-filter proxy, SQuAD QA, classification and
        generation tasks.
        """
        if not self.enable_rich_task_metrics:
            return
        if labels is None or logits is None or labels.dim() != 2 or \
                logits.dim() != 3:
            return

        try:
            pred_tokens = torch.argmax(logits.detach(), dim=-1)
        except Exception:
            return

        data_batch = ctx.data_batch if isinstance(ctx.data_batch, dict) else {}
        ref_outputs = data_batch.get('output', None)
        dirty_flags = data_batch.get('_is_dirty', None)
        batch_size = labels.shape[0]

        for idx in range(batch_size):
            label_row = labels[idx]
            valid_mask = label_row.ne(-100)
            if int(valid_mask.sum().item()) <= 0:
                continue

            pred_ids = pred_tokens[idx][valid_mask].detach().cpu().tolist()
            if len(pred_ids) == 0:
                continue
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

            ref_text = None
            if isinstance(ref_outputs, list) and idx < len(ref_outputs):
                ref_item = ref_outputs[idx]
                if ref_item is not None:
                    ref_text = str(ref_item)
            if ref_text is None:
                ref_ids = label_row[valid_mask].detach().cpu().tolist()
                ref_text = self.tokenizer.decode(ref_ids,
                                                 skip_special_tokens=True)

            ctx.text_metric_total += 1
            pred_norm = self._normalize_text(pred_text)
            ref_norm = self._normalize_text(ref_text)
            if pred_norm == ref_norm and ref_norm != "":
                ctx.text_em_correct += 1
            ctx.text_overlap_f1_sum += self._token_overlap_f1(
                pred_text, ref_text, qa_style=False)

            if self.dataset_name == 'gsm8k':
                ref_answer = self._extract_gsm8k_answer(ref_text)
                if ref_answer is not None:
                    pred_answer = self._extract_gsm8k_answer(pred_text)
                    ctx.gsm8k_em_total += 1
                    if pred_answer is not None and pred_answer == ref_answer:
                        ctx.gsm8k_em_correct += 1

                    if dirty_flags is not None:
                        if torch.is_tensor(dirty_flags):
                            is_dirty = bool(dirty_flags[idx].item()) \
                                if idx < int(dirty_flags.shape[0]) else False
                        elif isinstance(dirty_flags, list):
                            is_dirty = bool(dirty_flags[idx]) \
                                if idx < len(dirty_flags) else False
                        else:
                            is_dirty = bool(dirty_flags)

                        if is_dirty:
                            ctx.dirty_samples_seen += 1

                        should_filter = (pred_answer is None) or \
                            (pred_answer != ref_answer)
                        if should_filter:
                            if is_dirty:
                                ctx.dirty_samples_filtered += 1
                            else:
                                ctx.clean_samples_filtered += 1

            if self.dataset_name == 'squad':
                qa_pred = self._normalize_qa_text(pred_text)
                qa_ref = self._normalize_qa_text(ref_text)
                if qa_ref != "":
                    ctx.squad_total += 1
                    if qa_pred == qa_ref:
                        ctx.squad_em_correct += 1
                    ctx.squad_f1_sum += self._token_overlap_f1(
                        pred_text, ref_text, qa_style=True)

            if self.dataset_name in self.classification_datasets:
                y_true = self._map_class_label(self.dataset_name, ref_text)
                if y_true is not None:
                    y_pred = self._map_class_label(self.dataset_name, pred_text)
                    y_pred = y_pred if y_pred is not None else '__unk__'
                    ctx.cls_total += 1
                    if y_pred == y_true:
                        ctx.cls_correct += 1
                    ctx.cls_true_labels.append(y_true)
                    ctx.cls_pred_labels.append(y_pred)

            if self.dataset_name in self.generation_datasets:
                ref_tokens = [
                    t for t in self._normalize_text(ref_text).split(' ') if t
                ]
                pred_tokens_norm = [
                    t for t in self._normalize_text(pred_text).split(' ') if t
                ]
                if len(ref_tokens) > 0:
                    ctx.gen_total += 1
                    if len(pred_tokens_norm) > 0:
                        ctx.gen_nonempty_pred += 1
                    ctx.gen_overlap_f1_sum += self._token_overlap_f1(
                        pred_text, ref_text, qa_style=False)
                    ctx.gen_length_ratio_sum += (
                        float(len(pred_tokens_norm)) / float(len(ref_tokens)))

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1):
        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")
        for batch_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator'):
                # Build gradient accumulation upon accelerator
                for _ in range(self.grad_accum_step):
                    with self.accelerator.accumulate(self.ctx.model):
                        for hook in hooks_set["on_batch_start"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_forward"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_backward"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_end"]:
                            hook(self.ctx)

            else:
                for idx in range(self.grad_accum_step):
                    self.ctx.cur_batch_i = CtxVar(
                        batch_i * self.grad_accum_step + idx, LIFECYCLE.BATCH)

                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                    break

    # def _hook_on_fit_start_numerical_precision(self, ctx):
    #     if self.cfg.train.is_enable_half:
    #         if not ctx.cfg.llm.deepspeed.use and \
    #            not ctx.cfg.llm.accelerator.use:
    #             ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            # prepare model sharding
            ctx.model.sharding()

            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                if ctx.cfg.train.optimizer == 'AdamW':
                    ctx.optimizer = AdamW(ctx.model.parameters(),
                                          **ctx.cfg[ctx.cur_mode].optimizer)
                else:
                    ctx.optimizer = get_optimizer(
                        ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.optimizer.zero_grad()
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

            ctx.model, ctx.optimizer, ctx.train_loader, \
                ctx.val_loader, ctx.test_loader, ctx.scheduler = \
                self.accelerator.prepare(
                    ctx.model,
                    getattr(ctx, 'optimizer', None),
                    getattr(ctx, 'train_loader', None),
                    getattr(ctx, 'val_loader', None),
                    getattr(ctx, 'test_loader', None),
                    getattr(ctx, 'scheduler', None)
                )

        elif ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            # if ctx.cfg.train.is_enable_half:
            #     ctx.fp16 = ctx.model_engine.fp16_enabled()

        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.forward_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.nan_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.skipped_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.total_tokens = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.correct_tokens = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.text_metric_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.text_em_correct = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.text_overlap_f1_sum = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.gsm8k_em_correct = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.gsm8k_em_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.squad_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.squad_em_correct = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.squad_f1_sum = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.cls_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.cls_correct = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.cls_true_labels = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.cls_pred_labels = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.gen_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.gen_nonempty_pred = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.gen_overlap_f1_sum = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.gen_length_ratio_sum = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.dirty_samples_seen = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.dirty_samples_filtered = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.clean_samples_filtered = CtxVar(0, LIFECYCLE.ROUTINE)

    def _hook_on_epoch_start(self, ctx):
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
                self.cfg, ctx.cur_split)
            setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        ctx.forward_batches += 1
        if torch.isnan(loss) or not torch.isfinite(loss):
            ctx.nan_batches += 1
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        metrics = set(self.eval_metrics)
        auto_metric_eval_mode = self.enable_rich_task_metrics and \
            ctx.cur_mode in [MODE.VAL, MODE.TEST]
        if auto_metric_eval_mode or \
                ({'token_acc', 'token_accuracy'} & metrics):
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                labels_flat = labels.reshape(-1)
                preds_flat = predictions.reshape(-1)
                valid_mask = labels_flat.ne(-100)
                valid_tokens = int(valid_mask.sum().item())
                if valid_tokens > 0:
                    correct = int(
                        ((labels_flat == preds_flat) & valid_mask).sum().
                        item())
                    ctx.correct_tokens += correct
                    ctx.total_tokens += valid_tokens
        if auto_metric_eval_mode or {
                'gsm8k_exact_match', 'gsm8k_exact_match_accuracy', 'gsm8k_em',
                'gsm8k_accuracy', 'dirty_filter_recall',
                'dirty_filter_precision', 'clean_retention_rate'
        } & metrics:
            self._update_gsm8k_and_dirty_proxy_stats(ctx, labels, logits)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            ctx.optimizer.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
            ctx.optimizer.zero_grad()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                   ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            ctx.skipped_batches += 1
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        metrics = set(self.eval_metrics)
        if {'perplexity', 'ppl'} & metrics:
            perplexity = float(torch.exp(
                torch.tensor(min(float(avg_loss), 20.0))).item())
            if 'perplexity' in metrics:
                eval_results[f'{ctx.cur_split}_perplexity'] = perplexity
            if 'ppl' in metrics:
                eval_results[f'{ctx.cur_split}_ppl'] = perplexity
        if {'token_acc', 'token_accuracy'} & metrics:
            token_acc = (
                float(ctx.correct_tokens) / float(ctx.total_tokens)
                if int(ctx.total_tokens) > 0 else 0.0)
            if 'token_acc' in metrics:
                eval_results[f'{ctx.cur_split}_token_acc'] = token_acc
            if 'token_accuracy' in metrics:
                eval_results[f'{ctx.cur_split}_token_accuracy'] = token_acc
        nan_rate = (
            float(ctx.nan_batches) / float(ctx.forward_batches)
            if int(ctx.forward_batches) > 0 else 0.0)
        if 'nan_rate' in metrics:
            eval_results[f'{ctx.cur_split}_nan_rate'] = nan_rate
        if 'stability_nan_rate' in metrics:
            eval_results[f'{ctx.cur_split}_stability_nan_rate'] = nan_rate
        if 'skipped_batch_rate' in metrics:
            eval_results[f'{ctx.cur_split}_skipped_batch_rate'] = (
                float(ctx.skipped_batches) / float(ctx.forward_batches)
                if int(ctx.forward_batches) > 0 else 0.0)
        if {
                'gsm8k_exact_match', 'gsm8k_exact_match_accuracy', 'gsm8k_em',
                'gsm8k_accuracy'
        } & metrics:
            em_total = float(ctx.gsm8k_em_total)
            em_value = float(ctx.gsm8k_em_correct) / em_total \
                if em_total > 0 else 0.0
            if 'gsm8k_exact_match' in metrics:
                eval_results[f'{ctx.cur_split}_gsm8k_exact_match'] = em_value
            if 'gsm8k_exact_match_accuracy' in metrics:
                eval_results[
                    f'{ctx.cur_split}_gsm8k_exact_match_accuracy'] = em_value
            if 'gsm8k_em' in metrics:
                eval_results[f'{ctx.cur_split}_gsm8k_em'] = em_value
            if 'gsm8k_accuracy' in metrics:
                eval_results[f'{ctx.cur_split}_gsm8k_accuracy'] = em_value
        if {
                'dirty_filter_recall', 'dirty_filter_precision',
                'clean_retention_rate'
        } & metrics:
            dirty_seen = float(ctx.dirty_samples_seen)
            dirty_filtered = float(ctx.dirty_samples_filtered)
            clean_filtered = float(ctx.clean_samples_filtered)
            clean_seen = max(0.0, float(ctx.num_samples) - dirty_seen)
            filtered_total = dirty_filtered + clean_filtered
            dirty_recall = dirty_filtered / dirty_seen if dirty_seen > 0 else 0.0
            dirty_precision = \
                dirty_filtered / filtered_total if filtered_total > 0 else 0.0
            clean_retention = (
                (clean_seen - clean_filtered) / clean_seen
                if clean_seen > 0 else 0.0)
            clean_filter_fpr = (clean_filtered / clean_seen
                                if clean_seen > 0 else 0.0)
            filter_activation_rate = (
                filtered_total / float(ctx.num_samples)
                if float(ctx.num_samples) > 0 else 0.0)
            dirty_sample_rate = (
                dirty_seen / float(ctx.num_samples)
                if float(ctx.num_samples) > 0 else 0.0)

            ctx.dirty_filter_stats = {
                'dirty_samples_seen': dirty_seen,
                'dirty_samples_filtered': dirty_filtered,
                'clean_samples_filtered': clean_filtered,
            }

            if 'dirty_filter_recall' in metrics:
                eval_results[f'{ctx.cur_split}_dirty_filter_recall'] = \
                    dirty_recall
            if 'dirty_filter_precision' in metrics:
                eval_results[f'{ctx.cur_split}_dirty_filter_precision'] = \
                    dirty_precision
            if 'clean_retention_rate' in metrics:
                eval_results[f'{ctx.cur_split}_clean_retention_rate'] = \
                    clean_retention
            # Always emit interpretable companion metrics for clean-only
            # splits (e.g., test set with no dirty labels).
            eval_results[f'{ctx.cur_split}_clean_filter_false_positive_rate'] = \
                clean_filter_fpr
            eval_results[f'{ctx.cur_split}_filter_activation_rate'] = \
                filter_activation_rate
            eval_results[f'{ctx.cur_split}_dirty_sample_rate'] = \
                dirty_sample_rate

        self._append_auto_task_metrics(ctx, eval_results, avg_loss)

        # Fill extra metrics declared in cfg.eval.metrics via registered
        # metric functions when possible, so experiment logs have
        # consistent columns across methods.
        try:
            from federatedscope.core.monitors.metric_calculator import \
                SUPPORT_METRICS
        except Exception as import_error:
            SUPPORT_METRICS = {}
            logger.warning(f"Cannot import SUPPORT_METRICS: {import_error}")

        for metric_name in metrics:
            metric_key = f'{ctx.cur_split}_{metric_name}'
            if metric_key in eval_results:
                continue

            metric_item = SUPPORT_METRICS.get(metric_name, None)
            if metric_item is None or metric_item[0] is None:
                continue

            metric_func = metric_item[0]
            try:
                metric_val = metric_func(ctx=ctx, metric=metric_name)
            except TypeError:
                metric_val = metric_func(ctx=ctx)
            except Exception as metric_error:
                logger.debug(f"Metric `{metric_name}` failed: {metric_error}")
                continue

            if torch.is_tensor(metric_val):
                metric_val = float(metric_val.detach().cpu().item())
            elif hasattr(metric_val, 'item'):
                try:
                    metric_val = float(metric_val.item())
                except Exception:
                    pass
            eval_results[metric_key] = metric_val

        setattr(ctx, 'eval_metrics', eval_results)

        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
