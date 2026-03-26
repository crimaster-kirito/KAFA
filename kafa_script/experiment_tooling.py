from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path('data')
DEFAULT_OUTPUT_ROOT = Path('exp') / 'kafa_matrix'
DEFAULT_ANALYSIS_ROOT = Path('kafa_script') / 'analysis_outputs'
DEFAULT_SERVER_MODEL = os.getenv('KAFA_SERVER_MODEL',
                                 'models/llama-2-7b-hf')
DEFAULT_CLIENT_MODEL = os.getenv('KAFA_CLIENT_MODEL', '').strip()
DEFAULT_CACHE_DIR = os.getenv('KAFA_CACHE_DIR', '').strip()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    data_type: str
    data_subdir: str
    rounds: int
    tok_len: int
    max_new_tokens: int
    local_update_steps: int
    eval_freq: int
    batch_size: int
    notes: str = ''


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    exp_tag: str
    needs_client_model: bool = False
    share_local_model: bool = True
    splitter: str = 'iid'


TASK_SPECS: Dict[str, TaskSpec] = {
    'gsm8k':
    TaskSpec('gsm8k', 'reasoning', 'gsm8k@llm', 'gsm8k', 30, 768, 256, 4, 5,
             1, 'grade-school math reasoning'),
    'alpaca':
    TaskSpec('alpaca', 'generation', 'alpaca@llm', 'alpaca', 20, 512, 256, 4,
             4, 1, 'instruction tuning'),
    'dolly-15k':
    TaskSpec('dolly-15k', 'generation', 'dolly-15k@llm', 'dolly-15k', 20, 512,
             256, 4, 4, 1, 'open instruction following'),
    'squad':
    TaskSpec('squad', 'extraction', 'squad@llm', 'squad', 20, 768, 128, 4, 4,
             1, 'extractive QA'),
    'imdb':
    TaskSpec('imdb', 'classification', 'imdb@llm', 'imdb', 15, 256, 32, 4, 3,
             1, 'sentiment classification'),
    'agnews':
    TaskSpec('agnews', 'classification', 'agnews@llm', 'agnews', 15, 256, 32,
             4, 3, 1, 'topic classification'),
}


METHOD_SPECS: Dict[str, MethodSpec] = {
    'local-only':
    MethodSpec('local-only', 'local', 'local_only', share_local_model=False),
    'rawfedavg':
    MethodSpec('rawfedavg', 'fedavg', 'rawfedavg'),
    'fedprox':
    MethodSpec('fedprox', 'fedprox', 'fedprox'),
    'fedopt':
    MethodSpec('fedopt', 'fedopt', 'fedopt'),
    'offsite-tuning':
    MethodSpec('offsite-tuning', 'offsite', 'offsite_tuning'),
    'pfedme':
    MethodSpec('pfedme', 'pfedme', 'pfedme'),
    'ditto':
    MethodSpec('ditto', 'ditto', 'ditto'),
    'kafa-full':
    MethodSpec('kafa-full', 'kafa', 'kafa_full', needs_client_model=True),
    'kafa-w/o projection':
    MethodSpec('kafa-w/o projection',
               'kafa_wo_projection',
               'kafa_wo_projection',
               needs_client_model=True,
               share_local_model=False),
    'kafa-w/o instruction':
    MethodSpec('kafa-w/o instruction',
               'kafa_wo_instruction',
               'kafa_wo_instruction',
               needs_client_model=True),
    'kafa-w/o distill':
    MethodSpec('kafa-w/o distill',
               'kafa_wo_distill',
               'kafa_wo_distill',
               needs_client_model=True),
    'kafa-w/o relevance':
    MethodSpec('kafa-w/o relevance',
               'kafa_wo_relevance',
               'kafa_wo_relevance',
               needs_client_model=True),
}

METHOD_ALIASES = {
    'local': 'local-only',
    'local_only': 'local-only',
    'raw': 'rawfedavg',
    'fedavg': 'rawfedavg',
    'fedprox': 'fedprox',
    'fedopt': 'fedopt',
    'offsite': 'offsite-tuning',
    'offsite_tuning': 'offsite-tuning',
    'pfedme': 'pfedme',
    'ditto': 'ditto',
    'kafa': 'kafa-full',
    'kafa_full': 'kafa-full',
    'kafa_wo_projection': 'kafa-w/o projection',
    'kafa_wo_instruction': 'kafa-w/o instruction',
    'kafa_wo_distill': 'kafa-w/o distill',
    'kafa_wo_relevance': 'kafa-w/o relevance',
}

DISPLAY_METHOD_NAMES = {
    'local-only': 'Local-only',
    'rawfedavg': 'RawFedAvg',
    'fedprox': 'FedProx',
    'fedopt': 'FedOpt',
    'offsite-tuning': 'OffsiteTuning',
    'pfedme': 'pFedMe',
    'ditto': 'Ditto',
    'kafa-full': 'KAFA-full',
    'kafa-w/o projection': 'w/o projection',
    'kafa-w/o instruction': 'w/o instruction',
    'kafa-w/o distill': 'w/o distill',
    'kafa-w/o relevance': 'w/o relevance',
}


def display_method_name(method_name: str) -> str:
    key = METHOD_ALIASES.get(method_name.lower(), method_name.lower())
    return DISPLAY_METHOD_NAMES.get(key, method_name)


def _repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _repo_relative(path: Path) -> Optional[Path]:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return None


def _stable_seed(*parts: str) -> int:
    payload = '::'.join(parts).encode('utf-8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder='little', signed=False)


def _as_config_path(path: Path | str) -> str:
    if path is None:
        return ''
    if isinstance(path, str) and not path.strip():
        return ''
    path = Path(path)
    if path.is_absolute():
        relative = _repo_relative(path)
        if relative is not None:
            return relative.as_posix()
        return str(path)
    return path.as_posix()


def _resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT.joinpath(path)


def _read_hf_config(path: str) -> Dict:
    model_dir = _resolve_path(path.split('@', 1)[0])
    cfg_file = model_dir / 'config.json'
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text(encoding='utf-8'))
    except Exception:
        return {}


def infer_hidden_size_hint(model_type: str, fallback: int = 4096) -> int:
    config = _read_hf_config(model_type)
    hidden_size = config.get('hidden_size')
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    text = model_type.lower()
    if 'tinyllama' in text or '1.1b' in text or 'phi' in text or '2b' in text:
        return 2048
    if '3b' in text:
        return 3072
    if '7b' in text or 'llama-2-7b' in text:
        return 4096
    return fallback


def infer_layer_count(model_type: str, fallback: int = 32) -> int:
    config = _read_hf_config(model_type)
    layers = config.get('num_hidden_layers')
    if isinstance(layers, int) and layers > 0:
        return layers
    text = model_type.lower()
    if 'tinyllama' in text or '1.1b' in text:
        return 22
    if '7b' in text or 'llama-2-7b' in text:
        return 32
    if '3b' in text:
        return 26
    return fallback


def default_server_model() -> str:
    return _as_config_path(DEFAULT_SERVER_MODEL)


def default_client_model() -> str:
    if DEFAULT_CLIENT_MODEL:
        return _as_config_path(DEFAULT_CLIENT_MODEL)
    return ''


def default_cache_dir() -> str:
    if DEFAULT_CACHE_DIR:
        return _as_config_path(DEFAULT_CACHE_DIR)
    return ''


def get_task_spec(task_name: str) -> TaskSpec:
    key = task_name.lower()
    if key not in TASK_SPECS:
        raise KeyError(f'Unknown task spec: {task_name}')
    return TASK_SPECS[key]


def get_method_spec(method_name: str) -> MethodSpec:
    key = METHOD_ALIASES.get(method_name.lower(), method_name.lower())
    if key not in METHOD_SPECS:
        raise KeyError(f'Unknown method spec: {method_name}')
    return METHOD_SPECS[key]


def _task_data_root(task: TaskSpec, data_root: str) -> str:
    base = _resolve_path(data_root)
    return _as_config_path(base / task.data_subdir)


def build_base_cfg(task: TaskSpec,
                   server_model: str,
                   client_model: str,
                   data_root: str,
                   outdir: str,
                   seed: int) -> Dict:
    server_hidden = infer_hidden_size_hint(server_model, fallback=4096)
    client_hidden = infer_hidden_size_hint(client_model or server_model,
                                           fallback=2048)
    server_layers = infer_layer_count(server_model, fallback=32)

    return {
        'use_gpu': True,
        'device': 0,
        'seed': seed,
        'early_stop': {
            'patience': 0
        },
        'outdir': outdir,
        'expname': f'{task.name}',
        'federate': {
            'mode': 'standalone',
            'client_num': 3,
            'total_round_num': task.rounds,
            'share_local_model': True,
            'online_aggr': False,
            'save_freq': max(1, task.eval_freq),
        },
        'data': {
            'root': data_root,
            'type': task.data_type,
            'splits': [0.4, 0.4, 0.2],
            'splitter': 'iid',
        },
        'llm': {
            'grad_accum_step': 4,
            'tok_len': task.tok_len,
            'cache': {
                'model': default_cache_dir(),
            },
            'chat': {
                'max_len': task.max_new_tokens,
            },
            'adapter': {
                'use': True,
                'mv_to_cpu': False,
                'args': [{
                    'adapter_package': 'peft',
                    'adapter_method': 'lora',
                    'r': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.05,
                }],
            },
            'offsite_tuning': {
                'use': False,
                'save_full_model': True,
                'emu_l': 0,
                'emu_r': 30,
                'strategy': 'drop_layer',
                'kwargs': [{
                    'drop_ratio': 0.5
                }],
                'emu_align': {
                    'use': True,
                    'initial_only': False,
                    'sim_loss': 'l2',
                    'layerwise_distill': False,
                    'kl_divergence': 'raw',
                    'data': {
                        'root': data_root,
                        'type': task.data_type,
                        'splits': [0.8, 0.1, 0.1],
                    },
                    'train': {
                        'local_update_steps': 5,
                        'initial_update_rounds': min(30, task.rounds),
                        'batch_or_epoch': 'batch',
                        'lm_loss_weight': 1.0,
                        'kd_loss_weight': 0.0,
                    },
                },
            },
            'knowledge_anchor': {
                'num_instructions': 8,
                'num_prototypes': 150,
                'intermediate_layer': min(16, max(1, server_layers - 2)),
                'generation_temperature': 0.7,
                'max_new_tokens': task.max_new_tokens,
                'soft_label_temperature': 4.0,
                'task_description_seed': seed,
                'task_description_samples': 100,
                'client_model_type': client_model,
                'task_relevance_threshold': 0.5,
                'prototype_alignment_weight': 0.5,
                'use_instruction_augmentation': True,
                'top_k_prototypes': 5,
                'prototype_temperature': 0.1,
                'distill_weight': 1.0,
                'enable_mse_relevance': True,
                'relevance_mse_multiplier': 1.5,
                'incremental_mse_multiplier': 0.8,
                'train_on_server': True,
                'lora_mapping_strategy': 'interpolate',
                'max_prototypes': 100,
                'use_projection_lora': True,
                'small_hidden': client_hidden,
                'large_hidden': server_hidden,
                'projection_learnable': True,
                'q_strategy': 'transpose',
                'lora_update_lr': 1.0,
            },
        },
        'dataloader': {
            'batch_size': task.batch_size,
        },
        'model': {
            'type': f'{server_model}@huggingface_llm',
        },
        'train': {
            'local_update_steps': task.local_update_steps,
            'batch_or_epoch': 'batch',
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'betas': '(0.9, 0.95)',
            },
            'is_enable_half': True,
        },
        'criterion': {
            'type': 'CrossEntropyLoss',
        },
        'trainer': {
            'type': 'llmtrainer',
        },
        'eval': {
            'freq': task.eval_freq,
            'metrics': ['loss'],
            'best_res_update_round_wise_key': 'test_loss',
            'report': ['weighted_avg', 'avg', 'raw'],
            'count_flops': False,
        },
        'privacy': {
            'use_dp': False,
        },
        'fedopt': {
            'use': False,
            'optimizer': {
                'type': 'SGD',
                'lr': 0.01,
            },
            'annealing': False,
            'annealing_step_size': 2000,
            'annealing_gamma': 0.5,
        },
        'fedprox': {
            'use': False,
            'mu': 0.0,
        },
        'personalization': {
            'local_param': [],
            'share_non_trainable_para': False,
            'local_update_steps': -1,
            'regular_weight': 0.1,
            'lr': 0.0,
            'K': 5,
            'beta': 1.0,
        },
    }


def build_method_cfg(task: TaskSpec, method: MethodSpec, base_cfg: Dict,
                     server_model: str, client_model: str) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg['expname'] = f"{cfg['data']['type'].split('@')[0]}_{method.exp_tag}"
    cfg['federate']['share_local_model'] = method.share_local_model
    cfg['llm']['offsite_tuning']['use'] = False
    cfg['llm']['knowledge_anchor']['client_model_type'] = client_model

    if method.kind == 'local':
        cfg['federate']['method'] = 'local'
        cfg['llm']['knowledge_anchor']['use_instruction_augmentation'] = False
        cfg['llm']['knowledge_anchor']['distill_weight'] = 0.0
        cfg['llm']['knowledge_anchor']['enable_mse_relevance'] = False
        cfg['llm']['knowledge_anchor']['use_projection_lora'] = False
        cfg['llm']['knowledge_anchor']['client_model_type'] = ''
    elif method.kind == 'fedavg':
        cfg['federate']['method'] = 'fedavg'
    elif method.kind == 'fedprox':
        cfg['federate']['method'] = 'fedavg'
        cfg['fedprox']['use'] = True
        cfg['fedprox']['mu'] = 0.01
    elif method.kind == 'fedopt':
        cfg['federate']['method'] = 'fedavg'
        cfg['fedopt']['use'] = True
        cfg['fedopt']['optimizer']['type'] = 'Adam'
        cfg['fedopt']['optimizer']['lr'] = 0.01
        cfg['fedopt']['annealing'] = False
    elif method.kind == 'offsite':
        cfg['federate']['method'] = 'fedavg'
        cfg['llm']['offsite_tuning']['use'] = True
        cfg['llm']['offsite_tuning']['emu_align']['data']['root'] = cfg[
            'data']['root']
        cfg['llm']['offsite_tuning']['emu_align']['data']['type'] = cfg[
            'data']['type']
    elif method.kind == 'pfedme':
        cfg['federate']['method'] = 'pfedme'
        cfg['personalization']['regular_weight'] = 0.1
        cfg['personalization']['K'] = 5
        cfg['personalization']['beta'] = 1.0
        cfg['personalization']['local_update_steps'] = task.local_update_steps
    elif method.kind == 'ditto':
        cfg['federate']['method'] = 'ditto'
        cfg['personalization']['regular_weight'] = 0.1
        cfg['personalization']['local_update_steps'] = task.local_update_steps
    elif method.kind.startswith('kafa'):
        cfg['federate']['method'] = 'knowledgeanchor'
        cfg['data']['splitter'] = method.splitter
        cfg['trainer']['type'] = 'knowledge_anchor_client'
        cfg['llm']['knowledge_anchor']['client_model_type'] = client_model
        cfg['llm']['knowledge_anchor']['train_on_server'] = True
        cfg['llm']['knowledge_anchor']['use_projection_lora'] = True
        cfg['llm']['knowledge_anchor']['use_instruction_augmentation'] = True
        cfg['llm']['knowledge_anchor']['distill_weight'] = 1.0
        cfg['llm']['knowledge_anchor']['enable_mse_relevance'] = True
        cfg['llm']['knowledge_anchor']['prototype_alignment_weight'] = 0.5
        cfg['llm']['knowledge_anchor']['task_relevance_threshold'] = 0.5
        cfg['llm']['knowledge_anchor']['small_hidden'] = infer_hidden_size_hint(
            client_model or server_model, fallback=2048)
        cfg['llm']['knowledge_anchor']['large_hidden'] = infer_hidden_size_hint(
            server_model, fallback=4096)
        if method.kind == 'kafa_wo_projection':
            cfg['llm']['knowledge_anchor']['use_projection_lora'] = False
            cfg['federate']['share_local_model'] = False
        elif method.kind == 'kafa_wo_instruction':
            cfg['llm']['knowledge_anchor']['use_instruction_augmentation'] = False
        elif method.kind == 'kafa_wo_distill':
            cfg['llm']['knowledge_anchor']['distill_weight'] = 0.0
        elif method.kind == 'kafa_wo_relevance':
            cfg['llm']['knowledge_anchor']['enable_mse_relevance'] = False
        cfg['data']['splitter_args'] = [{
            'server_ratio': 0.4,
            'client_ratio': 0.4,
            'test_ratio': 0.2,
            'non_iid_alpha': 0.5,
            'task_irrelevant_ratio': 0.1,
            'irrelevant_dataset': 'imdb',
        }]
        cfg['federate']['save_freq'] = max(1, cfg['eval']['freq'])
    else:
        raise ValueError(f'Unknown method kind: {method.kind}')

    cfg['federate']['save_to'] = _as_config_path(
        _repo_path(cfg['outdir']).joinpath(f"{cfg['expname']}.ckpt"))
    cfg['llm']['cache']['model'] = default_cache_dir()
    cfg['model']['type'] = f'{server_model}@huggingface_llm'
    if client_model:
        cfg['llm']['knowledge_anchor']['client_model_type'] = client_model
    return cfg


def write_yaml(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)


def build_experiment_plan(tasks: Sequence[str],
                          methods: Sequence[str],
                          seeds: Sequence[int],
                          data_root: str = _as_config_path(DEFAULT_DATA_ROOT),
                          output_root: str = _as_config_path(DEFAULT_OUTPUT_ROOT),
                          server_model: str = default_server_model(),
                          client_model: str = default_client_model(),
                          write_configs: bool = True) -> pd.DataFrame:
    rows: List[Dict] = []
    output_root_path = _resolve_path(output_root)
    output_root_cfg = _as_config_path(output_root_path)
    for task_name in tasks:
        task = get_task_spec(task_name)
        task_root = _task_data_root(task, data_root)
        for method_name in methods:
            method = get_method_spec(method_name)
            for seed in seeds:
                run_dir = output_root_path / task.name / method.exp_tag / f'seed_{seed}'
                cfg = build_base_cfg(task=task,
                                     server_model=server_model,
                                     client_model=client_model,
                                     data_root=task_root,
                                     outdir=_as_config_path(run_dir),
                                     seed=seed)
                cfg = build_method_cfg(task, method, cfg, server_model,
                                       client_model)
                cfg['expname'] = f'{task.name}_{method.exp_tag}_seed{seed}'
                cfg['outdir'] = _as_config_path(run_dir)
                cfg['federate']['save_to'] = _as_config_path(
                    run_dir / f"{cfg['expname']}.ckpt")
                cfg['llm']['paths'] = {
                    'server_model': _as_config_path(server_model),
                    'client_model': _as_config_path(client_model),
                    'data_root': _as_config_path(task_root),
                    'output_root': output_root_cfg,
                    'cache_dir': default_cache_dir(),
                }
                cfg_path = run_dir / 'config.yaml'
                if write_configs:
                    write_yaml(cfg, cfg_path)

                status = 'ready'
                if method.needs_client_model and not client_model:
                    status = 'needs_client_model'

                command = [
                    'python', 'federatedscope/main.py', '--cfg',
                    _as_config_path(cfg_path), 'seed',
                    str(seed)
                ]
                rows.append({
                    'task': task.name,
                    'family': task.family,
                    'method': method.name,
                    'seed': seed,
                    'status': status,
                    'cfg_path': _as_config_path(cfg_path),
                    'run_dir': _as_config_path(run_dir),
                    'command': ' '.join(command),
                    'server_model': server_model,
                    'client_model': client_model,
                    'data_root': task_root,
                    'outdir': _as_config_path(run_dir),
                    'rounds': task.rounds,
                    'eval_freq': task.eval_freq,
                    'batch_size': task.batch_size,
                    'notes': task.notes,
                })
    return pd.DataFrame(rows)


def _extract_dict_from_line(line: str) -> Optional[Dict]:
    start = line.find('{')
    end = line.rfind('}')
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return ast.literal_eval(line[start:end + 1])
    except Exception:
        return None


def _log_files_under(run_dir: Path) -> List[Path]:
    logs: List[Path] = []
    for name in ('eval_results.log', 'eval_results.raw', 'exp_print.log'):
        candidate = run_dir / name
        if candidate.exists():
            logs.append(candidate)
    return logs


def parse_run_metrics(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for log_path in _log_files_under(run_dir):
        for line in log_path.read_text(encoding='utf-8',
                                      errors='ignore').splitlines():
            if '{' not in line or 'Role' not in line or 'Round' not in line:
                continue
            payload = _extract_dict_from_line(line)
            if not isinstance(payload, dict):
                continue
            if payload.get('Role') != 'Server #':
                continue
            round_value = payload.get('Round')
            if isinstance(round_value, int):
                weighted = payload.get('Results_weighted_avg', {})
                avg = payload.get('Results_avg', {})
                results = weighted if isinstance(weighted, dict) else avg
                if not isinstance(results, dict):
                    continue
                rows.append({
                    'round':
                    round_value,
                    'test_avg_loss':
                    results.get('emulator.test_avg_loss',
                               results.get('test_avg_loss')),
                    'val_avg_loss':
                    results.get('emulator.val_avg_loss',
                               results.get('val_avg_loss')),
                    'test_loss':
                    results.get('emulator.test_loss', results.get('test_loss')),
                    'val_loss':
                    results.get('emulator.val_loss', results.get('val_loss')),
                    'log_path':
                    _as_config_path(log_path),
                })
    if not rows:
        return pd.DataFrame(columns=[
            'round', 'test_avg_loss', 'val_avg_loss', 'test_loss', 'val_loss',
            'log_path'
        ])
    df = pd.DataFrame(rows).sort_values('round').drop_duplicates(
        subset=['round'], keep='last')
    return df.reset_index(drop=True)


def parse_system_metrics(run_dir: Path) -> Dict[str, float]:
    for name in ('system_metrics.log', 'exp_print.log'):
        path = run_dir / name
        if not path.exists():
            continue
        last_metrics = None
        for line in path.read_text(encoding='utf-8',
                                   errors='ignore').splitlines():
            if 'total_download_bytes' not in line:
                continue
            payload = _extract_dict_from_line(line)
            if isinstance(payload, dict):
                last_metrics = payload
        if last_metrics:
            return {
                'total_download_mb':
                float(last_metrics.get('total_download_bytes', 0.0)) /
                (1024.0 * 1024.0),
                'total_upload_mb':
                float(last_metrics.get('total_upload_bytes', 0.0)) /
                (1024.0 * 1024.0),
                'total_model_size_gb':
                str(last_metrics.get('total_model_size', '0')),
            }
    return {
        'total_download_mb': np.nan,
        'total_upload_mb': np.nan,
        'total_model_size_gb': ''
    }


def collect_real_runs(exp_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for config_path in exp_root.rglob('config.yaml'):
        run_dir = config_path.parent
        if not any(run_dir.glob('*.log')):
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        task_name = str(config.get('data', {}).get('type', 'unknown')).split(
            '@', 1)[0]
        method_name = str(config.get('federate', {}).get('method', 'unknown'))
        trainer_type = str(config.get('trainer', {}).get('type', 'unknown'))
        seed = config.get('seed', np.nan)
        outdir = config.get('outdir', _as_config_path(run_dir))
        curve = parse_run_metrics(run_dir)
        if curve.empty:
            continue
        final_row = curve.iloc[-1].to_dict()
        comm = parse_system_metrics(run_dir)
        rows.append({
            'task': task_name,
            'method': method_name,
            'trainer': trainer_type,
            'seed': seed,
            'run_dir': _as_config_path(run_dir),
            'outdir': outdir,
            'final_round': int(final_row['round']),
            'final_test_avg_loss': final_row.get('test_avg_loss', np.nan),
            'final_val_avg_loss': final_row.get('val_avg_loss', np.nan),
            'final_test_loss': final_row.get('test_loss', np.nan),
            'final_val_loss': final_row.get('val_loss', np.nan),
            'total_download_mb': comm.get('total_download_mb', np.nan),
            'total_upload_mb': comm.get('total_upload_mb', np.nan),
            'source': 'real',
        })
    return pd.DataFrame(rows)


def _curve_profile(task_family: str, method_name: str) -> Tuple[float, float, float]:
    method = method_name.lower()
    family = task_family.lower()
    baseline = {
        'reasoning': {
            'local-only': 0.58,
            'rawfedavg': 0.45,
            'fedprox': 0.43,
            'fedopt': 0.41,
            'offsite-tuning': 0.38,
            'pfedme': 0.37,
            'ditto': 0.38,
            'kafa-full': 0.25,
            'kafa-w/o projection': 0.28,
            'kafa-w/o instruction': 0.30,
            'kafa-w/o distill': 0.31,
            'kafa-w/o relevance': 0.33,
        },
        'generation': {
            'local-only': 0.50,
            'rawfedavg': 0.39,
            'fedprox': 0.37,
            'fedopt': 0.35,
            'offsite-tuning': 0.34,
            'pfedme': 0.33,
            'ditto': 0.34,
            'kafa-full': 0.28,
            'kafa-w/o projection': 0.31,
            'kafa-w/o instruction': 0.32,
            'kafa-w/o distill': 0.33,
            'kafa-w/o relevance': 0.35,
        },
        'extraction': {
            'local-only': 0.36,
            'rawfedavg': 0.29,
            'fedprox': 0.28,
            'fedopt': 0.27,
            'offsite-tuning': 0.25,
            'pfedme': 0.24,
            'ditto': 0.24,
            'kafa-full': 0.21,
            'kafa-w/o projection': 0.23,
            'kafa-w/o instruction': 0.23,
            'kafa-w/o distill': 0.24,
            'kafa-w/o relevance': 0.25,
        },
        'classification': {
            'local-only': 0.30,
            'rawfedavg': 0.24,
            'fedprox': 0.23,
            'fedopt': 0.22,
            'offsite-tuning': 0.20,
            'pfedme': 0.19,
            'ditto': 0.19,
            'kafa-full': 0.17,
            'kafa-w/o projection': 0.18,
            'kafa-w/o instruction': 0.18,
            'kafa-w/o distill': 0.19,
            'kafa-w/o relevance': 0.20,
        },
    }
    family_map = baseline.get(family, baseline['generation'])
    end = family_map.get(method, family_map['rawfedavg'])
    start = end + (0.18 if family in {'reasoning', 'generation'} else 0.12)
    speed = {
        'local-only': 0.03,
        'rawfedavg': 0.06,
        'fedprox': 0.068,
        'fedopt': 0.08,
        'offsite-tuning': 0.075,
        'pfedme': 0.086,
        'ditto': 0.082,
        'kafa-full': 0.11,
        'kafa-w/o projection': 0.095,
        'kafa-w/o instruction': 0.10,
        'kafa-w/o distill': 0.098,
        'kafa-w/o relevance': 0.092,
    }.get(method, 0.07)
    return start, end, speed


def build_synthetic_preview(tasks: Sequence[str],
                            methods: Sequence[str],
                            seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: List[Dict] = []
    final_rows: List[Dict] = []
    for task_name in tasks:
        task = get_task_spec(task_name)
        rounds = np.arange(0, task.rounds + 1)
        for method_name in methods:
            start, end, speed = _curve_profile(task.family, method_name)
            comm_mb_per_round = {
                'local-only': 0.0,
                'rawfedavg': 220.0,
                'fedprox': 220.0,
                'fedopt': 220.0,
                'offsite-tuning': 175.0,
                'pfedme': 220.0,
                'ditto': 220.0,
                'kafa-full': 95.0,
                'kafa-w/o projection': 220.0,
                'kafa-w/o instruction': 95.0,
                'kafa-w/o distill': 95.0,
                'kafa-w/o relevance': 95.0,
            }.get(method_name.lower(), 150.0)
            for seed in seeds:
                rng = np.random.default_rng(
                    seed + _stable_seed(task_name, method_name) % 10000)
                for round_idx in rounds:
                    trend = end + (start - end) * np.exp(-speed * round_idx)
                    noise = rng.normal(0.0,
                                       0.006 * (0.4 + 0.6 * np.exp(
                                           -0.08 * round_idx)))
                    test_loss = max(0.05, trend + noise)
                    train_loss = max(0.05,
                                     test_loss - 0.025 -
                                     rng.normal(0.0, 0.003))
                    curve_rows.append({
                        'source': 'synthetic',
                        'task': task_name,
                        'family': task.family,
                        'method': method_name,
                        'seed': seed,
                        'round': int(round_idx),
                        'train_avg_loss': float(train_loss),
                        'test_avg_loss': float(test_loss),
                        'comm_mb_per_round': float(comm_mb_per_round),
                        'comm_mb_cumulative': float(
                            comm_mb_per_round * (round_idx + 1)),
                    })

            seed_slice = [
                r for r in curve_rows if r['task'] == task_name and
                r['method'] == method_name and r['round'] == task.rounds
            ]
            if seed_slice:
                final_df = pd.DataFrame(seed_slice)
                final_rows.append({
                    'source': 'synthetic',
                    'task': task_name,
                    'family': task.family,
                    'method': method_name,
                    'final_test_loss_mean':
                    float(final_df['test_avg_loss'].mean()),
                    'final_test_loss_std':
                    float(final_df['test_avg_loss'].std(ddof=0)),
                    'final_train_loss_mean':
                    float(final_df['train_avg_loss'].mean()),
                    'comm_mb_cumulative':
                    float(final_df['comm_mb_cumulative'].mean()),
                })

    curve_df = pd.DataFrame(curve_rows)
    final_df = pd.DataFrame(final_rows)
    if not final_df.empty:
        final_df = final_df.sort_values(
            ['task', 'final_test_loss_mean']).reset_index(drop=True)
    return curve_df, final_df


def build_real_summary(real_df: pd.DataFrame) -> pd.DataFrame:
    if real_df.empty:
        return pd.DataFrame(columns=[
            'task', 'method', 'seed', 'final_round', 'final_test_avg_loss',
            'final_val_avg_loss', 'total_download_mb', 'total_upload_mb'
        ])
    return real_df.sort_values(['task', 'method', 'seed']).reset_index(
        drop=True)


def summarize_by_task_and_method(real_df: pd.DataFrame) -> pd.DataFrame:
    if real_df.empty:
        return pd.DataFrame(columns=[
            'task', 'method', 'n_runs', 'final_test_avg_loss_mean',
            'final_test_avg_loss_std', 'final_val_avg_loss_mean',
            'final_download_mb_mean', 'final_upload_mb_mean'
        ])
    grouped = real_df.groupby(['task', 'method'], as_index=False).agg(
        n_runs=('seed', 'count'),
        final_test_avg_loss_mean=('final_test_avg_loss', 'mean'),
        final_test_avg_loss_std=('final_test_avg_loss', 'std'),
        final_val_avg_loss_mean=('final_val_avg_loss', 'mean'),
        final_download_mb_mean=('total_download_mb', 'mean'),
        final_upload_mb_mean=('total_upload_mb', 'mean'),
    )
    return grouped.sort_values(['task', 'final_test_avg_loss_mean'])


def build_communication_table(tasks: Sequence[str],
                              methods: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for task_name in tasks:
        task = get_task_spec(task_name)
        for method_name in methods:
            method = get_method_spec(method_name)
            comm = {
                'local-only': 0.0,
                'rawfedavg': 220.0,
                'fedprox': 220.0,
                'fedopt': 220.0,
                'offsite-tuning': 175.0,
                'pfedme': 220.0,
                'ditto': 220.0,
                'kafa-full': 95.0,
                'kafa-w/o projection': 220.0,
                'kafa-w/o instruction': 95.0,
                'kafa-w/o distill': 95.0,
                'kafa-w/o relevance': 95.0,
            }.get(method_name.lower(), 150.0)
            rows.append({
                'task': task_name,
                'family': task.family,
                'method': method.name,
                'comm_mb_per_round': comm,
                'comm_mb_cumulative': comm * task.rounds,
                'rounds': task.rounds,
            })
    return pd.DataFrame(rows)


def render_report(real_df: pd.DataFrame,
                  summary_df: pd.DataFrame,
                  synthetic_final_df: pd.DataFrame,
                  communication_df: pd.DataFrame,
                  plot_paths: Optional[Dict[str, Path]],
                  output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines: List[str] = []
    report_lines.append('# KAFA Experiment Report')
    report_lines.append('')
    report_lines.append(
        'This report combines real log summaries with a clearly labeled '
        '`synthetic` sanity-check preview. Synthetic rows are for preflight '
        'trend validation only and must not be treated as paper results.')
    report_lines.append('')
    report_lines.append('## Real Runs')
    if real_df.empty:
        report_lines.append('- No real runs were found under the selected root.')
    else:
        report_lines.append(
            f'- Real runs found: {len(real_df)} rows across '
            f'{real_df["task"].nunique()} tasks and '
            f'{real_df["method"].nunique()} methods.')
        best = summary_df.groupby('task', as_index=False).first()
        for _, row in best.iterrows():
            report_lines.append(
                f"- `{row['task']}` best method so far: "
                f"`{display_method_name(str(row['method']))}` "
                f"with mean final test loss `{row['final_test_avg_loss_mean']:.4f}`"
            )
    report_lines.append('')
    report_lines.append('## Comparison Set')
    report_lines.append(
        '- `local-only`: checks the lower bound when each client keeps its own model.')
    report_lines.append(
        '- `RawFedAvg`: measures whether plain federated averaging can transfer useful knowledge without alignment.')
    report_lines.append(
        '- `FedProx`: adds proximal regularization to test whether a simple stability prior closes the gap.')
    report_lines.append(
        '- `FedOpt`: checks whether server-side adaptive optimization improves convergence over vanilla averaging.')
    report_lines.append(
        '- `OffsiteTuning`: isolates the effect of server-side emulator alignment and layer dropping.')
    report_lines.append(
        '- `pFedMe` and `Ditto`: personalized FL baselines that are often competitive on heterogeneous data.')
    report_lines.append(
        '- `KAFA-full`: combines projection, instruction anchoring, relevance filtering, and distillation.')
    report_lines.append(
        '- `w/o projection / instruction / distill / relevance`: ablations to identify which component drives gains.')
    report_lines.append('')
    report_lines.append('## Synthetic Preview')
    if synthetic_final_df.empty:
        report_lines.append('- Synthetic preview could not be generated.')
    else:
        for task_name in synthetic_final_df['task'].unique():
            task_rows = synthetic_final_df[synthetic_final_df['task'] ==
                                           task_name]
            top = task_rows.iloc[0]
            report_lines.append(
                f"- `{task_name}`: expected best method is "
                f"`{display_method_name(str(top['method']))}` "
                f"with final test loss around `{top['final_test_loss_mean']:.4f}` "
                f"and std `{top['final_test_loss_std']:.4f}`")
    report_lines.append('')
    report_lines.append('## Expected Ordering')
    report_lines.append(
        '- On `gsm8k`, the strongest expected gap is between `KAFA-full` and the ablations, especially `w/o instruction` and `w/o relevance`.')
    report_lines.append(
        '- On `alpaca` and `dolly-15k`, `KAFA-full` should still lead, but the main visible advantage is usually faster convergence and lower variance.')
    report_lines.append(
        '- On `squad`, `imdb`, and `agnews`, absolute gaps are often smaller, so communication cost and stability become part of the argument.')
    report_lines.append(
        '- If `w/o projection` beats `KAFA-full` consistently, treat that as a configuration or model-shape mismatch before trusting the numbers.')
    report_lines.append('')
    report_lines.append('## Plots')
    if plot_paths:
        for name, path in plot_paths.items():
            rel_path = _as_config_path(path.relative_to(output_dir)
                                        if path.is_relative_to(output_dir)
                                        else path)
            report_lines.append(f'- `{name}`: [{rel_path}]({rel_path})')
    else:
        report_lines.append('- No plots were generated.')
    report_lines.append('')
    report_lines.append('## Communication Estimate')
    for _, row in communication_df.iterrows():
        report_lines.append(
            f"- `{row['task']}` / `{display_method_name(str(row['method']))}`: "
            f"{row['comm_mb_per_round']:.1f} MB per round, "
            f"{row['comm_mb_cumulative']:.1f} MB total")
    report_lines.append('')
    report_lines.append('## Interpretation')
    report_lines.append(
        '- Reasoning tasks should benefit most from the full KAFA stack; '
        'instruction augmentation and relevance filtering should dominate.')
    report_lines.append(
        '- Generation tasks should show a smoother gap: KAFA should still '
        'lead, but convergence stability is the main advantage.')
    report_lines.append(
        '- Extraction and classification tasks usually have smaller absolute '
        'gains, so communication efficiency and variance reduction matter more.'
    )
    report_lines.append(
        '- If real logs reverse these trends, check model paths, dataset split '
        'logic, and the client projection / mapper settings first.')
    report_path = output_dir / 'EXPERIMENT_REPORT.md'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    return report_path


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


PLOT_METHOD_ORDER = [
    'local-only', 'rawfedavg', 'fedprox', 'fedopt', 'offsite-tuning',
    'pfedme', 'ditto', 'kafa-full', 'kafa-w/o projection',
    'kafa-w/o instruction', 'kafa-w/o distill', 'kafa-w/o relevance'
]

PLOT_COLORS = {
    'local-only': '#7a7a7a',
    'rawfedavg': '#4c78a8',
    'fedprox': '#72b7b2',
    'fedopt': '#1f9e89',
    'offsite-tuning': '#f58518',
    'pfedme': '#54a24b',
    'ditto': '#2e7d32',
    'kafa-full': '#d62728',
    'kafa-w/o projection': '#e45756',
    'kafa-w/o instruction': '#ff9d4d',
    'kafa-w/o distill': '#ffbf79',
    'kafa-w/o relevance': '#ff6f61',
}


def _prepare_plot_dir(output_dir: Path) -> Path:
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _plot_final_loss_bars(synthetic_final_df: pd.DataFrame,
                          output_path: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = [task for task in TASK_SPECS.keys()
             if task in set(synthetic_final_df['task'].unique())]
    if not tasks:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()
    for ax, task_name in zip(axes, tasks):
        task_rows = synthetic_final_df[synthetic_final_df['task'] == task_name]
        task_rows = task_rows.sort_values('final_test_loss_mean')
        labels = [display_method_name(m) for m in task_rows['method']]
        colors = [PLOT_COLORS.get(m.lower(), '#4c78a8')
                  for m in task_rows['method']]
        ax.bar(range(len(task_rows)), task_rows['final_test_loss_mean'],
               color=colors, width=0.75)
        ax.set_xticks(range(len(task_rows)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
        ax.set_title(task_name)
        ax.set_ylabel('Final test loss')
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(0, max(0.6, float(task_rows['final_test_loss_mean'].max()) +
                           0.06))

    for ax in axes[len(tasks):]:
        ax.axis('off')

    fig.suptitle('Synthetic final loss comparison by task', fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_convergence_curve(task_name: str, synthetic_curve_df: pd.DataFrame,
                            output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_rows = synthetic_curve_df[synthetic_curve_df['task'] == task_name]
    if task_rows.empty:
        return

    task_rows = task_rows.copy()
    method_order = [m for m in PLOT_METHOD_ORDER
                    if m in set(task_rows['method'].unique())]

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for method in method_order:
        method_rows = task_rows[task_rows['method'] == method]
        curve = method_rows.groupby('round', as_index=False).agg(
            mean_loss=('test_avg_loss', 'mean'),
            std_loss=('test_avg_loss', 'std'),
        ).sort_values('round')
        if curve.empty:
            continue
        color = PLOT_COLORS.get(method.lower(), '#4c78a8')
        ax.plot(curve['round'],
                curve['mean_loss'],
                label=display_method_name(method),
                color=color,
                linewidth=2)
        if 'std_loss' in curve and not curve['std_loss'].isna().all():
            upper = curve['mean_loss'] + curve['std_loss'].fillna(0.0)
            lower = curve['mean_loss'] - curve['std_loss'].fillna(0.0)
            ax.fill_between(curve['round'], lower, upper, color=color, alpha=0.12)

    ax.set_title(f'{task_name} synthetic convergence trend')
    ax.set_xlabel('Round')
    ax.set_ylabel('Test loss')
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc='upper right')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_tradeoff_scatter(synthetic_final_df: pd.DataFrame,
                           communication_df: pd.DataFrame,
                           output_path: Path) -> None:
    import matplotlib.pyplot as plt

    merged = synthetic_final_df[['task', 'family', 'method',
                                 'final_test_loss_mean']].merge(
                                     communication_df[['task', 'family',
                                                       'method',
                                                       'comm_mb_cumulative']],
                                     on=['task', 'family', 'method'],
                                     how='left')
    if merged.empty:
        return

    summary = merged.groupby('method', as_index=False).agg(
        mean_loss=('final_test_loss_mean', 'mean'),
        mean_comm=('comm_mb_cumulative', 'mean'),
    )

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    for _, row in summary.iterrows():
        method = row['method']
        color = PLOT_COLORS.get(method.lower(), '#4c78a8')
        ax.scatter(row['mean_comm'],
                   row['mean_loss'],
                   s=120,
                   color=color,
                   edgecolor='white',
                   linewidth=0.8)
        ax.text(row['mean_comm'] * 1.03 if row['mean_comm'] > 0 else 5,
                row['mean_loss'] + 0.002,
                display_method_name(method),
                fontsize=8)

    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_xlabel('Average cumulative communication (MB)')
    ax.set_ylabel('Mean final test loss')
    ax.set_title('Communication-efficiency trade-off on synthetic preview')
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_plots(synthetic_curve_df: pd.DataFrame,
                synthetic_final_df: pd.DataFrame,
                communication_df: pd.DataFrame,
                output_dir: Path) -> Dict[str, Path]:
    plot_dir = _prepare_plot_dir(output_dir)
    plot_paths: Dict[str, Path] = {}

    final_path = plot_dir / 'final_loss_by_task.png'
    _plot_final_loss_bars(synthetic_final_df, final_path)
    if final_path.exists():
        plot_paths['final_loss_by_task'] = final_path

    tradeoff_path = plot_dir / 'communication_tradeoff.png'
    _plot_tradeoff_scatter(synthetic_final_df, communication_df, tradeoff_path)
    if tradeoff_path.exists():
        plot_paths['communication_tradeoff'] = tradeoff_path

    for task_name in TASK_SPECS.keys():
        curve_path = plot_dir / f'convergence_{task_name}.png'
        _plot_convergence_curve(task_name, synthetic_curve_df, curve_path)
        if curve_path.exists():
            plot_paths[f'convergence_{task_name}'] = curve_path

    return plot_paths


def build_and_save_artifacts(tasks: Sequence[str],
                             methods: Sequence[str],
                             seeds: Sequence[int],
                             exp_root: Path = DEFAULT_OUTPUT_ROOT,
                             analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
                             data_root: str = _as_config_path(DEFAULT_DATA_ROOT),
                             server_model: str = default_server_model(),
                             client_model: str = default_client_model(),
                             write_configs: bool = True) -> Dict[str, Path]:
    plan = build_experiment_plan(tasks=tasks,
                                 methods=methods,
                                 seeds=seeds,
                                 data_root=data_root,
                                 output_root=_as_config_path(exp_root),
                                 server_model=server_model,
                                 client_model=client_model,
                                 write_configs=write_configs)
    save_frame(plan, analysis_root / 'experiment_plan.csv')
    real_df = collect_real_runs(exp_root)
    if not real_df.empty:
        task_filter = {task.lower() for task in tasks}
        real_df = real_df[real_df['task'].str.lower().isin(task_filter)]
    save_frame(real_df, analysis_root / 'real_runs.csv')
    summary_df = summarize_by_task_and_method(real_df)
    save_frame(summary_df, analysis_root / 'real_runs_summary.csv')
    synthetic_curve_df, synthetic_final_df = build_synthetic_preview(
        tasks=tasks, methods=methods, seeds=seeds)
    save_frame(synthetic_curve_df,
               analysis_root / 'synthetic_expectation_curve.csv')
    save_frame(synthetic_final_df,
               analysis_root / 'synthetic_expectation_final.csv')
    communication_df = build_communication_table(tasks, methods)
    save_frame(communication_df, analysis_root / 'communication_estimate.csv')
    plot_paths = build_plots(synthetic_curve_df=synthetic_curve_df,
                             synthetic_final_df=synthetic_final_df,
                             communication_df=communication_df,
                             output_dir=analysis_root)
    report_path = render_report(real_df=real_df,
                                summary_df=summary_df,
                                synthetic_final_df=synthetic_final_df,
                                communication_df=communication_df,
                                plot_paths=plot_paths,
                                output_dir=analysis_root)
    return {
        'plan': analysis_root / 'experiment_plan.csv',
        'real': analysis_root / 'real_runs.csv',
        'summary': analysis_root / 'real_runs_summary.csv',
        'synthetic_curve': analysis_root /
        'synthetic_expectation_curve.csv',
        'synthetic_final': analysis_root /
        'synthetic_expectation_final.csv',
        'communication': analysis_root / 'communication_estimate.csv',
        'plot_dir': analysis_root / 'plots',
        'report': report_path,
    }
