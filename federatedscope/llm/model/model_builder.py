import logging
import os

import torch

from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)


def get_model_from_huggingface(model_name, config, **kwargs):
    from transformers import AutoConfig, AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    if config.train.is_enable_half:
        kwargs['torch_dtype'] = torch.bfloat16

    kwargs.setdefault('trust_remote_code', True)

    if os.path.exists(model_name) and os.path.isdir(model_name):
        config_path = os.path.join(model_name, 'config.json')
        if os.path.exists(config_path):
            model_config = AutoConfig.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=kwargs.get('trust_remote_code', True))
            num_kv_heads = getattr(model_config, 'num_key_value_heads', None)
            num_heads = getattr(model_config, 'num_attention_heads', None)
            if num_kv_heads and num_heads and num_kv_heads < num_heads:
                logger.info(
                    "Detected GQA model: num_attention_heads=%s, "
                    "num_key_value_heads=%s", num_heads, num_kv_heads)

            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=model_config,
                    **kwargs)
            except Exception as error:
                message = str(error)
                if 'size mismatch' in message or 'mismatched' in message.lower():
                    raise RuntimeError(
                        "Model weights do not match the instantiated "
                        "architecture. KAFA experiments must not use "
                        "ignore_mismatched_sizes=True because that randomly "
                        "initializes attention weights. Upgrade transformers "
                        "so this model's GQA layout is supported, or use a "
                        "client model without GQA.") from error
                raise

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_model_from_modelscope(model_name, config, **kwargs):
    from modelscope import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config, **kwargs):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config,
                                           **kwargs)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          config=config,
                                          **kwargs)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    return model
