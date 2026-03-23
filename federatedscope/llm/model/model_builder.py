from federatedscope.llm.model.adapter_builder import AdapterModel
import torch


def get_model_from_huggingface(model_name, config, **kwargs):
    from transformers import AutoModelForCausalLM
    import os

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    if config.train.is_enable_half:
        kwargs['torch_dtype'] = torch.bfloat16

    # 如果是本地路径，尝试使用本地文件
    if os.path.exists(model_name) and os.path.isdir(model_name):
        # 检查是否有必要的模型文件
        required_files = ['config.json']
        has_required = all(os.path.exists(os.path.join(model_name, f)) for f in required_files)
        if has_required:
            # 对于本地模型，尝试多种加载方式以处理GQA等特殊架构
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                # 方法1：先加载config，然后使用config加载模型
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(model_name, local_files_only=True)
                
                # 检查是否是GQA架构（num_key_value_heads < num_attention_heads）
                num_kv_heads = getattr(model_config, 'num_key_value_heads', None)
                num_heads = getattr(model_config, 'num_attention_heads', None)
                if num_kv_heads and num_heads and num_kv_heads < num_heads:
                    logger.info(f"检测到GQA架构: num_attention_heads={num_heads}, num_key_value_heads={num_kv_heads}")
                
                # 尝试使用config加载模型
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=model_config,
                        **kwargs
                    )
                except RuntimeError as e:
                    if 'size mismatch' in str(e) or 'mismatched' in str(e).lower():
                        # 如果是尺寸不匹配（GQA问题），使用ignore_mismatched_sizes
                        logger.warning(f"检测到尺寸不匹配（可能是GQA架构问题），使用ignore_mismatched_sizes=True")
                        return AutoModelForCausalLM.from_pretrained(
                            model_name,
                            config=model_config,
                            ignore_mismatched_sizes=True,
                            **kwargs
                        )
                    else:
                        raise
            except Exception as e:
                logger.warning(f"使用config加载失败: {e}")
                # 方法2：回退到正常加载（可能从HuggingFace Hub缓存加载，会自动处理架构）
                logger.warning("尝试正常加载（可能从HuggingFace Hub缓存，会自动处理架构）")
                try:
                    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                except Exception as e2:
                    # 方法3：最后尝试使用ignore_mismatched_sizes
                    logger.warning(f"正常加载也失败: {e2}")
                    logger.warning("尝试使用ignore_mismatched_sizes=True作为最后手段")
                    return AutoModelForCausalLM.from_pretrained(
                        model_name,
                        ignore_mismatched_sizes=True,
                        **kwargs
                    )

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

    # Resize LLM model based on settings
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
