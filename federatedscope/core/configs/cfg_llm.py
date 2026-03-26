import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN(new_allowed=True)
    cfg.llm.tok_len = 128
    cfg.llm.retry_on_nan_loss = False

    # Training the reward model
    cfg.llm.reward_coeff = 0.1

    # use gradient accumulation to enlarge the batch size
    # e.g., bsz = dataloader.batch_size * train.grad_accu_step
    cfg.llm.grad_accum_step = 1

    # ---------------------------------------------------------------------- #
    # Cache for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.cache = CN()
    cfg.llm.cache.model = ''

    # ---------------------------------------------------------------------- #
    # Chat tools for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100

    # ---------------------------------------------------------------------- #
    # Deepspeed related options
    # ---------------------------------------------------------------------- #
    cfg.llm.deepspeed = CN()
    cfg.llm.deepspeed.use = False
    cfg.llm.deepspeed.ds_config = ''

    # ---------------------------------------------------------------------- #
    # HuggingFace accelerator related options
    # ---------------------------------------------------------------------- #
    cfg.llm.accelerator = CN()
    cfg.llm.accelerator.use = False
    cfg.llm.accelerator.config = ''

    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.args = [{}]
    # Move adapter to `cpu` after training, which can save memory but cost
    # more time.
    cfg.llm.adapter.mv_to_cpu = False

    # ---------------------------------------------------------------------- #
    # Offsite-tuning related options
    # ---------------------------------------------------------------------- #
    cfg.llm.offsite_tuning = CN()
    cfg.llm.offsite_tuning.use = False
    cfg.llm.offsite_tuning.strategy = 'drop_layer'
    cfg.llm.offsite_tuning.kwargs = [{}]
    cfg.llm.offsite_tuning.emu_l = 1  # Index of emulator layer left
    cfg.llm.offsite_tuning.emu_r = 10  # Index of emulator layer right

    # Used in `eval`
    cfg.llm.offsite_tuning.eval_type = 'emu'  # Choose one of `[emu, full]`

    # Used in `aggregator`
    cfg.llm.offsite_tuning.save_full_model = False

    # Emulator alignment will use dataset in Server
    cfg.llm.offsite_tuning.emu_align = CN()
    cfg.llm.offsite_tuning.emu_align.use = False
    cfg.llm.offsite_tuning.emu_align.initial_only = True
    cfg.llm.offsite_tuning.emu_align.sim_loss = 'l2'  # Choose one of
    # `['cos', 'l2']`
    cfg.llm.offsite_tuning.emu_align.layerwise_distill = False
    cfg.llm.offsite_tuning.emu_align.kl_divergence = 'raw'  # Choose one of
    # `['raw', 'logps']`
    cfg.llm.offsite_tuning.emu_align.restore_from = ''
    cfg.llm.offsite_tuning.emu_align.save_to = ''
    cfg.llm.offsite_tuning.emu_align.exit_after_align = False

    # Server held-out data
    cfg.llm.offsite_tuning.emu_align.data = CN()
    cfg.llm.offsite_tuning.emu_align.data.root = 'data'
    cfg.llm.offsite_tuning.emu_align.data.type = 'alpaca@llm'
    cfg.llm.offsite_tuning.emu_align.data.splits = [0.8, 0.1, 0.1]

    cfg.llm.offsite_tuning.emu_align.train = CN()
    cfg.llm.offsite_tuning.emu_align.train.local_update_steps = 10
    cfg.llm.offsite_tuning.emu_align.train.initial_update_rounds = 50
    cfg.llm.offsite_tuning.emu_align.train.batch_or_epoch = 'batch'
    cfg.llm.offsite_tuning.emu_align.train.lm_loss_weight = 0.1
    cfg.llm.offsite_tuning.emu_align.train.kd_loss_weight = 0.9

    cfg.llm.offsite_tuning.emu_align.train.optimizer = CN(new_allowed=True)
    cfg.llm.offsite_tuning.emu_align.train.optimizer.type = 'SGD'
    cfg.llm.offsite_tuning.emu_align.train.optimizer.lr = 0.01

    # Overwrite clients' labels to LLM generated text
    cfg.llm.offsite_tuning.llm_generated = CN()
    cfg.llm.offsite_tuning.llm_generated.use = False
    cfg.llm.offsite_tuning.llm_generated.ratio = 0.1

    # ---------------------------------------------------------------------- #
    # Knowledge Anchor related options (FedKA方法)
    # ---------------------------------------------------------------------- #
    cfg.llm.knowledge_anchor = CN()
    # Server-side configuration
    cfg.llm.knowledge_anchor.num_instructions = 8  # K=8: Number of Alpaca-style instruction templates
    cfg.llm.knowledge_anchor.num_prototypes = 150  # M=150: Number of prototypes in the bank
    cfg.llm.knowledge_anchor.intermediate_layer = 16  # l=16: Layer index for hidden state extraction
    cfg.llm.knowledge_anchor.generation_temperature = 0.7  # Temperature for instruction generation
    cfg.llm.knowledge_anchor.max_new_tokens = 256  # Max tokens for generation
    cfg.llm.knowledge_anchor.soft_label_temperature = 4.0  # T=4.0: Temperature for soft label distillation
    cfg.llm.knowledge_anchor.task_description_seed = 42  # seed=42: Fixed random seed for task description inference
    cfg.llm.knowledge_anchor.task_description_samples = 100  # Number of instructions to sample for task description
    
    # Client-side configuration
    cfg.llm.knowledge_anchor.client_model_type = ''  # Client-side small model (e.g., 'tinyllama@huggingface_llm')
    cfg.llm.knowledge_anchor.task_relevance_threshold = 0.5  # Cosine similarity threshold for task relevance
    cfg.llm.knowledge_anchor.prototype_alignment_weight = 0.5  # λ=0.5: Weight for anchor loss in total loss
    cfg.llm.knowledge_anchor.use_instruction_augmentation = True  # Use instruction augmentation with [SEP] separator
    cfg.llm.knowledge_anchor.top_k_prototypes = 5  # K_p=5: Top-K prototypes for dynamic matching
    cfg.llm.knowledge_anchor.prototype_temperature = 0.1  # τ=0.1: Temperature for prototype soft weights
    cfg.llm.knowledge_anchor.distill_weight = 1.0  # γ=1.0: Weight for distillation loss in anchor loss
    cfg.llm.knowledge_anchor.enable_mse_relevance = True  # Use MSE-based relevance rule
    cfg.llm.knowledge_anchor.relevance_mse_multiplier = 1.5  # irrelevant if min_mse > 1.5 * avg_proto_dist
    cfg.llm.knowledge_anchor.incremental_mse_multiplier = 0.8  # add to incremental bank if min_mse >= 0.8 * avg
    
    # Server-side training
    cfg.llm.knowledge_anchor.train_on_server = True  # Whether to fine-tune on server data
    
    # LoRA mapping configuration
    cfg.llm.knowledge_anchor.lora_mapping_strategy = 'interpolate'  # 'direct', 'many_to_one', 'interpolate'
    cfg.llm.knowledge_anchor.max_prototypes = 100  # Max prototypes before reclustering
    cfg.llm.knowledge_anchor.use_projection_lora = True
    cfg.llm.knowledge_anchor.small_hidden = 2048
    cfg.llm.knowledge_anchor.large_hidden = 4096
    cfg.llm.knowledge_anchor.projection_learnable = True
    cfg.llm.knowledge_anchor.q_strategy = 'transpose'  # 'transpose', 'pseudo_inverse', 'learnable'
    cfg.llm.knowledge_anchor.lora_update_lr = 1.0

    # ---------------------------------------------------------------------- #
    # Runtime path hints for experiment orchestration
    # ---------------------------------------------------------------------- #
    cfg.llm.paths = CN()
    cfg.llm.paths.server_model = ''
    cfg.llm.paths.client_model = ''
    cfg.llm.paths.data_root = ''
    cfg.llm.paths.output_root = ''
    cfg.llm.paths.cache_dir = ''


def assert_llm_cfg(cfg):
    if cfg.llm.offsite_tuning.emu_align.use:
        if cfg.llm.offsite_tuning.emu_align.restore_from != '':
            logger.warning(
                'Enabling `restore_from` in offsite_tuning emulator '
                'alignment will skip training the emulator.')


register_config("llm", extend_llm_cfg)
