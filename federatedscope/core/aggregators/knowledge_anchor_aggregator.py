"""
Knowledge Anchor Aggregator for Server
鏈嶅姟鍣ㄧ鐭ヨ瘑閿氱偣鑱氬悎鍣細
1. 鑱氬悎瀹㈡埛绔笂浼犵殑LoRA鍙傛暟锛堥渶瑕佷粠灏忔ā鍨嬫槧灏勫埌澶фā鍨嬶級
2. 鏁村悎瀹㈡埛绔笂浼犵殑澧為噺bank
3. 鏇存柊鐭ヨ瘑閿氱偣锛堝師鍨嬪簱锛?

娉ㄦ剰锛氳繖鏄熀浜庨璁粌妯″瀷鐨勫井璋冨満鏅紝浣跨敤LoRA绛夊弬鏁伴珮鏁堝井璋冩柟娉?

鏂板鍔熻兘锛氭湇鍔″櫒绔姇褰盠oRA
- 鏈嶅姟鍣ㄧ淮鎶ゅ叏灞€澶oRA锛坔idden_size=4096锛?
- 閫氳繃鎶曞奖鐭╅樀P/Q瀹炵幇澶oRA 鈫?灏廘oRA鐨勫弻鍚戣浆鎹?
- 鎵€鏈夋姇褰遍€昏緫闆嗕腑鍦ㄦ湇鍔″櫒绔紝瀹㈡埛绔棤闇€淇敼
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import logging
import os
from typing import Dict, List, Optional

from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.configs.cfg_llm import (
    _infer_small_kv_out_from_client_type,
    _normalize_knowledge_anchor_client_model_type,
)
from federatedscope.llm.misc.lora_mapper import LoRAMapper, create_lora_mapper
from federatedscope.llm.misc.projection_mapper import (
    ProjectionMapper,
    create_projection_mapper,
    infer_kv_out_dim,
)

logger = logging.getLogger(__name__)


def _resolve_kv_out_dims(config, large_model):
    """
    鎺ㄦ柇澶?灏忔ā鍨嬩笂 v_proj銆乲_proj 鐨?LoRA lora_B 琛岀淮锛圞/V 杈撳嚭缁达級銆?
    鑻ラ厤缃?projection_*_kv_out > 0 鍒欒鐩栬嚜鍔ㄦ帹鏂€?
    """
    ka = config.llm.knowledge_anchor
    large_override = getattr(ka, 'projection_large_kv_out', 0) or 0
    small_override = getattr(ka, 'projection_small_kv_out', 0) or 0

    if large_model is not None and hasattr(large_model, 'config'):
        large_kv = int(large_override) if int(large_override) > 0 else \
            infer_kv_out_dim(large_model.config)
    else:
        large_h = getattr(ka, 'large_hidden', 4096)
        large_kv = int(large_override) if int(large_override) > 0 else int(large_h)

    small_kv = None
    if int(small_override) > 0:
        small_kv = int(small_override)
    else:
        client_type = getattr(ka, 'client_model_type', '') or ''
        if client_type.strip():
            small_kv = _infer_small_kv_out_from_client_type(config, client_type)
        small_h = getattr(ka, 'small_hidden', 2048)
        if small_kv is None:
            small_kv = int(small_h)
            logger.warning(
                f"small_kv_out fallback to small_hidden={small_h} (MHA assumption). "
                f"If clients use GQA (e.g., TinyLlama), please set "
                f"llm.knowledge_anchor.projection_small_kv_out explicitly "
                f"or configure llm.knowledge_anchor.client_model_type.")

    return large_kv, small_kv


class KnowledgeAnchorAggregator(Aggregator):
    """
    鐭ヨ瘑閿氱偣鑱氬悎鍣?
    澶勭悊LoRA鑱氬悎鍜屽閲廱ank鏁村悎
    """
    def __init__(self, model=None, device='cpu', config=None, small_model=None):
        """
        Args:
            model: 鏈嶅姟鍣ㄧ澶фā鍨嬶紙Llama-7B锛?
            device: 璁惧
            config: 閰嶇疆瀵硅薄
            small_model: 瀹㈡埛绔皬妯″瀷锛堢敤浜嶭oRA鏄犲皠锛屽彲閫夛級
        """
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        
        # 鎶曞奖鏄犲皠鍣紙鏈嶅姟鍣ㄧ鎶曞奖LoRA锛?
        self.projection_mapper = None
        use_projection = getattr(config.llm.knowledge_anchor, 'use_projection_lora', True)
        if use_projection and model is not None:
            try:
                # 浠庨厤缃垨妯″瀷鑾峰彇hidden_size
                small_hidden = getattr(config.llm.knowledge_anchor, 'small_hidden', 2048)
                large_hidden = getattr(config.llm.knowledge_anchor, 'large_hidden', None)
                if large_hidden is None:
                    # 灏濊瘯浠庢ā鍨媍onfig鑾峰彇
                    if hasattr(model, 'config'):
                        large_hidden = getattr(model.config, 'hidden_size', 4096)
                    else:
                        large_hidden = 4096
                
                rank = getattr(config.train, 'lora_rank', 8)
                learnable = getattr(config.llm.knowledge_anchor, 'projection_learnable', True)
                q_strategy = getattr(config.llm.knowledge_anchor, 'q_strategy', 'transpose')
                large_kv_out, small_kv_out = _resolve_kv_out_dims(config, model)

                self.projection_mapper = create_projection_mapper(
                    small_hidden=small_hidden,
                    large_hidden=large_hidden,
                    rank=rank,
                    learnable=learnable,
                    q_strategy=q_strategy,
                    device=device,
                    large_kv_out=large_kv_out,
                    small_kv_out=small_kv_out)
                self._restore_or_save_projection_mapper()
                logger.info(
                    f"鉁?宸插垵濮嬪寲鎶曞奖鏄犲皠鍣? small_hidden={small_hidden}, "
                    f"large_hidden={large_hidden}, rank={rank}, "
                    f"large_kv_out={large_kv_out}, small_kv_out={small_kv_out}")
            except Exception as e:
                logger.warning(f"鍒涘缓鎶曞奖鏄犲皠鍣ㄥけ璐? {e}锛屽皢浣跨敤浼犵粺LoRA鏄犲皠")
                self.projection_mapper = None
        
        # LoRA鏄犲皠鍣紙浼犵粺鏂规硶锛屼綔涓哄洖閫€锛?
        self.lora_mapper = None
        if self.projection_mapper is None and small_model is not None and model is not None:
            try:
                self.lora_mapper = create_lora_mapper(
                    small_model, 
                    model, 
                    mapping_strategy=getattr(config.llm.knowledge_anchor, 'lora_mapping_strategy', 'interpolate')
                )
            except Exception as e:
                logger.warning(f"鍒涘缓LoRA鏄犲皠鍣ㄥけ璐? {e}")

        if self.projection_mapper is None and self.lora_mapper is None and \
                model is not None:
            client_model_type = getattr(config.llm.knowledge_anchor,
                                        'client_model_type', '')
            if client_model_type:
                try:
                    tmp_cfg = config.clone()
                    try:
                        tmp_cfg.defrost()
                    except Exception:
                        pass
                    tmp_cfg.model.type = _normalize_knowledge_anchor_client_model_type(client_model_type)
                    inferred_small_model = get_model(tmp_cfg, backend=config.backend)
                    self.lora_mapper = create_lora_mapper(
                        inferred_small_model,
                        model,
                        mapping_strategy=getattr(
                            config.llm.knowledge_anchor,
                            'lora_mapping_strategy', 'interpolate'))
                    logger.info(
                        "Initialized LoRA mapper from client_model_type in config."
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize LoRA mapper from client_model_type: {e}")
        
        # 鍏ㄥ眬澶oRA state_dict锛堟湇鍔″櫒缁存姢锛?
        self.global_large_lora = None
        
        # 鐭ヨ瘑閿氱偣鐩稿叧
        self.knowledge_anchor = None
        self.incremental_bank_buffer = []  # 涓存椂瀛樺偍瀹㈡埛绔笂浼犵殑澧為噺bank

    def _projection_cache_path(self):
        outdir = getattr(self.cfg, 'outdir', 'exp')
        return os.path.join(outdir, 'knowledge_anchor_cache',
                            'projection_mapper.pt')

    def _restore_or_save_projection_mapper(self):
        if self.projection_mapper is None:
            return
        cache_path = self._projection_cache_path()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        meta = {
            'small_hidden': self.projection_mapper.small_hidden,
            'large_hidden': self.projection_mapper.large_hidden,
            'rank': self.projection_mapper.rank,
            'q_strategy': self.projection_mapper.q_strategy,
            'large_kv_out': self.projection_mapper.large_kv_out,
            'small_kv_out': self.projection_mapper.small_kv_out,
        }
        if os.path.exists(cache_path):
            try:
                payload = torch.load(cache_path, map_location=self.device)
                if payload.get('meta') == meta:
                    self.projection_mapper.load_state_dict(
                        payload['state_dict'])
                    logger.info(f"鉁?宸蹭粠缂撳瓨鍔犺浇鎶曞奖鐭╅樀: {cache_path}")
                    return
                logger.warning(
                    "Projection mapper cache meta mismatch, regenerating mapper cache.")
            except Exception as e:
                logger.warning(f"鍔犺浇鎶曞奖鐭╅樀缂撳瓨澶辫触: {e}锛屽皢閲嶆柊淇濆瓨")
        torch.save({
            'meta': meta,
            'state_dict': self.projection_mapper.state_dict()
        }, cache_path)
        logger.info(f"鉁?宸蹭繚瀛樻姇褰辩煩闃电紦瀛? {cache_path}")

    def set_knowledge_anchor(self, knowledge_anchor: Dict):
        """璁剧疆鐭ヨ瘑閿氱偣"""
        self.knowledge_anchor = knowledge_anchor
    
    def get_global_large_lora(self) -> Dict[str, torch.Tensor]:
        """
        鑾峰彇鏈嶅姟鍣ㄥ叏灞€澶oRA state_dict
        
        Returns:
            Dict: 澶фā鍨婰oRA state_dict
        """
        if self.global_large_lora is not None:
            return self.global_large_lora
        
        # 浠嶱EFT妯″瀷鑾峰彇LoRA鍙傛暟
        if self.model is not None:
            try:
                from peft import PeftModel
                if isinstance(self.model, PeftModel):
                    # 鑾峰彇PEFT state_dict锛堝彧鍖呭惈LoRA鍙傛暟锛?
                    self.global_large_lora = self.model.get_peft_state_dict()
                    logger.info(
                        f"Loaded global large LoRA from PEFT model with "
                        f"{len(self.global_large_lora)} parameters.")
                else:
                    # 濡傛灉涓嶆槸PEFT妯″瀷锛屽皾璇曚粠state_dict涓彁鍙朙oRA鍙傛暟
                    full_state_dict = self.model.state_dict()
                    self.global_large_lora = self._extract_lora_params(full_state_dict)
                    logger.info(
                        f"Extracted global large LoRA from model state_dict "
                        f"with {len(self.global_large_lora)} parameters.")
            except Exception as e:
                logger.warning(f"鑾峰彇鍏ㄥ眬澶oRA澶辫触: {e}")
                self.global_large_lora = {}
        
        return self.global_large_lora if self.global_large_lora is not None else {}
    
    def broadcast_small_lora(self) -> Dict[str, torch.Tensor]:
        """
        骞挎挱灏廘oRA锛堜笅鎶曞奖锛氬ぇLoRA 鈫?灏廘oRA锛?
        
        鍦ㄦ湇鍔″櫒broadcast鍓嶈皟鐢紝灏嗗叏灞€澶oRA鎶曞奖涓哄皬LoRA骞惰繑鍥?
        
        Returns:
            Dict: 鎶曞奖鍚庣殑灏忔ā鍨婰oRA state_dict锛堢敤浜巄roadcast缁欏鎴风锛?
        """
        if self.projection_mapper is None:
            logger.warning(
                "Projection mapper is not initialized; skip down projection.")
            return {}
        
        # 鑾峰彇鍏ㄥ眬澶oRA
        large_lora = self.get_global_large_lora()
        if not large_lora:
            logger.warning("鍏ㄥ眬澶oRA涓虹┖锛屾棤娉曡繘琛屼笅鎶曞奖")
            return {}
        
        # 涓嬫姇褰憋細澶oRA 鈫?灏廘oRA
        small_lora = self.projection_mapper.down_project(large_lora)
        logger.info(
            f"Down projection complete: large LoRA ({len(large_lora)} params) "
            f"-> small LoRA ({len(small_lora)} params).")
        
        return small_lora

    def aggregate(self, agg_info):
        """
        鑱氬悎瀹㈡埛绔洿鏂?
        
        Args:
            agg_info: 鍖呭惈瀹㈡埛绔弽棣堢殑瀛楀吀
                - client_feedback: List of (sample_size, model_params, incremental_bank)
                - recover_fun: 鎭㈠鍑芥暟锛堢敤浜庣瀵嗗叡浜紝鍙€夛級
                
        Returns:
            Dict: 鑱氬悎鍚庣殑妯″瀷鍙傛暟
        """
        models = agg_info["client_feedback"]
        recover_fun = agg_info.get('recover_fun', None)
        
        # 鍒嗙LoRA鍙傛暟鍜屽閲廱ank
        lora_updates = []
        incremental_banks = []
        client_weights = []
        
        for i, feedback in enumerate(models):
            if isinstance(feedback, tuple):
                if len(feedback) == 3:
                    # (sample_size, model_params, incremental_bank)
                    sample_size, model_params, incremental_bank = feedback
                elif len(feedback) == 2:
                    # (sample_size, model_params)
                    sample_size, model_params = feedback
                    incremental_bank = None
                else:
                    raise ValueError(f"Unexpected feedback format: {len(feedback)} elements")
            else:
                sample_size = 1
                model_params = feedback
                incremental_bank = None
            
            # 鎻愬彇LoRA鍙傛暟
            lora_params = self._extract_lora_params(model_params)
            
            # 濡傛灉闇€瑕佹槧灏勶紝杩涜鏄犲皠
            if self.lora_mapper is not None:
                try:
                    mapped_lora = self.lora_mapper.map_lora_weights(lora_params)
                    lora_updates.append((sample_size, mapped_lora))
                except Exception as e:
                    logger.warning(f"LoRA鏄犲皠澶辫触锛屼娇鐢ㄥ師濮嬪弬鏁? {e}")
                    lora_updates.append((sample_size, lora_params))
            else:
                lora_updates.append((sample_size, lora_params))
            
            # 鏀堕泦澧為噺bank
            if incremental_bank is not None:
                incremental_banks.append(incremental_bank)
            
            client_weights.append(sample_size)
        
        # 鑱氬悎LoRA鍙傛暟锛堟鏃舵槸灏廘oRA锛?
        aggregated_small_lora = self._aggregate_loras(lora_updates, client_weights, recover_fun)
        
        # 鏁村悎澧為噺bank
        if incremental_banks:
            self._integrate_incremental_banks(incremental_banks)
        
        # 濡傛灉浣跨敤鎶曞奖鏄犲皠锛岃繘琛屼笂鎶曞奖骞舵洿鏂板叏灞€澶oRA
        if self.projection_mapper is not None and aggregated_small_lora:
            try:
                # 涓婃姇褰憋細灏廘oRA 鈫?澶oRA
                large_lora_delta = self.projection_mapper.up_project(aggregated_small_lora)
                logger.info(f"鉁?涓婃姇褰卞畬鎴? 灏廘oRA({len(aggregated_small_lora)}涓弬鏁? -> "
                           f"澶oRA澧為噺({len(large_lora_delta)}涓弬鏁?")
                
                # 鏇存柊鍏ㄥ眬澶oRA
                self._update_global_large_lora(large_lora_delta)
                
                # 杩斿洖鏇存柊鍚庣殑澶oRA锛堢敤浜庢洿鏂版湇鍔″櫒妯″瀷锛?
                return self.get_global_large_lora()
            except Exception as e:
                logger.warning(f"涓婃姇褰卞け璐? {e}锛岃繑鍥炶仛鍚堝悗鐨勫皬LoRA")
                return aggregated_small_lora
        else:
            # 涓嶄娇鐢ㄦ姇褰辨槧灏勶紝鐩存帴杩斿洖鑱氬悎缁撴灉
            return aggregated_small_lora

    def _extract_lora_params(self, model_params: Dict) -> Dict:
        """鎻愬彇LoRA鍙傛暟"""
        lora_params = {}
        for key, value in model_params.items():
            if 'lora' in key.lower() or 'adapter' in key.lower():
                lora_params[key] = value
        return lora_params

    def _aggregate_loras(self, lora_updates: List, client_weights: List, recover_fun=None) -> Dict:
        """
        鑱氬悎LoRA鍙傛暟锛堝姞鏉冨钩鍧囷級
        
        Args:
            lora_updates: List of (sample_size, lora_params)
            client_weights: 瀹㈡埛绔潈閲嶅垪琛?
            recover_fun: 鎭㈠鍑芥暟锛堝彲閫夛級
            
        Returns:
            Dict: 鑱氬悎鍚庣殑LoRA鍙傛暟
        """
        total_weight = sum(client_weights)
        
        # 鑾峰彇鎵€鏈塳ey
        all_keys = set()
        for _, lora_params in lora_updates:
            all_keys.update(lora_params.keys())
        
        # 鍔犳潈鑱氬悎
        aggregated = {}
        for key in all_keys:
            weighted_sum = None
            for i, (sample_size, lora_params) in enumerate(lora_updates):
                if key in lora_params:
                    weight = client_weights[i] / total_weight if total_weight > 0 else 1.0 / len(lora_updates)
                    param = param2tensor(lora_params[key])
                    
                    if recover_fun:
                        param = recover_fun(param)
                    
                    if weighted_sum is None:
                        weighted_sum = param * weight
                    else:
                        weighted_sum += param * weight
            
            if weighted_sum is not None:
                aggregated[key] = weighted_sum
        
        return aggregated

    def _integrate_incremental_banks(self, incremental_banks: List[List[Dict]]):
        """
        鏁村悎瀹㈡埛绔笂浼犵殑澧為噺bank
        
        Args:
            incremental_banks: List of incremental banks from clients
        """
        # 鏀堕泦鎵€鏈夋柊鍘熷瀷
        all_new_prototypes = []
        for bank in incremental_banks:
            if isinstance(bank, list):
                all_new_prototypes.extend(bank)
        
        if not all_new_prototypes:
            return False
        
        logger.info(f"鏀跺埌 {len(all_new_prototypes)} 涓柊鍘熷瀷锛屽紑濮嬫暣鍚?..")
        
        # 鎻愬彇鍘熷瀷鍚戦噺
        prototype_vectors = []
        for entry in all_new_prototypes:
            if isinstance(entry, dict) and 'prototype' in entry:
                prototype_vectors.append(entry['prototype'])
            elif isinstance(entry, torch.Tensor):
                prototype_vectors.append(entry)
        
        if not prototype_vectors:
            return False
        
        # 杞崲涓簍ensor
        prototype_tensor = torch.stack(
            [param2tensor(x).float().cpu() for x in prototype_vectors]
        )  # [num_new, hidden_dim]
        
        # 涓庣幇鏈夊師鍨嬪簱鍚堝苟
        if self.knowledge_anchor and 'prototype_bank' in self.knowledge_anchor:
            existing_prototypes = self.knowledge_anchor['prototype_bank'].get('prototypes', None)
            
            if existing_prototypes is not None:
                if not isinstance(existing_prototypes, torch.Tensor):
                    existing_prototypes = torch.tensor(existing_prototypes)
                existing_prototypes = existing_prototypes.float().cpu()
                # 鍚堝苟鍘熷瀷
                combined_prototypes = torch.cat([existing_prototypes, prototype_tensor], dim=0)
                
                # 閲嶆柊鑱氱被锛堝彲閫夛細濡傛灉鍘熷瀷鏁伴噺杩囧锛屽彲浠ラ噸鏂拌仛绫伙級
                max_prototypes = max(
                    getattr(self.cfg.llm.knowledge_anchor, 'max_prototypes', 100),
                    int(existing_prototypes.shape[0]))
                if len(combined_prototypes) > max_prototypes:
                    logger.info(f"鍘熷瀷鏁伴噺瓒呰繃{max_prototypes}锛岃繘琛岄噸鏂拌仛绫?..")
                    combined_prototypes = self._recluster_prototypes(combined_prototypes, max_prototypes)
                    self.knowledge_anchor['prototype_bank']['soft_labels'] = None
                
                # 鏇存柊鐭ヨ瘑閿氱偣
                self.knowledge_anchor['prototype_bank']['prototypes'] = combined_prototypes
                self.knowledge_anchor['prototype_bank']['num_prototypes'] = len(combined_prototypes)
                self.knowledge_anchor['prototype_bank']['avg_prototype_distance'] = \
                    self._compute_avg_pairwise_mse(combined_prototypes)
                
                logger.info(f"鍘熷瀷搴撳凡鏇存柊锛屽綋鍓嶅師鍨嬫暟: {len(combined_prototypes)}")
                return True
            else:
                # 濡傛灉娌℃湁鐜版湁鍘熷瀷锛岀洿鎺ヤ娇鐢ㄦ柊鍘熷瀷
                self.knowledge_anchor['prototype_bank']['prototypes'] = prototype_tensor
                self.knowledge_anchor['prototype_bank']['num_prototypes'] = len(prototype_tensor)
                self.knowledge_anchor['prototype_bank']['avg_prototype_distance'] = \
                    self._compute_avg_pairwise_mse(prototype_tensor)
                return True
        return False

    def _recluster_prototypes(self, prototypes: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        閲嶆柊鑱氱被鍘熷瀷锛堝綋鍘熷瀷鏁伴噺杩囧鏃讹級
        
        Args:
            prototypes: 鍘熷瀷tensor [num_prototypes, hidden_dim]
            num_clusters: 鐩爣鑱氱被鏁?
            
        Returns:
            torch.Tensor: 鑱氱被鍚庣殑鍘熷瀷涓績 [num_clusters, hidden_dim]
        """
        prototypes_np = prototypes.float().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(prototypes_np)
        
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return cluster_centers

    @staticmethod
    def _compute_avg_pairwise_mse(prototypes: torch.Tensor) -> float:
        if prototypes is None or prototypes.shape[0] <= 1:
            return 0.0
        prototypes = prototypes.float()
        diff = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
        mse = (diff * diff).mean(dim=-1)
        mask = ~torch.eye(mse.shape[0], dtype=torch.bool, device=mse.device)
        vals = mse[mask]
        return float(vals.mean().item()) if vals.numel() > 0 else 0.0

    def _update_global_large_lora(self, large_lora_delta: Dict[str, torch.Tensor]):
        """
        鏇存柊鍏ㄥ眬澶oRA锛堢疮鍔犲閲忥級
        
        Args:
            large_lora_delta: 澶oRA澧為噺锛堜粠涓婃姇褰卞緱鍒帮級
        """
        # 鑾峰彇褰撳墠鍏ㄥ眬澶oRA
        current_large_lora = self.get_global_large_lora()
        
        # Client LoRA parameters are absolute weights after local training, not
        # round deltas. Treat the up-projected aggregate as the new target and
        # optionally smooth it with an EMA to avoid hetero-projection shocks.
        lr = float(getattr(self.cfg.llm.knowledge_anchor, 'lora_update_lr',
                           1.0))
        lr = min(1.0, max(0.0, lr))
        
        for key, target in large_lora_delta.items():
            target = param2tensor(target).detach()
            if not torch.isfinite(target).all():
                logger.warning("Skip non-finite projected LoRA parameter: %s",
                               key)
                continue
            if key in current_large_lora:
                current = param2tensor(current_large_lora[key]).detach()
                if current.shape != target.shape:
                    logger.warning(
                        "Skip projected LoRA parameter with shape mismatch: "
                        "%s current=%s target=%s", key, tuple(current.shape),
                        tuple(target.shape))
                    continue
                target = target.to(device=current.device, dtype=current.dtype)
                current_large_lora[key] = current + lr * (target - current)
            else:
                current_large_lora[key] = target.clone()
        
        # 鏇存柊鍏ㄥ眬澶oRA
        self.global_large_lora = current_large_lora
        
        # 濡傛灉妯″瀷鏄疨EFT妯″瀷锛屽悓姝ユ洿鏂版ā鍨嬪弬鏁?
        if self.model is not None:
            try:
                from peft import PeftModel
                if isinstance(self.model, PeftModel):
                    # 鍙洿鏂癓oRA鍙傛暟锛屼笉褰卞搷base model
                    self.model.load_state_dict(current_large_lora, strict=False)
                    logger.debug("鉁?宸插悓姝ユ洿鏂癙EFT妯″瀷鐨凩oRA鍙傛暟")
            except Exception as e:
                logger.warning(f"鍚屾鏇存柊妯″瀷鍙傛暟澶辫触: {e}")
    
    def update(self, model_parameters):
        """
        鏇存柊妯″瀷鍙傛暟
        
        濡傛灉浣跨敤鎶曞奖鏄犲皠锛宮odel_parameters搴旇鏄ぇLoRA state_dict
        鍚﹀垯锛岀洿鎺ュ姞杞藉埌妯″瀷
        """
        if self.model is not None:
            if self.projection_mapper is not None:
                # 浣跨敤鎶曞奖鏄犲皠鏃讹紝鏇存柊鍏ㄥ眬澶oRA
                if isinstance(model_parameters, dict):
                    # 妫€鏌ユ槸鍚︽槸澶oRA鍙傛暟
                    is_large_lora = any('lora' in k.lower() for k in model_parameters.keys())
                    if is_large_lora:
                        self.global_large_lora = model_parameters
                        # 鍚屾鍒癙EFT妯″瀷
                        try:
                            from peft import PeftModel
                            if isinstance(self.model, PeftModel):
                                self.model.load_state_dict(model_parameters, strict=False)
                        except:
                            pass
                    else:
                        # 涓嶆槸LoRA鍙傛暟锛屾甯稿姞杞?
                        self.model.load_state_dict(model_parameters, strict=False)
                else:
                    self.model.load_state_dict(model_parameters, strict=False)
            else:
                # 涓嶄娇鐢ㄦ姇褰辨槧灏勶紝姝ｅ父鍔犺浇
                self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        """淇濆瓨妯″瀷"""
        assert self.model is not None
        
        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        """鍔犺浇妯″瀷"""
        assert self.model is not None
        
        import os
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt.get('cur_round', -1)
        else:
            raise ValueError(f"The file {path} does NOT exist")

