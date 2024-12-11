from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from loguru import logger
from diffusers.runtime_state import get_runtime_state

class CacheEntry:
    def __init__(
        self,
        cache_type: "str",
        num_cache_tensors: int = 1,
        tensors: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        self.cache_type: str = cache_type
        if tensors is None:
            self.tensors: List[torch.Tensor] = [None,] * num_cache_tensors
        elif isinstance(tensors, torch.Tensor):
            self.tensors = [tensors, ]
        elif isinstance(tensors, List):
            self.tensors = [tensors, ]

class CacheManager:
    def __init__(self):
        self.cache: Dict[Tuple[str, Any], CacheEntry] = {}

    def register_cache_entry(self, layer, layer_type: str, cache_type: str = "naive_cache"):
        self.cache[layer_type, layer] = CacheEntry(cache_type)

    def cache_update(
        self,
        new_kv: Union[torch.Tensor, List[torch.Tensor]],
        layer,
        slice_dim: int = 1,
        layer_type: str = "attn",
    ):
        return_list = False
        if isinstance(new_kv, List):
            return_list = True
            new_kv = torch.cat(new_kv, dim=-1)
        if get_runtime_state().num_pipeline_patch == 1 or not get_runtime_state().patch_mode:
            kv_cache = new_kv
            self.cache[layer_type, layer].tensors[0] = kv_cache
        else:
            start_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx
            ]
            end_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx + 1
            ]
            kv_cache = self.cache[layer_type, layer].tensors[0]
            kv_cache = self._update_kv_in_dim(
                kv_cache=kv_cache,
                new_kv=new_kv,
                dim=slice_dim,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
            )
            self.cache[layer_type, layer].tensors[0] = kv_cache
        if return_list:
            return torch.chunk(kv_cache, 2, dim=-1)
        else:
            return kv_cache
    
    def _update_kv_in_dim(
        self,
        kv_cache: torch.Tensor,
        new_kv: torch.Tensor,
        dim: int,
        start_idx: int,
        end_idx: int,
    ):
        if dim < 0:
            dim += kv_cache.dim()

        if dim == 0:
            kv_cache[start_idx:end_idx, ...] = new_kv
        elif dim == 1:
            kv_cache[:, start_idx:end_idx:, ...] = new_kv
        elif dim == 2:
            kv_cache[:, :, start_idx:end_idx, ...] = new_kv
        elif dim == 3:
            kv_cache[:, :, :, start_idx:end_idx, ...] = new_kv
        return kv_cache
    
_CACHE_MANAGER = CacheManager()

def get_cache_manager():
    global _CACHE_MANAGER
    if _CACHE_MANAGER is None:
        _CACHE_MANAGER = CacheManager()
    return _CACHE_MANAGER