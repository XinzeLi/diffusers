import torch.distributed
import random
from typing import List, Optional, Tuple
from abc import ABCMeta
from argparse import Namespace

import numpy as np
import torch
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.parallel_state import (
    get_pp_group,
    get_sp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)

class RuntimeState(metaclass=ABCMeta):
    config: Namespace
    num_pipeline_patch: int

    def __init__(self, config):
        self.config = config
        self.num_pipeline_patch = config.pipefusion_parallel_degree
        self.warmup_steps = 1

class DiTRuntimeState(RuntimeState):
    patch_mode: bool
    pipeline_patch_idx: int
    vae_scale_factor: int
    backbone_patch_size: int
    pp_patches_height: Optional[List[int]]
    pp_patches_start_idx_local: Optional[List[int]]
    pp_patches_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_start_idx_local: Optional[List[int]]
    pp_patches_token_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_num: Optional[List[int]]
    max_condition_sequence_length: int

    def __init__(self, config):
        super().__init__(config)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self._set_model_parameters(
            # vae_scale_factor=pipeline.vae_scale_factor,
            # backbone_patch_size=pipeline.transformer.config.patch_size,
            # backbone_in_channel=pipeline.transformer.config.in_channels,
            # backbone_inner_dim=pipeline.transformer.config.num_attention_heads
            # * pipeline.transformer.config.attention_head_dim,
            vae_scale_factor=8,
            backbone_patch_size=2,
            backbone_in_channel=16,
            backbone_inner_dim=24 * 64
        )
        self._calc_patches_metadata()

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0

    def set_patch_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0
    
    def _set_model_parameters(
        self,
        vae_scale_factor:int,
        backbone_patch_size:int,
        backbone_inner_dim:int,
        backbone_in_channel:int,
    ):
        self.vae_scale_factor = vae_scale_factor
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

    def _input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        self.config.height = height or self.config.height
        self.config.width = width or self.config.width
        self._calc_patches_metadata()
        self._reset_recv_buffer()

    def _calc_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor = self.vae_scale_factor
        latents_height = self.config.height // vae_scale_factor
        latents_width = self.config.width // vae_scale_factor # xinze: for 1024 width and 8 vae factor, latents width is 128

        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch

        num_pipeline_patch = (
            latents_height + pipeline_patches_height - 1
        ) // pipeline_patches_height

        pipeline_patches_height_list = [
            pipeline_patches_height for _ in range(num_pipeline_patch-1)
        ]

        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]# xinze: if using sp, then patch will be deeper patched.
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]# xinze: for height 1024, pp=sp=2, it is [0, 32 ,64, 96, 128]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]# xinze: for height 1024, pp=sp=2, it is [[32, 32], [32, 32]]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]# xinze: for height 1024, pp=sp=2, it is [[0, 32, 64], [64, 96, 128]]
        pp_patches_height = [
            sp_patches_height[sp_patch_idx]
            for sp_patches_height in pp_sp_patches_height
        ]# xinze: for height 1024, pp=sp=2, it is [32, 32]
        pp_patches_start_idx_local = [0] + [
            sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
        ]# xinze: for height 1024, pp=sp=2, it is [0, 32, 64]
        pp_patches_start_end_idx_global = [
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]# xinze: for height 1024, pp=sp=2, it is [[0, 32], [64, 96]] for rank 0 and [[32, 64], [96, 128]] for rank 1
        
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]# xinze: it is [[0, 1024], [2048, 3072]] for rank 0 and [[1024, 2048], [3072, 4096]] for rank 1

        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]# xinze: it is [1024, 1024]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]# xinze: it is [0, 1024, 2048]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )

        self.pp_patches_token_num = pp_patches_token_num


    def _reset_recv_buffer(self):
        get_pp_group().reset_buffer()
        get_pp_group.set_config(self, dtype=torch.float16)

def initialize_runtime_state(engine_config):
    global _RUNTIME

    _RUNTIME = DiTRuntimeState(config=engine_config)

def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME