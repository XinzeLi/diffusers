import torch
from torch import nn
from torch.nn import functional as F
from diffusers.runtime_state import get_runtime_state
from diffusers.parallel_state import get_pipeline_parallel_world_size, get_sequence_parallel_world_size
from loguru import logger

class CustomConv2d(nn.Module):
    def __init__(
        self, conv2d: nn.Conv2d
    ):
        super().__init__()
        self.module = conv2d
        self.module_type = type(self.module)
        self.activation_cache = None

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output
    
    # def __call__(self, *args, **kwargs):
    #     if self.module is None:
    #         self.module = self.origin_conv2d(*args, **kwargs)
    #     logger.info(f"the key word are {kwargs=}")
    #     return self
    
    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        idx = get_runtime_state().pipeline_patch_idx
        pp_patches_start_idx_local = get_runtime_state().pp_patches_start_idx_local
        h_begin = pp_patches_start_idx_local[idx] - padding
        h_end = pp_patches_start_idx_local[idx + 1] + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        result = F.conv2d(
            padded_input,
            self.module.weight,
            self.module.bias,
            stride=stride,
            padding="valid",
        )
        return result
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        #import sys;import pdb;debug=pdb.Pdb(stdin=sys.__stdin__, stdout=sys.__stdout__);debug.set_trace()
        if (
            (
                get_pipeline_parallel_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            )
            or self.module.kernel_size == (1, 1)
            or self.module.kernel_size == 1
        ):
            output = self.naive_forward(x)
        else:
            if (
                not get_runtime_state().patch_mode
                or get_runtime_state().num_pipeline_patch == 1
            ):
                self.activation_cache = x
                output = self.naive_forward(self.activation_cache)
            else:
                if self.activation_cache is None:
                    self.activation_cache = torch.zeros(
                        [
                            x.shape[0],
                            x.shape[1],
                            get_runtime_state().pp_patches_start_idx_local[-1],
                            x.shape[3],
                        ],
                        dtype=x.dtype,
                        device=x.device,
                    )

                self.activation_cache[
                    :,
                    :,
                    get_runtime_state()
                    .pp_patches_start_idx_local[
                        get_runtime_state().pipeline_patch_idx
                    ] : get_runtime_state()
                    .pp_patches_start_idx_local[
                        get_runtime_state().pipeline_patch_idx + 1
                    ],
                    :,
                ] = x
                output = self.sliced_forward(self.activation_cache)
        return output