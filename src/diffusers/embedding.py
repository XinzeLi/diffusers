import torch
from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from diffusers.runtime_state import get_runtime_state
from torch import nn
from loguru import logger

class CustomPatchEmbed(nn.Module): # xinze: the difference is that, we do the positional embedding as if the patch is the full picture. After embedding process, we crop the result according to the patch index and use it as the final embedding.
    def __init__(
        self, patch_embedding: PatchEmbed,
    ):
        super().__init__()
        self.module = patch_embedding
        self.module_type = type(self.module) # self.module.pos_embed is injected in the from_pretrained step.
        self.pos_embed = None
        self.activation_cache = None


    def forward(self, latent):
        
        height = (
            get_runtime_state().config.height
            // get_runtime_state().vae_scale_factor
        )
        width = latent.shape[-1]

        latent = self.module.proj(latent)
        if self.module.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC


        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)

        if getattr(self.module, "pos_embed_max_size", None):
            pos_embed = self.module.cropped_pos_embed(height, width)


        if get_runtime_state().patch_mode:
            start, end = get_runtime_state().pp_patches_token_start_end_idx_global[
                get_runtime_state().pipeline_patch_idx
            ]
            pos_embed = pos_embed[
                :,
                start:end,
                :,
            ]

        return (latent + pos_embed).to(latent.dtype)