import time
import os
import torch
import torch.distributed
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
import argparse
# from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter
from diffusers.group_coordinator import (
    GroupCoordinator, 
    RankGenerator, 
    SequenceParallelGroupCoordinator, 
    PipelineParallelGroupCoordinator,
)
from deepspeed.module_inject import replace_module
from diffusers.runtime_state import initialize_runtime_state, get_runtime_state
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from diffusers.models.embeddings import PatchEmbed
from diffusers.conv import CustomConv2d
from diffusers.embedding import CustomPatchEmbed
from diffusers.parallel_state import (
    init_model_parallel_group,
    init_world_group,
    initialize_model_parallel,
    init_distributed_environment,
    get_world_group,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_last_stage,
)
import torch.nn as nn




def convert_transformer(
        transformer: nn.Module,
        blocks_name: List[str] = ['transformer_blocks'],
    ) -> nn.Module:
    pp_rank = get_pipeline_parallel_rank()
    pp_world_size = get_pipeline_parallel_world_size()
    blocks_list = {
        block_name: getattr(transformer, block_name) for block_name in blocks_name
    }
    # print(f"the blocks list is {blocks_list}")
    num_blocks_list = [len(blocks) for blocks in blocks_list.values()]
    blocks_idx = {
        name: [sum(num_blocks_list[:i]), sum(num_blocks_list[: i + 1])]
        for i, name in enumerate(blocks_name)
    }
    
    num_blocks_per_stage = (
        sum(num_blocks_list) + pp_world_size - 1
    ) // pp_world_size
    stage_block_start_idx = pp_rank * num_blocks_per_stage
    stage_block_end_idx = min(
        (pp_rank + 1) * num_blocks_per_stage,
        sum(num_blocks_list),
    )
    for name, [blocks_start, blocks_end] in zip(
        blocks_idx.keys(), blocks_idx.values()
    ):
        if (
            blocks_end <= stage_block_start_idx
            or stage_block_end_idx <= blocks_start
        ):
            setattr(transformer, name, nn.ModuleList([]))
        elif stage_block_start_idx <= blocks_start:
            if blocks_end <= stage_block_end_idx:
                pass
            else:
                setattr(
                    transformer,
                    name,
                    blocks_list[name][: -(blocks_end - stage_block_end_idx)],
                )
                # self.stage_info.after_flags[name] = False
        elif blocks_start < stage_block_start_idx:
            if blocks_end <= stage_block_end_idx:
                setattr(
                    transformer,
                    name,
                    blocks_list[name][stage_block_start_idx - blocks_start :],
                )
                # self.stage_info.after_flags[name] = True
            else:  # blocks_end > stage_layer_end_idx
                setattr(
                    transformer,
                    name,
                    blocks_list[name][
                        stage_block_start_idx
                        - blocks_start : stage_block_end_idx
                        - blocks_end
                    ],
                )
                # self.stage_info.after_flags[name] = False

    return transformer

def _change_layer(submodule):
    if isinstance(submodule, nn.Conv2d):
        return CustomConv2d(conv2d=submodule)
        # return submodule
    elif isinstance(submodule, PatchEmbed):
        # print("YYYYYYYYYYY")
        # import sys;import pdb;debug=pdb.Pdb(stdin=sys.__stdin__, stdout=sys.__stdout__);debug.set_trace()
        return CustomPatchEmbed(patch_embedding=submodule)
    else:
        return submodule


def change_conv_embed(model: nn.Module, submodule_classes_to_change=[nn.Conv2d, PatchEmbed]):
    if model is None:
        return None
    for name, module in model.named_children():
        need_change = False
        for class_to_change in submodule_classes_to_change:
            if isinstance(module, class_to_change):
                need_change = True
                break
        if need_change:
            # print(f"the module {name} is changing!!!")
            new_layer = _change_layer(module)
            setattr(model, name, new_layer)

        for subname, submodule in module.named_children():
            need_change = False
            for class_to_change in submodule_classes_to_change:
                if isinstance(submodule, class_to_change):
                    need_change = True
                    break
            if need_change:
                # print(f"the submodule {subname} is changing!!!")
                new_layer = _change_layer(submodule)
                setattr(module, subname, new_layer)

    # model = replace_module(model=model, orig_class=nn.Conv2d, replace_fn=_change_layer)
    # model = replace_module(model=model, orig_class=PatchEmbed, replace_fn=_change_layer)

    return model




def main():
    parser = argparse.ArgumentParser(description="parallel diffuser arguments")
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        type=str,
        default="/maasjfs/hf_models/stable-diffusion-3.5-fp8/stable-diffusion-3-medium-diffusers",
        help="Name or path of the huggingface model to use."
    )
    model_group.add_argument(
        "--ulysses_degree",
        type=int,
        default=None,
        help="Ulysses SP degree. Used in attention layer"
    )
    model_group.add_argument(
        "--ring_degree",
        type=int,
        default=None,
        help="Ring SP degree. Used in attention layer"
    )
    model_group.add_argument(
        "--pipefusion_parallel_degree",
        type=int,
        default=1,
        help="Pipefusion parallel degree."
    )
    model_group.add_argument(
        "--prompt",
        type=str,
        nargs="*",
        default="",
        help="prompt."
    )
    model_group.add_argument(
        "--use_parallel_vae", 
        action="store_true",
        help="use distvae to parallel vae"
        )
    model_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="number of inference steps."
    )
    model_group.add_argument(
        "--use_cfg_parallel",
        action="store_true",
        help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set",
    )
    model_group.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    model_group.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    args = parser.parse_args()
    use_parallel_vae = args.use_parallel_vae
    ulysses_degree = args.ulysses_degree
    ring_degree = args.ring_degree
    PP_degree = args.pipefusion_parallel_degree
    if args.use_cfg_parallel:
        cfg_degree = 2
    else:
        cfg_degree = 1
    
    init_distributed_environment()
    
    
    
    # raise Exception(f"use_parallel_Vae is {use_parallel_vae}")
    
    # torch.cuda.synchronize(device=local_rank)
    # print(f"the world size is {torch.distributed.get_world_size()}")
    # print(f"Ulysses degree is {ulysses_degree}, ring degree is {ring_degree}, PP degree is {PP_degree}")
    # print(f"the local rank is {local_rank}")
    
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        pipeline_parallel_degree=PP_degree,
    )
    initialize_runtime_state(engine_config=args)

    # origin_conv2d = nn.Conv2d
    # nn.Conv2d = CustomConv2d(conv2d=origin_conv2d)
    # origin_PatchEmbed = embeddings.PatchEmbed
    # embeddings.PatchEmbed = CustomPatchEmbed(patch_embedding=origin_PatchEmbed)

    local_rank= int(os.environ.get("LOCAL_RANK", "0"))
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=args.model,
        # engine_config=engine_config,
        torch_dtype=torch.float16,
        use_parallel_vae=use_parallel_vae,
        PP_degree=PP_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    ).to(f"cuda:{local_rank}")
    
    
    

    if pipe.transformer is not None and PP_degree > 1:
        pipe.transformer = convert_transformer(pipe.transformer)
        pipe.transformer = change_conv_embed(model=pipe.transformer)
    
    args.num_inference_steps = 20
    # pipe.prepare_run(input_config)
    output = pipe(
        height=1024,
        width=1024,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        # output_type=args.output_type,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    start_time = time.time()
    for i in range(3):
        output = pipe(
            height=1024,
            width=1024,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            # output_type=args.output_type,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
    end_time = time.time()
    parallel_info = (
        # f"dp{args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{args.ulysses_degree}_ring{args.ring_degree}_"
        f"pp{args.pipefusion_parallel_degree}"
    )
    if is_pipeline_last_stage():
        image=output.images[0]
        if not os.path.exists("results"):
            os.mkdir("results")
        image.save(f"./results/SD3_result_{parallel_info}_rank{local_rank}.png")
        print(f"image saved to ./results/SD3_result_{parallel_info}_rank{local_rank}.png")
        if get_world_group().rank == get_world_group().world_size - 1:
            print(f"used time: {(end_time-start_time)/3:.2f} seconds")
    
    

if __name__ == "__main__":
    main()