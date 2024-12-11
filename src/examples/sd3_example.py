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
from diffusers.runtime_state import initialize_runtime_state, get_runtime_state
from typing import Any, Dict, List, Optional, Tuple, Union
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
    print(f"the blocks list is {blocks_list}")
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
    initialize_runtime_state(engine_config=args)
    

    if pipe.transformer is not None and PP_degree > 1:
        pipe.transformer = convert_transformer(pipe.transformer)
    start_time = time.time()
    # pipe.prepare_run(input_config)
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
            print(f"used time: {end_time-start_time:.2f} seconds")

if __name__ == "__main__":
    main()