from typing import List, Optional
import torch
import torch.distributed
from diffusers.group_coordinator import (
    GroupCoordinator,
    PipelineParallelGroupCoordinator,
    SequenceParallelGroupCoordinator,
    RankGenerator,
    generate_masked_orthogonal_rank_groups
)
import os
from yunchang import set_seq_parallel_pg
from yunchang.globals import PROCESS_GROUP

_WORLD: Optional[GroupCoordinator] = None
_SP: Optional[SequenceParallelGroupCoordinator] = None
_PP: Optional[PipelineParallelGroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None

def get_world_group() -> GroupCoordinator:
    return _WORLD

# SP
def get_sp_group() -> SequenceParallelGroupCoordinator:
    return _SP

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_sp_group().rank_in_group


def get_ulysses_parallel_world_size():
    return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank():
    return get_sp_group().ulysses_rank


def get_ring_parallel_world_size():
    return get_sp_group().ring_world_size

def get_ring_parallel_rank():
    return get_sp_group().ring_rank

# CFG
def get_cfg_group() -> GroupCoordinator:
    return _CFG

def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group



# PP
def get_pp_group() -> GroupCoordinator:
    return _PP


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )

def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: "nccl",
    parallel_mode: str,
    **kwargs,
) -> GroupCoordinator:
    assert parallel_mode in [
        "data",
        "pipeline",
        "tensor",
        "sequence",
        "classifier_free_guidance",
    ], f"parallel_mode {parallel_mode} is not supported"
    if parallel_mode == "pipeline":
        return PipelineParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )
    elif parallel_mode == "sequence":
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            **kwargs,
        )
    else:
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )


def initialize_model_parallel(
    classifier_free_guidance_degree: int = 1,
    # sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    # tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
) -> None:
    sequence_parallel_degree = ulysses_degree * ring_degree
    rank_generator: RankGenerator = RankGenerator(
        # tensor_parallel_degree,
        sequence_parallel_degree,
        pipeline_parallel_degree,
        classifier_free_guidance_degree,
        # data_parallel_degree,
        "sp-pp-cfg",
    )
    global _CFG
    global _PP
    global _SP
    backend = "nccl"
    _CFG = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("cfg"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )
    grouprankcfg = rank_generator.get_ranks("cfg")
    grouprankpp = rank_generator.get_ranks("pp")
    groupranksp = rank_generator.get_ranks("sp")
    print(f"cfg group ranks is {grouprankcfg}, pp group ranks is {grouprankpp}, sp group rank is {groupranksp}")

    _PP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("pp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="pipeline",
    )

    set_seq_parallel_pg(
        sp_ulysses_degree=ulysses_degree,
        sp_ring_degree=ring_degree,
        rank=get_world_group().rank_in_group,
        world_size=get_world_group().world_size,
    )
    _SP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("sp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="sequence",
        ulysses_group=PROCESS_GROUP.ULYSSES_PG,
        ring_group=PROCESS_GROUP.RING_PG,
    )
    print(f"ulysses group is {PROCESS_GROUP.ULYSSES_PG}, ring group is {PROCESS_GROUP.RING_PG}")
    print(f"ulysses world size is {_SP.ulysses_world_size}, ring group is {_SP.ring_world_size}")


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    local_rank= int(os.environ.get("LOCAL_RANK", "0"))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=-1,
        rank=-1,
    )
    torch.cuda.set_device(local_rank)
    global _WORLD
    ranks = list(range(torch.distributed.get_world_size()))
    _WORLD = init_world_group(ranks, local_rank, backend="nccl")