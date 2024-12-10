import os
from typing import Optional, Union, Tuple, Sequence
from datetime import timedelta
from loguru import logger
from collections.abc import Coroutine

import asyncio
import torch

import crossing_accelerator_cuda_extension
from parallel_context import (
    ParallelContext,
    AsyncModeContext,
    AsyncMode,
)

# types
Shape = Sequence[int]
Stream = torch.cuda.Stream
ProcessGroup = torch.distributed.ProcessGroup
Communicator = "torch.cuda.nccl.Communicator"

if int(os.getenv("CROSSING_ENABLE_MSCCLPP_ALLRECUDE", "0")) == 1:
    try:
        from crossing.accelerators.mscclpp_allreduce import MscclppAllReduce

        MSCCLPP_ALLRECUDE_ENABLED = True
    except:
        MSCCLPP_ALLRECUDE_ENABLED = False
else:
    MSCCLPP_ALLRECUDE_ENABLED = False

from contextlib import closing
import socket


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


MSCCLPP_MAX_SIZE_BYTES = 128 * 1024


class OverlapAsyncModeContext(AsyncModeContext):
    def __init__(self, parallel_ctx: "NcclParallelContext"):
        self.parallel_ctx = parallel_ctx

    def __enter__(self):
        self._coroutine_switch_condition = asyncio.Condition()
        self._comm_stream = self.parallel_ctx._tensor_parallel_comm_stream
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
            self.parallel_ctx._tensor_parallel_comm_stream = self._comm_stream
        self._old_async_mode = self.parallel_ctx._async_mode
        self.parallel_ctx._async_mode = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parallel_ctx._async_mode = self._old_async_mode
        if exc_type is not None:
            logger.error(
                f"An exception occurred in OverlapAsyncModeContext context scope: {exc_value}"
            )
        return False

    async def gather(self, task1, task2) -> Tuple:
        if isinstance(task1, Coroutine):
            task1 = asyncio.create_task(task1)
        if isinstance(task2, Coroutine):
            task2 = asyncio.create_task(task2)

        assert isinstance(
            task1, asyncio.Task
        ), f"task1 should be a asyncio.Task, but got {task1}"
        assert isinstance(
            task2, asyncio.Task
        ), f"task2 should be a asyncio.Task, but got {task2}"

        result1 = None
        result2 = None
        task1_completed = False
        task2_completed = False

        while not task1_completed or not task2_completed:
            done, pending = await asyncio.wait(
                [task1, task2], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task is task1:
                    # notify task2
                    async with self._coroutine_switch_condition:
                        self._coroutine_switch_condition.notify()
                    result1 = task.result()
                    task1_completed = True
                else:
                    assert task1_completed
                    # task_2 done
                    result2 = task.result()
                    task2_completed = True

        return result1, result2

    async def async_allreduce(self, tensor):
        comp_stream = torch.cuda.current_stream()
        # sync from comp to comm
        cuda_event_comp = torch.cuda.Event()
        cuda_event_comp.record(stream=comp_stream)
        self._comm_stream.wait_event(cuda_event_comp)

        # allreduce on comm stream
        with torch.cuda.stream(self._comm_stream):
            self.parallel_ctx.allreduce_in_tensor_parallel_group(tensor)

        # sync from comm to comp
        cuda_event_comm = torch.cuda.Event()
        cuda_event_comm.record(stream=self._comm_stream)
        # switch to another coroutine
        async with self._coroutine_switch_condition:
            self._coroutine_switch_condition.notify()
            await self._coroutine_switch_condition.wait()
        comp_stream.wait_event(cuda_event_comm)


def _init_torch_distributed(timeout=timedelta(hours=24)):
    assert torch.cuda.is_available()
    assert torch.distributed.is_available()

    env_rank = int(os.getenv("RANK", "0"))
    env_world_size = int(os.getenv("WORLD_SIZE", "1"))
    if env_world_size == 1:
        return

    if torch.distributed.is_initialized():
        assert env_world_size == torch.distributed.get_world_size()
        assert env_rank == torch.distributed.get_rank()
        return

    backend = "nccl"
    options = torch.distributed.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True

    torch.distributed.init_process_group(
        backend=backend,
        world_size=env_world_size,
        rank=env_rank,
        timeout=timeout,
        pg_options=options,
    )


def _check_send_recv_tensors(tensors: Union[torch.Tensor, Sequence[torch.Tensor]]):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        if not isinstance(tensors, (list, tuple)):
            raise ValueError(
                f"send_tensors must be a list of tensors, but got type {type(tensors)}"
            )

        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    "send_tensors must be a list of tensors, "
                    f"but got {i} th element type {type(tensor)}"
                )

    return tensors


class NcclParallelContext(ParallelContext):
    def __init__(
        self,
        *,
        ranks: Sequence[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        device_index: int = 0,
    ):
        _init_torch_distributed()

        self._ranks = ranks
        self._tensor_parallel_size = tensor_parallel_size
        self._pipeline_parallel_size = pipeline_parallel_size
        self._device_index = device_index
        self._torch_device = torch.device(f"cuda:{self._device_index}")
        self._parallel_size = tensor_parallel_size * pipeline_parallel_size

        if not isinstance(ranks, (list, tuple)) or not all(
            isinstance(rank, int) for rank in ranks
        ):
            raise ValueError(f"`ranks` should be a list of int")
        if len(ranks) != self._parallel_size:
            raise ValueError(f"`ranks` should has the same length as parallel_ctx.size")
        self._ranks = ranks

        if self._parallel_size > 1:
            self._main_group = torch.distributed.new_group(self._ranks, backend="nccl")
        else:
            self._main_group = None

        # for grouping
        self._process_mesh = torch.tensor(self._ranks)

        self._rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )

        # TODO: remove when tests don't rely on set_device in parallel_ctx
        if self._rank in self._ranks:
            torch.cuda.set_device(self._device_index)

        # tensor parallel property
        self._tensor_parallel_group: Optional[ProcessGroup] = None
        self._tensor_parallel_group_root: int = 0
        self._tensor_parallel_main_comm: Optional[Communicator] = None
        self._tensor_parallel_comm_stream: Optional[Stream] = None
        # async mode
        self._async_mode: Optional[AsyncModeContext] = None

        # pipeline parallel property
        self._pipeline_parallel_group: Optional[ProcessGroup] = None
        self._pipeline_stage_id: int = 0
        self._pipeline_parallel_group_prev_rank = 0
        self._pipeline_parallel_group_next_rank = 0

        with torch.cuda.device(self._device_index):
            self._init_pipeline_parallel_group()
            self._init_tensor_parallel_group()

        if self.tensor_parallel_size > 1 and MSCCLPP_ALLRECUDE_ENABLED:
            if self.tensor_parallel_rank == 0:
                port = _find_free_port()
                port_tensor = torch.tensor(
                    [port], dtype=torch.int32, device=self._torch_device
                )
            else:
                port_tensor = torch.empty(
                    (1,), dtype=torch.int32, device=self._torch_device
                )
            self.broadcast_in_tensor_parallel_group(port_tensor)
            port = port_tensor.item()
            self._mscclpp_allreduce = MscclppAllReduce(
                rank=self.tensor_parallel_rank,
                parallel_size=self.tensor_parallel_size,
                max_size_bytes=MSCCLPP_MAX_SIZE_BYTES,
                port=port,
                device=self._torch_device,
            )
        else:
            self._mscclpp_allreduce = None

    def _init_pipeline_parallel_group(self):
        if self.size > 1:
            self._pipeline_parallel_group = self._new_group(
                self._pipeline_parallel_size
            )
            if self._pipeline_parallel_group is not None:
                # set pipeline property
                group_ranks = torch.distributed.get_process_group_ranks(
                    self._pipeline_parallel_group
                )
                self._pipeline_stage_id = group_ranks.index(self._rank)
                self._pipeline_parallel_group_prev_rank = group_ranks[
                    (self._pipeline_stage_id - 1) % self._pipeline_parallel_size
                ]
                self._pipeline_parallel_group_next_rank = group_ranks[
                    (self._pipeline_stage_id + 1) % self._pipeline_parallel_size
                ]

    def _init_tensor_parallel_group(self):
        if self.size > 1:
            self._tensor_parallel_group = self._new_group(self._tensor_parallel_size)
            if self._tensor_parallel_group is not None:
                group_ranks = torch.distributed.get_process_group_ranks(
                    self._tensor_parallel_group
                )
                self._tensor_parallel_group_root = group_ranks[0]
                self._tensor_parallel_main_comm = self._create_communicator(
                    self._tensor_parallel_group
                )

    def _new_group(
        self, parallel_size: int
    ) -> Optional[torch.distributed.ProcessGroup]:
        remain_num_groups = self._process_mesh.size(-1)
        if remain_num_groups % parallel_size != 0:
            raise ValueError(
                f"The process mesh {self._process_mesh} cannot be divided further by parallel_size {parallel_size}"
            )

        num_groups = remain_num_groups // parallel_size
        self._process_mesh = self._process_mesh.view(
            *self._process_mesh.shape[:-1], parallel_size, num_groups
        )

        ret_group = None
        global_rank = torch.distributed.get_rank()
        # (past_num_groups, parallel_size, new_num_groups)
        process_mesh = self._process_mesh.view(-1, parallel_size, num_groups)
        for group_i in range(process_mesh.size(0)):
            for group_j in range(process_mesh.size(-1)):
                global_ranks = process_mesh[group_i, :, group_j].tolist()
                group = torch.distributed.new_group(global_ranks, backend="nccl")
                if global_rank in global_ranks:
                    ret_group = group

        return ret_group

    def _create_communicator(self, group: torch.distributed.ProcessGroup):
        backend = torch.distributed.get_backend(group)
        assert backend == "nccl"
        group_ranks = torch.distributed.get_process_group_ranks(group)
        if self._rank in group_ranks:
            group_rank = group_ranks.index(self._rank)
            unique_id = torch.cuda.nccl.unique_id()
            unique_id_tensor = torch.ByteTensor(list(unique_id)).cuda()
            torch.distributed.broadcast(
                unique_id_tensor, src=group_ranks[0], group=group
            )
            unique_id = unique_id_tensor.cpu().numpy().tobytes()
            comm = crossing_accelerator_cuda_extension._nccl_comm_init_rank(
                nranks=len(group_ranks), commId=unique_id, rank=group_rank
            )
            return comm
        return None

    def device_index(self) -> int:
        return self._device_index

    def torch_device(self) -> torch.device:
        return self._torch_device

    def allreduce(self, tensor: torch.Tensor, red_op: torch.distributed.ReduceOp):
        if self.size > 1:
            torch.distributed.all_reduce(tensor, op=red_op, group=self._main_group)

    @property
    def tensor_parallel_group(self):
        return self._tensor_parallel_group

    @property
    def size(self) -> int:
        return self._parallel_size

    @property
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    @property
    def tensor_parallel_rank(self) -> int:
        if self._tensor_parallel_group is None:
            return 0
        return self._tensor_parallel_group.rank()

    def broadcast_in_tensor_parallel_group(self, tensor: torch.Tensor):
        if self.tensor_parallel_size > 1:
            torch.distributed.broadcast(
                tensor,
                src=self._tensor_parallel_group_root,
                group=self._tensor_parallel_group,
            )

    def allreduce_in_tensor_parallel_group(self, tensor: torch.Tensor):
        if self.tensor_parallel_size > 1:
            tensor_size = tensor.numel() * tensor.element_size()
            if (
                self._mscclpp_allreduce
                and tensor_size <= MSCCLPP_MAX_SIZE_BYTES
                and tensor.dtype == torch.float16
                and tensor.is_contiguous()
            ):
                self._mscclpp_allreduce(tensor)
            else:
                crossing_accelerator_cuda_extension._nccl_all_reduce(
                    input=tensor,
                    output=tensor,
                    op=0,  # ncclRedOp_t::ncclSum
                    comm=self._tensor_parallel_main_comm,
                )

    def allgather_in_tensor_parallel_group(
        self, input: torch.Tensor, output: torch.Tensor
    ):
        if self.tensor_parallel_size > 1:
            crossing_accelerator_cuda_extension._nccl_all_gather(
                input=input,
                output=output,
                comm=self._tensor_parallel_main_comm,
            )

    def async_mode(self, async_mode_type: AsyncMode) -> AsyncModeContext:
        if async_mode_type == AsyncMode.OVERLAPPING:
            return OverlapAsyncModeContext(self)

        raise NotImplementedError

    async def async_allreduce_in_tensor_parallel_group(self, tensor: torch.Tensor):
        if self.tensor_parallel_size == 1:
            return

        if self._async_mode is None:
            return self.allreduce_in_tensor_parallel_group(tensor)

        return await self._async_mode.async_allreduce(tensor)

    @property
    def pipeline_parallel_group(self):
        return self._pipeline_parallel_group

    @property
    def pipeline_parallel_size(self) -> int:
        return self._pipeline_parallel_size

    @property
    def pipeline_stage_id(self) -> int:
        return self._pipeline_stage_id

    def recv_from_prev_stage(
        self, recv_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]
    ):
        """Receive tensor from previous stage in pipeline (forward receive)."""
        assert (
            self.pipeline_parallel_group is not None
        ), "pipeline_parallel_group is not initialized, pipeline_parallel_size need to be set greater than 1"

        recv_tensors = _check_send_recv_tensors(recv_tensors)
        p2p_ops = []
        for recv_tensor in recv_tensors:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_tensor,
                self._pipeline_parallel_group_prev_rank,
                group=self.pipeline_parallel_group,
            )
            p2p_ops.append(recv_op)

        if len(p2p_ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

    def send_to_next_stage(
        self, send_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]
    ):
        """Send tensor to next stage in pipeline (forward send)."""
        assert (
            self.pipeline_parallel_group is not None
        ), "pipeline_parallel_group is not initialized, pipeline_parallel_size need to be set greater than 1"

        send_tensors = _check_send_recv_tensors(send_tensors)
        p2p_ops = []
        for tensor in send_tensors:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor,
                self._pipeline_parallel_group_next_rank,
                group=self.pipeline_parallel_group,
            )
            p2p_ops.append(send_op)

        if len(p2p_ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

    def send_and_recv_between_neighborhoods(
        self,
        recv_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
        send_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    ):
        assert (
            self.pipeline_parallel_group is not None
        ), "pipeline_parallel_group is not initialized, pipeline_parallel_size need to be set greater than 1"

        recv_tensors = _check_send_recv_tensors(recv_tensors)
        send_tensors = _check_send_recv_tensors(send_tensors)
        p2p_ops = []

        for recv_tensor in recv_tensors:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_tensor,
                self._pipeline_parallel_group_prev_rank,
                group=self.pipeline_parallel_group,
            )
            p2p_ops.append(recv_op)

        for send_tensor in send_tensors:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_tensor,
                self._pipeline_parallel_group_next_rank,
                group=self.pipeline_parallel_group,
            )
            p2p_ops.append(send_op)

        if len(p2p_ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()