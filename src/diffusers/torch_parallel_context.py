from typing import Optional, Union, Sequence, ContextManager
from datetime import timedelta
from contextlib import nullcontext

import torch
import os

from diffusers.parallel_context import (
    AsyncMode,
    AsyncModeContext,
    ParallelContext,
)
from loguru import logger
# types
Shape = Sequence[int]
Stream = torch.cuda.Stream
ProcessGroup = torch.distributed.ProcessGroup


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


class TorchBasedParallelContext(ParallelContext):
    def __init__(
        self,
        *,
        ranks: Sequence[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        device_index: int = 0,
        backend: str = "nccl",
    ):
        self._init_torch_distributed()
        self._backend = backend

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
            self._main_group = torch.distributed.new_group(
                self._ranks, backend=self._backend
            )
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

    def _init_torch_distributed(self, timeout=timedelta(hours=24)):
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

    def backend(self) -> str:
        return self._backend

    def send_recv_comm_device(self) -> Union[str, torch.device]:
        # communication device for send/recv
        return self._torch_device

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
                # import sys;import pdb;debug=pdb.Pdb(stdin=sys.__stdin__, stdout=sys.__stdout__);debug.set_trace()
                self._pipeline_stage_id = group_ranks.index(self._rank)
                self._pipeline_parallel_group_prev_rank = group_ranks[
                    (self._pipeline_stage_id - 1 + self._pipeline_parallel_size) % self._pipeline_parallel_size
                ]
                self._pipeline_parallel_group_next_rank = group_ranks[
                    (self._pipeline_stage_id + 1) % self._pipeline_parallel_size
                ]
                from loguru import logger
                logger.info(f"the rank now is {self._rank=}, the next rank is {self._pipeline_parallel_group_next_rank=}, the last rank is {self._pipeline_parallel_group_prev_rank=}")

    def _init_tensor_parallel_group(self):
        if self.size > 1:
            self._tensor_parallel_group = self._new_group(self._tensor_parallel_size)
            if self._tensor_parallel_group is not None:
                group_ranks = torch.distributed.get_process_group_ranks(
                    self._tensor_parallel_group
                )
                self._tensor_parallel_group_root = group_ranks[0]

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
                group = torch.distributed.new_group(global_ranks, backend=self._backend)
                if global_rank in global_ranks:
                    ret_group = group

        return ret_group

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
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=self._tensor_parallel_group,
            )

    def allgather_in_tensor_parallel_group(
        self, input: torch.Tensor, output: torch.Tensor
    ):
        if self.tensor_parallel_size > 1:
            output1d = output.view(-1)
            local_elems = output1d.size(0) // self._tensor_parallel_size
            tensor_list = [
                output1d[i * local_elems : (i + 1) * local_elems]
                for i in range(self._tensor_parallel_size)
            ]
            torch.distributed.all_gather(
                tensor_list,
                input.view(-1),
                group=self._tensor_parallel_group,
            )

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
        logger.info(f"the current divice is {self._device_index=}, sending to {self._pipeline_parallel_group_prev_rank=}, in group {self.pipeline_parallel_group=}")
        if len(p2p_ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

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
        logger.info(f"the current divice is {self._device_index=}, sending to {self._pipeline_parallel_group_next_rank=}, in group {self.pipeline_parallel_group=}")
        if len(p2p_ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

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

    def tensor_parallel_reduce_context(self) -> ContextManager[None]:
        return nullcontext()

    def async_mode(self, async_mode_type: AsyncMode) -> AsyncModeContext:
        raise NotImplementedError

    async def async_allreduce_in_tensor_parallel_group(self, tensor: torch.Tensor):
        raise NotImplementedError
