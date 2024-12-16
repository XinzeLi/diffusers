import torch

from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Tuple, ContextManager
from contextlib import nullcontext
from enum import Enum



class AsyncMode(Enum):
    OVERLAPPING = 1


class AsyncModeContext(ABC):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def gather(self, *tasks) -> Tuple:
        pass


class ParallelContext(ABC):
    @abstractmethod
    def backend(self) -> str:
        pass

    @abstractmethod
    def device_index(self) -> int:
        pass

    @abstractmethod
    def torch_device(self) -> torch.device:
        pass

    @abstractmethod
    def send_recv_comm_device(self) -> Union[str, torch.device]:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def allreduce(self, tensor: torch.Tensor, red_op: torch.distributed.ReduceOp):
        pass

    @property
    @abstractmethod
    def tensor_parallel_size(self) -> int:
        pass

    @property
    @abstractmethod
    def tensor_parallel_rank(self) -> int:
        pass

    @abstractmethod
    def broadcast_in_tensor_parallel_group(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def allreduce_in_tensor_parallel_group(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def allgather_in_tensor_parallel_group(
        self, input: torch.Tensor, output: torch.Tensor
    ):
        pass

    @abstractmethod
    def async_mode(self, async_mode: AsyncMode) -> AsyncModeContext:
        pass

    @abstractmethod
    async def async_allreduce_in_tensor_parallel_group(self, tensor: torch.Tensor):
        pass

    @property
    @abstractmethod
    def pipeline_parallel_size(self) -> int:
        pass

    @property
    @abstractmethod
    def pipeline_stage_id(self) -> int:
        pass

    def in_pipeline_first_stage(self) -> bool:
        return self.pipeline_stage_id == 0

    def in_pipeline_last_stage(self) -> bool:
        return self.pipeline_stage_id == self.pipeline_parallel_size - 1

    @abstractmethod
    def recv_from_prev_stage(
        self, recv_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]
    ):
        pass

    @abstractmethod
    def send_to_next_stage(
        self, send_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]
    ):
        pass

    @abstractmethod
    def send_and_recv_between_neighborhoods(
        self,
        recv_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
        send_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    ):
        pass

    def tensor_parallel_reduce_context(self) -> ContextManager[None]:
        return nullcontext()

