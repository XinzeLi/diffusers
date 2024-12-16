import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
from typing import Any, Dict, List, Optional, Tuple, Union

class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

    def all_gather(# xinze: this is just adapted torch.all_gather_into_tensor and then reshape
            self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        world_size = self.world_size
        if world_size == 1:
            return input_
        if dim <0:
            dim += input_.dim()
        input_size = input_.size()
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device #xinze: (world_size, ) is a tuple.
        )
        # All-gather
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )
        from loguru import logger
        logger.info(f"the group is {self.device_group=}")
        if dim != 0:
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            tensor_list = [
                output_tensor.view(-1)
                .narrow(0, input_.numel() * i, input_.numel())
                .view_as(input_)
                for i in range(world_size)
            ]
            return tensor_list
        else:
            input_size = list(input_.size())
            input_size[dim] = input_size[dim] * world_size
            output_tensor = output_tensor.reshape(input_size)
            return output_tensor
        
    @property
    def prev_rank(self):
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]
    
    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]
    
    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank
    
    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]
    
    @property
    def group_next_rank(self):
        """Return the group rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group + 1) % world_size

    @property
    def group_prev_rank(self):
        """Return the group rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group - 1) % world_size

    @property
    def skip_rank(self):
        """Return the global rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(world_size - rank_in_group - 1) % world_size]

    @property
    def group_skip_rank(self):
        """Return the group rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (world_size - rank_in_group - 1) % world_size

class SequenceParallelGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        **kwargs,
    ):
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
        )

        ulysses_group = kwargs.get("ulysses_group", None)
        ring_group = kwargs.get("ring_group", None)
        if ulysses_group is None:
            raise RuntimeError(
                f"Please pass argument 'ulysses_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        if ring_group is None:
            raise RuntimeError(
                f"Please pass argument 'ring_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group

        self.ulysses_world_size = torch.distributed.get_world_size(
            self.ulysses_group
        )
        self.ulysses_rank = torch.distributed.get_rank(self.ulysses_group)
        self.ring_world_size = torch.distributed.get_world_size(self.ring_group)
        self.ring_rank = torch.distributed.get_rank(self.ring_group)

class PipelineParallelGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        **kwargs,
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.cpu_groups = []
        self.device_groups = []
        if len(group_ranks[0]) > 2 or len(group_ranks[0]) == 1:
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group
        # when pipeline parallelism is 2, we need to create two groups to avoid
        #   communication stall.
        # *_group_0_1 represents the group for communication from device 0 to
        #   device 1.
        # *_group_1_0 represents the group for communication from device 1 to
        #   device 0.
        elif len(group_ranks[0]) == 2:
            for ranks in group_ranks:
                device_group_0_1 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                device_group_1_0 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group_0_1 = torch.distributed.new_group(ranks, backend="gloo")
                cpu_group_1_0 = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_groups = [device_group_0_1, device_group_1_0]
                    self.cpu_groups = [cpu_group_0_1, cpu_group_1_0]
                    self.device_group = device_group_0_1
                    self.cpu_group = cpu_group_0_1

        assert self.cpu_group is not None
        assert self.device_group is not None

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.recv_buffer_set: bool = False
        self.recv_tasks_queue: List[Tuple[str, int]] = []
        self.receiving_tasks: List[Tuple[torch.distributed.Work, str, int]] = []
        self.dtype: Optional[torch.dtype] = None
        self.num_pipefusion_patches: Optional[int] = None

        self.recv_shape: Dict[str, Dict[int, torch.Size]] = {}
        self.send_shape: Dict[str, Dict[int, torch.Size]] = {}
        self.recv_buffer: Dict[str, Dict[int, torch.Size]] = {}

        self.skip_tensor_recv_buffer_set: bool = False
        self.recv_skip_tasks_queue: List[Union[int, Tuple[str, int]]] = []
        self.receiving_skip_tasks: List[Tuple[torch.distributed.Work, str, int]] = []
        self.skip_tensor_recv_buffer: Optional[
            Union[List[torch.Tensor], torch.Tensor]
        ] = None
        self.skip_device_group = None
        for ranks in group_ranks:
            skip_device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            if self.rank in ranks:
                self.skip_device_group = skip_device_group
        assert self.skip_device_group is not None

    def reset_buffer(self):
        self.recv_tasks_queue = []
        self.receiving_tasks = []
        self.recv_shape = {}
        self.send_shape = {}
        self.recv_buffer = {}

        self.recv_skip_tasks_queue = []
        self.receiving_skip_tasks = []
        self.skip_tensor_recv_buffer = {}

    def recv_next(self):
        if len(self.recv_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_tasks_queue) > 0:
            name, idx = self.recv_tasks_queue.pop(0)
            self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
            self.receiving_tasks.append(
                (self._pipeline_irecv(self.recv_buffer[name][idx]), name, idx)
            )
    
    def _pipeline_irecv(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor,
            src=self.prev_rank,
            group=(
                self.device_groups[(self.rank_in_group + 1) % 2]
                if self.world_size == 2
                else self.device_group
            ),
        )
    
    def get_pipeline_recv_data(
            self, idx: int = -1, name: str = "latent"
    ) -> torch.Tensor:
        receiving_task = self.receiving_tasks.pop(0)
        receiving_task[0].wait()
        return self.recv_buffer[name][idx]

    def _check_shape_and_buffer(
        self,
        tensor_send_to_next=None,
        recv_prev=False,
        name: Optional[str] = None,
        segment_idx: int = 0,
    ):
        pass

def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    def __init__(
        self,
        # tp: int,
        sp: int,
        pp: int,
        cfg: int,
        # dp: int,
        order: str,
        rank_offset: int = 0,
    ) -> None:
        # self.tp = tp
        self.sp = sp
        self.pp = pp
        self.cfg = cfg
        # self.dp = dp
        self.rank_offset = rank_offset
        self.world_size = sp * pp * cfg

        self.name_to_size = {
            # "tp": self.tp,
            "sp": self.sp,
            "pp": self.pp,
            "cfg": self.cfg,
            # "dp": self.dp,
        }
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split("-")
        token = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask
        )
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks
