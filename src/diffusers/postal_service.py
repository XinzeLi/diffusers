import torch

from typing import List, Union, Optional
from diffusers.parallel_context import ParallelContext
from loguru import logger

support_type = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.bfloat16,
    torch.cdouble,
    torch.quint8,
    torch.qint8,
    torch.qint32,
]

type2int = {}
for idx, dtype in enumerate(support_type):
    type2int[dtype] = idx

int64_type = torch.int64
int64_byte = torch.tensor([], dtype=int64_type).element_size()
byte_type = torch.uint8
min_copy_size = 4096


def is_cpu(device_type):
    return str(device_type) == "cpu"


def string2list(s: str):
    return list(s.encode())


def byte_list2string(t: List[int]):
    return bytes(t).decode()


def patch2long_byte(p):
    return (-p) % int64_byte


def jump2patch(p):
    return p + patch2long_byte(p)


def patching(p, tensor_list: List[torch.Tensor], device="cpu"):
    patch_size = patch2long_byte(p)
    if patch_size > 0:
        tensor_list.append(torch.empty(patch_size, dtype=byte_type, device=device))


def sum_tensor_len(tensor_list: List[torch.Tensor]):
    sum_len = 0
    for tensor in tensor_list:
        sum_len += len(tensor)
    return sum_len


class Shipment:
    def __init__(self, device: torch.device, buffer_device: Optional[torch.device]=None):
        # Record the device for accelerator
        # `device` is the device for accelerating computing and `buffer_device` is device for buffer to be sent/received
        self._device = device
        self._buffer_device = buffer_device or device
        # Buffer and shipment on cpu should be tensor
        # Use [] to speed up
        # We do not use None since len(None) would cause error
        # The structure of buffer, which is related to pack and unpack.
        # buffer = torch.tensor([the number of package n| package 0 | package 1 | ... | package n])
        # Each package has specific format:
        # package = [len(meta) | len(cpu tensors) | len(gpu tensors) | meta | cpu tensors | gpu tensors]
        # meta is the information of tensor, which has structure:
        # meta = [len(name) | name | dtype | ndim | shape | device | len(tensor)]
        self._buffer = []
        # Items packed in the buffer
        self._packed_items = {}
        # The store position on the buffer
        self._pack_item2position = {}
        # Not packed items
        self._warehouse = {}
        # Items packed but replaced by new items
        self._replaced_items = {}

        # Local variable for pack
        self.meta_int_list: List[int] = []
        # Local variable for look through items, or say, unpack
        # The part of buffer sent from gpu to cpu
        self.shipment_on_cpu = []
        # The point on gpu shipment marking the starting position of shipment on cpu
        self.cpu_shipment_start = 0
        # Convert the shipment on cpu from bytes to int64
        self.shipment_in_list = []
        # The pointer in shipment in list, which is used to mark the reading position of meta information
        # The data is int64, adding 1 to pointer_in_list is equal to move 8 bytes with a cpu or gpu pointer (with type bytes)
        self.pointer_in_list = 0

    # _bundling_single_int, _bundling, _pack are 3 functions related to pack
    def _bundling_single_int(self, single_int: int):
        self.meta_int_list.append(single_int)

    def _bundling(self, list_1d: Union[List[int], torch.Size]):
        self._bundling_single_int(len(list_1d))
        self.meta_int_list.extend(list_1d)

    def _pack(self):
        if len(self._replaced_items) > 0:
            for key, value in self._replaced_items.items():
                data_start_end = self._pack_item2position[key]
                self._buffer[data_start_end[0] : data_start_end[1]] = (
                    value.view(-1).view(byte_type).to(self._buffer_device)
                )

            self._replaced_items.clear()

        # No new items
        if len(self._warehouse) == 0:
            return

        # Update total pacakge number
        if len(self._buffer) > 0:
            self._buffer[0:int64_byte].view(int64_type)[0] += 1

        # Convert self._warehouse to byte tensors
        self.meta_int_list = []
        should_on_cpu2data_tensor_list = {True: [], False: []}
        self._bundling_single_int(len(self._warehouse))
        new_item2position = {}
        storage_start_position = 0
        # Go through all the cpu tensors before dealing with any gpu tensors
        for should_on_cpu in [True, False]:
            for key, value in self._warehouse.items():
                if is_cpu(value.device.type) == should_on_cpu:
                    # name
                    self._bundling(string2list(key))
                    # data type
                    self._bundling_single_int(type2int[value.dtype])
                    # shape
                    self._bundling(value.size())
                    # device
                    self._bundling_single_int(int(should_on_cpu))
                    # the length of value viewed as byte tensor
                    value_view_as_1d_byte_data = (
                        value.contiguous().view(-1).view(byte_type)
                    )
                    value_view_as_1d_byte_data_size = len(value_view_as_1d_byte_data)
                    self._bundling_single_int(value_view_as_1d_byte_data_size)
                    # data
                    should_on_cpu2data_tensor_list[should_on_cpu].append(
                        value_view_as_1d_byte_data
                    )
                    # storage pointers of start and end
                    storage_end_position = (
                        storage_start_position + value_view_as_1d_byte_data_size
                    )
                    new_item2position[key] = [
                        storage_start_position,
                        storage_end_position,
                    ]
                    storage_start_position = jump2patch(storage_end_position)
                    # An empty tensor patch, to make the offset divisible by int64_byte
                    patching(
                        value_view_as_1d_byte_data_size,
                        should_on_cpu2data_tensor_list[should_on_cpu],
                        value.device,
                    )

        # Calculate tensor length
        total_package_len = [
            len(self.meta_int_list) * int64_byte,
            sum_tensor_len(should_on_cpu2data_tensor_list[True]),
            sum_tensor_len(should_on_cpu2data_tensor_list[False]),
        ]
        self.meta_int_list = total_package_len + self.meta_int_list
        # Concatenate cpu tensors first
        if len(self._buffer) == 0:
            self.meta_int_list.insert(0, 1)
        cpu_data_tensor_list = should_on_cpu2data_tensor_list[True]
        if len(cpu_data_tensor_list) == 0:
            concat_cpu_tensor = torch.tensor(
                self.meta_int_list, dtype=int64_type, device=self._buffer_device
            ).view(byte_type)
            # Place the meta information before cpu and gpu tensors, move the offset correspondingly
            offset = len(concat_cpu_tensor)
        else:
            cpu_data_tensor_list.insert(
                0, torch.tensor(self.meta_int_list, dtype=int64_type).view(byte_type)
            )
            offset = len(cpu_data_tensor_list[0])
            concat_cpu_tensor = torch.cat(cpu_data_tensor_list).to(self._buffer_device)
        # Concatenate gpu tensors
        gpu_data_tensor_list = should_on_cpu2data_tensor_list[False]
        if is_cpu(self._buffer_device) and len(gpu_data_tensor_list) > 0:
            gpu_data_tensor_list = [torch.cat(gpu_data_tensor_list).to(self._buffer_device)]
        gpu_data_tensor_list.insert(0, concat_cpu_tensor)
        if len(self._buffer) > 0:
            gpu_data_tensor_list.insert(0, self._buffer)
            # Place the last package before this package, move the offset correspondingly
            offset += len(self._buffer)
        self._buffer = torch.cat(gpu_data_tensor_list)
        self.meta_int_list.clear()
        # Move the items from warehouse to the shipment
        self._packed_items.update(self._warehouse)
        self._warehouse.clear()
        # Maintain the packed item storage pointers
        for key, pointers in new_item2position.items():
            self._pack_item2position[key] = [pointers[0] + offset, pointers[1] + offset]

    # _unbundling_single_int, _unbunding, _deliver_to_host, _unpack are 4 functions related to unpack
    def _unbundling_single_int(self) -> int:
        single_int = self.shipment_in_list[self.pointer_in_list]
        self.pointer_in_list += 1
        return single_int

    def _unbundling(self):
        tensor_len = self._unbundling_single_int()
        p_start = self.pointer_in_list
        # Not using p_end would cause error:
        self.pointer_in_list += tensor_len
        item = self.shipment_in_list[p_start : self.pointer_in_list]
        return item

    def _deliver_to_host(self, size=min_copy_size):
        if size < min_copy_size:
            size = min_copy_size
        self.cpu_shipment_start += self.pointer_in_list * int64_byte
        p_end = min(self.cpu_shipment_start + size, len(self._buffer))
        self.shipment_on_cpu = self._buffer[self.cpu_shipment_start : p_end].to("cpu")
        self.shipment_in_list = self.shipment_on_cpu.view(int64_type).tolist()
        self.pointer_in_list = 0

    def _look_through_items(self, do_something):
        self.pointer_in_list = 0
        self.cpu_shipment_start = 0
        self._deliver_to_host()
        # Total pacakge number
        total_package_num = self._unbundling_single_int()
        for package_id in range(total_package_num):
            # Get length of meta, cpu tensors, gpu tensors
            total_meta_len = self._unbundling_single_int()
            total_cpu_tensor_len = self._unbundling_single_int()
            total_gpu_tensor_len = self._unbundling_single_int()
            # At least deliver this whole package to host
            if (
                len(self.shipment_on_cpu) - self.pointer_in_list * int64_byte
                < total_meta_len + total_cpu_tensor_len
            ):
                self._deliver_to_host(total_meta_len + total_cpu_tensor_len)
            # Assign pointers, which is related to the position of shipment copy
            # The pointer on shipment_on_cpu, marking the reading position of cpu tensors
            p_cpu = self.pointer_in_list * int64_byte + total_meta_len
            # The pointer on shipment on gpu, marking the reading position of gpu tensors
            p_gpu = self.cpu_shipment_start + p_cpu + total_cpu_tensor_len
            # The end of this package
            # total_gpu_tensor_len % 8 == 0, we do not need to use jump2patch
            p_end = p_gpu + total_gpu_tensor_len
            # The end position of meta information, which is also the start position of the cpu tensors
            p_meta_end = p_cpu
            # look into the shipment
            total_item_num = self._unbundling_single_int()
            for item_id in range(total_item_num):
                # name
                name_list = self._unbundling()
                name = byte_list2string(name_list)
                # data type
                dtype_id = self._unbundling_single_int()
                value_dtype = support_type[dtype_id]
                # shape
                # It supports scalar.
                # For example: torch.tensor([42]).view([]) == torch.tensor(42)
                shape = self._unbundling()
                # device
                on_cpu = self._unbundling_single_int()
                # data length
                data_len = self._unbundling_single_int()
                # p_gpu_end = p_gpu + data_len
                # data
                if on_cpu:
                    # on cpu
                    p_cpu_end = p_cpu + data_len
                    data = (
                        self.shipment_on_cpu[p_cpu:p_cpu_end]
                        .view(value_dtype)
                        .view(shape)
                    )
                    data_memory_start = self.cpu_shipment_start + p_cpu
                    data_memory_end = self.cpu_shipment_start + p_cpu_end
                    # Move the cpu pointer to the next cpu tensor
                    p_cpu = jump2patch(p_cpu_end)
                else:
                    # on cuda
                    p_gpu_end = p_gpu + data_len
                    data = self._buffer[p_gpu:p_gpu_end].view(value_dtype).view(shape).to(self._device)
                    data_memory_start = p_gpu
                    data_memory_end = p_gpu_end
                    # Move the gpu pointer to the next gpu tensor
                    p_gpu = jump2patch(p_gpu_end)
                # do something
                # Return False means keep going through the rest of the shipment
                # Return True means stopping looking for the item
                if do_something(name, data, data_memory_start, data_memory_end):
                    return

            assert (
                self.pointer_in_list * int64_byte == p_meta_end
            ), "expect self.pointer_in_list({}) * int64_byte({}) == p_meta_end({})".format(
                self.pointer_in_list, int64_byte, p_meta_end
            )
            assert p_gpu == p_end, "p_gpu ({}) must be equal to p_end ({})".format(
                p_gpu, p_end
            )
            # Move the meta pointer to the start of the next package
            self.pointer_in_list = (p_end - self.cpu_shipment_start) // int64_byte
            # Have not finish the package
            # and still need to take 3 integers from the cpu shipment
            if package_id < total_package_num - 1 and self.pointer_in_list + 3 > len(
                self.shipment_in_list
            ):
                self._deliver_to_host()

    def _unpack(self):
        def take_item_out(name, data, p_gpu, p_gpu_end):
            self._packed_items[name] = data
            self._pack_item2position[name] = [p_gpu, p_gpu_end]
            # Return False means keep going through the rest of the shipment
            return False

        self._look_through_items(take_item_out)

    # Make sure the item is packed before using this function
    def _remove_packed_item(self, item_name):
        self._warehouse.update(self._packed_items)
        self._warehouse.update(self._replaced_items)
        del self._warehouse[item_name]
        self._buffer = []
        self._packed_items.clear()
        self._replaced_items.clear()
        self._pack_item2position.clear()

    @staticmethod
    def from_buffer(buffer: torch.ByteTensor, device: torch.device=None):
        device = device or buffer.device
        shipment = Shipment(device, buffer_device=buffer.device)
        shipment._buffer = buffer
        shipment._unpack()
        return shipment

    def remove(self, item_name):
        if item_name in self._warehouse:
            del self._warehouse[item_name]
        else:
            assert (
                item_name in self._packed_items or item_name in self._replaced_items
            ), f"{item_name} not in shipment"
            self._remove_packed_item(item_name)

    def update(self, new_items: dict):
        if len(new_items) == 0:
            return
        for key, value in new_items.items():
            assert isinstance(
                value, torch.Tensor
            ), "parameter of {} must be a Tensor, but got a {}".format(key, type(value))
            assert (
                is_cpu(value.device.type) or value.device == self._device
            ), f"This shipment only accept tensors on cpu or {self._device} but gets {value.device}"

            not_found_item_with_same_meta = True
            # Packed or replaced
            for package in [self._packed_items, self._replaced_items]:
                if key in package:
                    old_value = package[key]
                    if (
                        old_value.dtype == value.dtype
                        and old_value.size() == value.size()
                        and old_value.device == value.device
                    ):
                        # Replace the item no matter value changed or not
                        # Different item with the same dtype, shape, device
                        if package == self._replaced_items:
                            # Found in the replaced items, directly replace
                            package[key] = value
                        else:
                            # Found in the packed items, move and replace it
                            del package[key]
                            self._replaced_items[key] = value

                        not_found_item_with_same_meta = False
                    else:
                        # Different item with same name, remove the old item first, add the new item later
                        self._remove_packed_item(key)

                    break

            if not_found_item_with_same_meta:
                # Add the new item
                self._warehouse[key] = value

    @property
    def buffer(self):
        self._pack()
        return self._buffer

    @property
    def content(self):
        return dict(self._packed_items, **self._warehouse, **self._replaced_items)

    def clear(self):
        self._warehouse.clear()
        self._buffer = []
        self._packed_items.clear()
        self._pack_item2position.clear()

    def is_empty(self) -> bool:
        return (
            len(self._warehouse) == 0
            and len(self._packed_items) == 0
            and len(self._replaced_items) == 0
        )


class PostalService:
    def __init__(self, parallel_ctx: ParallelContext):
        self.parallel_ctx = parallel_ctx
        self.send_recv_comm_device = self.parallel_ctx.send_recv_comm_device()

    def send_shipment(self, shipment: Shipment):
        # Pack items before shipping

        logger.info(f"{self.send_recv_comm_device=} is sending shape")
        self.parallel_ctx.send_to_next_stage(
            torch.tensor(
                [len(shipment.buffer)], device=self.send_recv_comm_device
            )
        )
        logger.info(f"{len(shipment.buffer)=} is sending")
        self.parallel_ctx.send_to_next_stage(shipment.buffer)
        #logger.info(f"Send: {shipment.buffer.shape=} is buffer")

    def recv_shipment(self) -> Shipment:
        logger.info(f"{self.send_recv_comm_device=} is receiving shape")
        shipment_volume = torch.empty(
            [1],
            dtype=torch.int64,
            device=self.send_recv_comm_device,
            requires_grad=False,
        )
        self.parallel_ctx.recv_from_prev_stage(shipment_volume)
        logger.info(f"{shipment_volume=} is receiving")
        
        buffer = torch.empty(
            shipment_volume.tolist(),
            dtype=byte_type,
            device=self.send_recv_comm_device,
            requires_grad=False,
        )
        self.parallel_ctx.recv_from_prev_stage(buffer)
        #logger.info(f"Receive: {buffer.shape=} is buffer")
        return Shipment.from_buffer(buffer, self.parallel_ctx.torch_device())

    def exchange_shipment(self, shipment: Shipment) -> Shipment:
        # Send and recv the length of shipment simultaneously
        recv_shipment_volume = torch.empty(
            [1],
            dtype=torch.int64,
            device=self.send_recv_comm_device,
            requires_grad=False,
        )
        send_shipment_volume = torch.tensor(
            [len(shipment.buffer)],
            dtype=torch.int64,
            device=self.send_recv_comm_device,
            requires_grad=False,
        )
        self.parallel_ctx.send_and_recv_between_neighborhoods(
            recv_shipment_volume, send_shipment_volume
        )
        # Send and recv shipment simultaneously
        recv_buffer = torch.empty(
            recv_shipment_volume.tolist(),
            dtype=byte_type,
            device=self.send_recv_comm_device,
            requires_grad=False,
        )
        send_buffer = shipment.buffer
        self.parallel_ctx.send_and_recv_between_neighborhoods(recv_buffer, send_buffer)
        return Shipment.from_buffer(recv_buffer, self.parallel_ctx.torch_device())

    def broadcast_shipment(self, shipment: Shipment) -> Shipment:
        if self.parallel_ctx.tensor_parallel_size == 1:
            return shipment

        # broadcast shipment volume
        shipment_volume = torch.tensor(len(shipment.buffer)).to(
            self.parallel_ctx.torch_device()
        )
        self.parallel_ctx.broadcast_in_tensor_parallel_group(shipment_volume)
        # broadcast shipment buffer
        if self.parallel_ctx.tensor_parallel_rank == 0:
            shipment_buffer = shipment.buffer
        else:
            shipment_buffer = torch.empty(
                shipment_volume,
                device=self.parallel_ctx.torch_device(),
                dtype=byte_type,
                requires_grad=False,
            )
        self.parallel_ctx.broadcast_in_tensor_parallel_group(shipment_buffer)

        if self.parallel_ctx.tensor_parallel_rank == 0:
            return shipment
        else:
            return Shipment.from_buffer(shipment_buffer)
