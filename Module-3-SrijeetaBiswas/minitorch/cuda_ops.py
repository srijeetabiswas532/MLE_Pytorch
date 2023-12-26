from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
        # print('you got here before matmul')
        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)  # converts ordinal into index for storage
            broadcast_index(out_index, out_shape, in_shape, in_index)  # broadcasting index from large shape to for the small shape

            st_pos_out = index_to_position(out_index, out_strides)
            st_pos_in = index_to_position(in_index, in_strides)

            out[st_pos_out] = fn(in_storage[st_pos_in])
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # shared local array
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # index
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # local index
    pos = cuda.threadIdx.x
    blockIdx = cuda.blockIdx.x

    # TODO: Implement for Task 3.3.
    # first input into shared array
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    # sync after a write
    cuda.syncthreads()

    if i < size:
        # skip size
        n = 1
        while n < BLOCK_DIM:
            # adding pairs of elements
            if pos % (2 * n) == 0:
                cache[pos] += cache[pos + n]
                cuda.syncthreads()
            # halves distance between pairs
            n *= 2
    if pos == 0:
        # writes final result
        out[blockIdx] = cache[0]
    # raise NotImplementedError("Need to implement for Task 3.3")


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # number of threads in a CUDA block
        BLOCK_DIM = 1024
        # shared memory
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        # local memory
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # position of block
        out_pos = cuda.blockIdx.x
        # position of thread within block
        pos = cuda.threadIdx.x

        # initialize with reduce_val provided in block shared array
        cache[pos] = reduce_value

        # check if current block is within bounds of input size
        if out_pos < out_size:
            # converts index to storage index
            to_index(out_pos, out_shape, out_index)
            # position in tensor
            pos2 = index_to_position(out_index, out_strides)
            # extracts value of index from out index array
            idx_to_be_reduced = out_index[reduce_dim]
            # global index for the thread where pos is the position of the thread within the block
            global_idx = pos + idx_to_be_reduced * BLOCK_DIM

            out_index[reduce_dim] = global_idx

            # check if global index is within valid bounds of the tensor being reduced
            if global_idx < a_shape[reduce_dim]:
                # loading corresponding element from input tensor into cache
                a_pos = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[a_pos]
                cuda.syncthreads()

                n = 1
                while n < BLOCK_DIM:
                    # we can perform parallel reduction by skipping by half each time
                    # adding pairs of elements
                    if pos % (2 * n) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + n])
                        cuda.syncthreads()
                    # halves distance between pairs
                    n *= 2
            # first thread in the block usually writes final result back to output tensor
            if pos == 0:
                # writes final result
                out[pos2] = cache[0]

        # TODO: Implement for Task 3.3.

        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # shared memory per tensor
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # 2D positions of threads
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # local indices
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    # valid bounds
    if i >= 0 and i < size and j >= 0 and j < size:
        # updating shared memory from storages
        # stride is (size, 1)
        a_shared[local_i, local_j] = a[i * size + j]
        b_shared[local_i, local_j] = b[i * size + j]
        cuda.syncthreads()

        dp = 0.0
        # calculating dot product
        for m in range(size):
            dp += a_shared[local_i, m] * b_shared[m, local_j]
        out[i * size + j] = dp
    # raise NotImplementedError("Need to implement for Task 3.3")


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    # shared memories
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    # position of threads
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    a_shared[pi, pj] = 0
    b_shared[pi, pj] = 0
    cuda.syncthreads()
    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4

    # move across shared dim by block dim
    dp = 0.0
    for block in range(0, a_shape[-1], BLOCK_DIM):
        # copy shared memory for a matrix
        k_a = block + pj  # global col idx
        # global row idx
        if i < a_shape[1] and k_a < a_shape[2]:
            a_lin = (a_strides[1] * i) + (a_strides[2] * k_a) + (batch * a_batch_stride)
            a_shared[pi, pj] = a_storage[a_lin]

        # copy shared memory for b matrix
        # global row idx
        k_b = block + pi
        if j < b_shape[2] and k_b < b_shape[1]:
            b_lin = (b_strides[1] * k_b) + (b_strides[2] * j) + (batch * b_batch_stride)
            b_shared[pi, pj] = b_storage[b_lin]

        # sync threads
        cuda.syncthreads()
        # compute the dot product for position c[i, j]
        for k in range(BLOCK_DIM):
            # global batch index
            k_k = block + k
            # shared dimension between a and b
            if k_k < a_shape[-1]:
                dp += a_shared[pi, k] * b_shared[k, pj]
    if i < out_shape[1] and j < out_shape[2]:
        out_lin = (out_strides[1] * i) + (out_strides[2] * j) + (out_strides[0] * batch)
        # one global write
        out[out_lin] = dp
    # raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
