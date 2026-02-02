# Memory management

Differently from most R objects, tensors created in torch have their
memory managed by LibTorch allocators. This means that functions like
[`object.size()`](https://rdrr.io/r/utils/object.size.html) or
`lobstr::mem_used()` do **not** correctly report memory used.

The R garbage collector is very lazy, and is only called when R needs
more memory from the OS. Since R is not aware of large chunks of memory
that might be in use by torch tensors, it might not call the garbage
collector as often as it would if it knew that tensors are using more
memory. It’s common then that, even tensors that are no longer in use by
the R session are still alive in the R session (and thus using memory)
because they still din’t get garbage collected.

To solve this problem, the torch package has implemented strategies to
automatically call the R garbage collector when LibTorch is allocating
more memory. The strategies are different depending on where the memory
is being allocated: on the CPU, GPU (CUDA devices) or
[MPS](https://developer.apple.com/documentation/metalperformanceshaders)
(on Apple Silicon machines/ or equipped with AMD GPU’s).

## CPU

On the CPU, torch will possibly call the R garbage collector in two
moments:

1.  Every 4GB of memory allocated by LibTorch we make a call to the R
    garbage collector so it cleans up dangling tensors. The 4GB
    threshold can be controled by setting the option
    `torch.threshold_call_gc`, for example using:

        options(torch.threshold_call_gc = 4000)

    This option must be set before calling
    [`library(torch)`](https://torch.mlverse.org/docs) or calling any
    torch function for the first time, as this setting is applied when
    torch starts up.

2.  If torch fails allocating enough memory for for creating a new
    tensor, the garbage collector is called and the allocation is
    retried. Note: in some OS’s (specially the UNIX based) it’s very
    hard for an allocation to fail if it’s not too large, because the
    system tries to use swap. If too much swaping is used it’s possible
    that the system hangs completely.

    If your R session is hanging, and you are convinced that it should
    have enough memory for the operations, try setting a lower value for
    the `torch.threshold_call_gc` option, with this you will call the GC
    more often and make sure tensors are quickly released from memory.
    Note though, that calling the GC too often adds a lot of overhead,
    so this will probably slow down the program execution.

## CUDA

CUDA memory tends to be scarcer than CPU memory, also, allocation must
be faster otherwise allocation overhead can counterbalance the speed up
of GPU. To make allocations very fast and to avoid segmentation,
LibTorch uses a caching allocator to manage the GPU memory, ie. once
LibTorch allocated CUDA memory it won’t give it back to the operation
system, instead it reuses that memory for future allocations. This means
that `nvidia-smi` or `nvtop` will not report the amount of memory used
by tensors, but the memory LibTorch has reserved from the OS. You can
use
[`torch::cuda_memory_summary()`](https://torch.mlverse.org/docs/dev/reference/cuda_memory_stats.md)
to query exactly the memory used by LibTorch.

Like the CPU allocator, torch’s CUDA allocator will also call the R
garbage collector in some situations to cleanup tensors that might be
dangling. In torch’s implementation the R garbage collector is called
whenever reusing a cached block fails. In this case, GC is called and we
retry getting a new block. However, unlike in the CPU case, that
allocations failures are very rare, reusing a block is not common in
programs where LibTorch never reserves a large chunk of memory, causing
the GC to be called at almost every allocation (and calling GC is time
consuming).

To fix this, the torch allocator will call a faster GC in some moments
and make a full collection in others.

1.  We don’t make any collection if the current reserved memory (cached
    memory) divided by the total GPU memory is smaller than 20%. This
    can be controlled by the `torch.cuda_allocator_reserved_rate` and
    the default is 0.2.
2.  We make a full collection if the current allocated memory (memory
    used by tensors) divided by the total device memory is larger than
    80%. This can be controlled via the
    `torch.cuda_allocator_allocated_rate` and the default is 0.8.
3.  We make a full collection if the current allocated memory is larger
    divided by current reserved memory is larger than 80%. This is
    controlled by the `torch.cuda_allocator_allocated_reserved_rate` and
    the default is 0.8.
4.  In all other cases a light collection is made. Equivalent to calling
    `gc(full = FALSE`) in R.

These options can help tuning allocation performance depending on the
program you are running.

### CUDA Memory Snapshots

To assist debugging CUDA memory usage, R torch provides functionality
for generating CUDA memory snapshots, similar to the [PyTorch Python
implementation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html).
Snapshots record the state of allocated CUDA memory at any point in
time, and optionally records the history of allocation events that led
up to that snapshot.

To generate a snapshot:

    # Enable memory history recording, capturing tracebacks and allocation events
    cuda_record_memory_history(enabled = "all", max_entries = 1e6)

    # Run your code

    # Save a snapshot file
    cuda_dump_memory_snapshot("my_snapshot.pickle")

Generated snapshots can be visualized interactively using the official
PyTorch Memory Visualizer at
[pytorch.org/memory_viz](https://torch.mlverse.org/docs/dev/articles/pytorch.org/memory_viz).
Simply drag and drop your saved snapshot (.pickle) into the visualizer,
which runs locally in your browser without uploading any data.

### LibTorch CUDA Allocation Options

Besides the R specific options you can set LibTorch options via
environment variables as described below. The behavior of caching
allocator can be controlled via environment variable
PYTORCH_CUDA_ALLOC_CONF. The format is
`PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2><value2>..`.
Available options:

- `max_split_size_mb` prevents the allocator from splitting blocks
  larger than this size (in MB). This can help prevent fragmentation and
  may allow some borderline workloads to complete without running out of
  memory. Performance cost can range from ‘zero’ to ‘substatial’
  depending on allocation patterns. Default value is unlimited, i.e. all
  blocks can be split. The memory_stats() and memory_summary() methods
  are useful for tuning. This option should be used as a last resort for
  a workload that is aborting due to ‘out of memory’ and showing a large
  amount of inactive split blocks.
- `roundup_power2_divisions` helps with rounding the requested
  allocation size to nearest power-2 division and making better use of
  the blocks. In the current CUDACachingAllocator, the sizes are rounded
  up in multiple of blocks size of 512, so this works fine for smaller
  sizes. However, this can be inefficient for large near-by allocations
  as each will go to different size of blocks and re-use of those blocks
  are minimized. This might create lots of unused blocks and will waste
  GPU memory capacity. This option enables the rounding of allocation
  size to nearest power-2 division. For example, if we need to round-up
  size of 1200 and if number of divisions is 4, the size 1200 lies
  between 1024 and 2048 and if we do 4 divisions between them, the
  values are 1024, 1280, 1536, and 1792. So, allocation size of 1200
  will be rounded to 1280 as the nearest ceiling of power-2 division.
- `garbage_collection_threshold` helps actively reclaiming unused GPU
  memory to avoid triggering expensive sync-and-reclaim-all operation
  (release_cached_blocks), which can be unfavorable to latency-critical
  GPU applications (e.g., servers). Upon setting this threshold (e.g.,
  0.8), the allocator will start reclaiming GPU memory blocks if the GPU
  memory capacity usage exceeds the threshold (i.e., 80% of the total
  memory allocated to the GPU application). The algorithm prefers to
  free old & unused blocks first to avoid freeing blocks that are
  actively being reused. The threshold value should be between greater
  than 0.0 and less than 1.0.

Notice that the garbage collector refered below is not the R garbage
collector but LibTorch’s collector, that releases memory from the cache
to the OS.

## MPS

Memory management in MPS devices is very similar to the strategy used in
CUDA devices, except that here there’s currently no configuration or
tuning possible. The R garbage collector will be called whenever there’s
no more available memory for the GPU and thus, possibly deleting some
Tensors. Allocation is then retried and if it fails, a OOM error is
raised.
