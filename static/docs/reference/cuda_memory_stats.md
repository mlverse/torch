# Returns a dictionary of CUDA memory allocator statistics for a given device.

The return value of this function is a dictionary of statistics, each of
which is a non-negative integer.

## Usage

``` r
cuda_memory_stats(device = cuda_current_device())

cuda_memory_summary(device = cuda_current_device())
```

## Arguments

- device:

  Integer value of the CUDA device to return capabilities of.

## Core statistics

- "allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  number of allocation requests received by the memory allocator.

- "allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  amount of allocated memory.

- "segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  number of reserved segments from cudaMalloc().

- "reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  amount of reserved memory.

- "active.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  number of active memory blocks.

- "active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  amount of active memory.

- "inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  number of inactive, non-releasable memory blocks.

- "inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
  amount of inactive, non-releasable memory.

For these core statistics, values are broken down as follows.

Pool type:

- all: combined statistics across all memory pools.

- large_pool: statistics for the large allocation pool (as of October
  2019, for size \>= 1MB allocations).

- small_pool: statistics for the small allocation pool (as of October
  2019, for size \< 1MB allocations).

Metric type:

- current: current value of this metric.

- peak: maximum value of this metric.

- allocated: historical total increase in this metric.

- freed: historical total decrease in this metric.

## Additional metrics

- "num_alloc_retries": number of failed cudaMalloc calls that result in
  a cache flush and retry.

- "num_ooms": number of out-of-memory errors thrown.
