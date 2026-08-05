# Configure the CPU memory allocator

Controls the CPU block cache that compensates for R's deferred garbage
collection. R doesn't free tensors immediately (unlike Python's
reference counting), so without caching, each operation allocates fresh
memory via expensive system calls. The cache reuses freed blocks,
significantly improving performance for training loops and repeated
computations.

## Usage

``` r
set_cpu_allocator_config(
  threshold_call_gc = 4000L,
  cache_enabled = TRUE,
  cache_max_size_mb = threshold_call_gc,
  cache_min_block_size = 1024L
)
```

## Arguments

- threshold_call_gc:

  Cumulative allocation threshold (in MB) that triggers R's garbage
  collector. Default: 4000 (4 GB).

- cache_enabled:

  Whether to enable the CPU block cache. Default: `TRUE`.

- cache_max_size_mb:

  Maximum total cached memory in MB. When the cache exceeds this limit,
  freed blocks are returned to the OS instead of cached. Default: same
  as `threshold_call_gc`.

- cache_min_block_size:

  Minimum block size (in bytes) to cache. Blocks smaller than this are
  handled by the system allocator directly. Default: 1024 (1 KB).

## Why a CPU cache is needed

Python frees tensors immediately via reference counting: when a variable
is reassigned, the old tensor's memory is returned to the system
allocator, which efficiently reuses it for the next allocation. R uses
garbage collection (GC) instead, meaning old tensors accumulate in
memory until GC runs. For large tensors (\> ~1 MB), the system allocator
uses `mmap`/`munmap` system calls, which are expensive when many dead
blocks are still mapped.

The CPU block cache solves this by intercepting freed blocks and holding
them in a fast in-process pool keyed by size. Subsequent allocations of
the same size skip the system allocator entirely and reuse cached
blocks. The cache is automatically flushed when R's GC is triggered by
the allocator, so memory is eventually returned to the OS.

## When to tune these settings

The defaults work well for most training workloads. Consider adjusting
if:

- You're running on a memory-constrained system: lower
  `cache_max_size_mb` to reduce memory retention.

- You're seeing out-of-memory errors: call
  [`cpu_cache_flush()`](https://torch.mlverse.org/docs/dev/reference/cpu_cache_flush.md)
  to release cached blocks, or lower `cache_max_size_mb`.

- You want to disable caching entirely for debugging: set
  `cache_enabled = FALSE`.

## Options

These settings can also be configured via R options before loading
torch:

- `torch.threshold_call_gc`

- `torch.cpu_cache_enabled`

- `torch.cpu_cache_max_size_mb`

- `torch.cpu_cache_min_block_size`

## See also

[`cpu_cache_flush()`](https://torch.mlverse.org/docs/dev/reference/cpu_cache_flush.md)
to manually release cached memory.
