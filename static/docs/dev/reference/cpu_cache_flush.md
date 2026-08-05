# Flush the CPU memory cache

Releases all cached CPU memory blocks back to the operating system. This
does not affect tensors that are currently in use — only blocks that
were previously freed and are being held in the cache for reuse.

## Usage

``` r
cpu_cache_flush()
```

## Details

Call this when you want to reduce memory usage, for example between a
training phase and an inference phase, or before allocating a large
non-torch object.

## See also

[`set_cpu_allocator_config()`](https://torch.mlverse.org/docs/dev/reference/set_cpu_allocator_config.md)
to configure cache behavior.
