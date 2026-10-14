# Empty cache

Releases all unoccupied cached memory currently held by the caching
allocator so that those can be used in other GPU application and visible
in `nvidia-smi`.

## Usage

``` r
cuda_empty_cache()
```

## Note

`cuda_empty_cache()` doesn’t increase the amount of GPU memory available
for torch. However, it may help reduce fragmentation of GPU memory in
certain cases. See Memory management article for more details about GPU
memory management.
