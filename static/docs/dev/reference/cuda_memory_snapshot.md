# Capture CUDA Memory State Snapshot

Saves a snapshot of the CUDA memory state at the time it was called. The
resulting binary output is in pickle format and can be visualized using
the interactive snapshot viewer available at
[pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz).

## Usage

``` r
cuda_memory_snapshot()
```

## Value

Raw binary data representing the snapshot in pickle format.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
snapshot <- cuda_memory_snapshot()
} # }
}
```
