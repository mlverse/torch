# Save CUDA Memory State Snapshot to File

Calls
[`cuda_memory_snapshot()`](https://torch.mlverse.org/docs/dev/reference/cuda_memory_snapshot.md)
and saves the resulting binary snapshot to a specified file using
`writeBin`. The resulting file can be visualized using the interactive
snapshot viewer available at
[pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz).

## Usage

``` r
cuda_dump_memory_snapshot(filepath)
```

## Arguments

- filepath:

  Character; the path to the file where the snapshot will be saved.

## Value

None; snapshot is saved directly to the file.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
cuda_dump_memory_snapshot("snapshot.bin")
} # }
}
```
