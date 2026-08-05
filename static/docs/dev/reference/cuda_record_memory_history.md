# Enable Recording of Memory Allocation Stack Traces

Enables recording of stack traces associated with memory allocations,
allowing users to identify the source of memory allocation in CUDA
snapshots.

## Usage

``` r
cuda_record_memory_history(
  enabled,
  context = "all",
  stacks = "all",
  max_entries = 1
)
```

## Arguments

- enabled:

  Character or `NULL`. Controls memory history recording. Options:

  `NULL`

  :   Disable recording of memory history.

  `"state"`

  :   Record currently allocated memory information.

  `"all"`

  :   Record the history of all allocation and free events (default).

- context:

  Character or `NULL`. Controls traceback recording. Options:

  `NULL`

  :   Do not record any tracebacks.

  `"state"`

  :   Record tracebacks for currently allocated memory.

  `"alloc"`

  :   Record tracebacks for allocation events.

  `"all"`

  :   Record tracebacks for both allocation and free events (default).

- stacks:

  Character. Defines the stack trace frames to include. Options:

  `"all"`

  :   Include all frames (default).

- max_entries:

  Integer. The maximum number of allocation/free events to retain.

## Value

None; function invoked for side effects.

## Details

Alongside tracking stack traces for each current allocation and free
event, this function can also keep a historical log of all allocation
and free events.

Use
[`cuda_memory_snapshot()`](https://torch.mlverse.org/docs/dev/reference/cuda_memory_snapshot.md)
to retrieve recorded information. Visualization can be performed using
[pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz).

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
cuda_record_memory_history(enabled = 'all', context = 'all', stacks = 'all', max_entries = 10000)
} # }
}
```
