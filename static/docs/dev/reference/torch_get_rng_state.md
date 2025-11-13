# RNG state management

Low level functionality to set and change the RNG state. It's
recommended to use
[`torch_manual_seed()`](https://torch.mlverse.org/docs/dev/reference/torch_manual_seed.md)
for most cases.

## Usage

``` r
torch_get_rng_state()

torch_set_rng_state(state)

cuda_get_rng_state(device = NULL)

cuda_set_rng_state(state, device = NULL)
```

## Arguments

- state:

  A tensor with the current state or a list containing the state for
  each device - (for CUDA).

- device:

  The cuda device index to get or set the state. If `NULL` gets the
  state for all available devices.

## Functions

- `torch_set_rng_state()`: Sets the RNG state for the CPU

- `cuda_get_rng_state()`: Gets the RNG state for CUDA.

- `cuda_set_rng_state()`: Sets the RNG state for CUDA.
