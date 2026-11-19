# Sets the seed for generating random numbers.

Sets the seed for generating random numbers.

## Usage

``` r
torch_manual_seed(seed)

local_torch_manual_seed(seed, .env = parent.frame())

with_torch_manual_seed(code, ..., seed)
```

## Arguments

- seed:

  integer seed.

- .env:

  environment that will take the modifications from manual_seed.

- code:

  expression to run in the context of the seed

- ...:

  unused currently.

## Functions

- `local_torch_manual_seed()`: Modifies the torch seed in the
  environment scope.

- `with_torch_manual_seed()`: A with context to change the seed during
  the function execution.

## Note

Currently the `local_torch_manual_seed` and `with_torch_manual_seed`
won't work with Tensors in the MPS device. You can sample the tensors on
CPU and move them to MPS if reproducibility is required.
