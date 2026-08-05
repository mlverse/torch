# Rrelu

Randomized leaky ReLU.

## Usage

``` r
nnf_rrelu(input, lower = 1/8, upper = 1/3, training = FALSE, inplace = FALSE)

nnf_rrelu_(input, lower = 1/8, upper = 1/3, training = FALSE)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- lower:

  lower bound of the uniform distribution. Default: 1/8

- upper:

  upper bound of the uniform distribution. Default: 1/3

- training:

  bool wether it's a training pass. DEfault: FALSE

- inplace:

  can optionally do the operation in-place. Default: FALSE
