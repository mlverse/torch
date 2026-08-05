# Leaky_relu

Applies element-wise, \\LeakyReLU(x) = max(0, x) + negative_slope \*
min(0, x)\\

## Usage

``` r
nnf_leaky_relu(input, negative_slope = 0.01, inplace = FALSE)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- negative_slope:

  Controls the angle of the negative slope. Default: 1e-2

- inplace:

  can optionally do the operation in-place. Default: FALSE
