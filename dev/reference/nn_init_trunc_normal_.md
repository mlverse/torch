# Truncated normal initialization

Fills the input Tensor with values drawn from a truncated normal
distribution.

## Usage

``` r
nn_init_trunc_normal_(tensor, mean = 0, std = 1, a = -2, b = 2)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- mean:

  the mean of the normal distribution

- std:

  the standard deviation of the normal distribution

- a:

  the minimum cutoff value

- b:

  the maximum cutoff value

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_trunc_normal_(w)
}
#> torch_tensor
#> -1.1191 -0.2470 -0.9165  0.1052  0.4447
#> -0.1379  0.7516  0.5126 -0.9974 -1.7605
#> -0.0899 -0.7023 -1.6851 -1.0832 -1.2718
#> [ CPUFloatType{3,5} ]
```
