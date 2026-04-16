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
#> -0.0104 -0.1521 -0.3241 -0.2120  0.9374
#> -0.1427 -0.9092 -0.1942  1.1288  0.7253
#> -0.1213  0.9259  1.0805  0.6749  0.8277
#> [ CPUFloatType{3,5} ]
```
