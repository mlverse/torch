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
#> -0.7648 -0.9943 -0.5423  1.3335  1.6922
#> -1.6981 -1.3253 -0.1057  1.1288 -0.8256
#> -0.3623 -0.0575  1.7201  0.5932  0.9368
#> [ CPUFloatType{3,5} ]
```
