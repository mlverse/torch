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
#> -1.4153 -0.7180 -1.3599  0.0546 -1.5598
#> -0.0817  0.7041 -0.0849  0.7229 -0.6368
#> -0.0643  1.9459 -1.5179  0.6895 -0.1274
#> [ CPUFloatType{3,5} ]
```
