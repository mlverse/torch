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
#>  0.7807  1.3855  0.9967 -0.5050  0.1181
#> -1.2707  0.1259 -1.5156  0.4365  0.5365
#> -1.0938 -0.9878 -0.5554  1.4905  0.2744
#> [ CPUFloatType{3,5} ]
```
