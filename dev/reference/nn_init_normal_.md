# Normal initialization

Fills the input Tensor with values drawn from the normal distribution

## Usage

``` r
nn_init_normal_(tensor, mean = 0, std = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- mean:

  the mean of the normal distribution

- std:

  the standard deviation of the normal distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_normal_(w)
}
#> torch_tensor
#>  0.7149  0.1543 -0.0525 -0.8500 -1.1841
#> -0.2916  0.0597 -0.0982  0.6300  0.0994
#>  1.2370 -1.1706 -0.7220 -2.4324 -0.6581
#> [ CPUFloatType{3,5} ]
```
