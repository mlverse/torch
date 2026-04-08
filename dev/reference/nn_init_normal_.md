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
#> -0.4603 -0.0726  1.3397 -0.8258 -0.3120
#>  1.0238 -0.7553  0.7671 -1.4030 -0.3022
#>  1.9689  0.3646 -0.0560 -1.3254  0.3247
#> [ CPUFloatType{3,5} ]
```
