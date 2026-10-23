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
#> -0.1723 -1.3459 -0.4152 -1.4291 -0.2426
#> -0.9324 -0.3222 -0.4081 -0.7191  0.1844
#> -0.5836  0.4227  1.0624 -0.5659  0.3914
#> [ CPUFloatType{3,5} ]
```
