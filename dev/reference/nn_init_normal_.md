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
#> -0.0464 -0.5554  0.3804  1.0657 -0.3335
#>  0.0940  0.0245  0.2999 -1.1642 -2.4909
#> -0.8044  0.2943 -1.9367  2.0579  1.0442
#> [ CPUFloatType{3,5} ]
```
