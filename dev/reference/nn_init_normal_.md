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
#>  0.5254  1.6339 -1.0790  0.2694  2.1016
#> -1.9265  0.1392  0.3186 -0.6944  0.3353
#> -0.1361  0.3092  1.2665 -0.1792  1.0175
#> [ CPUFloatType{3,5} ]
```
